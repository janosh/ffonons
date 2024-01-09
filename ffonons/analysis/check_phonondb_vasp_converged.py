"""Extract VASP calculation parameters from phononDB and check convergence."""


# %%
import lzma
import os
import tarfile
import warnings
import zipfile
from collections.abc import Sequence
from multiprocessing import Pool

import numpy as np
import pandas as pd
import yaml
from pymatgen.analysis.magnetism.analyzer import DEFAULT_MAGMOMS
from pymatgen.core import Structure
from pymatgen.io.vasp import Incar, Kpoints
from pymatgen.io.vasp.sets import MPRelaxSet, MPStaticSet
from tqdm import tqdm

from ffonons import DATA_DIR

# %%
warnings.filterwarnings("ignore")

__author__ = "Aakash Naik"
__date__ = "2024-01-09"

db_name = "phonon-db"
ph_docs_dir = f"{DATA_DIR}/{db_name}"

# Change the parent directory where all zip files from togoDB are downloaded
os.chdir(ph_docs_dir)

dbs_files = [file for file in os.listdir() if not file.endswith(("csv", "xlsx"))]


def get_density_from_kmesh(mesh: Sequence[int], struct: Structure) -> float:
    """Get kpoints grid density using a kpoints mesh.

    Args:
        mesh (tuple): The kpoint mesh as a tuple of three integers.
        struct (pymatgen.Structure): The pymatgen structure object.

    Returns:
        float: kpoints grid density averaged over dimensions.
    """
    rec_cell = struct.lattice.reciprocal_lattice.matrix
    density = np.matrix([np.linalg.norm(b) / np.array(mesh) for b in rec_cell])

    return np.average(density)


def get_mp_kppa_kppvol_from_mesh(
    mesh: Sequence[int], struct: Structure, default_grid: float
) -> tuple[float, float, float]:
    """Get materials project kpoints grid density using a kpoints mesh.

    Args:
        mesh (list): The kpoint mesh.
        struct (pymatgen.Structure): The pymatgen structure object.
        default_grid (float): The default materials project kpoints grid density.

    Returns:
        tuple[float, float, float]:
            - kpoint grid density (kppa)
            - the kpoints per volume (kppvol)
            - reference kpoints grid density (kppa_ref)
    """
    vol = struct.lattice.reciprocal_lattice.volume  # reciprocal volume
    lattice = struct.lattice
    lengths = lattice.abc  # a, b, c lengths

    mult = np.mean([a * b for a, b in zip(mesh, lengths, strict=False)])
    real_vol = lengths[0] * lengths[1] * lengths[2]
    n_grid_magnitude = mult**3 / (real_vol)

    kpts_per_atom = int(round(n_grid_magnitude * len(struct)))
    kpts_per_vol = int(round(kpts_per_atom / (vol * len(struct))))

    if default_grid is not None:
        kpts_per_atom_ref = int(default_grid * vol * len(struct))
    else:
        kpts_per_atom_ref = None

    return kpts_per_atom, kpts_per_vol, kpts_per_atom_ref


def get_comp_calc_params(file: str) -> pd.DataFrame:
    """Extract calculation parameters for a given database file."""
    zip_file_path = file
    mpid = zip_file_path.split("-")[0]

    df = pd.DataFrame(index=[mpid])
    df["Composition"] = ""

    # Specify the name of the tar.lzma file inside the zip archive
    tar_lzma_file_name = "vasp-settings.tar.lzma"
    # Specify the path to your xz file
    xz_file_path = "phonopy_params.yaml.xz"

    # Open the zip file
    # Read the contents of the tar.lzma file
    # Open the xz archive from the lzma file
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref, zip_ref.open(
        xz_file_path, "r"
    ) as lzma_file, lzma.open(lzma_file, "rb") as lzma_file_content:
        content = lzma_file_content.read()
        phonopy_yaml = yaml.safe_load(content.decode("utf-8"))

        # get sc matrix
        sc_mat = phonopy_yaml["supercell_matrix"]

    # Open zip and tar.lzma files
    # Open the tar archive from the lzma file
    with zipfile.ZipFile(zip_file_path) as zip_ref, zip_ref.open(
        tar_lzma_file_name
    ) as lzma_file, lzma.open(lzma_file, "rb") as lzma_file_content, tarfile.open(
        fileobj=lzma_file_content, mode="r"
    ) as tar:
        # List the contents of the tar archive
        file_list = tar.getnames()
        # extract potcar settings
        user_potcar_settings = {}
        potcar_title = []
        for file_name in file_list:
            if "PAW_dataset.txt" in file_name:
                paw_data = tar.extractfile(file_name)
                content = paw_data.readlines()
                for con in content:
                    element = con.decode("utf-8").strip().split("</c><c>")[1].strip()
                    for data in con.decode("utf-8").strip().split("</c><c>"):
                        if "PAW" in data:
                            potcar_title.append(
                                data.replace("</c>", "").replace("</rc>", "").strip()
                            )
                            potcar_used = (
                                data.replace("</c>", "")
                                .replace("</rc>", "")
                                .strip()
                                .split(" ")[1]
                                .strip()
                            )
                            user_potcar_settings[element] = potcar_used

        # Read specific files from the tar archive without extracting them
        poscar_file_name = "vasp-settings/POSCAR-unitcell"
        poscar_file = tar.extractfile(poscar_file_name)
        content = poscar_file.read().decode("utf-8")
        poscar = Structure.from_str(content, fmt="poscar")
        df.loc[
            mpid, "Reduced_formula"
        ] = poscar.composition.get_reduced_formula_and_factor()[0]
        df.loc[mpid, "Composition"] = poscar.composition

        for el in poscar.composition.chemical_system.split("-"):
            if el in DEFAULT_MAGMOMS:
                df.loc[mpid, "Require_U_correction"] = True
            else:
                df.loc[mpid, "Require_U_correction"] = False

        mp_relax = MPRelaxSet(
            structure=poscar, user_potcar_settings=user_potcar_settings
        )
        df["mp_relax_kpts"] = ""
        df.loc[mpid, "mp_relax_kpts"] = mp_relax.kpoints.kpts[0]

        # get mp kpoint grid density
        kppa, kpt_per_vol, kppa_ref = get_mp_kppa_kppvol_from_mesh(
            struct=poscar, mesh=mp_relax.kpoints.kpts[0], default_grid=64
        )

        df.loc[mpid, "mp_relax_grid_desity"] = kppa
        df.loc[mpid, "mp_relax_reciprocal_density"] = kpt_per_vol

        # aiida k-point density
        kpt_dens = get_density_from_kmesh(mesh=mp_relax.kpoints.kpts[0], struct=poscar)
        df.loc[mpid, "mp_relax_grid_desity_aiida"] = kpt_dens

        df.loc[mpid, "mp_default_kpoint_grid_density"] = kppa_ref

        # get supercell structure
        sc = poscar.copy().make_supercell(scaling_matrix=sc_mat)

        # mp-static for sc
        mp_static = MPStaticSet(structure=sc, user_potcar_settings=user_potcar_settings)
        df["mp_static_kpts_sc"] = ""
        df.loc[mpid, "mp_static_kpts_sc"] = mp_static.kpoints.kpts[0]

        # get mp kpoint grid density
        kppa_sc, kppvol_sc, kppa_ref_sc = get_mp_kppa_kppvol_from_mesh(
            struct=sc, mesh=mp_static.kpoints.kpts[0], default_grid=100
        )

        df.loc[mpid, "mp_static_grid_desity_sc"] = kppa_sc
        df.loc[mpid, "mp_static_reciprocal_density_sc"] = kppvol_sc

        # aiida k-point density
        kpt_dens = get_density_from_kmesh(mesh=mp_static.kpoints.kpts[0], struct=sc)
        df.loc[mpid, "mp_static_grid_desity_aiida_sc"] = kpt_dens

        # kppa_ref_sc = int(100 * vol_sc * len(sc))
        df.loc[mpid, "mp_default_kpoint_grid_density_sc_static"] = kppa_ref_sc

        # mp-relax for sc
        mp_relax_sc = MPRelaxSet(
            structure=sc, user_potcar_settings=user_potcar_settings
        )
        df["mp_relax_kpts_sc"] = ""
        df.loc[mpid, "mp_relax_kpts_sc"] = mp_relax_sc.kpoints.kpts[0]

        # get mp kpoint grid density
        kppa_sc, kppvol_sc, kppa_ref_sc = get_mp_kppa_kppvol_from_mesh(
            struct=sc, mesh=mp_relax_sc.kpoints.kpts[0], default_grid=64
        )

        df.loc[mpid, "mp_relax_grid_desity_sc"] = kppa_sc
        df.loc[mpid, "mp_relax_reciprocal_density_sc"] = kppvol_sc

        # aiida k-point density
        kpt_dens = get_density_from_kmesh(mesh=mp_relax_sc.kpoints.kpts[0], struct=sc)
        df.loc[mpid, "mp_relax_grid_desity_aiida_sc"] = kpt_dens

        # kppa_ref_sc = int(64 * vol_sc * len(sc))
        df.loc[mpid, "mp_default_kpoint_grid_density_sc_relax"] = kppa_ref_sc

        # check for potcars title match
        mp_set_potcar = [potcar.TITEL for potcar in mp_static.potcar]
        if sorted(mp_set_potcar) != sorted(potcar_title):
            raise ValueError(
                f"POTCARs do not match: {mp_set_potcar=} vs togo={potcar_title}"
            )

        df["potcar_ENMAX"] = np.nan
        df.loc[mpid, "potcar_ENMAX"] = int(
            max([potcar.ENMAX for potcar in mp_static.potcar])
        )
        df["potcar_1.3_ENMAX"] = np.nan
        df.loc[mpid, "potcar_1.3_ENMAX"] = int(
            1.3 * max([potcar.ENMAX for potcar in mp_static.potcar])
        )

        # extract togo calc parameters
        for file_name in file_list:
            if "INCAR" in file_name:
                incar_file = tar.extractfile(file_name)
                content = incar_file.read().decode("utf-8")
                incar = Incar.from_str(content)
                name = file_name.split("/")[-1]

                df.loc[mpid, f"{name}_ENCUT"] = incar.get("ENCUT")

                if incar.get("ISPIN"):
                    df.loc[mpid, f"{name}_Magnetization"] = True
                else:
                    df.loc[mpid, f"{name}_Magnetization"] = False
            if "KPOINTS" in file_name:
                kpoint_file = tar.extractfile(file_name)
                content = kpoint_file.read().decode("utf-8")
                kpoint = Kpoints.from_str(content)
                name = file_name.split("/")[-1]

                df[f"{name}_kpts"] = ""

                df.loc[mpid, f"{name}_kpts"] = kpoint.kpts[0]

                if "force" in name:
                    # get mp kpoint grid density from togo k-points for sc
                    kppa_sc, kppvol_sc, _ = get_mp_kppa_kppvol_from_mesh(
                        struct=sc, mesh=kpoint.kpts[0], default_grid=None
                    )

                    df.loc[mpid, f"{name}_grid_desity_sc"] = kppa_sc
                    df.loc[mpid, f"{name}_reciprocal_density_sc"] = kppvol_sc

                    # aiida k-point density
                    kpt_dens = get_density_from_kmesh(mesh=kpoint.kpts[0], struct=sc)
                    df.loc[mpid, f"{name}_grid_desity_aiida_sc"] = kpt_dens

                else:
                    # get mp kpoint grid density from togo k-points for unit cell
                    kppa, kpt_per_vol, _ = get_mp_kppa_kppvol_from_mesh(
                        struct=poscar,
                        mesh=kpoint.kpts[0],
                        default_grid=None,
                    )

                    df.loc[mpid, f"{name}_grid_desity"] = kppa
                    df.loc[mpid, f"{name}_reciprocal_density"] = kpt_per_vol

                    # aiida k-point density
                    kpt_dens = get_density_from_kmesh(
                        mesh=kpoint.kpts[0], struct=poscar
                    )
                    df.loc[mpid, f"{name}_grid_desity_aiida"] = kpt_dens

    return df


# %%
rows = []
with Pool(processes=4, maxtasksperchild=1) as pool, tqdm(
    total=len(dbs_files), desc="Extracting Calc data"
) as pbar:
    for result in pool.imap_unordered(get_comp_calc_params, dbs_files, chunksize=1):
        pbar.update()
        rows.append(result)

df = pd.concat(rows)
df = df.sort_index()

df.to_csv("togo-vasp-params.csv")
