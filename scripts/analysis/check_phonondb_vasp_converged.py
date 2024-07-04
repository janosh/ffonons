"""Extract VASP calculation parameters from Togo phononDB and check convergence to
ensure phononDB serves as sufficiently accurate reference data for a reliable MLFF
phonon benchmark.
"""

# %%
import json
import lzma
import os
import tarfile
import warnings
import zipfile
from collections.abc import Sequence
from glob import glob

import numpy as np
import pandas as pd
import yaml
from monty.json import MontyEncoder
from pymatgen.core import Structure
from pymatgen.entries.compatibility import needs_u_correction
from pymatgen.io.vasp import Incar, Kpoints
from pymatgen.io.vasp.sets import BadInputSetWarning, MPRelaxSet, MPStaticSet
from pymatviz.enums import Key
from tqdm import tqdm

from ffonons import DATA_DIR
from ffonons.enums import DB

__author__ = "Aakash Naik, Janosh Riebesell"
__date__ = "2024-01-09"

# suppress "BadInputSetWarning: POTCAR data with symbol P is not known by pymatgen..."
# and UserWarning: No Pauling electronegativity for Ne. Setting to NaN.
for category in (UserWarning, BadInputSetWarning):
    warnings.filterwarnings(action="ignore", category=category, module="pymatgen")

csv_path = f"{DATA_DIR}/{DB.phonon_db}/togo-vasp-params.csv.bz2"


# %%
def get_density_from_kmesh(mesh: Sequence[int], struct: Structure) -> float:
    """Get kpoints grid density using a kpoints mesh.

    Args:
        mesh (tuple): The kpoint mesh as a tuple of three integers.
        struct (pymatgen.Structure): The pymatgen structure object.

    Returns:
        float: kpoints grid density averaged over dimensions.
    """
    recip_cell = struct.lattice.reciprocal_lattice.matrix
    density = np.matrix([np.linalg.norm(vec) / np.array(mesh) for vec in recip_cell])

    return density.mean()


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


def get_vasp_calc_params(zip_file_path: str) -> dict:
    """Extract calculation parameters for a given database file.

    Args:
        zip_file_path (str): The database file path.

    Returns:
        dict: DFT calculation parameters.
    """
    params = {}
    try:  # bail on corrupted ZIP files
        zip_ref = zipfile.ZipFile(zip_file_path)
    except zipfile.BadZipFile:
        print(f"Corrupted file: {zip_file_path!r}, deleting...")
        os.remove(zip_file_path)
        return None, None

    # get supercell matrix from phonopy_params.yaml.xz
    with (
        zip_ref.open("phonopy_params.yaml.xz", mode="r") as lzma_file,
        lzma.open(lzma_file, "rb") as lzma_file_content,
    ):
        phonopy_yaml = yaml.safe_load(lzma_file_content.read().decode("utf-8"))

        supercell_mat = phonopy_yaml["supercell_matrix"]

    # involved process to get at the calc params:
    # 1. open the zip file
    # 2. read the contents of the vasp-settings.tar.lzma file
    # 3. open the phonopy_params.yaml.xz archive from the lzma file
    with (
        zip_ref.open("vasp-settings.tar.lzma") as lzma_file,
        lzma.open(lzma_file, "rb") as lzma_file_content,
        tarfile.open(fileobj=lzma_file_content, mode="r") as tar,
    ):
        file_list = tar.getnames()  # list contents of the tar archive
        user_potcar_settings: dict[str, str] = {}  # extract POTCAR settings
        potcar_title = []
        for file_name in file_list:
            if "PAW_dataset.txt" not in file_name:
                continue
            paw_lines = tar.extractfile(file_name).readlines()
            for con in paw_lines:
                element = con.decode("utf-8").strip().split("</c><c>")[1].strip()
                for data in con.decode("utf-8").strip().split("</c><c>"):
                    if "PAW" not in data:
                        continue
                    potcar_title.append(
                        data.replace("</c>", "").replace("</rc>", "").strip()
                    )
                    potcar_used = data.replace("</c>", "").replace("</rc>", "")
                    user_potcar_settings[element] = (
                        potcar_used.strip().split(" ")[1].strip()
                    )

        # Read specific files from the tar archive without extracting them
        poscar_file_name = "vasp-settings/POSCAR-unitcell"
        poscar_file = tar.extractfile(poscar_file_name)
        paw_lines = poscar_file.read().decode("utf-8")
        struct = Structure.from_str(paw_lines, fmt="poscar")
        params[Key.reduced_formula] = (
            struct.composition.get_reduced_formula_and_factor()[0]
        )
        params[Key.formula] = struct.formula

        params["requires_u_correction"] = bool(needs_u_correction(struct.composition))

        mp_relax = MPRelaxSet(
            structure=struct, user_potcar_settings=user_potcar_settings
        )
        params["mp_relax_kpts"] = mp_relax.kpoints.kpts[0]

        # get MP kpoint grid density
        kppa, kpt_per_vol, kppa_ref = get_mp_kppa_kppvol_from_mesh(
            struct=struct, mesh=mp_relax.kpoints.kpts[0], default_grid=64
        )

        params["mp_relax_grid_density"] = kppa
        params["mp_relax_reciprocal_density"] = kpt_per_vol

        # AiiDA k-point density
        kpt_dens = get_density_from_kmesh(mesh=mp_relax.kpoints.kpts[0], struct=struct)
        params["mp_relax_grid_density_aiida"] = kpt_dens

        params["mp_default_kpoint_grid_density"] = kppa_ref

        # get supercell structure
        supercell = struct.copy().make_supercell(scaling_matrix=supercell_mat)

        # mp-static for supercell
        mp_static = MPStaticSet(
            structure=supercell, user_potcar_settings=user_potcar_settings
        )
        params["mp_static_kpts_sc"] = mp_static.kpoints.kpts[0]

        # get MP kpoint grid density
        kppa_sc, kppvol_sc, kppa_ref_sc = get_mp_kppa_kppvol_from_mesh(
            struct=supercell, mesh=mp_static.kpoints.kpts[0], default_grid=100
        )

        params["mp_static_grid_density_sc"] = kppa_sc
        params["mp_static_reciprocal_density_sc"] = kppvol_sc

        # AiiDA k-point density
        kpt_dens = get_density_from_kmesh(
            mesh=mp_static.kpoints.kpts[0], struct=supercell
        )
        params["mp_static_grid_density_aiida_sc"] = kpt_dens

        params["mp_default_kpoint_grid_density_sc_static"] = kppa_ref_sc

        # mp-relax for supercell
        mp_relax_sc = MPRelaxSet(
            structure=supercell, user_potcar_settings=user_potcar_settings
        )
        params["mp_relax_kpts_sc"] = mp_relax_sc.kpoints.kpts[0]

        # get MP kpoint grid density
        kppa_sc, kppvol_sc, kppa_ref_sc = get_mp_kppa_kppvol_from_mesh(
            struct=supercell, mesh=mp_relax_sc.kpoints.kpts[0], default_grid=64
        )

        params["mp_relax_grid_density_sc"] = kppa_sc
        params["mp_relax_reciprocal_density_sc"] = kppvol_sc

        # AiiDA k-point density
        kpt_dens = get_density_from_kmesh(
            mesh=mp_relax_sc.kpoints.kpts[0], struct=supercell
        )
        params["mp_relax_grid_density_aiida_sc"] = kpt_dens

        params["mp_default_kpoint_grid_density_sc_relax"] = kppa_ref_sc

        # check for potcars title match
        mp_set_potcar = [potcar.TITEL for potcar in mp_static.potcar]
        if sorted(mp_set_potcar) != sorted(potcar_title):
            raise ValueError(
                f"POTCARs do not match: {mp_set_potcar=} vs togo={potcar_title}"
            )

        params["potcar_enmax"] = int(max(potcar.ENMAX for potcar in mp_static.potcar))
        params["potcar_1.3_enmax"] = int(
            1.3 * max(potcar.ENMAX for potcar in mp_static.potcar)
        )

        # extract togo calc parameters
        for file_name in file_list:
            if "INCAR" in file_name:
                incar_file = tar.extractfile(file_name)
                incar = Incar.from_str(incar_file.read().decode("utf-8"))
                name = file_name.split("/")[-1]

                params[f"{name}_encut"] = incar.get("ENCUT")
                params[f"{name}_magnetization"] = bool(incar.get("ISPIN"))

            if "KPOINTS" in file_name:
                kpoint_file = tar.extractfile(file_name)
                kpoint = Kpoints.from_str(kpoint_file.read().decode("utf-8"))
                name = file_name.split("/")[-1]

                params[f"{name}_kpts"] = kpoint.kpts[0]

                if "force" in name:
                    # get MP kpoint grid density from togo k-points for supercell
                    kppa_sc, kppvol_sc, _ = get_mp_kppa_kppvol_from_mesh(
                        struct=supercell, mesh=kpoint.kpts[0], default_grid=None
                    )

                    params[f"{name}_grid_density_sc"] = kppa_sc
                    params[f"{name}_reciprocal_density_sc"] = kppvol_sc

                    # AiiDA k-point density
                    kpt_dens = get_density_from_kmesh(
                        mesh=kpoint.kpts[0], struct=supercell
                    )
                    params[f"{name}_grid_density_aiida_sc"] = kpt_dens

                else:
                    # get MP kpoint grid density from togo k-points for unit cell
                    kppa, kpt_per_vol, _ = get_mp_kppa_kppvol_from_mesh(
                        struct=struct, mesh=kpoint.kpts[0], default_grid=None
                    )

                    params[f"{name}_grid_density"] = kppa
                    params[f"{name}_reciprocal_density"] = kpt_per_vol

                    # AiiDA k-point density
                    kpt_dens = get_density_from_kmesh(
                        mesh=kpoint.kpts[0], struct=struct
                    )
                    params[f"{name}_grid_density_aiida"] = kpt_dens

    return params, struct


# %%
# directory to where Togo DB ZIP files were downloaded
ph_docs_dir = f"{DATA_DIR}/{DB.phonon_db}"
db_files = sorted(  # sort by MP ID
    glob(f"{ph_docs_dir}/*.zip"), key=lambda path: int(path.split("-")[-3])
)
print(f"found {len(db_files)=:,}")
mp_togo_id_map = {f'mp-{file.split("-")[-3]}': file.split("-")[-2] for file in db_files}

params = locals().get("results", {})  # prevent accidental results overwrite
structures = locals().get("structures", {})  # prevent accidental results overwrite

for file in tqdm(db_files):
    mp_id = f'mp-{file.split("-")[-3]}'
    if mp_id in params and mp_id in structures:
        continue
    params, struct = get_vasp_calc_params(file)
    if params and struct:
        params[mp_id], structures[mp_id] = params, struct


# %%
df_params = pd.DataFrame(params).T.sort_index().convert_dtypes().round(5)

# Togo DB phonons were calculated without magnetism or U-corrections (check this by
# ensuring all ISPIN values are False). any offending materials should be excluded from
# the analysis.
assert sum(df_params.filter(like="magnetization").sum()) == 0
assert not any(df_params[Key.needs_u_correction])

df_params.to_csv(csv_path)


# %%
with lzma.open(f"{DATA_DIR}/{DB.phonon_db}/structures.json.lzma", "wb") as lzma_file:
    json_str = json.dumps(structures, cls=MontyEncoder)
    lzma_file.write(json_str.encode("utf-8"))


# %% load CSV file
df_params = pd.read_csv(csv_path, index_col=Key.mat_id)
