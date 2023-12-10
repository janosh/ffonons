"""This model defines utility functions for loading existing DFT or ML phonon
band structures and DOSs.
"""
import gzip
import json
import lzma
import os
import re
from collections import defaultdict
from collections.abc import Sequence
from glob import glob
from pathlib import Path
from typing import Any, Literal
from zipfile import ZipFile

import numpy as np
import phonopy
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phonopy.units import VaspToTHz
from pymatgen.core import Structure
from pymatgen.io.phonopy import (
    get_ph_bs_symm_line,
    get_ph_dos,
    get_pmg_structure,
)
from pymatgen.io.vasp import Kpoints
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos
from tqdm import tqdm

from ffonons import DATA_DIR, FIGS_DIR, bs_key, dos_key, formula_key, id_key
from ffonons.dbs.phonondb import get_phonopy_kpath

__author__ = "Janine George, Aakash Naik, Janosh Riebesell"
__date__ = "2023-11-24"


def load_pymatgen_phonon_docs(
    which_db: Literal["mp", "phonon_db"],
) -> dict[str, dict[str, dict]]:
    """Load existing DFT/ML phonon band structure and DOS docs from disk for a
    specified database.

    Args:
        which_db ("mp" | "phonon_db"): Which database to load docs from.

    Returns:
        dict[str, dict[str, dict]]: Outer key is material ID, 2nd-level key is the
            model name, and third-level key is either "phonon_dos" or
            "phonon_bandstructure".
    """
    mp_docs_paths = glob(f"{DATA_DIR}/{which_db}/*.json.gz")
    mp_docs = defaultdict(dict)

    for doc_path in tqdm(mp_docs_paths, desc=f"Loading {which_db} docs"):
        with gzip.open(doc_path, "rt") as file:
            doc_dict = json.load(file)

        doc_dict[dos_key] = PhononDos.from_dict(doc_dict[dos_key])
        doc_dict[bs_key] = PhononBandStructureSymmLine.from_dict(doc_dict[bs_key])

        mp_id, formula, model = re.search(
            rf".*/{which_db}/(mp-\d+)-([A-Z][^-]+)-(.*).json.gz", doc_path
        ).groups()
        assert mp_id.startswith("mp-"), f"Invalid {mp_id=}"

        mp_docs[mp_id][model] = doc_dict | {
            formula_key: formula,
            id_key: mp_id,
            "dir_path": os.path.dirname(doc_path),
        }

    return mp_docs


def parse_phonondb_docs(
    phonopy_yaml: str | None = None,
    out_dir: str = FIGS_DIR,
    supercell: list[int] = (2, 2, 2),
    primitive_matrix: str | list[list[float]] = "auto",
    nac: bool = True,
    poscar: str = "POSCAR",
    force_sets: str = "FORCE_SETS",
    born: str = "BORN",
    code: str = "vasp",
    kpath_scheme: str = "seekpath",
    symprec: float = 1e-5,
    **kwargs: Any,
) -> dict[str, Any]:
    """Get phonon data from phonopy and save it to disk. Returns the structure and the
    phonon data as dict.

    Args:
        phonopy_yaml (str, optional): path to phonopy yaml file. Defaults to None.
        out_dir (str, optional): path to output directory. Defaults to FIGS_DIR.
        supercell (list[int], optional): supercell matrix. Defaults to (2, 2, 2).
        primitive_matrix (str | list[list[float]], optional): primitive matrix.
            Defaults to "auto".
        nac (bool, optional): whether to use non-analytical corrections.
            Defaults to True.
        poscar (str, optional): path to POSCAR file. Defaults to "POSCAR".
        force_sets (str, optional): path to FORCE_SETS file. Defaults to "FORCE_SETS".
        born (str, optional): path to BORN file. Defaults to "BORN".
        code (str, optional): code used for phonon calculations. Defaults to "vasp".
        kpath_scheme (str, optional): kpath scheme. Defaults to "seekpath".
            Important to use "seekpath" if not using primitive cell as input!
        symprec (float, optional): precision for symmetry determination.
            Defaults to 1e-5.
        **kwargs: additional parameters that can be passed to this method as a dict

    Returns:
        dict: phonon data
    """
    os.makedirs(out_dir, exist_ok=True)
    if code == "vasp":
        factor = VaspToTHz
    if phonopy_yaml.lower().endswith(".zip"):
        # open zip archive and only read the phonopy_params.yaml.xz file from it
        with ZipFile(phonopy_yaml, "r") as zip_file:
            yaml_xz = zip_file.open("phonopy_params.yaml.xz")
            # wrap uncompressed yaml.xz file content with io.IOBase object
            # to make it compatible with phonopy.load
            phonopy_yaml = lzma.open(yaml_xz, "rt")

    if phonopy_yaml is None:
        phonon = phonopy.load(
            supercell_matrix=supercell,
            primitive_matrix=primitive_matrix,
            unitcell_filename=poscar,
            force_sets_filename=force_sets,
            born_filename=born,
            factor=factor,
            is_nac=nac,
        )
    else:
        phonon = phonopy.load(
            phonopy_yaml,
            factor=factor,
            is_nac=nac,
        )

    k_path_dict, k_path_concrete = get_phonopy_kpath(
        structure=get_pmg_structure(phonon.primitive),
        kpath_scheme=kpath_scheme,
        symprec=symprec,
    )

    q_points, connections = get_band_qpoints_and_path_connections(
        k_path_concrete, npoints=kwargs.get("npoints_band", 101)
    )

    # phonon band structures will always be computed
    filename_band_yaml = f"{out_dir}/phonon-band-structure.yaml"

    # TODO: potentially add kwargs to avoid computation of eigenvectors
    phonon.run_band_structure(
        q_points,
        path_connections=connections,
        with_eigenvectors=kwargs.get("band_structure_eigenvectors", False),
        is_band_connection=kwargs.get("band_structure_eigenvectors", False),
    )
    phonon.write_yaml_band_structure(filename=filename_band_yaml)
    bs_symm_line = get_ph_bs_symm_line(
        filename_band_yaml,
        labels_dict=k_path_dict,
        has_nac=phonon.nac_params is not None,
    )
    os.remove(filename_band_yaml)

    # will determine if imaginary modes are present in the structure
    imaginary_modes = bs_symm_line.has_imaginary_freq(
        tol=kwargs.get("tol_imaginary_modes", 1e-5)
    )

    # gets data for visualization on website - yaml is also enough
    if kwargs.get("band_structure_eigenvectors"):
        bs_symm_line.write_phononwebsite(f"{out_dir}/phonon-website.json")

    # get phonon density of states
    filename_dos_yaml = f"{out_dir}/phonon-dos.yaml"

    kpoint_density_dos = kwargs.get("kpoint_density_dos", 7000)
    kpoint = Kpoints.automatic_density(
        structure=get_pmg_structure(phonon.primitive),
        kppa=kpoint_density_dos,
        force_gamma=True,
    )
    phonon.run_mesh(kpoint.kpts[0])
    phonon.run_total_dos()
    phonon.write_total_dos(filename=filename_dos_yaml)
    dos = get_ph_dos(filename_dos_yaml)
    # rm filename_dos_yaml
    os.remove(filename_dos_yaml)

    # compute vibrational part of free energies per formula unit
    temperature_range = np.arange(
        kwargs.get("tmin", 0), kwargs.get("tmax", 500), kwargs.get("tstep", 10)
    )

    free_energies = [
        dos.helmholtz_free_energy(
            structure=get_pmg_structure(phonon.primitive), t=temperature
        )
        for temperature in temperature_range
    ]

    entropies = [
        dos.entropy(structure=get_pmg_structure(phonon.primitive), t=temperature)
        for temperature in temperature_range
    ]

    internal_energies = [
        dos.internal_energy(
            structure=get_pmg_structure(phonon.primitive), t=temperature
        )
        for temperature in temperature_range
    ]

    heat_capacities = [
        dos.cv(structure=get_pmg_structure(phonon.primitive), t=temperature)
        for temperature in temperature_range
    ]

    # will compute thermal displacement matrices
    # for the primitive cell (phonon.primitive!)
    # only this is available in phonopy
    if kwargs.get("create_thermal_displacements"):
        phonon.run_mesh(kpoint.kpts[0], with_eigenvectors=True, is_mesh_symmetry=False)
        freq_min_thermal_displacements = kwargs.get(
            "freq_min_thermal_displacements", 0.0
        )
        phonon.run_thermal_displacement_matrices(
            t_min=kwargs.get("tmin_thermal_displacements", 0),
            t_max=kwargs.get("tmax_thermal_displacements", 500),
            t_step=kwargs.get("tstep_thermal_displacements", 100),
            freq_min=freq_min_thermal_displacements,
        )

        temperature_range_thermal_displacements = np.arange(
            kwargs.get("tmin_thermal_displacements", 0),
            kwargs.get("tmax_thermal_displacements", 500),
            kwargs.get("tstep_thermal_displacements", 100),
        )
        for idx, temp in enumerate(temperature_range_thermal_displacements):
            phonon.thermal_displacement_matrices.write_cif(
                phonon.primitive,
                idx,
                filename=Path(out_dir) / f"therm-displace-mat-{temp}K.cif",
            )
        _disp_mat = phonon._thermal_displacement_matrices  # noqa: SLF001
        tdisp_mat = _disp_mat.thermal_displacement_matrices.tolist()

        tdisp_mat_cif = _disp_mat.thermal_displacement_matrices_cif.tolist()

    else:
        tdisp_mat = None
        tdisp_mat_cif = None

    return {
        "unit_cell": get_pmg_structure(phonon.unitcell),
        "primitive": get_pmg_structure(phonon.primitive),
        "supercell_matrix": phonon.supercell_matrix,
        "nac_params": phonon.nac_params,
        "phonon_bandstructure": bs_symm_line,
        "phonon_dos": dos,
        "free_energies": free_energies,
        "internal_energies": internal_energies,
        "heat_capacities": heat_capacities,
        "entropies": entropies,
        "temperatures": temperature_range.tolist(),
        "has_imaginary_modes": imaginary_modes,
        "thermal_displacement_data": {
            "temperatures_thermal_displacements": temperature_range_thermal_displacements.tolist(),  # noqa: E501
            "thermal_displacement_matrix_cif": tdisp_mat_cif,
            "thermal_displacement_matrix": tdisp_mat,
            "freq_min_thermal_displacements": freq_min_thermal_displacements,
        }
        if kwargs.get("create_thermal_displacements")
        else None,
    }


def get_gnome_pmg_structures(
    zip_path: str = f"{DATA_DIR}/gnome/stable-cifs-by-id.zip",
    ids: int | Sequence[str] = 10,
) -> dict[str, Structure]:
    """Load structures from GNoME ZIP file.

    Args:
        zip_path (str): Path to GNoME ZIP file. Defaults to
            f"{DATA_DIR}/gnome/stable-cifs-by-id.zip".
        ids (int | Sequence[str]): number of structures to load or list of material IDs.
            Defaults to 10.

    Returns:
        dict[str, Structure]: dict of structures with material ID as key
    """
    structs: dict[str, Structure] = {}
    with ZipFile(zip_path, "r") as zip_ref:
        if isinstance(ids, int):
            file_list = zip_ref.namelist()[:ids]
        elif isinstance(ids, Sequence):
            file_list = [f"by_id/{mp_id}.CIF" for mp_id in ids]
        else:
            raise TypeError(f"Invalid {ids=}")

        desc = "Loading GNoME structures"
        for filename in tqdm(file_list, desc=desc, disable=len(file_list) < 100):
            if filename.endswith(".CIF"):
                mat_id = filename.split("/")[-1].split(".")[0]
                with zip_ref.open(filename) as file:
                    struct = Structure.from_str(file.read().decode(), "cif")

                struct.properties["id"] = mat_id
                structs[mat_id] = struct

    return structs
