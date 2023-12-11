"""This model defines utility functions for loading existing DFT or ML phonon
band structures and DOSs.
"""
import io
import json
import os
import re
from collections import defaultdict
from collections.abc import Sequence
from glob import glob
from typing import Any, Literal
from zipfile import ZipFile

import numpy as np
import phonopy
import yaml
from monty.io import zopen
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phonopy.units import VaspToTHz
from pymatgen.core import Structure
from pymatgen.io.phonopy import (
    get_ph_bs_symm_line_from_dict,
    get_pmg_structure,
)
from pymatgen.io.vasp import Kpoints
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos
from tqdm import tqdm

from ffonons import DATA_DIR, bs_key, dos_key, formula_key, id_key
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
    # glob json.gz or .json.lzma files
    paths = glob(f"{DATA_DIR}/{which_db}/*.json.gz") + glob(
        f"{DATA_DIR}/{which_db}/*.json.lzma"
    )
    ph_docs = defaultdict(dict)

    for path in tqdm(paths, desc=f"Loading {which_db} docs"):
        with zopen(path, "rt") as file:
            doc_dict = json.load(file)

        doc_dict[dos_key] = PhononDos.from_dict(doc_dict[dos_key])
        doc_dict[bs_key] = PhononBandStructureSymmLine.from_dict(doc_dict[bs_key])

        mp_id, formula, model = re.search(
            rf".*/{which_db}/(mp-\d+)-([A-Z][^-]+)-(.*).json.*", path
        ).groups()
        assert mp_id.startswith("mp-"), f"Invalid {mp_id=}"

        ph_docs[mp_id][model] = doc_dict | {
            formula_key: formula,
            id_key: mp_id,
            "dir_path": os.path.dirname(path),
        }

    return ph_docs


def parse_phonondb_docs(
    phonopy_doc_path: str | None = None,
    supercell: list[int] = (2, 2, 2),
    primitive_matrix: str | list[list[float]] = "auto",
    nac: bool = True,
    poscar: str = "POSCAR",
    force_sets: str = "FORCE_SETS",
    born: str = "BORN",
    code: str = "vasp",
    kpath_scheme: str = "seekpath",
    symprec: float = 1e-5,
    out_dir: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Get phonon data from phonopy and save it to disk. Returns the structure and the
    phonon data as dict.

    Args:
        phonopy_doc_path (str, optional): path to phonopy yaml file. Defaults to None.
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
        out_dir (str, optional): path to output directory. Defaults to None.
            Must be specified if passing band_structure_eigenvectors or
            band_structure_eigenvectors as kwargs.
        **kwargs: additional parameters that can be passed to this method as a dict

    Returns:
        dict: phonon data
    """
    if code == "vasp":
        factor = VaspToTHz
    if phonopy_doc_path.lower().endswith(".zip"):
        # open zip archive and only read the phonopy_params.yaml.xz file from it
        with ZipFile(phonopy_doc_path, "r") as zip_file:
            try:
                yaml_xz = zip_file.open("phonopy_params.yaml.xz")
                phonopy_doc_path = zopen(yaml_xz, "rt")
            except Exception as exc:
                available_files = zip_file.namelist()
                raise RuntimeError(
                    f"Failed to read {phonopy_doc_path=}, {available_files=}"
                ) from exc

    if phonopy_doc_path is None:
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
            phonopy_doc_path,
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
    # TODO: potentially add kwargs to avoid computation of eigenvectors
    phonon.run_band_structure(
        q_points,
        path_connections=connections,
        with_eigenvectors=kwargs.get("band_structure_eigenvectors", False),
        is_band_connection=kwargs.get("band_structure_eigenvectors", False),
    )

    # convert bands to pymatgen PhononBandStructureSymmLine
    bands_io = io.StringIO()  # create in-memory io like like object to write yaml to
    phonon._band_structure._write_yaml(w=bands_io, comment=None)  # noqa: SLF001
    # parse bands_io.getvalue() as YAML
    bands_dict = yaml.safe_load(bands_io.getvalue())
    bs_symm_line = get_ph_bs_symm_line_from_dict(
        bands_dict,
        labels_dict=k_path_dict,
        has_nac=phonon.nac_params is not None,
    )

    # will determine if imaginary modes are present in the structure
    imaginary_modes = bs_symm_line.has_imaginary_freq(
        tol=kwargs.get("tol_imaginary_modes", 1e-5)
    )

    # gets data for visualization on website - yaml is also enough
    if kwargs.get("band_structure_eigenvectors"):
        os.makedirs(out_dir, exist_ok=True)
        bs_symm_line.write_phononwebsite(f"{out_dir}/phonon-website.json")

    # convert phonon DOS to pymatgen PhononDos
    kpoint_density_dos = kwargs.get("kpoint_density_dos", 7000)
    kpoint = Kpoints.automatic_density(
        structure=get_pmg_structure(phonon.primitive),
        kppa=kpoint_density_dos,
        force_gamma=True,
    )
    phonon.run_mesh(kpoint.kpts[0])
    phonon.run_total_dos()
    ph_dos = PhononDos(
        frequencies=phonon.total_dos.frequency_points, densities=phonon.total_dos.dos
    )

    # compute vibrational part of free energies per formula unit
    temperature_range = np.arange(
        kwargs.get("tmin", 0), kwargs.get("tmax", 500), kwargs.get("tstep", 10)
    )

    free_energies = [
        ph_dos.helmholtz_free_energy(
            structure=get_pmg_structure(phonon.primitive), t=temperature
        )
        for temperature in temperature_range
    ]

    entropies = [
        ph_dos.entropy(structure=get_pmg_structure(phonon.primitive), t=temperature)
        for temperature in temperature_range
    ]

    internal_energies = [
        ph_dos.internal_energy(
            structure=get_pmg_structure(phonon.primitive), t=temperature
        )
        for temperature in temperature_range
    ]

    heat_capacities = [
        ph_dos.cv(structure=get_pmg_structure(phonon.primitive), t=temperature)
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
        os.makedirs(out_dir, exist_ok=True)
        for idx, temp in enumerate(temperature_range_thermal_displacements):
            phonon.thermal_displacement_matrices.write_cif(
                phonon.primitive,
                idx,
                filename=f"{out_dir}/therm-displace-mat-{temp}K.cif",
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
        "phonon_dos": ph_dos,
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
