import copy
import io
import json
import lzma
import os
import re
import sys
from dataclasses import dataclass
from glob import glob
from typing import Any, Literal
from zipfile import ZipFile

import numpy as np
import pandas as pd
import phonopy
import requests
import yaml
from bs4 import BeautifulSoup
from monty.json import MontyEncoder
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phonopy.units import VaspToTHz
from pymatgen.core import Structure
from pymatgen.io.phonopy import get_ph_bs_symm_line_from_dict, get_pmg_structure
from pymatgen.io.vasp import Kpoints
from pymatgen.phonon import PhononBandStructureSymmLine, PhononDos
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.symmetry.kpath import KPathSeek

from ffonons import DATA_DIR

__author__ = "Janine George, Aakash Naik, Janosh Riebesell"
__date__ = "2023-12-07"

db_name = "phonon-db"
ph_docs_dir = f"{DATA_DIR}/{db_name}"
togo_id_key = "togo_id"

id_map_path = f"{DATA_DIR}/{db_name}/map-mp-id-togo-id.csv"
mp_to_togo_id = pd.read_csv(id_map_path, index_col=0)[togo_id_key].to_dict()
togo_to_mp_id = {val: key for key, val in mp_to_togo_id.items()}


# %%
def fetch_togo_doc_by_id(doc_id: str, out_path: str = "") -> str:
    """Download the phonopy file for a given MP ID. The file is saved to out_path which
    defaults to "data/phonon-db/{mp_id}-{togo_id}-pbe.zip". out_path is returned.
    If the file already exists, it is skipped but its path is still returned.
    """
    if doc_id.startswith("mp-"):
        togo_id = mp_to_togo_id[doc_id]
        mp_id = doc_id
    else:
        togo_id = doc_id
        mp_id = togo_to_mp_id[doc_id]

    filename = f"{mp_id}-{togo_id}-pbe.zip"
    out_path = out_path or f"{ph_docs_dir}/{filename}"
    if os.path.isfile(out_path):
        print(f"{out_path=} already exists. skipping")
        return out_path

    download_url = f"https://mdr.nims.go.jp/download_all/{togo_id}.zip"
    resp = requests.get(download_url, allow_redirects=True, timeout=15)

    with open(out_path, "wb") as file:
        file.write(resp.content)

    return out_path


def scrape_and_fetch_togo_docs_from_page(
    url: str, on_error: Literal["raise", "warn", "ignore"] = "ignore"
) -> pd.DataFrame | str:
    """Extract togo ID, MP ID from Togo DB index pages and download their phonopy files.

    Args:
        url (str): URL of the Togo DB index page
        on_error ("raise" | "warn" | "ignore"): what to do if an error occurs.
            Defaults to "raise".

    Returns:
        pd.DataFrame | str: DataFrame with togo ID, MP ID, and download URLs for
            phonopy files. If an error occurs, returns the error message.
    """
    response = requests.get(url, timeout=15)

    if on_error == "raise":
        response.raise_for_status()
    elif on_error in ("warn", "ignore"):
        try:
            response.raise_for_status()
        except Exception as exc:
            msg = f"{url=} failed with {exc}"
            if on_error == "warn":
                print(msg, file=sys.stderr)
            return msg

    # parse the HTML content of the page
    soup = BeautifulSoup(response.text, "html.parser")

    # extract all links from the page
    tables = [
        tr.prettify() for tr in soup.find_all("tr") if "document_" in tr.prettify()
    ]

    doc_ids, mp_ids, download_urls = [], [], []

    for table in tables:
        soup1 = BeautifulSoup(table, "html.parser")

        # find the relevant elements within the table row
        doc_id = soup1.find("tr")["id"].split("_")[-1]
        link_element = soup1.find_all("a", class_="")[0]
        mp_id = f"mp-{link_element.text.strip().split()[-1]}"
        out_path = f"{ph_docs_dir}/{mp_id}-{doc_id}-pbe.zip"

        if os.path.isfile(out_path):
            print(f"{out_path=} already exists. skipping")
            continue

        doc_ids += [doc_id]
        mp_ids += [mp_id]

        download_url = f"https://mdr.nims.go.jp/download_all/{doc_id}.zip"
        resp = requests.get(download_url, allow_redirects=True, timeout=15)
        if resp.status_code != 200:
            continue  # skip if download failed
        download_urls += [download_url]
        with open(out_path, "wb") as file:
            file.write(resp.content)

    df_out = pd.DataFrame(index=mp_ids, columns=["doc_ids", "download_urls"])
    df_out["doc_ids"] = doc_ids
    df_out["download_urls"] = download_urls

    return df_out


def phonondb_doc_to_pmg_lzma(
    zip_path: str, existing: Literal["skip", "overwrite", "raise"] = "skip"
) -> tuple[Structure, dict[str, Any]]:
    """Convert a zipped phonon DB doc to a pymatgen Structure and dict of phonon data.

    Args:
        zip_path (str): path to the zipped phonon DB doc
        existing ("skip" | "overwrite" | "raise"): What to do if output file already
            exists. Defaults to "skip".

    Returns:
        tuple[Structure, dict[str, Any]]: Structure and dict of phonon data
    """
    mat_id = "-".join(zip_path.split("/")[-1].split("-")[:2])
    if matches := glob(f"{ph_docs_dir}/{mat_id}-*-pbe.json.lzma"):
        if existing == "skip":
            print(f"{matches[0]} already exists. skipping")
            return matches[0]
        if existing == "raise":
            raise RuntimeError(f"{matches[0]} already exists.")
        # else: overwrite i.e. continue

    phonondb_doc = parse_phonondb_docs(zip_path, is_nac=False)

    assert re.match(r"mp-\d+", mat_id), f"Invalid {mat_id=}"

    struct = phonondb_doc.unit_cell
    formula = struct.formula.replace(" ", "")
    pmg_doc_path = f"{ph_docs_dir}/{mat_id}-{formula}-pbe.json.lzma"

    with lzma.open(pmg_doc_path, "wt") as file:
        json.dump(phonondb_doc, file, cls=MontyEncoder)

    return pmg_doc_path


def get_phonopy_kpath(
    structure: Structure, kpath_scheme: str, symprec: float, **kwargs: Any
) -> tuple:
    """Get high-symmetry points in k-space in phonopy format.

    Args:
        structure (Structure): pymatgen structure object
        kpath_scheme (str): kpath scheme
        symprec (float): precision for structure symmetry determination
        **kwargs: additional params passed to HighSymmKpath or KPathSeek

    Returns:
        tuple: kpoints and path
    """
    if kpath_scheme in ("setyawan_curtarolo", "latimer_munro", "hinuma"):
        high_symm_kpath = HighSymmKpath(
            structure, path_type=kpath_scheme, symprec=symprec, **kwargs
        )
        kpath = high_symm_kpath.kpath
    elif kpath_scheme == "seekpath":
        high_symm_kpath = KPathSeek(structure, symprec=symprec, **kwargs)
        kpath = high_symm_kpath._kpath  # noqa: SLF001

    path = copy.deepcopy(kpath["path"])

    for path_idx, label_set in enumerate(kpath["path"]):
        for label_idx, label in enumerate(label_set):
            path[path_idx][label_idx] = kpath["kpoints"][label]
    return kpath["kpoints"], path


@dataclass
class PhononDBDocParsed:
    """Dataclass for phonon DB docs."""

    structure: Structure
    primitive: Structure
    supercell_matrix: list[list[int]]  # 3x3 matrix
    nac_params: dict[str, Any]  # non-analytical corrections based on Born charges
    phonon_bandstructure: PhononBandStructureSymmLine
    phonon_dos: PhononDos
    free_energies: list[float]  # vibrational part of free energies per formula unit
    internal_energies: list[float]  # vibrational part of internal energies per f.u.
    heat_capacities: list[float]
    entropies: list[float]
    temps: list[float]  # temperatures
    has_imaginary_modes: bool  # whether imaginary modes are present anywhere in the BS
    thermal_displacement_data: dict[str, Any] | None = None
    mp_id: str | None = None  # material ID
    formula: str | None = None  # chemical formula


def parse_phonondb_docs(
    phonopy_doc_path: str | None = None,
    phonopy_params: dict[str, Any] | None = None,
    supercell: list[int] = (2, 2, 2),
    primitive_matrix: str | list[list[float]] = "auto",
    is_nac: bool = False,
    poscar: str = "POSCAR",
    force_sets: str = "FORCE_SETS",
    born: str = "BORN",
    code: str = "vasp",
    kpath_scheme: str = "seekpath",
    symprec: float = 1e-5,
    out_dir: str | None = None,
    **kwargs: Any,
) -> PhononDBDocParsed:
    """Get phonon data from phonopy and save it to disk. Returns the structure and the
    phonon data as dict.

    Args:
        phonopy_doc_path (str): Path to zipped phonopy doc containing YAML file with
            phonopy params.
        phonopy_params (dict[str, Any]): phonopy params as dict (only used if
            phonopy_doc_path is None).
        supercell (list[int], optional): supercell matrix. Defaults to (2, 2, 2).
        primitive_matrix (str | list[list[float]], optional): primitive matrix.
            Defaults to "auto".
        is_nac (bool, optional): whether to use apply non-analytical corrections from
            Born charges. Defaults to False (since Born charges are not available from
            MLIPs).
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
        PhononDBDoc: dataclass with phonon data
    """
    if code == "vasp":
        factor = VaspToTHz

    if phonopy_doc_path:
        # open zip archive and only read the phonopy_params.yaml.xz file from it
        try:
            with ZipFile(phonopy_doc_path) as zip_file:
                yaml_xz = zip_file.open("phonopy_params.yaml.xz")
                phonopy_params = lzma.open(yaml_xz, "rt")
        except Exception as exc:
            exc.add_note(f"Failed to load {phonopy_doc_path=}")
            raise

    if phonopy_params:
        phonon = phonopy.load(phonopy_params, factor=factor, is_nac=is_nac)
    else:
        phonon = phonopy.load(
            supercell_matrix=supercell,
            primitive_matrix=primitive_matrix,
            unitcell_filename=poscar,
            force_sets_filename=force_sets,
            born_filename=born,
            factor=factor,
            is_nac=is_nac,
        )

    pmg_struct = get_pmg_structure(phonon.primitive)
    k_path_dict, k_path_concrete = get_phonopy_kpath(
        structure=pmg_struct, kpath_scheme=kpath_scheme, symprec=symprec
    )

    q_points, connections = get_band_qpoints_and_path_connections(
        k_path_concrete, npoints=kwargs.get("npoints_band", 101)
    )

    # phonon band structures will always be computed
    # TODO: potentially add kwargs to avoid computation of eigenvectors
    with_bs_eig_vecs = kwargs.get("band_structure_eigenvectors", False)
    phonon.run_band_structure(
        q_points,
        path_connections=connections,
        with_eigenvectors=with_bs_eig_vecs,
        is_band_connection=with_bs_eig_vecs,
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
    has_imag_modes = bs_symm_line.has_imaginary_freq(
        tol=kwargs.get("tol_imaginary_modes", 1e-5)
    )

    # gets data for visualization on website - yaml is also enough
    if kwargs.get("band_structure_eigenvectors"):
        os.makedirs(out_dir, exist_ok=True)
        bs_symm_line.write_phononwebsite(f"{out_dir}/phonon-website.json")

    # convert phonon DOS to pymatgen PhononDos
    kpoint_density_dos = kwargs.get("kpoint_density_dos", 7000)
    struct = get_pmg_structure(phonon.primitive)
    kpoint = Kpoints.automatic_density(
        structure=struct,
        kppa=kpoint_density_dos,
        force_gamma=True,
    )
    phonon.run_mesh(kpoint.kpts[0])
    phonon.run_total_dos()
    ph_dos = PhononDos(
        frequencies=phonon.total_dos.frequency_points, densities=phonon.total_dos.dos
    )

    # compute vibrational part of free energies per formula unit
    temp_range = np.arange(
        kwargs.get("tmin", 0), kwargs.get("tmax", 500), kwargs.get("tstep", 10)
    )

    free_energies = [
        ph_dos.helmholtz_free_energy(temp=temp, structure=struct) for temp in temp_range
    ]

    entropies = [ph_dos.entropy(temp=temp, structure=struct) for temp in temp_range]

    internal_energies = [
        ph_dos.internal_energy(temp=temp, structure=struct) for temp in temp_range
    ]

    heat_capacities = [ph_dos.cv(temp=temp, structure=struct) for temp in temp_range]

    # will compute thermal displacement matrices
    # for the primitive cell (phonon.primitive!)
    # only this is available in phonopy
    therm_disp_mat = therm_disp_mat_cif = None  # initialize
    if kwargs.get("create_thermal_displacements"):
        phonon.run_mesh(kpoint.kpts[0], with_eigenvectors=True, is_mesh_symmetry=False)
        freq_min_thermal_displacements = kwargs.get("freq_min_thermal_displacements", 0)
        t_min, t_max, t_step = (
            kwargs.get(f"{key}_thermal_displacements", val)
            for key, val in (("tmin", 0), ("tmax", 500), ("tstep", 100))
        )
        phonon.run_thermal_displacement_matrices(
            t_min=t_min,
            t_max=t_max,
            t_step=t_step,
            freq_min=freq_min_thermal_displacements,
        )

        temp_range_thermal_displacements = np.arange(t_min, t_max, t_step)
        os.makedirs(out_dir, exist_ok=True)
        for idx, temp in enumerate(temp_range_thermal_displacements):
            cif_path = f"{out_dir}/therm-displace-mat-{temp}K.cif"

            phonon.thermal_displacement_matrices.write_cif(
                phonon.primitive, idx, filename=cif_path
            )
        td_matrices = phonon._thermal_displacement_matrices  # noqa: SLF001
        therm_disp_mat = td_matrices.thermal_displacement_matrices.tolist()

        therm_disp_mat_cif = td_matrices.thermal_displacement_matrices_cif.tolist()

    return PhononDBDocParsed(
        unit_cell=get_pmg_structure(phonon.unitcell),
        primitive=struct,
        supercell_matrix=phonon.supercell_matrix,
        nac_params=phonon.nac_params,
        phonon_bandstructure=bs_symm_line,
        phonon_dos=ph_dos,
        free_energies=free_energies,
        internal_energies=internal_energies,
        heat_capacities=heat_capacities,
        entropies=entropies,
        temps=temp_range.tolist(),
        has_imaginary_modes=has_imag_modes,
        thermal_displacement_data={
            "temps_thermal_displacements": temp_range_thermal_displacements.tolist(),
            "thermal_displacement_matrix_cif": therm_disp_mat_cif,
            "thermal_displacement_matrix": therm_disp_mat,
            "freq_min_thermal_displacements": freq_min_thermal_displacements,
        }
        if kwargs.get("create_thermal_displacements")
        else None,
    )
