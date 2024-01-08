"""This model defines utility functions for loading existing DFT or ML phonon
band structures and DOSs from disk.
"""
import json
import re
from collections import defaultdict
from collections.abc import Sequence
from glob import glob
from typing import TYPE_CHECKING, Literal
from zipfile import ZipFile

import pandas as pd
from monty.io import zopen
from monty.json import MontyDecoder
from pymatgen.core import Structure
from tqdm import tqdm

from ffonons import DATA_DIR, dft_key, find_last_dos_peak, formula_key, id_key

if TYPE_CHECKING:
    from atomate2.common.schemas.phonons import PhononBSDOSDoc

    from ffonons.dbs.phonondb import PhononDBDocParsed

__author__ = "Janosh Riebesell"
__date__ = "2023-11-24"


NestedDict = dict[str, dict[str, dict]]


def load_pymatgen_phonon_docs(
    which_db: Literal["mp", "phonon-db"],
    with_df: bool = True,
    imaginary_freq_tol: float = 0.1,
) -> NestedDict | tuple[NestedDict, pd.DataFrame]:
    """Load existing DFT/ML phonon band structure and DOS docs from disk for a
    specified database.

    Args:
        which_db ("mp" | "phonon-db"): Which database to load docs from.
        with_df (bool): Whether to return a pandas DataFrame as well with last phonon
            DOS peak frequencies, DOS MAE and presence of imaginary modes as columns.
            Defaults to True.
        imaginary_freq_tol (float): Tolerance for classifying a frequency as imaginary.
            Defaults to 0.1. See pymatgen's PhononBandStructureSymmLine
            has_imaginary_freq() method.

    Returns:
        dict[str, dict[str, dict]] | tuple[dict, pd.DataFrame]: Outer key is material
            ID, 2nd-level key is the model name, and third-level key is either
            "phonon_dos" or "phonon_bandstructure".
    """
    # glob json.gz or .json.lzma files
    paths = glob(f"{DATA_DIR}/{which_db}/*.json.gz") + glob(
        f"{DATA_DIR}/{which_db}/*.json.lzma"
    )
    if len(paths) == 0:
        raise FileNotFoundError(f"No files found in {DATA_DIR}/{which_db}")
    ph_docs = defaultdict(dict)

    for path in tqdm(paths, desc=f"Loading {which_db} docs"):
        with zopen(path, "rt") as file:
            ph_doc: PhononBSDOSDoc | PhononDBDocParsed = json.load(
                file, cls=MontyDecoder
            )

        mp_id, formula, model = re.search(
            rf".*/{which_db}/(mp-\d+)-([A-Z][^-]+)-(.*).json.*", path
        ).groups()
        if not mp_id.startswith("mp-"):
            raise ValueError(f"Invalid {mp_id=}")

        ph_doc.file_path = path
        setattr(ph_doc, id_key, mp_id)

        ph_docs[mp_id][model] = ph_doc

    if not with_df:
        return ph_docs

    summary_dict: dict[str, dict] = defaultdict(dict)
    for mp_id, docs in ph_docs.items():
        for model_key, ph_doc in docs.items():
            summary_dict[mp_id][formula_key] = ph_doc.structure.formula
            col_key = model_key.replace("-", "_")

            # last phonon DOS peak
            ph_dos = ph_doc.phonon_dos
            last_peak = find_last_dos_peak(ph_dos)
            summary_dict[mp_id][f"last_phdos_peak_{col_key}_THz"] = last_peak

            # max frequency from band structure
            ph_bs = ph_doc.phonon_bandstructure
            summary_dict[mp_id][f"max_freq_{col_key}_THz"] = ph_bs.bands.max()
            summary_dict[mp_id][f"min_freq_{col_key}_THz"] = ph_bs.bands.min()
            summary_dict[mp_id][f"band_width_{col_key}_THz"] = ph_bs.width()

            if model_key != dft_key and dft_key in docs:  # calculate DOS MAE and R2
                pbe_dos = ph_docs[mp_id][dft_key].phonon_dos
                summary_dict[mp_id][f"phdos_mae_{col_key}_THz"] = ph_dos.mae(pbe_dos)
                summary_dict[mp_id][f"phdos_r2_{col_key}"] = ph_dos.r2_score(pbe_dos)

            # has imaginary modes
            tol = imaginary_freq_tol
            has_imag_modes = ph_bs.has_imaginary_freq(tol=tol)
            summary_dict[mp_id][f"imaginary_freq_{col_key}"] = has_imag_modes
            has_imag_gamma_mode = ph_bs.has_imaginary_gamma_freq(tol=tol)
            summary_dict[mp_id][f"imaginary_gamma_freq_{col_key}"] = has_imag_gamma_mode

    # convert_dtypes() turns boolean cols imaginary_(gamma_)freq to bool
    df_summary = pd.DataFrame(summary_dict).T.convert_dtypes()
    df_summary.index.name = id_key
    if formula_key in df_summary:
        df_summary = df_summary.set_index(formula_key, append=True)

    return ph_docs, df_summary


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
    with ZipFile(zip_path) as zip_ref:
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
