"""This model defines utility functions for loading existing DFT or ML phonon
band structures and DOSs from disk.
"""
import json
import os
import re
from collections import defaultdict
from collections.abc import Sequence
from glob import glob
from typing import Literal
from zipfile import ZipFile

import pandas as pd
from monty.io import zopen
from pymatgen.core import Structure
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos
from tqdm import tqdm

from ffonons import (
    DATA_DIR,
    bs_key,
    dft_key,
    dos_key,
    find_last_dos_peak,
    formula_key,
    id_key,
)

__author__ = "Janosh Riebesell"
__date__ = "2023-11-24"


NestedDict = dict[str, dict[str, dict]]


def load_pymatgen_phonon_docs(
    which_db: Literal["mp", "phonon_db"],
    with_df: bool = True,
    imaginary_freq_tol: float = 0.1,
) -> NestedDict | tuple[NestedDict, pd.DataFrame]:
    """Load existing DFT/ML phonon band structure and DOS docs from disk for a
    specified database.

    Args:
        which_db ("mp" | "phonon_db"): Which database to load docs from.
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

    if with_df:
        summary_dict: dict[str, dict] = defaultdict(dict)
        for mp_id, docs in ph_docs.items():
            if dft_key not in docs:
                continue

            for model_key, doc in docs.items():
                summary_dict[mp_id][formula_key] = doc[formula_key]
                col_key = model_key.replace("-", "_")

                # last phonon DOS peak
                phonon_dos: PhononDos = doc[dos_key]
                last_peak = find_last_dos_peak(phonon_dos)
                summary_dict[mp_id][f"last_phdos_peak_{col_key}_THz"] = last_peak

                # max frequency from band structure
                ph_bs: PhononBandStructureSymmLine = doc[bs_key]
                summary_dict[mp_id][f"max_freq_{col_key}_THz"] = ph_bs.bands.max()
                summary_dict[mp_id][f"min_freq_{col_key}_THz"] = ph_bs.bands.min()
                summary_dict[mp_id][f"band_width_{col_key}_THz"] = ph_bs.width()

                if model_key != dft_key:
                    # DOS MAE
                    pbe_dos = ph_docs[mp_id][dft_key][dos_key]
                    summary_dict[mp_id][f"phdos_mae_{col_key}_THz"] = phonon_dos.mae(
                        pbe_dos
                    )

                # has imaginary modes
                tol = imaginary_freq_tol
                has_imag_modes = ph_bs.has_imaginary_freq(tol=tol)
                summary_dict[mp_id][f"imaginary_freq_{col_key}"] = has_imag_modes
                has_imag_gamma_mode = ph_bs.has_imaginary_gamma_freq(tol=tol)
                summary_dict[mp_id][
                    f"imaginary_gamma_freq_{col_key}"
                ] = has_imag_gamma_mode

        # convert_dtypes() turns boolean cols imaginary_(gamma_)freq to bool
        df_summary = pd.DataFrame(summary_dict).T.convert_dtypes()
        df_summary.index.name = id_key
        df_summary = df_summary.set_index(formula_key, append=True)

        return ph_docs, df_summary

    return ph_docs


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
