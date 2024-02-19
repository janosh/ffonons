"""This model defines utility functions for loading existing DFT or ML phonon
band structures and DOSs from disk.
"""
import json
import os
import re
from collections import defaultdict
from collections.abc import Sequence
from glob import glob
from pathlib import Path
from typing import Literal
from zipfile import ZipFile

import pandas as pd
from atomate2.common.schemas.phonons import PhononBSDOSDoc
from monty.io import zopen
from monty.json import MontyDecoder
from pymatgen.core import Structure
from tqdm import tqdm

from ffonons import DATA_DIR, find_last_dos_peak
from ffonons.dbs.phonondb import PhononDBDocParsed
from ffonons.enums import DB, Key

__author__ = "Janosh Riebesell"
__date__ = "2023-11-24"


PhDocs = dict[str, dict[str, PhononBSDOSDoc | PhononDBDocParsed]]


def load_pymatgen_phonon_docs(which_db: Literal["mp", "phonon-db"]) -> PhDocs:
    """Load existing DFT/ML phonon band structure and DOS docs from disk for a
    specified database.

    Args:
        which_db ("mp" | "phonon-db"): Which database to load docs from.

    Returns:
        dict[str, dict[str, dict]]: Outer key is material ID, 2nd-level key is the model
            name (or DFT) mapped to a PhononBSDOSDoc.
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
        setattr(ph_doc, Key.mat_id, mp_id)
        ph_docs[mp_id][model] = ph_doc

    return ph_docs


def get_df_summary(
    ph_docs: PhDocs | DB = None,
    imaginary_freq_tol: float = 0.1,
    cache_path: str | Path = "",
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """Get a pandas DataFrame with last phonon DOS peak frequencies, band widths, DOS
    MAE, DOS R^2, presence of imaginary modes (at Gamma or anywhere) and other metrics.

    Args:
        ph_docs (PhDocs | WhichDB): Nested dicts (1st level: material IDs,
            2nd level: model name) of PhononBSDOSDoc or PhononDBDocParsed objects.
            Can also be a database name (str), see ffonons.WhichDB.
        imaginary_freq_tol (float): Tolerance for classifying a frequency as imaginary.
            Defaults to 0.1. See pymatgen's PhononBandStructureSymmLine
            has_imaginary_freq() method.
        cache_path (str | Path): Path to cache file. Set to None to disable caching.
            Defaults to f"{DATA_DIR}/{ph_docs}/df-summary.csv.gz" if ph_docs is a str,
            else to None.
        refresh_cache (bool): Whether to refresh the cache. Defaults to False.

    Returns:
        pd.DataFrame: Summary metrics for each material and model in ph_docs.
    """
    if isinstance(ph_docs, str):
        cache_path = cache_path or f"{DATA_DIR}/{ph_docs}/df-summary.csv.gz"

    if not refresh_cache and os.path.isfile(cache_path or ""):
        return pd.read_csv(
            cache_path, index_col=[Key.mat_id, Key.model]
        ).convert_dtypes()

    if isinstance(ph_docs, str | type(None)):
        ph_docs = load_pymatgen_phonon_docs(which_db=ph_docs or "phonon-db")

    summary_dict: dict[tuple[str, str], dict] = defaultdict(dict)
    for mat_id, docs in ph_docs.items():
        for model_key, ph_doc in docs.items():
            id_mod_key = mat_id, model_key
            summary_dict[id_mod_key][Key.formula] = ph_doc.structure.formula
            summary_dict[id_mod_key][Key.n_sites] = len(ph_doc.structure)

            # last phonon DOS peak
            ph_dos = ph_doc.phonon_dos
            last_peak = find_last_dos_peak(ph_dos)
            summary_dict[id_mod_key][Key.last_dos_peak] = last_peak

            # max frequency from band structure
            ph_bs = ph_doc.phonon_bandstructure
            summary_dict[id_mod_key]["max_freq_THz"] = ph_bs.bands.max()
            summary_dict[id_mod_key]["min_freq_THz"] = ph_bs.bands.min()

            if model_key != Key.dft and Key.dft in docs:  # calculate DOS MAE and R2
                pbe_dos = ph_docs[mat_id][Key.dft].phonon_dos
                summary_dict[id_mod_key][Key.dos_mae] = ph_dos.mae(pbe_dos)
                summary_dict[id_mod_key]["phdos_r2"] = ph_dos.r2_score(pbe_dos)

            # has imaginary modes
            has_imag_modes = ph_bs.has_imaginary_freq(tol=imaginary_freq_tol)
            summary_dict[id_mod_key]["imaginary_freq"] = has_imag_modes
            has_imag_gamma_mode = ph_bs.has_imaginary_gamma_freq(tol=imaginary_freq_tol)
            summary_dict[id_mod_key]["imaginary_gamma_freq"] = has_imag_gamma_mode

    # convert_dtypes() turns boolean cols imaginary_(gamma_)freq to bool
    df_summary = pd.DataFrame(summary_dict).T.convert_dtypes()
    df_summary.index.names = [Key.mat_id, "model"]

    if cache_path:
        df_summary.to_csv(cache_path)

    return df_summary


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

                struct.properties[Key.mat_id] = mat_id
                structs[mat_id] = struct

    return structs
