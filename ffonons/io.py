"""This model defines utility functions for loading existing DFT or ML phonon
band structures and DOSs from disk.
"""

import json
import os
import re
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from glob import glob
from pathlib import Path
from typing import Literal
from zipfile import ZipFile

import numpy as np
import pandas as pd
from atomate2.common.schemas.phonons import PhononBSDOSDoc
from monty.io import zopen
from monty.json import MontyDecoder
from pymatgen.core import Structure
from pymatviz.enums import Key
from tqdm import tqdm

from ffonons import DATA_DIR
from ffonons.dbs.phonondb import PhononDBDocParsed
from ffonons.enums import DB, PhKey

__author__ = "Janosh Riebesell"
__date__ = "2023-11-24"


PhDocs = dict[str, dict[str, PhononBSDOSDoc | PhononDBDocParsed]]


def load_pymatgen_phonon_docs(
    which_db: Literal["mp", "phonon-db"], materials_ids: Sequence[str] = ()
) -> PhDocs:
    """Load existing DFT/ML phonon band structure and DOS docs from disk for a
    specified database.

    Args:
        which_db ("mp" | "phonon-db"): Which database to load docs from.
        materials_ids (Sequence[str]): List of material IDs to load. Defaults to ().

    Returns:
        dict[str, dict[str, dict]]: Outer key is material ID, 2nd-level key is the model
            name (or DFT) mapped to a PhononBSDOSDoc.
    """
    # glob json.gz or .json.lzma files
    paths = glob(f"{DATA_DIR}/{which_db}/*.json.gz") + glob(
        f"{DATA_DIR}/{which_db}/*.json.lzma"
    )
    if materials_ids:
        paths = [
            path for path in paths if any(mp_id in path for mp_id in materials_ids)
        ]
    if len(paths) == 0:
        raise FileNotFoundError(f"No files found in {DATA_DIR}/{which_db}")
    ph_docs = defaultdict(dict)

    for path in tqdm(paths, desc=f"Loading {which_db} docs"):
        try:
            with zopen(path, mode="rt") as file:
                ph_doc: PhononBSDOSDoc | PhononDBDocParsed = json.load(
                    file, cls=MontyDecoder
                )
        except Exception as exc:
            print(f"error loading {path=}: {exc}")
            continue

        path_regex = rf".*/{which_db}/(mp-\d+)-([A-Z][^-]+)-(.*).json.*"
        try:
            mp_id, _formula, model = re.search(path_regex, path).groups()
        except (ValueError, AttributeError):
            raise ValueError(
                f"Can't parse MP ID and model from {path=}, should match {path_regex=}"
            ) from None
        if not mp_id.startswith("mp-"):
            raise ValueError(f"Invalid {mp_id=}")

        ph_doc.file_path = path
        setattr(ph_doc, Key.mat_id, mp_id)
        ph_docs[mp_id][model] = ph_doc

    return ph_docs


def update_key_name(directory: str, key_map: dict[str, str]) -> None:
    """Load all phonon docs in a directory and update the name of a key, then save
    updated doc back to disk.

    Args:
        directory (str): Path to the directory containing the phonon docs.
        key_map (dict[str, str]): Mapping of old key names to new key names.

    Example:
        update_key_name(f"{DATA_DIR}/{which_db}/", {"supercell_matrix": "supercell"})
    """
    paths = glob(f"{directory}/*.json.gz") + glob(f"{directory}/*.json.lzma")

    for path in tqdm(paths, desc="Updating key name"):
        try:
            with zopen(path, mode="rt") as file:
                ph_doc: PhononBSDOSDoc | PhononDBDocParsed = json.load(file)
        except Exception as exc:
            print(f"Error loading {path=}: {exc}")
            continue

        for old_key, new_key in key_map.items():
            if old_key in ph_doc:
                ph_doc[new_key] = ph_doc.pop(old_key)

        with zopen(path, mode="wt") as file:
            json.dump(ph_doc, file)


def get_df_summary(
    ph_docs: PhDocs | DB = None,
    *,  # force keyword-only arguments
    imaginary_freq_tol: float = 0.01,
    cache_path: str | Path = "",
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """Get a pandas DataFrame with last phonon DOS peak frequencies, band widths, DOS
    MAE, DOS R^2, presence of imaginary modes (at Gamma or anywhere) and other metrics.

    Inspect for correlations with
        import seaborn as sns
        sns.PairGrid(data=df_summary).map(sns.histplot)

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
    if isinstance(ph_docs, str) and cache_path is not None:
        cache_path = (
            cache_path
            or f"{DATA_DIR}/{ph_docs}/df-summary-tol={imaginary_freq_tol}.csv.gz"
        )

    if not refresh_cache and os.path.isfile(cache_path or ""):
        # print days since file was cached
        n_days = (
            datetime.now(tz=UTC)
            - datetime.fromtimestamp(os.path.getmtime(cache_path), tz=UTC)
        ).days
        print(f"Using cached df_summary from {cache_path!r} (days old: {n_days}). ")
        return pd.read_csv(
            cache_path, index_col=[Key.mat_id, Key.model]
        ).convert_dtypes()

    if isinstance(ph_docs, str | type(None)):
        ph_docs = load_pymatgen_phonon_docs(which_db=ph_docs or "phonon-db")

    summary_dict: dict[tuple[str, str], dict] = defaultdict(dict)
    for mat_id, docs in ph_docs.items():  # iterate over materials
        supercell = docs[Key.pbe].supercell
        # assert all off-diagonal elements are zero (check assumes all positive values)
        if not supercell.trace() == supercell.sum():
            raise ValueError(f"Non-diagonal {supercell=}")

        for model, ph_doc in docs.items():  # iterate over models for each material
            id_model = mat_id, model
            summary_dict[id_model][Key.formula] = ph_doc.structure.formula
            summary_dict[id_model][Key.n_sites] = len(ph_doc.structure)
            summary_dict[id_model][Key.supercell] = ", ".join(
                map(str, np.diag(supercell))
            )

            # last phonon DOS peak
            ph_dos = ph_doc.phonon_dos
            last_peak = ph_dos.get_last_peak()
            summary_dict[id_model][Key.last_ph_dos_peak] = last_peak

            # min/max frequency from band structure
            ph_bs = ph_doc.phonon_bandstructure
            summary_dict[id_model][Key.max_ph_freq] = ph_bs.bands.max()
            summary_dict[id_model][Key.min_ph_freq] = ph_bs.bands.min()

            if model != Key.pbe and Key.pbe in docs:  # calculate DOS MAE and R2
                pbe_dos = ph_docs[mat_id][Key.pbe].phonon_dos
                summary_dict[id_model][Key.ph_dos_mae] = ph_dos.mae(pbe_dos)
                summary_dict[id_model][PhKey.ph_dos_r2] = ph_dos.r2_score(pbe_dos)

            # has imaginary modes
            has_imag_modes = ph_bs.has_imaginary_freq(tol=imaginary_freq_tol)
            summary_dict[id_model][Key.has_imag_ph_modes] = has_imag_modes
            has_imag_gamma_mode = ph_bs.has_imaginary_gamma_freq(tol=imaginary_freq_tol)
            summary_dict[id_model][Key.has_imag_ph_gamma_modes] = has_imag_gamma_mode

    # convert_dtypes() turns boolean cols imaginary_(gamma_)freq to bool
    df_summary = pd.DataFrame(summary_dict).T.convert_dtypes()
    df_summary.index.names = [str(Key.mat_id), "model"]

    if cache_path:
        df_summary.to_csv(cache_path)

    return df_summary


def get_gnome_pmg_structures(
    zip_path: str = f"{DATA_DIR}/gnome/stable-cifs-by-id.zip",
    ids: int | Sequence[str] = 10,
    pbar_desc: str = "Loading GNoME structures",
    pbar_disable: bool | int = 100,
) -> dict[str, Structure]:
    """Load structures from GNoME ZIP file.

    Args:
        zip_path (str): Path to GNoME ZIP file. Defaults to
            f"{DATA_DIR}/gnome/stable-cifs-by-id.zip".
        ids (int | Sequence[str]): number of structures to load or list of material IDs.
            Defaults to 10.
        pbar_desc (str): tqdm progress bar description. Defaults to "Loading GNoME
            structures".
        pbar_disable (bool | int): Disable progress bar if True or if number of
            structures is less than this value. Defaults to 100.

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

        if isinstance(pbar_disable, int):
            pbar_disable = len(file_list) < pbar_disable

        for filename in tqdm(file_list, desc=pbar_desc, disable=pbar_disable):
            if filename.endswith(".CIF"):
                mat_id = filename.split("/")[-1].split(".")[0]
                with zip_ref.open(filename) as file:
                    struct = Structure.from_str(file.read().decode(), "cif")

                struct.properties[Key.mat_id] = mat_id
                structs[mat_id] = struct

    return structs
