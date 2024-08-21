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
from typing import TYPE_CHECKING, Literal
from zipfile import ZipFile

import numpy as np
import pandas as pd
from atomate2.common.schemas.phonons import PhononBSDOSDoc
from monty.io import zopen
from monty.json import MontyDecoder
from pymatgen.core import Structure
from pymatviz.enums import Key
from tqdm import tqdm

from ffonons.enums import DB, PhKey

if TYPE_CHECKING:
    from ffonons.dbs.phonondb import PhononDBDocParsed
else:
    PhononDBDocParsed = object

__author__ = "Janosh Riebesell"
__date__ = "2023-11-24"


PhDocs = dict[str, dict[str, PhononBSDOSDoc | PhononDBDocParsed]]


def load_pymatgen_phonon_docs(
    docs_to_load: DB | Sequence[str],
    *,
    materials_ids: Sequence[str] = (),
    glob_patt: str = "",
    verbose: bool = True,
) -> PhDocs:
    """Load existing DFT/ML phonon band structure and DOS docs from disk for a
    specified database.

    Args:
        docs_to_load ("mp" | "phonon-db" | Sequence[str]): Database name to load docs
            for or list of file paths to load.
        materials_ids (Sequence[str]): List of material IDs to load. Defaults to ().
        glob_patt (str): Glob pattern to match files to load from the database
            directory. Defaults to "". If set, only files matching this pattern will be
            loaded. Ignored if docs_to_load is a list of file paths.
        verbose (bool): Whether to print progress bar. Defaults to True.

    Returns:
        dict[str, dict[str, dict]]: Outer key is material ID, 2nd-level key is the model
            name (or DFT) mapped to a PhononBSDOSDoc.
    """
    from ffonons import DATA_DIR

    if len(docs_to_load) == 0:
        return {}
    if isinstance(docs_to_load, str):
        if glob_patt == "":
            paths = glob(f"{DATA_DIR}/{docs_to_load}/*.json.gz") + glob(
                f"{DATA_DIR}/{docs_to_load}/*.json.lzma"
            )
        else:
            paths = glob(f"{DATA_DIR}/{docs_to_load}/{glob_patt}")
    elif {*map(type, docs_to_load)} == {str}:
        paths = docs_to_load
    else:
        raise TypeError(f"Invalid {docs_to_load=}, should be str or list of str")

    if materials_ids:
        paths = [
            path for path in paths if any(mp_id in path for mp_id in materials_ids)
        ]

    if len(paths) == 0:
        err_msg = f"No files found in {DATA_DIR}/{docs_to_load}"
        if glob_patt:
            err_msg += f" and {glob_patt=}"
        if materials_ids:
            err_msg += f" and {materials_ids=}"
        raise FileNotFoundError(err_msg)

    ph_docs = defaultdict(dict)

    pbar = tqdm(paths, desc=f"Loading {len(paths)} docs")
    for path in pbar if verbose else paths:
        if verbose:
            pbar.set_postfix_str(path.split("/")[-1])
        try:
            with zopen(path, mode="rt") as file:
                ph_doc: PhononBSDOSDoc | PhononDBDocParsed = json.load(
                    file, cls=MontyDecoder
                )
        except Exception as exc:
            print(f"error loading {path=}: {exc}")
            continue

        path_regex = r".*/(mp-\d+)-([A-Z][^-]+)-(.*).json.*"
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


def get_df_summary(
    ph_docs: PhDocs | DB = DB.phonon_db,
    *,  # force keyword-only arguments
    imaginary_freq_tol: float = 0.01,
    cache_path: str | Path = "",
    refresh_cache: bool | str | Literal["incremental"] = "incremental",  # noqa: PYI051
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
            Defaults to 0.01. See pymatgen's PhononBandStructureSymmLine
            has_imaginary_freq() method.
        cache_path (str | Path): Path to cache file. Set to None to disable caching.
            Default = f"{DATA_DIR}/{ph_docs}/df-summary-tol={imaginary_freq_tol}.csv.gz"
        refresh_cache (bool | str): If True, reload all phonon docs in given database
            directory. Will write a new summary CSV after. If a string, use as a
            glob pattern to only reload matching files for speed. Has no effect when
            ph_docs is a list of documents and not a str (as in a database name) other
            than writing a new CSV cache file. Defaults to "incremental".

    Returns:
        pd.DataFrame: Summary metrics for each material and model in ph_docs.
    """
    from ffonons import DATA_DIR

    if isinstance(ph_docs, str) and cache_path is not None:
        cache_path = (
            cache_path
            or f"{DATA_DIR}/{ph_docs}/df-summary-tol={imaginary_freq_tol}.csv.gz"
        )

    df_cached = None
    if os.path.isfile(cache_path or ""):
        df_cached = pd.read_csv(
            cache_path, index_col=[Key.mat_id, Key.model]
        ).convert_dtypes()
        if not refresh_cache:
            n_days = (
                datetime.now(tz=UTC)
                - datetime.fromtimestamp(os.path.getmtime(cache_path), tz=UTC)
            ).days
            print(f"Using cached df_summary from {cache_path!r} (days old: {n_days}). ")
            return df_cached

    if (
        isinstance(ph_docs, str)
        and refresh_cache == "incremental"
        and isinstance(df_cached, pd.DataFrame)
    ):
        all_files = glob(f"{DATA_DIR}/{ph_docs}/*.json.gz") + glob(
            f"{DATA_DIR}/{ph_docs}/*.json.lzma"
        )

        loaded_mat_id_model_combos = tuple(df_cached.index)

        def id_model_combo_already_loaded(path: str) -> bool:
            mat_id, _formula, model = re.search(
                r".*/(mp-\d+)-([A-Z][^-]+)-(.*).json.*", path
            ).groups()
            return (mat_id, model) in loaded_mat_id_model_combos

        files_to_load = [
            path for path in all_files if not id_model_combo_already_loaded(path)
        ]
        ph_docs = files_to_load

    glob_patt = refresh_cache if isinstance(refresh_cache, str) else ""
    loaded_docs = load_pymatgen_phonon_docs(docs_to_load=ph_docs, glob_patt=glob_patt)

    summary_dict: dict[tuple[str, str], dict] = defaultdict(dict)
    for mat_id, docs in loaded_docs.items():  # iterate over materials
        for model, ph_doc in docs.items():  # iterate over models for each material
            if (mat_id, model) in getattr(df_cached, "index", ()) and not refresh_cache:
                # Skip if this entry already exists in the cache
                continue

            id_model = mat_id, model
            summary_dict[id_model][Key.formula] = ph_doc.structure.formula
            summary_dict[id_model][Key.n_sites] = len(ph_doc.structure)
            supercell = getattr(
                ph_doc, "supercell", getattr(ph_doc, "supercell_matrix", None)
            )
            # assert all off-diagonal elements are zero (check assumes supercell matrix
            # has only positive values)
            if (
                isinstance(supercell, np.ndarray)
                and supercell.trace() != supercell.sum()
            ):
                raise ValueError(f"Non-diagonal {supercell=}")
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
                pbe_dos = docs[Key.pbe].phonon_dos
                summary_dict[id_model][Key.ph_dos_mae] = ph_dos.mae(pbe_dos)
                summary_dict[id_model][PhKey.ph_dos_r2] = ph_dos.r2_score(pbe_dos)

            # has imaginary modes
            has_imag_modes = ph_bs.has_imaginary_freq(tol=imaginary_freq_tol)
            summary_dict[id_model][Key.has_imag_ph_modes] = has_imag_modes
            has_imag_gamma_mode = ph_bs.has_imaginary_gamma_freq(tol=imaginary_freq_tol)
            summary_dict[id_model][Key.has_imag_ph_gamma_modes] = has_imag_gamma_mode

    # convert_dtypes() turns boolean cols imaginary_(gamma_)freq to bool
    new_df = pd.DataFrame(summary_dict).T.convert_dtypes()
    idx_names = [str(Key.mat_id), str(Key.model)]
    if len(new_df.index.names) == len(idx_names):
        new_df.index.names = idx_names
    elif len(new_df.index.names) == 1:
        new_df.index.names = [idx_names[0]]

    # Concatenate the existing DataFrame with the new one
    if df_cached is None:
        df_summary = new_df
    else:
        df_cached.update(new_df)  # Update existing rows
        df_cached = pd.concat(  # Add new rows not in df_cached
            [df_cached, new_df[~new_df.index.isin(df_cached.index)]]
        )
        df_summary = df_cached

    if cache_path:
        df_summary.to_csv(cache_path)

    return df_summary


def get_gnome_pmg_structures(
    zip_path: str = "",
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
    from ffonons import DATA_DIR

    zip_path = zip_path or f"{DATA_DIR}/gnome/stable-cifs-by-id.zip"

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
