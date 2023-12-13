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

from monty.io import zopen
from pymatgen.core import Structure
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos
from tqdm import tqdm

from ffonons import DATA_DIR, bs_key, dos_key, formula_key, id_key

__author__ = "Janosh Riebesell"
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
