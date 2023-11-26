# %%
import gzip
import json
import os
import re
from collections import defaultdict
from glob import glob

from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos
from tqdm import tqdm

from ffonons import DATA_DIR, bs_key, dos_key, formula_key, id_key

__author__ = "Janosh Riebesell"
__date__ = "2023-11-24"


# %% load all docs
docs_paths = glob(f"{DATA_DIR}/phonon-bs-dos/*.json.gz")
all_docs = defaultdict(dict)

for doc_path in tqdm(docs_paths, desc="Loading docs"):
    with gzip.open(doc_path, "rt") as file:
        doc_dict = json.load(file)

    doc_dict[dos_key] = PhononDos.from_dict(doc_dict[dos_key])
    doc_dict[bs_key] = PhononBandStructureSymmLine.from_dict(doc_dict[bs_key])

    mp_id, formula, model = re.search(
        r".*/phonon-bs-dos/(mp-.*)-(.*)-(.*).json.gz", doc_path
    ).groups()
    assert mp_id.startswith("mp-"), f"Invalid {mp_id=}"

    all_docs[mp_id][model] = doc_dict | {
        formula_key: formula,
        id_key: mp_id,
        "dir_path": os.path.dirname(doc_path),
    }
