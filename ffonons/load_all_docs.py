# %%
import gzip
import json
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
all_docs = defaultdict(dict)

for phonon_doc in tqdm(glob(f"{DATA_DIR}/**/phonon-bs-dos-*.json.gz"), desc="Loading"):
    with gzip.open(phonon_doc, "rt") as file:
        doc_dict = json.load(file)

    doc_dict[dos_key] = PhononDos.from_dict(doc_dict[dos_key])
    doc_dict[bs_key] = PhononBandStructureSymmLine.from_dict(doc_dict[bs_key])

    mp_id, formula, model = re.search(
        r".*/(mp-.*)-(.*)/phonon-bs-dos-(.*).json.gz", phonon_doc
    ).groups()
    assert mp_id.startswith("mp-"), f"Invalid {mp_id=}"

    all_docs[mp_id][model] = doc_dict | {formula_key: formula, id_key: mp_id}
