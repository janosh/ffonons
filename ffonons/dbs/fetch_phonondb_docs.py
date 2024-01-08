# %%
import re
from glob import glob

import pandas as pd
from tqdm import tqdm

from ffonons import DATA_DIR
from ffonons.dbs.phonondb import (
    fetch_togo_doc_by_id,
    phonondb_doc_to_pmg_lzma,
    scrape_and_fetch_togo_docs_from_page,
)

db_name = "phonon-db"
ph_docs_dir = f"{DATA_DIR}/{db_name}"
togo_id_key = "togo_id"
__author__ = "Janine George, Aakash Nair, Janosh Riebesell"
__date__ = "2023-12-07"
phonondb_base_url = "https://mdr.nims.go.jp/collections/8g84ms862"


# %% get all phonon_db page urls
urls = [f"{phonondb_base_url}?{page=}" for page in range(1, 1005)]

dfs = []
for url in tqdm(urls, desc="Downloading Togo Phonopy DB"):
    dfs += [scrape_and_fetch_togo_docs_from_page(url, on_error="ignore")]


# %%
df = pd.concat(df for df in dfs if isinstance(df, pd.DataFrame))
df = df.sort_index()


# %% convert phonondb docs to lzma compressed JSON which is much faster to load
zip_files = glob(f"{ph_docs_dir}/mp-*-pbe.zip")

for zip_path in (pbar := tqdm(zip_files, desc="Parsing PhononDB docs to PMG lzma")):
    mat_id = "-".join(zip_path.split("/")[-1].split("-")[:2])
    assert re.match(r"mp-\d+", mat_id), f"Invalid {mat_id=}"
    existing_docs = glob(f"{ph_docs_dir}/{mat_id}-*-pbe.json.lzma")
    if len(existing_docs) > 1:
        raise RuntimeError(f"> 1 doc for {mat_id=}: {existing_docs}")

    pbar.set_postfix_str(f"{mat_id}")
    phonondb_doc_to_pmg_lzma(zip_path, existing="overwrite")


# %%
id_formula_map = {
    "mp-2998": "BaTiO3",
    "mp-4651": "SrTiO3",
    "mp-2892": "BaNd2O4",
    "mp-6586": "K2NaAlF6",
    "mp-3996": "GaAsO4",
    "mp-3978": "SrSiO3",
    "mp-3472": "PbSO4",
    "mp-2798": "SiP",
    "mp-2789": "NO2",
    "mp-2657": "TiO2",
    "mp-2659": "LiN3",
    "mp-2667": "CsAu",
    "mp-2672": "K2O2",
    "mp-2691": "CdSe",
    "mp-2697": "SrO2",
    "mp-2706": "SnF4",
    "mp-2739": "TeO2",
    "mp-2741": "CaF2",
    "mp-2758": "SrSe",
    "mp-2763": "Nd2O3",
    "mp-2782": "ZnP2",
    "mp-2784": "Na2Te",
}
for mp_id in id_formula_map:
    zip_path = fetch_togo_doc_by_id(mp_id)
    assert zip_path.endswith(".zip"), f"Invalid {zip_path=}"

    try:
        pmg_doc_path = phonondb_doc_to_pmg_lzma(zip_path)
    except Exception as exc:
        print(f"{mp_id=}: {exc}")
        continue
