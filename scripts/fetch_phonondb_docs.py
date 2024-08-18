"""Module to fetch and parse Togo PhononDB docs for MP materials."""

# %%
import os
import re
from glob import glob
from zipfile import BadZipFile

import pandas as pd
from mp_api.client import MPRester
from pymatviz.enums import Key
from tqdm import tqdm

from ffonons import DATA_DIR
from ffonons.dbs.phonondb import (
    fetch_togo_doc_by_id,
    map_mp_to_togo_id,
    phonondb_doc_to_pmg_lzma,
    scrape_and_fetch_togo_docs_from_page,
)
from ffonons.enums import DB

__author__ = "Janine George, Aakash Nair, Janosh Riebesell"
__date__ = "2023-12-07"

db_name = DB.phonon_db
ph_docs_dir = f"{DATA_DIR}/{db_name}"
phonondb_base_url = "https://mdr.nims.go.jp/collections/8g84ms862"


# %% get all phonon_db page urls
urls = [f"{phonondb_base_url}?{page=}" for page in range(1, 1005)]

dfs_fetched = []
for url in tqdm(urls, desc="Downloading Togo Phonopy DB"):
    dfs_fetched += [scrape_and_fetch_togo_docs_from_page(url, on_error="ignore")]


# %%
df_fetched = pd.concat(df for df in dfs_fetched if isinstance(df, pd.DataFrame))
df_fetched = df_fetched.sort_index()


# %% 5 Togo materials with single site: mp-39, mp-23155, mp-111, mp-753304, mp-632250
docs = MPRester(use_document_model=False).materials.search(
    num_sites=(3, 3), fields=["nsites", Key.mat_id, "formula_pretty", Key.volume]
)

df_mp = pd.DataFrame(docs).set_index(Key.mat_id, drop=False)


# %% fetch Togo docs by MP ID
pbar = tqdm(
    # fetch all PhononDB docs
    map_mp_to_togo_id,
    # fetch docs for MP materials with above specified number of sites
    # df_mp.query(f"{Key.mat_id} in {list(mp_to_togo_id)}").index,
    desc="Fetching Togo docs",
)
n_prev_zip_files = glob(f"{ph_docs_dir}/*.zip")
for mp_id in pbar:
    try:
        pbar.set_postfix_str(f"{mp_id}")
        zip_path = fetch_togo_doc_by_id(mp_id)
        if not zip_path.endswith(".zip"):
            raise ValueError(f"Unexpected {zip_path=}")

        try:
            pmg_doc_path = phonondb_doc_to_pmg_lzma(
                zip_path, existing="skip-silent", on_read_error="raise"
            )
        except (ValueError, RuntimeError, BadZipFile) as exc:
            # TODO look into frequent error: is not a zip file, maybe from 404 response?
            print(f"{mp_id=}: {exc}")
            # if bad zip file, remove it
            if isinstance(exc, BadZipFile):
                print(f"Removing bad {zip_path=}")
                os.remove(zip_path)
            continue
    except KeyboardInterrupt:
        break

n_new_zip_files = glob(f"{ph_docs_dir}/*.zip")
print(f"{n_new_zip_files - n_prev_zip_files} new zip files fetched")


# %% convert phonondb docs to lzma compressed JSON which is much faster to load
zip_files = glob(f"{ph_docs_dir}/mp-*-pbe.zip")
lzma_files = glob(f"{ph_docs_dir}/mp-*-pbe.json.lzma")

ids_todo = {
    re.match(r"(mp-\d+)-", zip_path.split("/")[-1])[1] for zip_path in zip_files
} - {re.match(r"(mp-\d+)-", lzma_path.split("/")[-1])[1] for lzma_path in lzma_files}

print(
    f"total downloaded: {len(zip_files)}, total converted: {len(lzma_files)}, "
    f"left todo: {len(ids_todo)}"
)


# %%
for mat_id in (pbar := tqdm(ids_todo, desc="Parsing PhononDB docs to PMG lzma")):
    # mat_id = "-".join(zip_path.split("/")[-1].split("-")[:2])
    if not re.match(r"mp-\d+", mat_id):
        raise ValueError(f"Invalid {mat_id=}")
    existing_lzma_docs = glob(f"{ph_docs_dir}/{mat_id}-*-pbe.json.lzma")
    if len(existing_lzma_docs) > 1:
        raise RuntimeError(f"> 1 doc for {mat_id=}: {existing_lzma_docs}")
    zip_docs = glob(f"{ph_docs_dir}/{mat_id}-*-pbe.zip")
    if len(zip_docs) > 1:
        raise RuntimeError(f"> 1 doc for {mat_id=}: {zip_docs}")

    pbar.set_postfix_str(mat_id)
    phonondb_doc_to_pmg_lzma(zip_docs[0], existing="skip")
