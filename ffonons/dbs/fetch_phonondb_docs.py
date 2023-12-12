# %%
import json
import lzma
import os
import re
import sys
from glob import glob

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from ffonons import DATA_DIR, bs_key, dos_key, struct_key
from ffonons.io import parse_phonondb_docs

db_name = "phonon_db"
ph_docs_dir = f"{DATA_DIR}/{db_name}"

__author__ = "Janine George, Aakash Nair, Janosh Riebesell"
__date__ = "2023-12-07"


# %%
def get_mp_doc_ids_from_url(url: str) -> pd.DataFrame:
    """Extract doc id, MP ID from Togo DB and download the phonopy file."""
    # Send an HTTP request to the URL
    response = requests.get(url, timeout=15)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract all links from the page
        tables = [
            tr.prettify() for tr in soup.find_all("tr") if "document_" in tr.prettify()
        ]

        doc_ids, mp_ids, download_urls = [], [], []

        for table in tables:
            soup1 = BeautifulSoup(table, "html.parser")

            # Find the relevant elements within the table row
            doc_id = soup1.find("tr")["id"].split("_")[-1]
            link_element = soup1.find_all("a", class_="")[0]
            mp_id = f"mp-{link_element.text.strip().split()[-1]}"
            out_path = f"{ph_docs_dir}/{mp_id}-{doc_id}-pbe.zip"

            if os.path.isfile(out_path):
                print(f"{out_path=} already exists. skipping")
                continue

            doc_ids += doc_id
            mp_ids += mp_id

            download_url = f"https://mdr.nims.go.jp/download_all/{doc_id}.zip"
            download_urls += download_url
            resp = requests.get(download_url, allow_redirects=True, timeout=15)
            with open(out_path, "wb") as file:
                file.write(resp.content)

        df_out = pd.DataFrame(index=mp_ids, columns=["doc_ids", "download_urls"])
        df_out["doc_ids"] = doc_ids
        df_out["download_urls"] = download_urls

        return df_out
    # Print an error message if the request was not successful
    print(f"Error: Unable to fetch the page. Status code: {response.status_code}")
    return None


# %% get all phonon_db page urls
urls = [
    f"https://mdr.nims.go.jp/collections/8g84ms862?{page=}" for page in range(1, 1005)
]

rows = []
for url in tqdm(urls, desc="Downloading Togo Phonopy DB"):
    try:
        result = get_mp_doc_ids_from_url(url)
        rows += result
    except Exception as exc:
        print(exc, file=sys.stderr)


# %%
df = pd.concat(map(bool, rows))
df = df.sort_index()


# %%
desc = "Processing PhononDB docs"
for zip_path in (pbar := tqdm(glob(f"{ph_docs_dir}/mp-*-pbe.zip"), desc=desc)):
    mat_id = "-".join(zip_path.split("/")[-1].split("-")[:2])
    assert re.match(r"mp-\d+", mat_id), f"Invalid {mat_id=}"
    existing_docs = glob(f"{ph_docs_dir}/{mat_id}-*-pbe.json.lzma")
    # if material was already processed, skip
    if len(existing_docs) == 1:
        continue
    if len(existing_docs) > 1:
        raise RuntimeError(f"> 1 doc for {mat_id=}: {existing_docs}")

    pbar.set_description(f"{mat_id}")
    phonon_db_results = parse_phonondb_docs(zip_path, nac=False)

    struct = phonon_db_results["unit_cell"]
    formula = struct.formula.replace(" ", "")
    id_formula = f"{mat_id}-{formula}"

    dft_doc_dict = {
        dos_key: phonon_db_results["phonon_dos"].as_dict(),
        bs_key: phonon_db_results["phonon_bandstructure"].as_dict(),
        struct_key: struct.as_dict(),
        "mp_id": mat_id,
    }
    dft_doc_path = f"{ph_docs_dir}/{mat_id}-{formula}-pbe.json.lzma"

    with lzma.open(dft_doc_path, "wt") as file:
        file.write(json.dumps(dft_doc_dict))
