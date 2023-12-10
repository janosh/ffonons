# %%
import os
import sys

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from ffonons import DATA_DIR

out_dir = f"{DATA_DIR}/phonon_db"

__author__ = "Janine George, Aakash Nair, Janosh Riebesell"
__date__ = "2023-12-07"


# %%
def get_mp_doc_ids_from_url(url: str) -> pd.DataFrame:
    """Extract doc id, MP ID from togo db and download the phonopy file."""
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
            mp_id = link_element.text.strip().split()[-1]
            out_path = f"{out_dir}/mp-{mp_id}-{doc_id}-pbe.zip"

            if os.path.isfile(out_path):
                print(f"{out_path=} already exists. skipping")
                continue

            # Get the extracted information
            doc_ids.append(doc_id)
            mp_ids.append(f"mp-{mp_id}")

            # curate download url
            download_url = "https://mdr.nims.go.jp/download_all/" + doc_id + ".zip"
            download_urls.append(download_url)
            z = requests.get(download_url, allow_redirects=True, timeout=15)
            with open(out_path, "wb") as file:
                file.write(z.content)

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
        rows.append(result)
    except Exception as exc:
        print(exc, file=sys.stderr)


# %%
df = pd.concat(map(bool, rows))
df = df.sort_index()
