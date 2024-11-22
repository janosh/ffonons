import json
import lzma
import os
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.phonon import PhononBandStructureSymmLine, PhononDos

from ffonons import TEST_FILES
from ffonons.dbs.phonondb import (
    PhononDBDocParsed,
    fetch_togo_doc_by_id,
    get_phonopy_kpath,
    parse_phonondb_docs,
    phonondb_doc_to_pmg_lzma,
    scrape_and_fetch_togo_docs_from_page,
)
from ffonons.enums import KpathScheme


# Mock global variables
@pytest.fixture(autouse=True)
def _mock_mp_togo_id_maps() -> Generator[None, None, None]:
    with (
        patch("ffonons.dbs.phonondb.map_mp_to_togo_id", {"mp-1": "1"}),
        patch("ffonons.dbs.phonondb.map_togo_to_mp_id", {"1": "mp-1"}),
    ):
        yield


def test_fetch_togo_doc_by_id_existing_file(tmp_path: Path) -> None:
    file_path = tmp_path / "mp-1-1-pbe.zip"
    file_path.touch()

    result = fetch_togo_doc_by_id("mp-1", str(file_path))

    assert result == str(file_path)


@patch("ffonons.dbs.phonondb.requests.get")
def test_fetch_togo_doc_by_id_download(mock_get: MagicMock, tmp_path: Path) -> None:
    mock_get.return_value.content = b"mock content"

    result = fetch_togo_doc_by_id("mp-1", f"{tmp_path}/mp-1-1-pbe.zip")

    assert result == f"{tmp_path}/mp-1-1-pbe.zip"
    assert (tmp_path / "mp-1-1-pbe.zip").read_bytes() == b"mock content"


@patch("ffonons.dbs.phonondb.requests.get")
@patch("ffonons.dbs.phonondb.BeautifulSoup")
@patch("ffonons.dbs.phonondb.os.path.isfile", return_value=False)
@patch("ffonons.dbs.phonondb.open", new_callable=MagicMock)
def test_scrape_and_fetch_togo_docs_from_page(
    mock_open: MagicMock,  # noqa: ARG001
    mock_isfile: MagicMock,  # noqa: ARG001
    mock_beau_soup: MagicMock,
    mock_get: MagicMock,
) -> None:
    mock_get.return_value.text = "<html>mock content</html>"
    mock_beau_soup.return_value.find_all.return_value = [
        MagicMock(prettify=lambda: '<tr id="document_1"><a class="">1</a></tr>')
    ]
    mock_get.return_value.status_code = 200

    result = scrape_and_fetch_togo_docs_from_page("http://mock.url")

    assert isinstance(result, pd.DataFrame)
    assert "doc_ids" in result.columns
    assert "download_urls" in result.columns


def test_get_phonopy_kpath() -> None:
    struct = Structure(
        lattice=Lattice.cubic(3),
        species=("Fe", "Fe"),
        coords=((0, 0, 0), (0.5, 0.5, 0.5)),
    )
    result = get_phonopy_kpath(struct, KpathScheme.seekpath, symprec=1e-5)

    assert isinstance(result, tuple)
    assert len(result) == 2


phonondb_zip_file_path = f"{TEST_FILES}/phonondb/mp-643101-k3569900j-pbe.zip"


def test_phonondb_doc_to_pmg_lzma(tmp_path: Path) -> None:
    pmg_doc_path = f"{tmp_path}/mp-643101-k3569900j-pbe.json.xz"
    out_path = phonondb_doc_to_pmg_lzma(
        phonondb_zip_file_path, pmg_doc_path=pmg_doc_path
    )
    assert pmg_doc_path == out_path
    assert os.path.isfile(out_path)

    # load file and check content
    with lzma.open(out_path, "rt") as file:
        doc = json.load(file)
    assert isinstance(doc, dict)
    assert list(doc) == [
        "structure",
        "primitive",
        "supercell",
        "nac_params",
        "phonon_bandstructure",
        "phonon_dos",
        "free_energies",
        "internal_energies",
        "heat_capacities",
        "entropies",
        "temps",
        "has_imaginary_modes",
        "thermal_displacement_data",
        "mp_id",
        "formula",
        "@module",
        "@class",
        "@version",
    ]


def test_parse_phonondb_docs() -> None:
    ph_doc = parse_phonondb_docs(phonondb_zip_file_path)
    assert isinstance(ph_doc, PhononDBDocParsed)
    for key, typ in dict(
        structure=Structure,
        primitive=Structure,
        supercell=np.ndarray,
        nac_params=(dict, type(None)),
        phonon_bandstructure=PhononBandStructureSymmLine,
        phonon_dos=PhononDos,
        free_energies=list,
        internal_energies=list,
        heat_capacities=list,
        entropies=list,
        temps=list,
        has_imaginary_modes=(bool, np.bool_),
        thermal_displacement_data=(dict, type(None)),
        mp_id=(str, type(None)),
        formula=(str, type(None)),
    ).items():
        val = getattr(ph_doc, key)
        assert isinstance(val, typ), f"{key=}, {typ=}, {val=}"
