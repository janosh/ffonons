from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from atomate2.common.schemas.phonons import PhononBSDOSDoc
from pymatgen.core import Lattice, Structure
from pymatgen.phonon import PhononBandStructureSymmLine, PhononDos

import ffonons
from ffonons.enums import DB
from tests.conftest import mace_ph_doc_path, phonondb_ph_doc_path


def test_load_pymatgen_phonon_docs(mock_data_dir: Path) -> None:
    mp_dir = mock_data_dir / "mp"
    mp_dir.mkdir()
    (mp_dir / "mp-1-NaCl-pbe.json.gz").touch()
    (mp_dir / "mp-2-MgO-ml_model.json.lzma").touch()

    mock_ph_doc = MagicMock(spec=PhononBSDOSDoc)
    mock_ph_doc.structure = Structure(
        Lattice.cubic(5.0), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    mock_ph_doc.supercell = np.eye(3) * 2
    mock_ph_doc.phonon_bandstructure = MagicMock(spec=PhononBandStructureSymmLine)
    mock_ph_doc.phonon_dos = MagicMock(spec=PhononDos)

    with (
        patch("ffonons.io.zopen"),
        patch("ffonons.io.json.load", return_value=mock_ph_doc),
        patch("ffonons.io.re.search") as mock_search,
    ):
        # Mock the regex search to return the expected groups
        mock_search.return_value.groups.side_effect = [
            ("mp-1", "NaCl", "pbe"),
            ("mp-2", "MgO", "ml_model"),
        ]
        result = ffonons.io.load_pymatgen_phonon_docs(docs_to_load=DB.mp)

    assert len(result) == 2
    assert "mp-1" in result
    assert "mp-2" in result
    assert "pbe" in result["mp-1"]
    assert "ml_model" in result["mp-2"]
    assert hasattr(result["mp-1"]["pbe"], "file_path")


def test_load_pymatgen_phonon_docs_with_glob_pattern(mock_data_dir: Path) -> None:
    mp_dir = mock_data_dir / "mp"
    mp_dir.mkdir()
    (mp_dir / "mp-1-NaCl-pbe.json.gz").touch()
    (mp_dir / "mp-2-MgO-ml_model.json.lzma").touch()
    (mp_dir / "mp-3-CaF2-pbe.json.gz").touch()

    mock_ph_doc = MagicMock(spec=ffonons.io.PhononBSDOSDoc)

    with (
        patch("ffonons.io.zopen"),
        patch("ffonons.io.json.load", return_value=mock_ph_doc),
        patch("ffonons.io.re.search") as mock_search,
    ):
        mock_search.return_value.groups.side_effect = [
            ("mp-1", "NaCl", "pbe"),
            ("mp-3", "CaF2", "pbe"),
        ]
        result = ffonons.io.load_pymatgen_phonon_docs(
            docs_to_load=DB.mp, glob_patt="*-pbe.json.gz"
        )

    assert len(result) == 2
    assert "mp-1" in result
    assert "mp-3" in result
    assert "mp-2" not in result


def test_load_pymatgen_phonon_docs_with_materials_ids(mock_data_dir: Path) -> None:
    mp_dir = mock_data_dir / "mp"
    mp_dir.mkdir()
    (mp_dir / "mp-1-NaCl-pbe.json.gz").touch()
    (mp_dir / "mp-2-MgO-ml_model.json.lzma").touch()

    mock_ph_doc = MagicMock(spec=ffonons.io.PhononBSDOSDoc)

    with (
        patch("ffonons.io.zopen"),
        patch("ffonons.io.json.load", return_value=mock_ph_doc),
        patch("ffonons.io.re.search") as mock_search,
    ):
        mock_search.return_value.groups.side_effect = [("mp-1", "NaCl", "pbe")]
        result = ffonons.io.load_pymatgen_phonon_docs(
            docs_to_load=DB.mp, materials_ids=["mp-1"]
        )

    assert len(result) == 1
    assert "mp-1" in result
    assert "mp-2" not in result


def test_load_pymatgen_phonon_docs_with_file_paths() -> None:
    mock_ph_doc = MagicMock(spec=ffonons.io.PhononBSDOSDoc)

    with (
        patch("ffonons.io.zopen"),
        patch("ffonons.io.json.load", return_value=mock_ph_doc),
        patch("ffonons.io.re.search") as mock_search,
    ):
        mock_search.return_value.groups.side_effect = [
            ("mp-1", "NaCl", "pbe"),
            ("mp-2", "MgO", "ml_model"),
        ]
        result = ffonons.io.load_pymatgen_phonon_docs(
            docs_to_load=[
                "/path/to/mp-1-NaCl-pbe.json.gz",
                "/path/to/mp-2-MgO-ml_model.json.lzma",
            ]
        )

    assert len(result) == 2
    assert "mp-1" in result
    assert "mp-2" in result


@pytest.mark.parametrize("docs_to_load", [None, {42: 51}, 123])
def test_load_pymatgen_phonon_docs_invalid_input(
    docs_to_load: dict | int | None,
) -> None:
    with pytest.raises(TypeError, match=f"Invalid {docs_to_load=}"):
        ffonons.io.load_pymatgen_phonon_docs(docs_to_load=docs_to_load)


def test_load_pymatgen_phonon_docs_file_read_error(mock_data_dir: Path) -> None:
    mp_dir = mock_data_dir / "mp"
    mp_dir.mkdir()
    (mp_dir / "mp-1-NaCl-pbe.json.gz").touch()

    with (
        patch("ffonons.io.zopen", side_effect=Exception("File read error")),
        patch("builtins.print") as mock_print,
    ):
        result = ffonons.io.load_pymatgen_phonon_docs(docs_to_load=DB.mp)

    assert len(result) == 0
    mock_print.assert_called_once()
    assert "File read error" in mock_print.call_args[0][0]


def test_load_pymatgen_phonon_docs_invalid_file_name() -> None:
    nonexistent_file = f"{phonondb_ph_doc_path}.nonexistent"
    with pytest.raises(
        FileNotFoundError, match=f"No such file or directory: {nonexistent_file!r}"
    ):
        ffonons.io.load_pymatgen_phonon_docs(docs_to_load=[nonexistent_file])


def test_load_pymatgen_phonon_docs_invalid_mp_id() -> None:
    with patch("ffonons.io.re.search") as mock_search:
        mock_search.return_value.groups.return_value = ("invalid-1", "NaCl", "pbe")

        with pytest.raises(ValueError, match="Invalid mp_id="):
            ffonons.io.load_pymatgen_phonon_docs(
                docs_to_load=[phonondb_ph_doc_path, mace_ph_doc_path]
            )
