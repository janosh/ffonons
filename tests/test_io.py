from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from atomate2.common.schemas.phonons import PhononBSDOSDoc
from pymatgen.core import Lattice, Structure
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos

from ffonons.enums import Key
from ffonons.io import (
    get_df_summary,
    get_gnome_pmg_structures,
    load_pymatgen_phonon_docs,
    update_key_name,
)


@pytest.fixture()
def mock_data_dir(tmp_path: Path) -> Generator[Path, None, None]:
    with patch("ffonons.io.DATA_DIR", str(tmp_path)):
        yield tmp_path


@pytest.fixture()
def mock_phonon_docs() -> dict[str, dict[str, PhononBSDOSDoc]]:
    structure = Structure(
        Lattice.cubic(5.0), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    mock_band_structure = MagicMock(spec=PhononBandStructureSymmLine)
    mock_band_structure.bands = np.array([[-1, 0, 1], [2, 3, 4]])
    mock_band_structure.has_imaginary_freq.return_value = True
    mock_band_structure.has_imaginary_gamma_freq.return_value = False
    mock_dos = MagicMock(spec=PhononDos)
    mock_dos.get_last_peak.return_value = 10.5
    mock_dos.mae.return_value = 0.1
    mock_dos.r2_score.return_value = 0.95

    return {
        "mp-1": {
            "pbe": PhononBSDOSDoc(
                structure=structure,
                supercell=np.eye(3) * 2,
                phonon_bandstructure=mock_band_structure,
                phonon_dos=mock_dos,
                has_imaginary_modes=False,
            ),
            "ml_model": PhononBSDOSDoc(
                structure=structure,
                supercell=np.eye(3) * 2,
                phonon_bandstructure=mock_band_structure,
                phonon_dos=mock_dos,
                has_imaginary_modes=False,
            ),
        }
    }


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
        result = load_pymatgen_phonon_docs(which_db="mp")

    assert len(result) == 2
    assert "mp-1" in result
    assert "mp-2" in result
    assert "pbe" in result["mp-1"]
    assert "ml_model" in result["mp-2"]
    assert hasattr(result["mp-1"]["pbe"], "file_path")


def test_update_key_name(mock_data_dir: Path) -> None:
    test_dir = mock_data_dir / "test_update"
    test_dir.mkdir()
    (test_dir / "test_file.json.gz").touch()

    with (
        patch("ffonons.io.json.load") as mock_json_load,
        patch("ffonons.io.json.dump") as mock_json_dump,
    ):
        mock_json_load.return_value = {"old_key": "value"}
        update_key_name(str(test_dir), {"old_key": "new_key"})

    mock_json_dump.assert_called_once()
    args, _ = mock_json_dump.call_args
    assert "new_key" in args[0]
    assert "old_key" not in args[0]


def test_get_df_summary(mock_phonon_docs: dict[str, dict[str, PhononBSDOSDoc]]) -> None:
    with patch("ffonons.io.load_pymatgen_phonon_docs", return_value=mock_phonon_docs):
        df_summary = get_df_summary("mp", cache_path=None)

    assert isinstance(df_summary, pd.DataFrame)
    assert df_summary.index.names == [str(Key.mat_id), "model"]
    assert set(df_summary.columns) == {
        Key.formula,
        Key.n_sites,
        Key.supercell,
        Key.last_dos_peak,
        Key.max_freq,
        Key.min_freq,
        Key.dos_mae,
        Key.ph_dos_r2,
        Key.has_imag_freq,
        Key.has_imag_gamma_freq,
    }


def test_get_df_summary_with_cache(
    mock_data_dir: Path, mock_phonon_docs: dict[str, dict[str, PhononBSDOSDoc]]
) -> None:
    cache_path = mock_data_dir / "mp" / "df-summary-tol=0.01.csv.gz"
    cache_path.parent.mkdir(parents=True)

    assert not cache_path.exists()

    # Test cache creation
    with patch("ffonons.io.load_pymatgen_phonon_docs", return_value=mock_phonon_docs):
        df_summary = get_df_summary("mp", cache_path=str(cache_path))

    assert cache_path.exists()

    # Test cache loading
    with patch("ffonons.io.load_pymatgen_phonon_docs") as mock_load:
        df_summary_cached = get_df_summary("mp", cache_path=str(cache_path))

    mock_load.assert_not_called()
    pd.testing.assert_frame_equal(df_summary, df_summary_cached)


def test_get_gnome_pmg_structures(tmp_path: Path) -> None:
    mock_zip_path = tmp_path / "test.zip"
    mock_zip_path.touch()

    with patch("ffonons.io.ZipFile") as mock_zipfile:
        mock_zipfile.return_value.__enter__.return_value.namelist.return_value = [
            "by_id/mp-1.CIF",
            "by_id/mp-2.CIF",
        ]
        mock_zip_open = (
            mock_zipfile.return_value.__enter__.return_value.open.return_value.__enter__
        )
        mock_zip_open.return_value.read.return_value = b"mock CIF content"

        with patch("ffonons.io.Structure") as mock_structure:
            mock_structure.from_str.return_value = MagicMock(properties={})
            structures = get_gnome_pmg_structures(str(mock_zip_path), ids=2)

    assert len(structures) == 2
    assert "mp-1" in structures
    assert "mp-2" in structures
    for struct in structures.values():
        assert Key.mat_id in struct.properties


def test_get_gnome_pmg_structures_with_specific_ids(tmp_path: Path) -> None:
    mock_zip_path = tmp_path / "test.zip"
    mock_zip_path.touch()

    with patch("ffonons.io.ZipFile") as mock_zipfile:
        mock_zipfile.return_value.__enter__.return_value.namelist.return_value = [
            "by_id/mp-1.CIF",
            "by_id/mp-2.CIF",
            "by_id/mp-3.CIF",
        ]
        mock_zip_open = (
            mock_zipfile.return_value.__enter__.return_value.open.return_value.__enter__
        )
        mock_zip_open.return_value.read.return_value = b"mock CIF content"

        with patch("ffonons.io.Structure") as mock_structure:
            mock_structure.from_str.return_value = MagicMock(properties={})
            structures = get_gnome_pmg_structures(
                str(mock_zip_path), ids=["mp-1", "mp-3"]
            )

    assert len(structures) == 2
    assert "mp-1" in structures
    assert "mp-3" in structures
    assert "mp-2" not in structures
