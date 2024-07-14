from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from atomate2.common.schemas.phonons import PhononBSDOSDoc
from pymatgen.core import Lattice, Structure
from pymatgen.phonon import PhononBandStructureSymmLine, PhononDos
from pymatviz.enums import Key

from ffonons.enums import PhKey
from ffonons.io import (
    get_df_summary,
    get_gnome_pmg_structures,
    load_pymatgen_phonon_docs,
    update_key_name,
)


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
        result = load_pymatgen_phonon_docs(docs_to_load="mp")

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
        Key.last_ph_dos_peak,
        Key.max_ph_freq,
        Key.min_ph_freq,
        Key.ph_dos_mae,
        PhKey.ph_dos_r2,
        Key.has_imag_ph_modes,
        Key.has_imag_ph_gamma_modes,
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


# --- test_find_last_dos_peak is in test_io.py because it's used by
# ffonons.io.get_df_summary. get_last_peak was first implemented as find_last_dos_peak()
# in ffonons and later upstreamed into pymatgen
def test_get_last_peak(mp_661_mace_dos: PhononDos, mp_2789_pbe_dos: PhononDos) -> None:
    # this test was written before find_last_dos_peak() became PhononDos.get_last_peak()
    # in pymatgen
    last_mp_661_peak = mp_661_mace_dos.get_last_peak()
    assert last_mp_661_peak == pytest.approx(19.55269, abs=0.01)

    # example material with only 1 high peak and all others tiny: mp-2789
    last_mp_2789_peak = mp_2789_pbe_dos.get_last_peak()
    assert last_mp_2789_peak == pytest.approx(55.659, abs=0.01)
