from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pymatgen.phonon import PhononDos
from pymatviz.enums import Key

import ffonons


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
            structures = ffonons.io.get_gnome_pmg_structures(str(mock_zip_path), ids=2)

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
            structures = ffonons.io.get_gnome_pmg_structures(
                str(mock_zip_path), ids=["mp-1", "mp-3"]
            )

    assert len(structures) == 2
    assert "mp-1" in structures
    assert "mp-3" in structures
    assert "mp-2" not in structures


# --- test_find_last_dos_peak is in test_io.py because it's used by
# ffonons.io.ffonons.io.get_df_summary. get_last_peak was first implemented as
# find_last_dos_peak() in ffonons and later upstreamed into pymatgen
def test_get_last_peak(mp_661_mace_dos: PhononDos, mp_2789_pbe_dos: PhononDos) -> None:
    # this test was written before find_last_dos_peak() became PhononDos.get_last_peak()
    # in pymatgen
    last_mp_661_peak = mp_661_mace_dos.get_last_peak()
    assert last_mp_661_peak == pytest.approx(19.55269, abs=0.01)

    # example material with only 1 high peak and all others tiny: mp-2789
    last_mp_2789_peak = mp_2789_pbe_dos.get_last_peak()
    assert last_mp_2789_peak == pytest.approx(55.659, abs=0.01)
