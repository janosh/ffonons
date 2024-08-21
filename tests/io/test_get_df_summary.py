from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from pymatgen.core import Lattice, Structure
from pymatviz.enums import Key

from ffonons.enums import DB, PhKey
from ffonons.io import get_df_summary
from tests.conftest import mock_phonon_docs


def test_get_df_summary_basic(capsys: pytest.CaptureFixture[str]) -> None:
    with patch("ffonons.io.load_pymatgen_phonon_docs", return_value=mock_phonon_docs):
        df_summary = get_df_summary(DB.phonon_db, cache_path=None)

    assert isinstance(df_summary, pd.DataFrame)
    assert df_summary.index.names == [str(Key.mat_id), str(Key.model)]
    expected_columns = {
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
    assert set(df_summary.columns) >= expected_columns

    ref_key = "missing_key"
    with patch("ffonons.io.load_pymatgen_phonon_docs", return_value=mock_phonon_docs):
        df_summary = get_df_summary(DB.phonon_db, ref_key=ref_key, cache_path=None)

    mat_id = next(iter(mock_phonon_docs))
    stdout, stderr = capsys.readouterr()
    assert (
        f"Skipping {mat_id=}, no {ref_key} doc found, please (re-)generate!\n" in stdout
    )
    assert stderr == ""


def test_get_df_summary_with_cache(mock_data_dir: Path) -> None:
    cache_path = mock_data_dir / DB.mp / "df-summary-tol=0.01.csv.gz"
    cache_path.parent.mkdir(parents=True)

    assert not cache_path.exists()

    # Test cache creation
    with patch("ffonons.io.load_pymatgen_phonon_docs", return_value=mock_phonon_docs):
        df_summary = get_df_summary(DB.mp, cache_path=str(cache_path))

    assert cache_path.exists()

    # Test cache loading
    with patch("ffonons.io.load_pymatgen_phonon_docs") as mock_load:
        df_summary_cached = get_df_summary(
            DB.mp, cache_path=str(cache_path), refresh_cache=False
        )

    mock_load.assert_not_called()
    # check_dtype=False since reloaded df has dtype=int32 on Windows, orig has int64
    pd.testing.assert_frame_equal(df_summary, df_summary_cached, check_dtype=False)


def test_get_df_summary_refresh_cache(mock_data_dir: Path) -> None:
    cache_path = mock_data_dir / DB.mp / "df-summary-tol=0.01.csv.gz"
    cache_path.parent.mkdir(parents=True)

    # Create initial cache
    with patch("ffonons.io.load_pymatgen_phonon_docs", return_value=mock_phonon_docs):
        get_df_summary(DB.mp, cache_path=str(cache_path))

    # Modify mock_phonon_docs
    mock_phonon_docs["mp-1"]["ml_model"].phonon_dos.get_last_peak.return_value = 11.0

    # Refresh cache
    with patch("ffonons.io.load_pymatgen_phonon_docs", return_value=mock_phonon_docs):
        df_summary_refreshed = get_df_summary(
            DB.mp, cache_path=str(cache_path), refresh_cache=True
        )

    assert df_summary_refreshed.loc[("mp-1", "ml_model"), Key.last_ph_dos_peak] == 11.0


def test_get_df_summary_incremental_refresh(mock_data_dir: Path) -> None:
    cache_path = mock_data_dir / DB.mp / "df-summary-tol=0.01.csv.gz"
    cache_path.parent.mkdir(parents=True)

    # Create initial cache
    with patch("ffonons.io.load_pymatgen_phonon_docs", return_value=mock_phonon_docs):
        get_df_summary(DB.mp, cache_path=str(cache_path))

    # Add new material to mock_phonon_docs
    new_doc = deepcopy(mock_phonon_docs["mp-1"]["ml_model"])
    new_structure = Structure(
        Lattice.cubic(4.2), ["Mg", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    new_doc.structure = new_structure
    mock_phonon_docs["mp-2"] = {"pbe": new_doc, "ml_model": new_doc}

    # Perform incremental refresh
    with patch("ffonons.io.load_pymatgen_phonon_docs", return_value=mock_phonon_docs):
        df_summary_incremental = get_df_summary(
            DB.mp, cache_path=str(cache_path), refresh_cache="incremental"
        )

    assert "mp-2" in df_summary_incremental.index.get_level_values(0)


def test_get_df_summary_imaginary_freq_tol() -> None:
    with patch("ffonons.io.load_pymatgen_phonon_docs", return_value=mock_phonon_docs):
        df_summary = get_df_summary(
            DB.phonon_db, cache_path=None, imaginary_freq_tol=0.1
        )

    assert Key.has_imag_ph_modes in df_summary.columns
    assert Key.has_imag_ph_gamma_modes in df_summary.columns
