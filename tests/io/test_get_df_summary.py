from pathlib import Path
from unittest.mock import patch

import pandas as pd
from pymatviz.enums import Key

import ffonons
from ffonons.enums import DB, PhKey
from ffonons.io import get_df_summary
from tests.conftest import mock_phonon_docs


def test_get_df_summary_basic() -> None:
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


def test_get_df_summary_with_cache(mock_data_dir: Path) -> None:
    cache_path = mock_data_dir / "mp" / "df-summary-tol=0.01.csv.gz"
    cache_path.parent.mkdir(parents=True)

    assert not cache_path.exists()

    # Test cache creation
    with patch("ffonons.io.load_pymatgen_phonon_docs", return_value=mock_phonon_docs):
        df_summary = ffonons.io.get_df_summary("mp", cache_path=str(cache_path))

    assert cache_path.exists()

    # Test cache loading
    with patch("ffonons.io.load_pymatgen_phonon_docs") as mock_load:
        df_summary_cached = ffonons.io.get_df_summary(
            "mp", cache_path=str(cache_path), refresh_cache=False
        )

    mock_load.assert_not_called()
    # check_dtype=False since reloaded df has dtype=int32 on Windows, orig has int64
    pd.testing.assert_frame_equal(df_summary, df_summary_cached, check_dtype=False)
