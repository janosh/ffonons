import json
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from atomate2.common.schemas.phonons import PhononBSDOSDoc
from monty.io import zopen
from pymatgen.core import Structure
from pymatgen.phonon import PhononBandStructureSymmLine, PhononDos
from pymatgen.phonon.dos import PhononDosFingerprint
from pymatviz.enums import Key

from ffonons import TEST_FILES

summary_csv_url = "https://github.com/janosh/ffonons/raw/3d1d39e9/data/phonon-db/df-summary-tol=0.01.csv.gz"
df_preds_mock = pd.read_csv(summary_csv_url, index_col=[0, 1])


@pytest.fixture
def mock_data_dir(tmp_path: Path) -> Generator[Path, None, None]:
    with (
        patch("ffonons.DATA_DIR", str(tmp_path)),
        patch("ffonons.dbs.mp.DATA_DIR", str(tmp_path)),
    ):
        yield tmp_path


def get_mock_ph_doc(formula: str = "Na Cl") -> PhononBSDOSDoc:
    struct = Structure(np.eye(3) * 5, formula.split(), [[0, 0, 0], [0.5, 0.5, 0.5]])
    mock_band_structure = MagicMock(spec=PhononBandStructureSymmLine)
    mock_band_structure.bands = np.array([[-1, 0, 1], [2, 3, 4]])
    mock_band_structure.has_imaginary_freq.return_value = True
    mock_band_structure.has_imaginary_gamma_freq.return_value = False

    frequencies = np.linspace(0, 20, 100)
    densities = np.random.default_rng().random(100)
    mock_dos = MagicMock(spec=PhononDos)
    mock_dos.frequencies = frequencies
    mock_dos.densities = densities
    mock_dos.get_last_peak.return_value = 11
    mock_dos.mae.return_value = 0.1
    mock_dos.r2_score.return_value = 0.95

    ph_dos_fp = PhononDosFingerprint(
        frequencies=np.array([frequencies]),
        densities=densities,
        n_bins=100,
        bin_width=0.2,
    )
    mock_dos.get_dos_fp.return_value = ph_dos_fp

    return PhononBSDOSDoc(
        structure=struct,
        supercell=np.eye(3) * 2,
        phonon_bandstructure=mock_band_structure,
        phonon_dos=mock_dos,
        has_imaginary_modes=False,
    )


mock_phonon_docs = {
    "mp-1": {"pbe": get_mock_ph_doc(), "ml_model": get_mock_ph_doc(formula="Al O")}
}


@pytest.fixture
def mock_phonon_docs_fixture() -> dict[str, dict[str, PhononBSDOSDoc]]:
    """Prefer mock_phonon_docs over mock_phonon_docs_fixture since more sensitive to
    side-effects since fixture will regenerate the mock_phonon_docs for every test
    function.
    """
    return mock_phonon_docs


@pytest.fixture(scope="session")
def mp_661_mace_dos() -> PhononDos:
    mace_ph_dos_path = f"{TEST_FILES}/phonondb/mp-661-Al2N2-mace-y7uhwpje.json.lzma"
    with zopen(mace_ph_dos_path, mode="rt") as file:
        return PhononDos.from_dict(json.load(file)[Key.ph_dos])


@pytest.fixture(scope="session")
def mp_2789_pbe_dos() -> PhononDos:
    phonondb_ph_dos_path = f"{TEST_FILES}/phonondb/mp-2789-N12O24-pbe.json.lzma"
    with zopen(phonondb_ph_dos_path, mode="rt") as file:
        return PhononDos.from_dict(json.load(file)[Key.ph_dos])
