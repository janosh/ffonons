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


@pytest.fixture
def mock_phonon_docs() -> dict[str, dict[str, PhononBSDOSDoc]]:
    structure = Structure(np.eye(3) * 5, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    mock_band_structure = MagicMock(spec=PhononBandStructureSymmLine)
    mock_band_structure.bands = np.array([[-1, 0, 1], [2, 3, 4]])
    mock_band_structure.has_imaginary_freq.return_value = True
    mock_band_structure.has_imaginary_gamma_freq.return_value = False
    mock_dos = MagicMock(spec=PhononDos)
    mock_dos.get_last_peak.return_value = 10.5
    mock_dos.mae.return_value = 0.1
    mock_dos.r2_score.return_value = 0.95

    phonon_doc = PhononBSDOSDoc(
        structure=structure,
        supercell=np.eye(3) * 2,
        phonon_bandstructure=mock_band_structure,
        phonon_dos=mock_dos,
        has_imaginary_modes=False,
    )
    return {"mp-1": {"pbe": phonon_doc, "ml_model": phonon_doc}}


@pytest.fixture(scope="session")
def mp_661_mace_dos() -> PhononDos:
    mace_ph_dos_path = f"{TEST_FILES}/phonondb/mp-661-Al2N2-mace-y7uhwpje.json.xz"
    with zopen(mace_ph_dos_path, mode="rt") as file:
        return PhononDos.from_dict(json.load(file)[Key.ph_dos])


@pytest.fixture(scope="session")
def mp_2789_pbe_dos() -> PhononDos:
    phonondb_ph_dos_path = f"{TEST_FILES}/phonondb/mp-2789-N12O24-pbe.json.xz"
    with zopen(phonondb_ph_dos_path, mode="rt") as file:
        return PhononDos.from_dict(json.load(file)[Key.ph_dos])
