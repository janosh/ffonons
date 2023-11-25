import gzip
import json
import os

import pytest
from pymatgen.phonon.dos import PhononDos

from ffonons import DATA_DIR, FIGS_DIR, ROOT, dos_key, find_last_dos_peak


@pytest.fixture(scope="session")
def mace_dos() -> PhononDos:
    with gzip.open(f"{DATA_DIR}/mp-149-Si2/phonon-bs-dos-mace.json.gz", "rt") as file:
        doc_dict = json.load(file)

    return PhononDos.from_dict(doc_dict[dos_key])


def test_root() -> None:
    assert os.path.isdir(ROOT)
    assert os.path.isdir(FIGS_DIR)
    assert os.path.isdir(DATA_DIR)


def test_find_last_dos_peak(mace_dos: PhononDos) -> None:
    last_peak = find_last_dos_peak(mace_dos)
    assert last_peak == pytest.approx(10.026070, abs=0.01)
