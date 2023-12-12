import json
import os

import pytest
from monty.io import zopen
from pymatgen.phonon.dos import PhononDos

from ffonons import DATA_DIR, FIGS_DIR, ROOT, dos_key, find_last_dos_peak


@pytest.fixture(scope="session")
def mace_dos() -> PhononDos:
    with zopen(f"{DATA_DIR}/phonon-bs-dos/mp-149-Si2-mace.json.gz", "rt") as file:
        doc_dict = json.load(file)

    return PhononDos.from_dict(doc_dict[dos_key])


def test_root() -> None:
    assert os.path.isdir(ROOT)
    assert os.path.isdir(FIGS_DIR)
    assert os.path.isdir(DATA_DIR)


def test_find_last_dos_peak(mace_dos: PhononDos) -> None:
    last_peak = find_last_dos_peak(mace_dos)
    assert last_peak == pytest.approx(10.057336, abs=0.01)
