import json
import os

import pytest
from monty.io import zopen
from pymatgen.phonon import PhononDos

from ffonons import DATA_DIR, FIGS_DIR, ROOT, dos_key, find_last_dos_peak


@pytest.fixture(scope="session")
def mp_149_mace_dos() -> PhononDos:
    with zopen(f"{DATA_DIR}/mp/mp-149-Si2-mace.json.gz", "rt") as file:
        return PhononDos.from_dict(json.load(file)[dos_key])


@pytest.fixture(scope="session")
def mp_2789_pbe_dos() -> PhononDos:
    with zopen(f"{DATA_DIR}/phonon-db/mp-2789-N12O24-pbe.json.lzma", "rt") as file:
        return PhononDos.from_dict(json.load(file)[dos_key])


def test_root() -> None:
    assert os.path.isdir(ROOT)
    assert os.path.isdir(FIGS_DIR)
    assert os.path.isdir(DATA_DIR)


def test_find_last_dos_peak(
    mp_149_mace_dos: PhononDos, mp_2789_pbe_dos: PhononDos
) -> None:
    last_mp_149_peak = find_last_dos_peak(mp_149_mace_dos)
    assert last_mp_149_peak == pytest.approx(10.057, abs=0.01)

    # example material with only 1 high peak and all others tiny: mp-2789
    last_mp_2789_peak = find_last_dos_peak(mp_2789_pbe_dos)
    assert last_mp_2789_peak == pytest.approx(55.659, abs=0.01)
