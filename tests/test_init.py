import json
import os

import pytest
from monty.io import zopen
from pymatgen.phonon import PhononDos

from ffonons import DATA_DIR, PDF_FIGS, ROOT
from ffonons.enums import Key


@pytest.fixture(scope="session")
def mp_661_mace_dos() -> PhononDos:
    with zopen(
        f"{DATA_DIR}/phonon-db/mp-661-Al2N2-mace-y7uhwpje.json.lzma", "rt"
    ) as file:
        return PhononDos.from_dict(json.load(file)[Key.dos])


@pytest.fixture(scope="session")
def mp_2789_pbe_dos() -> PhononDos:
    with zopen(f"{DATA_DIR}/phonon-db/mp-2789-N12O24-pbe.json.lzma", "rt") as file:
        return PhononDos.from_dict(json.load(file)[Key.dos])


def test_root() -> None:
    assert os.path.isdir(ROOT)
    assert os.path.isdir(PDF_FIGS)
    assert os.path.isdir(DATA_DIR)


def test_find_last_dos_peak(
    mp_661_mace_dos: PhononDos, mp_2789_pbe_dos: PhononDos
) -> None:
    # this test was written before find_last_dos_peak() became PhononDos.get_last_peak()
    # in pymatgen
    last_mp_661_peak = mp_661_mace_dos.get_last_peak()
    assert last_mp_661_peak == pytest.approx(19.55269, abs=0.01)

    # example material with only 1 high peak and all others tiny: mp-2789
    last_mp_2789_peak = mp_2789_pbe_dos.get_last_peak()
    assert last_mp_2789_peak == pytest.approx(55.659, abs=0.01)
