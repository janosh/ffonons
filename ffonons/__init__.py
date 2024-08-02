"""The ffonons package helps predict and analyze phonons with machine learning
force fields.
"""

import os
from datetime import UTC, datetime
from typing import Literal, get_args

import plotly.express as px
import pymatviz as pmv
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
from pymatgen.electronic_structure.dos import CompleteDos, Dos
from pymatgen.phonon import PhononBandStructureSymmLine, PhononDos

from ffonons.enums import PhKey

__author__ = "Janosh Riebesell"
__date__ = "2023-11-15"

pmv.set_plotly_template("pymatviz_white")

PKG_DIR = os.path.dirname(__file__)
ROOT = os.path.dirname(PKG_DIR)
DATA_DIR = f"{ROOT}/data"
PDF_FIGS = f"{ROOT}/figs"
PAPER_DIR = f"{ROOT}/figs/phonon-db"
SITE_FIGS = f"{ROOT}/site/src/figs"
SOFT_PES_DIR = f"{PDF_FIGS}/soft-pes"
TEST_FILES = f"{ROOT}/tests/files"

today = f"{datetime.now(tz=UTC):%Y-%m-%d}"
px.defaults.labels |= PhKey.val_label_dict()

speed_of_light = 299792458  # [m/s]
thz_to_per_cm = 1e12 / (speed_of_light * 100)  # convert THz to cm^-1: 33.356410
AnyDos = CompleteDos | PhononDos | Dos
AnyBandStructure = BandStructureSymmLine | PhononBandStructureSymmLine

KpathScheme = Literal["setyawan_curtarolo", "latimer_munro", "hinuma", "seekpath"]
ValidKpathSchemes = get_args(KpathScheme)
(
    setyawan_curtarolo_kpath_scheme,
    latimer_munro_kpath_scheme,
    hinuma_kpath_scheme,
    seekpath_kpath_scheme,
) = ValidKpathSchemes
