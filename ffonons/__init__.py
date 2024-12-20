"""The ffonons package helps predict and analyze phonons with machine learning
force fields.
"""

import os
from datetime import UTC, datetime

import plotly.express as px
import pymatviz as pmv
import scipy.constants
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
from pymatgen.electronic_structure.dos import CompleteDos, Dos
from pymatgen.phonon import PhononBandStructureSymmLine, PhononDos

from ffonons import io
from ffonons.enums import PhKey

__author__ = "Janosh Riebesell"
__date__ = "2023-11-15"

pmv.set_plotly_template("pymatviz_white")

PKG_DIR = os.path.dirname(__file__)
ROOT = os.path.dirname(PKG_DIR)
DATA_DIR = f"{ROOT}/data"
os.makedirs(PDF_FIGS := f"{ROOT}/figs", exist_ok=True)
os.makedirs(PAPER_DIR := f"{ROOT}/figs/phonon-db", exist_ok=True)
SITE_FIGS = f"{ROOT}/site/src/figs"
SOFT_PES_DIR = f"{PDF_FIGS}/soft-pes"
TEST_FILES = f"{ROOT}/tests/files"

today = f"{datetime.now(tz=UTC):%Y-%m-%d}"
px.defaults.labels |= {key.value: key.label for key in PhKey}

# convert THz to cm^-1: 33.356410
thz_to_per_cm = scipy.constants.tera / (scipy.constants.c * scipy.constants.centi)
AnyDos = CompleteDos | PhononDos | Dos
AnyBandStructure = BandStructureSymmLine | PhononBandStructureSymmLine
