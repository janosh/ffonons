"""The ffonons package helps predict and analyze phonons with machine learning
force fields.
"""

import os
from datetime import UTC, datetime

import plotly.express as px
import plotly.io as pio
import pymatviz  # noqa: F401, import registers pymatviz_white template
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
from pymatgen.electronic_structure.dos import CompleteDos, Dos
from pymatgen.phonon import PhononBandStructureSymmLine, PhononDos

__author__ = "Janosh Riebesell"
__date__ = "2023-11-15"

px.defaults.template = "pymatviz_white"
pio.templates.default = "pymatviz_white"

PKG_DIR = os.path.dirname(__file__)
ROOT = os.path.dirname(PKG_DIR)
DATA_DIR = f"{ROOT}/data"
PDF_FIGS = f"{ROOT}/figs"
PAPER_DIR = f"{ROOT}/../thesis/figs/phonons"
SITE_FIGS = f"{ROOT}/site/src/figs"
SOFT_PES_DIR = f"{PDF_FIGS}/soft-pes"

today = f"{datetime.now(tz=UTC):%Y-%m-%d}"


speed_of_light = 299792458  # [m/s]
thz_to_per_cm = 1e12 / (speed_of_light * 100)  # convert THz to cm^-1: 33.356410
AnyDos = CompleteDos | PhononDos | Dos
AnyBandStructure = BandStructureSymmLine | PhononBandStructureSymmLine
