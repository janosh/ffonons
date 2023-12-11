"""Unpublished Python package supporting the analysis and results in the
2nd pymatgen publication.
"""

import os
from typing import Literal

import numpy as np
import plotly.express as px
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
from pymatgen.electronic_structure.dos import CompleteDos, Dos
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos

__author__ = "Janosh Riebesell"
__date__ = "2023-11-15"

px.defaults.template = "plotly_dark"

PKG_DIR = os.path.dirname(__file__)
ROOT = os.path.dirname(PKG_DIR)
DATA_DIR = f"{ROOT}/data"
FIGS_DIR = f"{ROOT}/figs"

bs_key = "phonon_bandstructure"
dos_key = "phonon_dos"
id_key = "material_id"
formula_key = "formula"
struct_key = "structure"
dos_peak_key = "last phdos peak"

WhichDB = Literal["mp", "phonon_db", "gnome"]
name_case_map = dict(mp="MP", mace="MACE", chgnet="CHGNet")

AnyDos = CompleteDos | PhononDos | Dos
AnyBandStructure = BandStructureSymmLine | PhononBandStructureSymmLine


def find_last_dos_peak(dos: AnyDos, min_ratio: float = 0.1) -> float:
    """Find the last peak in the phonon DOS defined as the highest frequency with a DOS
    value at least threshold_ratio times the height of the overall highest DOS peak.
    A peak is any local maximum of the DOS data.

    Args:
        dos (Dos): pymatgen Dos object (electronic or phonon)
        min_ratio (float, optional): Minimum ratio of the height of the last peak
            to the height of the highest peak. Defaults to 0.1.

    Returns:
        float: frequency of the last DOS peak in THz
    """
    first_deriv = np.gradient(dos.densities, dos.frequencies)
    second_deriv = np.gradient(first_deriv, dos.frequencies)

    maxima_idx = (
        (first_deriv[:-1] > 0) & (first_deriv[1:] < 0) & (second_deriv[:-1] < 0)
    )  # maxima of the first derivative
    # get mean of the two frequencies around the maximum
    maxima_frequencies = (
        dos.frequencies[:-1][maxima_idx] + dos.frequencies[1:][maxima_idx]
    ) / 2

    # Filter maxima based on the threshold
    max_dos = max(dos.densities)
    threshold = min_ratio * max_dos
    return max(maxima_frequencies[dos.densities[:-1][maxima_idx] >= threshold])
