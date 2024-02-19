"""Unpublished Python package supporting the analysis and results in the
2nd pymatgen publication.
"""
import os
from datetime import UTC, datetime

import numpy as np
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

today = f"{datetime.now(tz=UTC):%Y-%m-%d}"


speed_of_light = 299792458  # [m/s]
thz_to_per_cm = 1e12 / (speed_of_light * 100)  # convert THz to cm^-1: 33.356410
AnyDos = CompleteDos | PhononDos | Dos
AnyBandStructure = BandStructureSymmLine | PhononBandStructureSymmLine


def find_last_dos_peak(dos: AnyDos, min_ratio: float = 0.1) -> float:
    """Find the last peak in the phonon DOS defined as the highest frequency with a DOS
    value at least threshold * height of the overall highest DOS peak.
    A peak is any local maximum of the DOS as a function of frequency.

    Args:
        dos (Dos): pymatgen Dos object (electronic or phonon)
        min_ratio (float, optional): Minimum ratio of the height of the last peak
            to the height of the highest peak. Defaults to 0.1 = 10%. In case no peaks
            are high enough to match, the threshold is reset to half the height of the
            second-highest peak.

    Returns:
        float: frequency of the last DOS peak in THz
    """
    first_deriv = np.gradient(dos.densities, dos.frequencies)
    second_deriv = np.gradient(first_deriv, dos.frequencies)

    maxima = (  # maxima indices of the first DOS derivative w.r.t. frequency
        (first_deriv[:-1] > 0) & (first_deriv[1:] < 0) & (second_deriv[:-1] < 0)
    )
    # get mean of the two nearest frequencies around the maximum as better estimate
    maxima_freqs = (dos.frequencies[:-1][maxima] + dos.frequencies[1:][maxima]) / 2

    # filter maxima based on the threshold
    max_dos = max(dos.densities)
    threshold = min_ratio * max_dos
    filtered_maxima_freqs = maxima_freqs[dos.densities[:-1][maxima] >= threshold]

    if len(filtered_maxima_freqs) == 0:
        # if no maxima reach the threshold (i.e. 1 super high peak and all other peaks
        # tiny), use half the height of second highest peak as threshold
        second_highest_peak = sorted(dos.densities)[-2]
        threshold = second_highest_peak / 2
        filtered_maxima_freqs = maxima_freqs[dos.densities[:-1][maxima] >= threshold]

    return max(filtered_maxima_freqs)
