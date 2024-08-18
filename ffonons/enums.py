"""Enums for labeling ffonons package data and models.

LabelEnum extends the built-in StrEnum class, allowing the addition of optional label
and description attributes. The Key, DB, and Model enums inherit from LabelEnum.
"""

from pymatviz.enums import LabelEnum

__author__ = "Janosh Riebesell"
__date__ = "2024-02-19"


class PhKey(LabelEnum):
    """Keys for accessing the data in phonon docs and dataframes."""

    togo_id = "togo_id", "Togo DB ID"

    # model metrics
    dos_mae = "ph_dos_mae_thz", "MAE<sub>ph DOS</sub> (THz)"
    ph_dos_r2 = "ph_dos_r2", "R<sup>2</sup><sub>ph DOS</sub>"
    last_ph_dos_peak_thz_ml = (
        "last_ph_dos_peak_thz_ml",
        "ω<sub>max</sub><sup>ML</sup> (THz)",
    )
    last_ph_dos_peak_thz_pbe = (
        "last_ph_dos_peak_thz_dft",
        "ω<sub>max</sub><sup>DFT</sup> (THz)",
    )
    mae_last_ph_dos_peak_thz = (
        "mae_last_ph_dos_peak_thz",
        "MAE<sub>ω<sub>max</sub></sub> (THz)",
    )
    r2_last_ph_dos_peak_thz = (
        "r2_last_ph_dos_peak_thz",
        "R<sup>2</sup><sub>ω<sub>max</sub></sub> (THz)",
    )
    min_freq_pbe = "min_freq_thz_pbe", "Ω<sub>min</sub><sup>PBE</sup> (THz)"
    min_freq_ml = "min_freq_thz_ml", "Ω<sub>min</sub><sup>ML</sup> (THz)"
    min_freq_rel = (
        "min_freq_rel",
        "Ω<sub>min</sub><sup>ML</sup> / Ω<sub>min</sub><sup>DFT</sup>",
    )
    max_freq_pbe = "max_freq_thz_pbe", "Ω<sub>max</sub><sup>PBE</sup> (THz)"
    max_freq_ml = "max_freq_thz_ml", "Ω<sub>max</sub><sup>ML</sup> (THz)"
    max_freq_rel = (
        "max_freq_rel",
        "Ω<sub>max</sub><sup>ML</sup> / Ω<sub>max</sub><sup>DFT</sup>",
    )
    mae_max_freq_thz = "mae_max_freq_thz", "MAE<sub>Ω<sup>max</sup></sub> (THz)"
    r2_max_freq_thz = "r2_max_freq_thz", "R<sup>2</sup><sub>Ω<sup>max</sup></sub> (THz)"

    # classification metrics for presence of imaginary modes
    roc_auc_imag_freq = "roc_auc_imag_freq", "ROC AUC"
    prec_imag_freq = "prec_imag_freq", "Prec."
    recall_imag_freq = "recall_imag_freq", "Recall"
    f1_imag_freq = "f1_imag_freq", "F1"
    acc_imag_freq = "acc_imag_freq", "Acc."
    fpr_imag_freq = "fpr_imag_freq", "FPR"
    fnr_imag_freq = "fnr_imag_freq", "FNR"


class DB(LabelEnum):
    """Database names."""

    mp = "mp", "Materials Project"
    phonon_db = "phonon-db", "Togo PhononDB"
    one_off = "one-off", "One-off materials"  # e.g. from experiment


class Model(LabelEnum):
    """Model names."""

    # key, label, color
    m3gnet_ms = "m3gnet", "M3GNet-MS", "blue"
    chgnet_030 = "chgnet-v0.3.0", "CHGNet v0.3.0", "orange"
    mace_mp = "mace-y7uhwpje", "MACE-MP", "green"
    gnome = "gnome", "GNoME", "red"
    pbe = "pbe", "PBE", "gray"
    sevennet_0 = "sevennet-v0", "SevenNet v0", "purple"


class KpathScheme(LabelEnum):
    """K-path schemes for band structure plots."""

    setyawan_curtarolo = "setyawan_curtarolo", "Setyawan-Curtarolo"
    latimer_munro = "latimer_munro", "Latimer-Munro"
    hinuma = "hinuma", "Hinuma"
    seekpath = "seekpath", "Seekpath"
