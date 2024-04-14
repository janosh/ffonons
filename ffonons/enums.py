"""Enums for labeling ffonons package data and models.

LabelEnum extends the built-in StrEnum class, allowing the addition of optional label
and description attributes. The Key, DB, and Model enums inherit from LabelEnum.
"""

from enum import StrEnum
from typing import Self

__author__ = "Janosh Riebesell"
__date__ = "2024-02-19"


class LabelEnum(StrEnum):
    """StrEnum with optional label and description attributes plus dict() method."""

    def __new__(
        cls, val: str, label: str | None = None, desc: str | None = None
    ) -> Self:
        """Create a new class."""
        member = str.__new__(cls, val)
        member._value_ = val
        member.__dict__ |= dict(label=label, desc=desc)
        return member

    @property
    def label(self) -> str:
        """Make label read-only."""
        return self.__dict__["label"]

    @property
    def description(self) -> str:
        """Make description read-only."""
        return self.__dict__["desc"]

    @classmethod
    def key_val_dict(cls) -> dict[str, str]:
        """Map of keys to values."""
        return {key: str(val) for key, val in cls.__members__.items()}

    @classmethod
    def val_label_dict(cls) -> dict[str, str | None]:
        """Map of values to labels."""
        return {str(val): val.label for val in cls.__members__.values()}

    @classmethod
    def val_desc_dict(cls) -> dict[str, str | None]:
        """Map of values to descriptions."""
        return {str(val): val.description for val in cls.__members__.values()}

    @classmethod
    def label_desc_dict(cls) -> dict[str | None, str | None]:
        """Map of labels to descriptions."""
        return {str(val.label): val.description for val in cls.__members__.values()}


class Key(LabelEnum):
    """Keys for accessing the data in phonon docs and dataframes."""

    bs = "phonon_bandstructure", "Phonon band structure"
    composition = "composition", "Chemical composition"
    dft = "dft", "DFT"
    dos = "phonon_dos", "Phonon density of states"
    formula = "formula", "Chemical formula"
    last_dos_peak = "last_ph_dos_peak_thz", "Last phonon DOS peak"
    mat_id = "material_id", "Material ID"
    model = "model", "Model"
    n_sites = "n_sites", "Number of sites"
    pbe = "pbe", "PBE"
    pbesol = "pbesol", "PBEsol"
    reduced_formula = "reduced_formula", "Reduced chemical formula"
    needs_u_correction = "requires_u_correction", "Requires Hubbard U correction"
    struct = "structure", "Structure"
    supercell = "supercell", "Supercell"
    togo_id = "togo_id", "Togo DB ID"
    volume = "volume", "Volume"

    # model metrics
    dos_mae = "ph_dos_mae_thz", "MAE<sub>ph DOS</sub> (THz)"
    ph_dos_r2 = "ph_dos_r2", "R<sup>2</sup><sub>ph DOS</sub>"
    has_imag_gamma_freq = "has_imag_gamma_freq", "Has imaginary Γ modes"
    has_imag_freq = "has_imag_freq", "Has imaginary modes"
    last_ph_dos_peak_thz = "last_ph_dos_peak_thz", "ω<sub>max</sub> (THz)"
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
    max_freq = "max_freq_thz", "Ω<sub>max</sub> (THz)"  # aka band width
    min_freq = "min_freq_thz", "Ω<sub>min</sub> (THz)"
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
