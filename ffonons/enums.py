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
    dos_mae = "phdos_mae_thz", "phonon DOS MAE"
    formula = "formula", "Chemical formula"
    last_dos_peak = "last_phdos_peak_thz", "Last phonon DOS peak"
    mat_id = "material_id", "Material ID"
    model = "model", "Model"
    n_sites = "n_sites", "Number of sites"
    pbe = "pbe", "PBE"
    pbesol = "pbesol", "PBEsol"
    reduced_formula = "reduced_formula", "Reduced chemical formula"
    struct = "structure", "Structure"
    supercell = "supercell", "Supercell"
    togo_id = "togo_id", "Togo DB ID"
    volume = "volume", "Volume"

    # model metrics
    phdos_mae_thz = "phdos_mae_thz", "MAE<sub>ph DOS</sub> (THz)"
    phdos_r2 = "phdos_r2", "R<sup>2</sup><sub>ph DOS</sub>"
    imaginary_gamma_freq = "imaginary_gamma_freq", "Imag. Γ"
    has_imag_modes = "imaginary_freq", "Has imaginary modes"
    last_phdos_peak_thz = "last_phdos_peak_thz", "ω<sub>max</sub> (THz)"
    last_phdos_peak_thz_ml = (
        "last_phdos_peak_thz_ml",
        "ω<sub>max</sub><sup>ML</sup> (THz)",
    )
    last_phdos_peak_thz_pbe = (
        "last_phdos_peak_thz_dft",
        "ω<sub>max</sub><sup>DFT</sup> (THz)",
    )
    mae_last_phdos_peak_thz = (
        "mae_last_phdos_peak_thz",
        "MAE<sub>ω<sub>max</sub></sub> (THz)",
    )
    r2_last_phdos_peak_thz = (
        "r2_last_phdos_peak_thz",
        "R<sup>2</sup><sub>ω<sub>max</sub></sub> (THz)",
    )
    max_freq = "max_freq_thz", "Ω<sub>max</sub> (THz)"
    min_freq = "min_freq_thz", "Ω<sub>min</sub> (THz)"
    max_freq_pbe = "max_freq_thz_pbe", "Ω<sub>max</sub><sup>PBE</sup> (THz)"
    max_freq_ml = "max_freq_thz_ml", "Ω<sub>max</sub><sup>ML</sup> (THz)"
    max_freq_rel = (
        "max_freq_rel",
        "Ω<sub>max</sub><sup>ML</sup> / Ω<sub>max</sub><sup>DFT</sup>",
    )
    mae_max_freq_thz = "mae_max_freq_thz", "MAE<sub>Ω<sub>max</sub></sub> (THz)"
    r2_max_freq_thz = "r2_max_freq_thz", "R<sup>2</sup><sub>Ω<sub>max</sub></sub> (THz)"


class DB(LabelEnum):
    """Database names."""

    mp = "mp", "Materials Project"
    phonon_db = "phonon-db", "Phonon Database"
    one_off = "one-off", "One-off materials"  # e.g. from experiment


class Model(LabelEnum):
    """Model names."""

    m3gnet_ms = "m3gnet", "M3GNet-MS", "blue"
    chgnet_030 = "chgnet-v0.3.0", "CHGNet v0.3.0", "orange"
    mace_mp = "mace-y7uhwpje", "MACE-MP", "green"
    gnome = "gnome", "GNoME", "red"
