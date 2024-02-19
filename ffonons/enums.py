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
    def val_dict(cls) -> dict[str, str]:
        """Return the Enum as dictionary."""
        return {key: str(val) for key, val in cls.__members__.items()}

    @classmethod
    def label_dict(cls) -> dict[str, str]:
        """Return the Enum as dictionary."""
        return {str(val): val.label for key, val in cls.__members__.items()}


class Key(LabelEnum):
    """Keys for accessing the data in phonon docs and dataframes."""

    bs = "phonon_bandstructure", "Phonon band structure"
    composition = "composition", "Chemical composition"
    dft = "pbe", "DFT"
    dos = "phonon_dos", "Phonon density of states"
    dos_mae = "phdos_mae_THz", "Mean absolute error of phonon DOS"
    formula = "formula", "Chemical formula"
    last_dos_peak = "last_phdos_peak_THz", "Last phonon DOS peak"
    mat_id = "material_id", "Material ID"
    max_freq = "max_freq_THz", "Maximum phonon frequency"
    min_freq = "min_freq_THz", "Minimum phonon frequency"
    model = "model", "Model"
    n_sites = "n_sites", "Number of sites"
    reduced_formula = "reduced_formula", "Reduced chemical formula"
    struct = "structure", "Structure"
    togo_id = "togo_id", "Togo DB ID"


class DB(LabelEnum):
    """Database names."""

    mp = "mp", "Materials Project"
    phonon_db = "phonon-db", "Phonon Database"
    gnome = "gnome", "GNoME"
    one_off = "one-off", "One-off materials"  # e.g. from experiment


class Model(LabelEnum):
    """Model names."""

    m3gnet_ms = "m3gnet", "M3GNet-MS"
    mace_mp = "mace-y7uhwpje", "MACE-MP"
    chgnet_030 = "chgnet-v0.3.0", "CHGNet v0.3.0"
    gnome = "gnome", "GNoME"
