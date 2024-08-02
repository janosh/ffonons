import inspect
from enum import StrEnum
from typing import cast

from pymatviz.enums import LabelEnum

import ffonons.enums


def test_label_enum() -> None:
    # assert all classes in enums are LabelEnums
    for enum in dir(ffonons.enums):
        if inspect.isclass(enum) and issubclass(enum, StrEnum):
            assert issubclass(enum, LabelEnum)
            enum = cast(LabelEnum, enum)  # help mypy
            val_dict = enum.key_val_dict()
            assert isinstance(val_dict, dict)
            label_dict = enum.val_label_dict()
            assert isinstance(label_dict, dict)
            assert val_dict != label_dict
            assert len(val_dict) == len(label_dict)
