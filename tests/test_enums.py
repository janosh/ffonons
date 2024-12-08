import inspect
from enum import StrEnum

from pymatviz.enums import LabelEnum

import ffonons.enums


def test_label_enum() -> None:
    # assert all classes in enums are LabelEnums
    for enum in dir(ffonons.enums):
        if inspect.isclass(enum) and issubclass(enum, StrEnum):
            assert issubclass(enum, LabelEnum)
