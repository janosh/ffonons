import os
import re

import plotly.express as px
import plotly.io as pio

from ffonons import DATA_DIR, PDF_FIGS, ROOT, today
from ffonons.enums import PhKey


def test_dir_globals() -> None:
    assert os.path.isdir(ROOT)
    assert os.path.isdir(PDF_FIGS)
    assert os.path.isdir(DATA_DIR)


def test_plotly_defaults() -> None:
    assert px.defaults.template == "pymatviz_white"
    assert pio.templates.default == "pymatviz_white"

    assert px.defaults.labels[PhKey.togo_id] == PhKey.togo_id.label
    assert px.defaults.labels[PhKey.ph_dos_mae] == PhKey.ph_dos_mae.label
    assert px.defaults.labels[PhKey.ph_dos_r2] == PhKey.ph_dos_r2.label


def test_today() -> None:
    assert re.match(r"20\d{2}-\d{2}-\d{2}", today)
