"""Calculate the phonon DOS MAEs for all materials."""

# %%
import plotly.express as px
from pymatviz.io import save_fig
from pymatviz.utils import add_ecdf_line

from ffonons import PAPER_DIR
from ffonons.enums import Key, Model
from ffonons.io import get_df_summary

__author__ = "Janosh Riebesell"
__date__ = "2023-12-17"


# %%
df_summary = get_df_summary(which_db := "phonon-db")
df_model = df_summary.xs(Model.mace_mp, level=1)


# %% plot histogram of all phDOS MAEs
fig = px.histogram(df_model[Key.dos_mae], nbins=350)

add_ecdf_line(fig)

fig.layout.xaxis.update(title=Key.dos_mae.label, range=[0, 16])
fig.layout.yaxis.update(title="Number of Materials")
fig.layout.legend = dict(x=0.95, y=0.8, xanchor="right", yanchor="top")
fig.layout.margin = dict(l=10, r=10, b=10, t=10)
fig.layout.font.size = 20

fig.show()
save_fig(fig, f"{PAPER_DIR}/phdos-mae-hist.pdf")
