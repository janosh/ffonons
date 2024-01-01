"""Calculate the phonon DOS MAEs for all materials."""

# %%
import numpy as np
import plotly.express as px
from pymatviz.io import save_fig

from ffonons import PAPER_DIR
from ffonons.io import load_pymatgen_phonon_docs
from ffonons.plots import pretty_label_map

__author__ = "Janosh Riebesell"
__date__ = "2023-12-17"


# %%
ph_docs, df_summary = load_pymatgen_phonon_docs(which_db := "phonon-db")
model_key = "mace-y7uhwpje"
srs_ph_dos_mae = df_summary[f"phdos_mae_{model_key.replace('-', '_')}_THz"].dropna()
srs_ph_dos_mae.name = f"{pretty_label_map[model_key]}"


# %% plot histogram of all phDOS MAEs
fig = px.histogram(srs_ph_dos_mae, nbins=350)

hist_counts, bin_edges = np.histogram(srs_ph_dos_mae, bins=350, range=[0, 16])
# calculate the cumulative fraction of materials with MAE < x
cumulative_percentage = np.cumsum(hist_counts) / np.sum(hist_counts)

fig.add_scatter(
    x=bin_edges[1:],
    y=cumulative_percentage,
    name="Cumulative",
    yaxis="y2",
    line=dict(color="navy", width=3),
    hovertemplate="x: %{x:.2f} THz<br>Cumulative: %{y:.0%}",
)
fig.layout.yaxis2 = dict(
    title="Cumulative", overlaying="y", side="right", showgrid=False
)

fig.layout.xaxis.update(title="Phonon DOS MAE (THz)", range=[0, 16])
fig.layout.yaxis.update(title="Number of Materials")
fig.layout.legend = dict(x=0.95, y=0.8, xanchor="right", yanchor="top")
fig.layout.margin = dict(l=10, r=10, b=10, t=10)
fig.layout.font.size = 20

fig.show()
save_fig(fig, f"{PAPER_DIR}/phdos-mae-hist.pdf")
