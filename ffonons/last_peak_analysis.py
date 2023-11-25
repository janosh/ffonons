# %%
import itertools
from collections import defaultdict

import pandas as pd
import plotly.express as px
from pymatviz.utils import add_identity_line

from ffonons import dos_key, find_last_dos_peak, name_case_map
from ffonons.load_all_docs import all_docs
from ffonons.plots import plot_phonon_dos

__author__ = "Janosh Riebesell"
__date__ = "2023-11-24"


# %% compute last phonon DOS peak for each model and MP
last_peaks = defaultdict(lambda: defaultdict(float))

for mp_id, docs in all_docs.items():
    for model, doc in docs.items():
        phonon_dos = doc[dos_key]
        last_peak = find_last_dos_peak(phonon_dos)
        key = name_case_map[model]
        last_peaks[mp_id][key] = last_peak


df_last_peak = pd.DataFrame(last_peaks).T

# iterate over all pairs of columns
for col1, col2 in itertools.combinations(df_last_peak.columns, 2):
    diff = df_last_peak[col1] - df_last_peak[col2]
    df_last_peak[f"{col1} - {col2} ({len(diff.dropna())} points)"] = diff
    print(f"{col1} - {col2}: {diff.mean():.2f} +- {diff.std():.2f}")


# %% plot histogram of MP vs model last phonon DOS peaks
fig = px.histogram(df_last_peak.filter(like="-"), nbins=800, opacity=0.6)
fig.layout.title = "Histogram of difference in last phonon DOS peak frequencies"
fig.layout.xaxis.title = "Difference in Last Phonon DOS Peak Frequency (THz)"
fig.layout.yaxis.title = "Number of Materials"
fig.layout.legend = dict(x=1, y=1, xanchor="right", yanchor="top")

for idx, trace in enumerate(fig.data):
    trace.marker.color = px.colors.qualitative.Dark2[idx]
    if "MP" not in trace.name:
        trace.visible = "legendonly"

fig.show()


# %% plot MP vs model last phonon DOS peaks as scatter
fig = px.scatter(df_last_peak.set_index("MP")[["MACE", "CHGNet"]].query("index < 50"))
fig.layout.xaxis.title = "MP last phonon DOS peak frequency (THz)"
fig.layout.yaxis.title = "Model last phonon DOS peak frequency (THz)"
fig.layout.legend = dict(x=1, y=1, xanchor="right", yanchor="top")

add_identity_line(fig)
fig.show()


# %% plot location of last phonon DOS peak in DOS
for mp_id, docs in all_docs.items():
    for model, doc in docs.items():
        phonon_dos = doc[dos_key]
        ax = plot_phonon_dos({model: phonon_dos})
        ax.set_title(f"{mp_id} {model}", fontsize=22, fontweight="bold")
        last_peak = find_last_dos_peak(phonon_dos)
        ax.axvline(last_peak, color="red", linestyle="--")
        # save_fig(ax, f"{FIGS_DIR}/{mp_id}-{model}/{model}-dos.pdf")
