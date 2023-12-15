# %%
import itertools
from collections import defaultdict
from glob import glob

import numpy as np
import pandas as pd
import plotly.express as px
from pymatgen.util.string import htmlify
from pymatviz.io import save_fig
from pymatviz.utils import add_identity_line
from sklearn.metrics import r2_score

from ffonons import (
    FIGS_DIR,
    WhichDB,
    dft_key,
    dos_key,
    find_last_dos_peak,
    formula_key,
    id_key,
)
from ffonons.io import load_pymatgen_phonon_docs

__author__ = "Janosh Riebesell"
__date__ = "2023-11-24"


# %% compute last phonon DOS peak for each model and MP
last_peaks = defaultdict(lambda: defaultdict(float))

which_db: WhichDB = "phonon_db"
ph_docs = load_pymatgen_phonon_docs(which_db)


# %%
for mp_id, docs in ph_docs.items():
    for model_key, doc in docs.items():
        phonon_dos = doc[dos_key]
        last_peak = find_last_dos_peak(phonon_dos)
        last_peaks[mp_id][model_key] = last_peak
        last_peaks[mp_id][formula_key] = doc[formula_key]


df_last_peak = pd.DataFrame(last_peaks).T
df_last_peak.index.name = id_key
df_last_peak = df_last_peak.set_index(formula_key, append=True)

# don't rerun this line after adding more cols to df_last_peak
model_cols = list(set(df_last_peak) - {dft_key})


# iterate over all pairs of columns
print("MAE of last phonon DOS peak frequencies:")
for col1, col2 in itertools.combinations(df_last_peak, 2):
    diff = df_last_peak[col1] - df_last_peak[col2]
    df_last_peak[f"{col1} - {col2} ({len(diff.dropna())} points)"] = diff
    MAE = diff.abs().mean()
    R2 = r2_score(*df_last_peak[[col1, col2]].dropna().to_numpy().T)
    print(f"{col1} vs {col2} = {MAE:.2f} THz, {R2 = :.2f}")


# %% plot histogram of MP vs model last phonon DOS peaks
fig = px.histogram(df_last_peak.filter(like=" - "), nbins=800, opacity=0.6)
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
# assert len(df_last_peak.query(f"{which_db} > 50")) < 2  # only one outlier
# post-hoc PES hardening shift in THz (ML-predictions will be boosted by this value)
pes_shift = 0  # 0.6

# std dev of MP last phonon DOS peak
print(f"PBE last phonon DOS peak std dev: {df_last_peak[dft_key].std():.2f} THz")

df_plot = df_last_peak.copy().reset_index().round(2).query(f"{dft_key} < 80")
df_plot[model_cols] += pes_shift

fig = px.scatter(df_plot, x=dft_key, y=model_cols, hover_name=id_key)
db_label_map = {"phonon_db": "PhononDB PBE", "gnome": "GNOME", "mp": "MP"}
fig.layout.xaxis.title = f"{db_label_map[which_db]} last phDOS peak frequency (THz)"
fig.layout.yaxis.title = "MLFF last phDOS peak frequency (THz)"
fig.layout.legend = dict(x=0, y=1, yanchor="top")
fig.layout.update(width=800, height=600)

for trace in fig.data:
    trace.marker.size = 10
    trace.marker.opacity = 0.8

    targets, preds = df_plot[[dft_key, trace.name]].dropna().to_numpy().T
    MAE = np.abs(targets - (preds + pes_shift)).mean()
    R2 = r2_score(targets, (preds + pes_shift))
    trace.name = f"MACE ({MAE=:.2f}, R<sup>2</sup>={R2:.2f}, n={len(targets)})"


# increase legend font size
fig.layout.legend.update(x=0.01, y=0.83, borderwidth=1, bordercolor="lightgray")


# annotate outliers from parity line
mace_col = model_cols[0]
assert mace_col.startswith("mace")
df_outliers = df_plot.query(
    f"{dft_key} - `{mace_col}` > 4.5 or `{mace_col}` - {dft_key} > 3"
)
style = "font-size: 11px;"  # span style for material ID
for _idx, row in df_outliers.iterrows():
    too_low = row[dft_key] > row[mace_col]  # to change arrow direction
    fig.add_annotation(
        x=row[dft_key],
        y=row[mace_col],
        text=f"{htmlify(row['formula'])}<br><span {style=}>{row[id_key]}</span>",
        xanchor="center" if too_low else "right",
        yanchor="top" if too_low else "middle",
        xshift=0 if too_low else -10,
        yshift=-5 if too_low else 0,
        showarrow=False,
        # arrowhead=1,
        # ax=(0 if too_low else 1) * 20,
        # ay=too_low * 1000,
        # standoff=9,  # shorten arrow at end
    )

# fix axis range to start at 0
xy_max = (df_plot[model_cols].max().max()) // 10 * 10 + 10  # round up to nearest 10
fig.update_xaxes(range=[0, xy_max])
fig.update_yaxes(range=[0, xy_max])

add_identity_line(fig)
fig.layout.margin = dict(l=12, r=12, b=12, t=12)
fig.show()
parity_fig_path = f"{FIGS_DIR}/{which_db}/parity-pbe-vs-model-last-peak.pdf"
save_fig(fig, parity_fig_path, width=600, height=400)


# %% get 5 material IDs for each model with most over-softened DOS (i.e. last
# peak most underestimated)
for col in df_last_peak.filter(like=" - "):
    most_underestimated = df_last_peak[col].sort_values()
    print(most_underestimated[:5])
    for mp_id in most_underestimated[:5].index:
        print(glob(f"{FIGS_DIR}/{mp_id}*/*")[0])


# %% get 5 material IDs for each model with most accurate DOS (i.e. last peak
# closest to MP)
for col in df_last_peak.filter(like=" - "):
    most_accurate = df_last_peak[col].abs().sort_values()
    print(most_accurate[:5])
    for mp_id in most_accurate[:5].index:
        print(glob(f"{FIGS_DIR}/{mp_id}*/*")[0])
