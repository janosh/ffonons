# %%
import itertools
from collections import defaultdict
from glob import glob

import numpy as np
import pandas as pd
import plotly.express as px
from pymatviz.io import save_fig
from pymatviz.utils import add_identity_line
from sklearn.metrics import r2_score

from ffonons import FIGS_DIR, dos_key, find_last_dos_peak, name_case_map
from ffonons.load_all_docs import all_docs

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
print("MAE of last phonon DOS peak frequencies:")
for col1, col2 in itertools.combinations(df_last_peak.columns, 2):
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


# %% compute MAE and R2 of MP vs model last phonon DOS peaks
MAEs = defaultdict(dict)
R2s = defaultdict(dict)
shift = 0  # 1  # post-hoc PES hardening shift in THz (ML-predicted frequencies will be
# boosted by this amount)

for model in ("MACE", "CHGNet"):
    targets, preds = df_last_peak[["MP", model]].dropna().to_numpy().T
    MAEs["MP"][model] = np.abs(targets - (preds + shift)).mean()
    R2s["MP"][model] = r2_score(targets, (preds + shift))


# %% plot MP vs model last phonon DOS peaks as scatter
assert len(df_last_peak.query("MP > 50")) < 2  # only one outlier

# std dev of MP last phonon DOS peak
print(f"MP last phonon DOS peak std dev: {df_last_peak['MP'].std():.2f} THz")

fig = px.scatter(
    df_last_peak.set_index("MP")[["MACE", "CHGNet"]].query("index < 50") + shift
)
fig.layout.xaxis.title = "MP last phonon DOS peak frequency (THz)"
fig.layout.yaxis.title = "Model last phonon DOS peak frequency (THz)"
fig.layout.legend = dict(x=0, y=1, yanchor="top")
fig.layout.update(width=800, height=600)

for trace in fig.data:
    trace.marker.size = 10
    trace.marker.opacity = 0.8
    MAE, R2 = MAEs["MP"][trace.name], R2s["MP"][trace.name]
    trace.name = f"{trace.name} ({MAE=:.2f}, R<sup>2</sup>={R2:.2f})"

add_identity_line(fig)
fig.layout.margin = dict(l=8, r=8, b=8, t=8)
fig.show()
save_fig(fig, f"{FIGS_DIR}/mp-vs-model-last-peak-scatter.pdf")


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
