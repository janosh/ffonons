"""Analyze highest phonon DOS peaks vs DFT for all materials."""

# %%
import itertools
from glob import glob

import numpy as np
import plotly.express as px
from pymatgen.util.string import htmlify
from pymatviz.io import save_fig
from pymatviz.utils import add_identity_line
from sklearn.metrics import r2_score

from ffonons import FIGS_DIR, PAPER_DIR, dft_key, id_key, pretty_label_map
from ffonons.io import load_pymatgen_phonon_docs

__author__ = "Janosh Riebesell"
__date__ = "2023-11-24"


# %%
ph_docs, df_summary = load_pymatgen_phonon_docs(which_db := "phonon_db")
model_key = "mace-y7uhwpje"

df_summary.to_csv(
    f"{PAPER_DIR}/{model_key}-phonon-db-summary.csv",
    float_format="%.6g",
)
print("MAE of last phonon DOS peak frequencies:")
last_peak_cols = df_summary.filter(like="last_phdos_peak").columns
for col1, col2 in itertools.combinations(last_peak_cols, 2):
    diff = df_summary[col1] - df_summary[col2]
    df_summary[f"{col1} - {col2} (n={len(diff.dropna()):,})"] = diff
    MAE = diff.abs().mean()
    R2 = r2_score(*df_summary[[col1, col2]].dropna().to_numpy().T)
    print(f"{col1} vs {col2} = {MAE:.2f} THz, {R2 = :.2f}")


# %% plot histogram of MP vs model last phonon DOS peaks
fig = px.histogram(df_summary.filter(like=" - ").dropna(), nbins=120, opacity=0.6)
# fig.layout.title = "Histogram of difference in last phonon DOS peak frequencies"
fig.layout.xaxis.title = "Difference in Last Phonon DOS Peak Frequency (THz)"
fig.layout.yaxis.title = "Number of Materials"
fig.layout.legend = dict(x=1, y=1, xanchor="right", yanchor="top")

fig.layout.margin = dict(l=5, r=5, b=5, t=5)
fig.show()


# %% plot MP vs model last phonon DOS peaks as scatter
# assert len(df_summary.query(f"{which_db} > 50")) < 2  # only one outlier
# post-hoc PES hardening shift in THz (ML-predictions will be boosted by this value)
pes_shift = 0  # 0.6
peak_pbe_col = f"last_phdos_peak_{dft_key}_THz"
peak_ml_col = f"last_phdos_peak_{model_key.replace('-', '_')}_THz"
for col in (peak_pbe_col, peak_ml_col):
    assert col in df_summary, f"{col=} not in df_summary"

# std dev of MP last phonon DOS peak
print(f"PBE last phonon DOS peak std dev: {df_summary[peak_pbe_col].std():.2f} THz")

df_plot = df_summary.copy().reset_index().query(f"{peak_pbe_col} < 80").dropna()
df_plot[last_peak_cols] += pes_shift

fig = px.scatter(df_plot, x=peak_pbe_col, y=last_peak_cols, hover_name=id_key)

for trace in fig.data:
    trace.marker.size = 10
    trace.marker.opacity = 0.8

    targets, preds = df_plot[[peak_pbe_col, trace.name]].dropna().to_numpy().T
    MAE = np.abs(targets - (preds + pes_shift)).mean()
    R2 = r2_score(targets, (preds + pes_shift))
    pretty_label = pretty_label_map[model_key]
    trace.name = (
        f"{pretty_label} ({MAE=:.2f}, R<sup>2</sup>={R2:.2f}, n={len(targets)})"
    )

# increase legend font size
fig.layout.legend.update(x=0.01, y=0.98, borderwidth=1, bordercolor="lightgray")

# annotate outliers from parity line
df_outliers = df_plot.query(
    f"{peak_pbe_col} - `{peak_ml_col}` > 6 or `{peak_ml_col}` - {peak_pbe_col} > 3"
)
for idx in df_outliers.index:
    row = df_outliers.loc[idx]
    too_low = row[peak_pbe_col] > row[peak_ml_col]  # to annotation offset
    err = row[peak_ml_col] - row[peak_pbe_col]
    fig.add_annotation(
        x=row[peak_pbe_col],
        y=row[peak_ml_col],
        text=f"{htmlify(row['formula'])}<br>{row[id_key]}<br>{err:.0f} THz = "
        f"{abs(err / row[peak_pbe_col]):.0%}",
        font=dict(size=9),
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

# set axis range to start at 0
xy_max = (df_plot[last_peak_cols].max().max()) // 10 * 10 + 10  # round up to nearest 10
fig.update_xaxes(range=[0, xy_max])
fig.update_yaxes(range=[0, xy_max])
db_label_map = {"phonon_db": "PhononDB PBE", "gnome": "GNOME", "mp": "MP"}
fig.layout.xaxis.title = f"{db_label_map[which_db]} last phDOS peak frequency (THz)"
fig.layout.yaxis.title = "MLFF last phDOS peak frequency (THz)"
fig.layout.legend.update(x=0.01, y=0.98, yanchor="top", title=None)
add_identity_line(fig)  # annotations change plot range so add parity line after

fig.layout.margin = dict(l=12, r=12, b=12, t=12)
fig.show()
img_name = "parity-pbe-vs-model-last-peak"
parity_fig_path = f"{FIGS_DIR}/{which_db}/{img_name}.pdf"
save_fig(fig, parity_fig_path, width=500, height=300)
save_fig(fig, f"{PAPER_DIR}/{img_name}.pdf", width=700, height=450)


# %%
for err_dir in ("over", "under"):
    print(f"\nmaterials with most {err_dir}estimated last phDOS peak")
    for col in df_summary.filter(like=" - "):
        srs_sort_diff = df_summary[col].sort_values().dropna()
        # print file paths to band structure plots
        for (mp_id, _formula), diff in getattr(
            srs_sort_diff, "head" if err_dir == "under" else "tail"
        )(3).items():
            print(f"\n- {mp_id} {_formula} off by {diff:.2f} THz")
            print("  " + glob(f"{FIGS_DIR}/{which_db}/{mp_id}-bands*.pdf")[0])


# %%
print("materials with most accurate last phDOS peak")
for col in df_summary.filter(like=" - "):
    most_accurate = df_summary[col].abs().sort_values()
    for (mp_id, formula), diff in most_accurate[:5].items():
        print(f"\n{mp_id} {formula} off by {diff:.2f} THz")
        print(glob(f"{FIGS_DIR}/{which_db}/{mp_id}-bands*.pdf")[0])
