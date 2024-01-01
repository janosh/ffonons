"""Analyze highest phonon DOS peaks vs DFT for all materials."""

# %%
import itertools
from glob import glob

import numpy as np
import plotly.express as px
from pymatgen.util.string import htmlify
from pymatviz.io import save_fig
from pymatviz.utils import add_identity_line, annotate_metrics
from sklearn.metrics import r2_score

from ffonons import FIGS_DIR, PAPER_DIR, dft_key, id_key
from ffonons.io import load_pymatgen_phonon_docs
from ffonons.plots import pretty_label_map

__author__ = "Janosh Riebesell"
__date__ = "2023-11-24"


# %%
ph_docs, df_summary = load_pymatgen_phonon_docs(which_db := "phonon-db")
model_key = "mace-y7uhwpje"

df_summary.to_csv(
    f"{PAPER_DIR}/{model_key}-phonon-db-summary.csv",
    float_format="%.6g",
)
print("MAE of last phonon DOS peak frequencies:")

for like in ("last_phdos_peak", "band_width"):
    for col1, col2 in itertools.combinations(df_summary.filter(like=like), 2):
        diff = df_summary[col1] - df_summary[col2]
        df_summary[f"{col1} - {col2} (n={len(diff.dropna()):,})"] = diff
        MAE = diff.abs().mean()
        R2 = r2_score(*df_summary[[col1, col2]].dropna().to_numpy().T)
        print(f"{col1} vs {col2} = {MAE:.2f} THz, {R2 = :.2f}")


# %% parity plot of max_freq of bands vs last_phdos_peak
x_col, y_col = "max_freq_pbe_THz", "last_phdos_peak_pbe_THz"  # PBE
x_col, y_col = "max_freq_mace_y7uhwpje_THz", "last_phdos_peak_mace_y7uhwpje_THz"  # MACE

fig = px.scatter(
    df_summary.reset_index().dropna(),
    x=x_col,
    y=y_col,
    hover_name=id_key,
    hover_data=["formula"],
)

annotate_metrics(fig=fig, xs=df_summary.dropna()[x_col], ys=df_summary.dropna()[y_col])
add_identity_line(fig)
fig.layout.margin = dict(l=5, r=5, b=5, t=5)
fig.show()


# %% plot histogram of MP vs model last phonon DOS peaks
fig = px.histogram(
    df_summary.filter(like=(query := " - band_width")).dropna(), nbins=120, opacity=0.6
)
x_title_map = {
    " - band_width": "Band Width (THz)",
    " - last_phdos_peak": "Last Phonon DOS Peak (THz)",
}
fig.layout.xaxis.title = f"Difference in {x_title_map[query]} (THz)"
fig.layout.yaxis.title = "Number of Materials"
fig.layout.legend = dict(x=1, y=1, xanchor="right", yanchor="top")

fig.layout.margin = dict(l=5, r=5, b=5, t=5)
fig.show()


# %% plot MP vs model last phonon DOS peaks as scatter
# peak_pbe_col = f"last_phdos_peak_{dft_key}_THz"
# peak_ml_col = f"last_phdos_peak_{model_key.replace('-', '_')}_THz"
# axes_label, file_suffix = "last phDOS peak (THz)", "last-peak"

band_width_pbe_col = f"band_width_{dft_key}_THz"
band_width_ml_col = f"band_width_{model_key.replace('-', '_')}_THz"
axes_label, file_suffix = "band width (THz)", "band-width"

x_col, y_col = band_width_pbe_col, band_width_ml_col

# std dev of MP last phonon DOS peak
print(f"PBE last phonon DOS peak std dev: {df_summary[x_col].std():.2f} THz")

# post-hoc PES hardening shift in THz (ML-predictions will be boosted by this value)
pes_shift = 0  # 0.6
df_plot = df_summary.copy().reset_index().dropna()
df_plot[y_col] += pes_shift

fig = px.scatter(df_plot, x=x_col, y=[y_col], hover_name=id_key, hover_data=["formula"])

for trace in fig.data:
    trace.marker.size = 10
    trace.marker.opacity = 0.8

    targets, preds = df_plot[[x_col, trace.name]].dropna().to_numpy().T
    MAE = np.abs(targets - (preds + pes_shift)).mean()
    R2 = r2_score(targets, (preds + pes_shift))
    pretty_label = pretty_label_map[model_key]
    trace.name = (
        f"{pretty_label} ({MAE=:.2f}, R<sup>2</sup>={R2:.2f}, n={len(targets)})"
    )

# increase legend font size
fig.layout.legend.update(x=0.01, y=0.98, borderwidth=1, bordercolor="lightgray")

# annotate outliers from parity line
df_outliers = df_plot.query(f"{x_col} - `{y_col}` > 6 or `{y_col}` - {x_col} > 5")
for idx in df_outliers.index:
    row = df_outliers.loc[idx]
    too_low = row[x_col] > row[y_col]  # to annotation offset
    err = row[y_col] - row[x_col]
    fig.add_annotation(
        x=row[x_col],
        y=row[y_col],
        text=f"{htmlify(row['formula'])}<br>{row[id_key]}<br>{err:.0f} THz = "
        f"{abs(err / row[x_col]):.0%}",
        # xanchor="center" if too_low else "right",
        # yanchor="top" if too_low else "middle",
        # xshift=0 if too_low else -10,
        # yshift=-5 if too_low else 0,
        # showarrow=False,
        arrowhead=1,
        ax=(20 if too_low else -90),
        ay=(50 if too_low else 30),
        standoff=6,  # shorten arrow at end
    )

# set axis range to start at 0
xy_max = (df_plot[y_col].max()) // 10 * 10 + 10  # round up to nearest 10
fig.update_xaxes(range=[0, xy_max])
fig.update_yaxes(range=[0, xy_max])

dft_label = pretty_label_map[dft_key]
# title = f"{dft_label} std dev = {df_plot[x_col].std():.3g} THz"
# fig.layout.title.update(text=title, x=0.5, y=0.97)
fig.layout.xaxis.title.update(text=f"{dft_label} {axes_label}", standoff=6)
fig.layout.yaxis.title.update(text=f"ML {axes_label}", standoff=6)
fig.layout.legend.update(x=0.01, y=0.98, yanchor="top", title=None)
add_identity_line(fig)  # annotations change plot range so add parity line after

fig.layout.width, fig.layout.height = 500, 340
fig.layout.margin = dict(l=3, r=3, b=3, t=3)
fig.show()
img_name = f"parity-pbe-vs-ml-{file_suffix}"
parity_fig_path = f"{FIGS_DIR}/{which_db}/{img_name}.pdf"
save_fig(fig, parity_fig_path)
save_fig(fig, f"{PAPER_DIR}/{img_name}.pdf")


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
