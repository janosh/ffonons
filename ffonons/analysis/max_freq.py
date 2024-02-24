"""Analyze ML vs DFT highest frequencies in phonon band structures."""

# %%
import numpy as np
import plotly.express as px
from pymatviz.io import save_fig
from pymatviz.utils import add_identity_line, annotate_metrics
from sklearn.metrics import r2_score

from ffonons import PAPER_DIR, PDF_FIGS
from ffonons.enums import DB, Key, Model
from ffonons.io import get_df_summary

# from pymatgen.util.string import htmlify

__author__ = "Janosh Riebesell"
__date__ = "2023-11-24"


# %%
imaginary_freq_tol = 0.01
df_summary = get_df_summary(
    which_db := DB.phonon_db, imaginary_freq_tol=imaginary_freq_tol
)


# %% parity plot of max_freq of bands vs last_phdos_peak
x_col, y_col = Key.max_freq, Key.last_dos_peak
model = Model.mace_mp  #  Key.pbe

df_plot = df_summary.xs(model, level=1)
fig = px.scatter(
    df_plot.reset_index(),
    x=x_col,
    y=y_col,
    hover_name=Key.mat_id,
    hover_data=[Key.formula],
)

annotate_metrics(
    fig=fig, xs=df_plot[x_col], ys=df_plot[y_col], suffix=f"N={len(df_plot):,}"
)
add_identity_line(fig)
title = f"{model.label} {x_col.label} vs {y_col.label}"
fig.layout.title.update(text=title, x=0.5, y=0.97)
fig.layout.margin = dict(l=5, r=5, b=5, t=35)
fig.show()


# %% plot histogram of MP vs model last phonon DOS peaks
# df_diff =
fig = px.histogram(
    df_summary.filter(like=(query := " - band_width")), nbins=120, opacity=0.6
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
# prop = Key.last_dos_peak
prop = Key.max_freq
df_plot = df_summary.unstack(level=1)[prop].dropna().round(2).copy()
hover_cols = [Key.formula, Key.n_sites]
df_plot[hover_cols] = df_summary.xs(Key.pbe, level=1)[hover_cols]
x_col = Key.pbe
y_cols = list({*Model.key_val_dict().values()} & {*df_plot})
# post-hoc PES hardening shift in THz (ML-predictions will be boosted by this value)
pes_shift = 0  # 0.6
df_plot[y_cols] += pes_shift

# std dev of MP last phonon DOS peak
print(f"PBE last phonon DOS peak std dev: {df_plot[x_col].std():.2f} THz")


fig = px.scatter(
    df_plot.reset_index(),
    x=x_col,
    y=y_cols,
    hover_name=Key.mat_id,
    hover_data=hover_cols,
    size=Key.n_sites,
)

for trace in fig.data:
    trace.marker.size = 6  # clashes with size=Key.n_sites
    trace.marker.opacity = 0.6

    targets, preds = df_plot[[x_col, trace.name]].dropna().to_numpy().T
    MAE = np.abs(targets - (preds + pes_shift)).mean()
    R2 = r2_score(targets, (preds + pes_shift))
    trace_label = Model.val_label_dict()[trace.name]
    trace.name = f"<b>{trace_label}</b><br>{MAE=:.2f}, R<sup>2</sup>={R2:.2f}"

# # annotate outliers from parity line
# for y_col in y_cols:
#     # get severe outliers
#     df_outliers = df_plot.query(f"`{y_col}` > {x_col} * 3 or 3 * `{y_col}` < {x_col}")
#     for mat_id in df_outliers.index:
#         xi, yi, formula = df_outliers.loc[mat_id, [x_col, y_col, Key.formula]]
#         ml_too_low = xi > yi  # ML underpredicts, used in annotation offset
#         err = yi - xi
#         fig.add_annotation(
#             x=xi,
#             y=yi,
#             text=f"{htmlify(mat_id)}<br>{formula}<br>{err:.0f} THz = "
#             f"{abs(err / xi):.0%}",
#             # xanchor="center" if too_low else "right",
#             # yanchor="top" if too_low else "middle",
#             # xshift=0 if too_low else -10,
#             # yshift=-5 if too_low else 0,
#             # showarrow=False,
#             arrowhead=1,
#             ax=(20 if ml_too_low else -90),
#             ay=(50 if ml_too_low else 30),
#             standoff=6,  # shorten arrow at end
#         )

#     # set axis range to start at 0
#     xy_max = (df_plot[y_col].max()) // 10 * 10 + 10  # round up to nearest 10
#     fig.update_xaxes(range=[0, xy_max])
#     fig.update_yaxes(range=[0, xy_max])

fig.layout.xaxis.title = Key.val_label_dict()[f"{prop}_pbe"]
fig.layout.yaxis.title = Key.val_label_dict()[f"{prop}_ml"]
fig.layout.legend.update(
    x=0.01,
    y=0.98,
    yanchor="top",
    borderwidth=1,
    bordercolor="lightgray",
    font_size=14,
    title=dict(
        text=f"  N={len(df_plot)}, imag. tol={imaginary_freq_tol}", font_size=14
    ),
    itemsizing="constant",
)
add_identity_line(fig)  # annotations change plot range so add parity line after

fig.layout.margin = dict(l=3, r=3, b=3, t=3)
fig.show()
file_suffix = prop.replace("_", "-").replace("-thz", "")
img_name = f"parity-pbe-vs-ml-{file_suffix}"


# %%
save_fig(fig, f"{PDF_FIGS}/{which_db}/{img_name}.pdf")
save_fig(fig, f"{PAPER_DIR}/{img_name}.svg", width=600, height=400)


# %%
srs_dft = df_summary.xs(Key.pbe, level=1)[Key.last_dos_peak]
for key, df_model in df_summary.groupby(level=1):
    if key == Key.pbe:
        continue

    print("  most accurate")
    diff = (srs_dft - df_model[Key.last_dos_peak]).dropna()
    most_accurate = diff.abs().sort_values().head(5)
    for (mp_id, formula), diff in most_accurate[:5].items():
        print(f"  - {diff:.2f} THz: {mp_id} {formula}")
        # print(glob(f"{FIGS_DIR}/{which_db}/{mp_id}-bands*.pdf")[0])

    print(f"{Model.val_label_dict()[key]} {Key.last_dos_peak} (N={len(df_model)})")
    for err_dir in ("over", "under"):
        print(f"  most {err_dir}estimated")
        diff = (srs_dft - df_model[Key.last_dos_peak]).dropna()
        for mp_id, diff_val in getattr(diff, "head" if err_dir == "under" else "tail")(
            3
        ).items():
            print(f"  - {diff_val:.2f} THz: {mp_id}")
            # print("  " + glob(f"{FIGS_DIR}/{which_db}/{mp_id}-bands*.pdf")[0])
