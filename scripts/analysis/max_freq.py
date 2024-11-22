"""Analyze ML vs DFT highest frequencies in phonon band structures."""

# %%
import numpy as np
import pandas as pd
import plotly.express as px
import pymatviz as pmv
from IPython.display import display
from pymatviz.enums import Key
from sklearn.metrics import r2_score

import ffonons
from ffonons.enums import DB, Model, PhKey

# from pymatgen.util.string import htmlify

__author__ = "Janosh Riebesell"
__date__ = "2023-11-24"


# %%
imaginary_freq_tol = 0.01
df_summary = ffonons.io.get_df_summary(
    which_db := DB.phonon_db, imaginary_freq_tol=imaginary_freq_tol
)
idx_n_avail = df_summary[Key.max_ph_freq].unstack().dropna(thresh=4).index


# %% parity plot of max_freq of bands vs last_phdos_peak
x_col, y_col = Key.max_ph_freq, Key.last_ph_dos_peak
model = Model.mace_mp0  # Key.pbe

df_plot = df_summary.xs(model, level=1)
fig = px.scatter(
    df_plot.reset_index(),
    x=x_col,
    y=y_col,
    hover_name=Key.mat_id,
    hover_data=[Key.formula],
)

pmv.powerups.annotate_metrics(
    fig=fig, xs=df_plot[x_col], ys=df_plot[y_col], suffix=f"N={len(df_plot):,}"
)
pmv.powerups.add_identity_line(fig)
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
# prop = Key.last_ph_dos_peak
prop = Key.max_ph_freq
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

fig.layout.xaxis.title = PhKey.val_label_dict()[f"{prop}_pbe"]
fig.layout.yaxis.title = PhKey.val_label_dict()[f"{prop}_ml"]
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
# annotations change x/y-range so add parity line after to ensure it spans whole plot
pmv.powerups.add_identity_line(fig)

fig.layout.margin = dict(l=3, r=3, b=3, t=3)
fig.show()
file_suffix = prop.replace("_", "-").replace("-thz", "")
img_name = f"parity-pbe-vs-ml-{file_suffix}"


# %%
pmv.save_fig(fig, f"{ffonons.PDF_FIGS}/{which_db}/{img_name}.pdf")
pmv.save_fig(fig, f"{ffonons.PAPER_DIR}/{img_name}.svg", width=600, height=400)


# %%
df_dft = df_summary.xs(Key.pbe, level=1)

# print dataframe with 5 largest absolute differences between ML and DFT max_freq
for model in Model:
    if model == Key.pbe or model not in df_summary.index.get_level_values(1):
        continue

    df_ml = df_summary.xs(model, level=1)
    dft_col = f"{Key.max_ph_freq}_dft"
    df_ml[dft_col] = df_dft[Key.max_ph_freq]
    df_ml["diff"] = df_ml[Key.max_ph_freq] - df_dft[Key.max_ph_freq]
    df_ml["pct_diff"] = df_ml["diff"] / df_dft[Key.max_ph_freq]

    cols = [
        Key.formula,
        Key.supercell,
        Key.max_ph_freq,
        dft_col,
        "diff",
        "pct_diff",
    ]
    col_map = PhKey.val_label_dict()
    styler = df_ml[cols].sort_values("diff", key=abs).tail(10).style.set_caption(model)
    float_cols = [*df_ml[cols].select_dtypes(include="number")]
    styler.format("{:.2f}", subset=float_cols).background_gradient(subset=float_cols)

    display(styler.format("{:.0%}", subset="pct_diff"))


# %% print largest max freq error for each model
df_max_freq_err = pd.DataFrame()
df_dft = df_summary.xs(Key.pbe, level=1)
max_err_key = "Max Ph Freq Max Error"

for model in Model:
    if model == Key.pbe or model not in df_summary.index.get_level_values(1):
        continue

    df_model = df_summary.loc[idx_n_avail].xs(model, level=1)
    if len(df_model) != len(idx_n_avail):
        raise ValueError(f"{len(df_model)=} {len(idx_n_avail)=}")
    worst_mat_id = abs(df_dft[Key.max_ph_freq] - df_model[Key.max_ph_freq]).idxmax()
    formula = df_model.loc[worst_mat_id, Key.formula]

    dct = {Key.mat_id.label: worst_mat_id, "Formula": formula}
    ml_freq = dct["Max Ph Freq ML"] = df_model.loc[worst_mat_id, Key.max_ph_freq]
    dft_freq = dct["Max Ph Freq DFT"] = df_dft.loc[worst_mat_id, Key.max_ph_freq]
    max_err = dct[max_err_key] = ml_freq - dft_freq
    dct["Max Error Rel"] = max_err / df_dft[Key.max_ph_freq].max()
    df_max_freq_err[model.label.split()[0].replace("-MS", "")] = dct

df_max_freq_err = df_max_freq_err.T.sort_values(by=max_err_key)
df_max_freq_err = df_max_freq_err.convert_dtypes().round(4)
df_max_freq_err.index.name = "Model"
display(df_max_freq_err)
df_max_freq_err.to_csv(f"{ffonons.PAPER_DIR}/max-phonon-freq-errors.csv")
