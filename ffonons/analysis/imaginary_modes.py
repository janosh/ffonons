"""Calculate confusion matrix for whether PBE and MACE both predict imaginary modes
for each material at the Gamma point.
"""

# %%
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from pymatviz.io import save_fig
from pymatviz.utils import add_identity_line
from sklearn.metrics import confusion_matrix

from ffonons import FIGS_DIR, PAPER_DIR, dft_key, formula_key, id_key
from ffonons.io import load_pymatgen_phonon_docs
from ffonons.plots import pretty_label_map

__author__ = "Janosh Riebesell"
__date__ = "2023-12-15"


# %% compute last phonon DOS peak for each model and MP
imaginary_freq_tol = 1e-1
ph_docs, df_summary = load_pymatgen_phonon_docs(
    which_db := "phonon-db", imaginary_freq_tol=imaginary_freq_tol
)
model_key = "mace-y7uhwpje"

print(f'{df_summary.filter(like="imaginary_").dropna().shape=}')


# %% plot confusion matrix
for col in ("imaginary_gamma_freq_", "imaginary_freq_"):
    y_true = df_summary.dropna()[f"{col}{dft_key.replace('-', '_')}"]
    y_pred = df_summary.dropna()[f"{col}{model_key.replace('-', '_')}"]
    conf_mat = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=(False, True))
    # make percentages
    conf_mat = (conf_mat / conf_mat.sum() * 100).round(1)

    if "gamma" in col:
        label1, label2 = axis_labels = ("Stable", "Unstable")
    else:
        axis_labels = {"No": "Negative", "Yes": "Positive"}
        label1, label2 = axis_labels.values()
    annos = (
        (f"True<br>{label1}", f"False<br>{label2}"),
        (f"False<br>{label1}", f"True<br>{label2}"),
    )

    annotated_vals = np.array(annos, dtype=object) + "<br>" + conf_mat.astype(str) + "%"
    fig = ff.create_annotated_heatmap(
        z=np.rot90(conf_mat.T),
        x=list(axis_labels),
        y=list(reversed(axis_labels)),
        annotation_text=np.rot90(annotated_vals.T),
        colorscale="blues",
        xgap=7,
        ygap=7,
    )
    # annotate accuracy in top left corner
    acc = conf_mat.diagonal().sum() / conf_mat.sum()
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.45,
        y=-0.12,
        text=f"Acc={acc:.0%}, N={len(y_true)}, Tol={imaginary_freq_tol:.2}",
        showarrow=False,
        font=dict(size=(font_size := 26)),
    )

    fig.layout.xaxis.title = pretty_label_map[model_key]
    fig.layout.yaxis.update(title=pretty_label_map[dft_key], tickangle=-90)
    fig.layout.font.size = font_size
    fig.layout.height = fig.layout.width = 425  # force same width and height
    fig.layout.margin = dict(l=10, r=10, t=10, b=40)

    fig.layout.update(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    # remove border line
    fig.layout.xaxis.update(showline=False, showgrid=False)
    fig.layout.yaxis.update(showline=False, showgrid=False)

    fig.show()

    img_name = f"{col.replace('_', '-')}confusion-matrix"
    save_fig(fig, f"{PAPER_DIR}/{img_name}.pdf")
    save_fig(fig, f"{FIGS_DIR}/{which_db}/{img_name}.pdf")


# %% repeat confusion matrix calculation to check for consistency
dft_imag_col = f"imaginary_gamma_freq_{dft_key}"
ml_imag_col = f"imaginary_gamma_freq_{model_key.replace('-', '_')}"
for pbe, mace in ((True, True), (True, False), (False, True), (False, False)):
    both = df_summary.dropna().query(f"{dft_imag_col}=={pbe} and {ml_imag_col}=={mace}")
    print(f"{pbe=} {mace=} : {len(both)/len(df_summary.dropna()):.1%}")

# kinetically unstable materials
n_unstable = df_summary[dft_imag_col].mean()
print(f"DFT unstable rate {n_unstable:.0%}")
n_unstable = df_summary[ml_imag_col].mean()
print(f"ML unstable rate {n_unstable:.0%}")


# %% plot imaginary modes confusion matrix as parity plot using min. freq. across all
# bands and k-points (with shaded regions for TP, FP, FN, TN)
pbe_key = "min_freq_pbe_THz"
y_key, color_key = "min_freq_THz", "model"
df_melt = df_summary.reset_index().melt(
    id_vars=[id_key, formula_key, pbe_key],
    value_vars=(y_cols := set(df_summary.filter(like="min_freq_")) - {pbe_key}),
    var_name=color_key,
    value_name=y_key,
)
df_melt["model"] = df_melt["model"].str.split("_").str[2:-1].str.join("-")

fig = px.scatter(
    df_melt.dropna(),
    x=pbe_key,
    y=y_key,
    hover_data=[id_key, formula_key],
    color=color_key,
)

imag_tol = 0.1
# add dashed dynamic instability decision boundary separators
fig.add_vline(x=-imag_tol, line=dict(width=1, dash="dash"))
fig.add_hline(y=-imag_tol, line=dict(width=1, dash="dash"))
add_identity_line(fig)

# add transparent rectangle with TN, TP, FN, FP labels in each quadrant
for sign_x, sign_y, label, color in (
    (-1, -1, "TP", "green"),
    (-1, 1, "FN", "orange"),
    (1, -1, "FP", "red"),
    (1, 1, "TN", "blue"),
):
    common = dict(row="all", col="all")
    fig.add_shape(
        type="rect",
        x0=-imag_tol,
        y0=-imag_tol,
        x1=10 * sign_x,
        y1=10 * sign_y,
        fillcolor=color,
        line=dict(width=0),  # remove edge line
        layer="below",
        opacity=0.2,
        **common,
    )
    fig.add_annotation(
        x=int(sign_x > 0),
        y=int(sign_y > 0),
        xshift=-15 * sign_x,
        yshift=-5 * sign_y,
        text=label,
        showarrow=False,
        font=dict(size=16, color=color),
        xref="x domain",
        yref="y domain",
        **common,
    )
min_xy = df_melt[[y_key, pbe_key]].min().min() - 0.4
max_xy = df_melt[[y_key, pbe_key]].max().max() + 0.4
fig.update_xaxes(range=[min_xy, max_xy], title="PBE min. freq. (THz)")
fig.update_yaxes(range=[min_xy, max_xy], title="ML min. freq. (THz)")
fig.layout.legend.update(x=0.02, y=0.9, title="")
fig.layout.margin = dict(l=10, r=10, t=10, b=10)

fig.show()
