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

from ffonons import FIGS_DIR, PAPER_DIR, DBs, Key
from ffonons.io import get_df_summary
from ffonons.plots import model_labels, pretty_labels

__author__ = "Janosh Riebesell"
__date__ = "2023-12-15"


# %% compute last phonon DOS peak for each model and MP
imaginary_freq_tol = 0.05
df_summary = get_df_summary(
    which_db := DBs.phonon_db, imaginary_freq_tol=imaginary_freq_tol
)


# %% plot confusion matrix
model_key = "mace-y7uhwpje"
model_key = "chgnet-v0.3.0"

for col in ("imaginary_gamma_freq", "imaginary_freq"):
    df_dft, df_ml = (df_summary.xs(key, level=1) for key in (Key.dft, model_key))
    y_true, y_pred = df_dft[col], df_ml[col]
    y_true = y_true.loc[y_pred.index.droplevel(1)]
    conf_mat = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize="all")

    label1, label2 = (
        ("Γ-Stable", "Γ-Unstable") if "gamma" in col else ("Stable", "Unstable")
    )
    annos = (
        (f"True<br>{label1}", f"False<br>{label2}"),
        (f"False<br>{label1}", f"True<br>{label2}"),
    )
    conf_mat_pct = (100 * conf_mat).astype(int).astype(str)
    annotated_vals = np.array(annos, dtype=object) + "<br>" + conf_mat_pct + "%"
    fig = ff.create_annotated_heatmap(
        z=np.rot90(conf_mat.T),
        x=(label1, label2),
        y=(label2, label1),
        annotation_text=np.rot90(annotated_vals.T),
        colorscale="blues",
        xgap=7,
        ygap=7,
    )
    # annotate accuracy, number of materials, and tolerance for imaginary frequencies
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

    fig.layout.xaxis.title = pretty_labels[model_key]
    fig.layout.yaxis.update(title=pretty_labels[Key.dft], tickangle=-90)
    fig.layout.font.size = font_size
    fig.layout.height = fig.layout.width = 425  # force same width and height
    fig.layout.margin = dict(l=10, r=10, t=10, b=40)

    fig.layout.update(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    # remove border line
    fig.layout.xaxis.update(showline=False, showgrid=False)
    fig.layout.yaxis.update(showline=False, showgrid=False)

    fig.show()

    img_name = f"{col.replace('_', '-')}{model_key}-confusion-matrix"
    save_fig(fig, f"{PAPER_DIR}/{img_name}.pdf")
    save_fig(fig, f"{FIGS_DIR}/{which_db}/{img_name}.pdf")


# %% repeat confusion matrix calculation to check for consistency
dft_imag_col = "imaginary_gamma_freq"
ml_imag_col = "imaginary_gamma_freq"
for pbe, mace in ((True, True), (True, False), (False, True), (False, False)):
    both = df_summary.query(f"{dft_imag_col}=={pbe} and {ml_imag_col}=={mace}")
    print(f"{pbe=} {mace=} : {len(both)/len(df_summary):.1%}")

# kinetically unstable materials
n_unstable = df_summary[dft_imag_col].mean()
print(f"DFT unstable rate {n_unstable:.0%}")
n_unstable = df_summary[ml_imag_col].mean()
print(f"ML unstable rate {n_unstable:.0%}")


# %% plot imaginary modes confusion matrix as parity plot using min. freq. across all
# k-points (with shaded regions for TP, FP, FN, TN)
y_col, color_col = "min_freq_THz", "model"

df_melt = (
    df_summary.unstack(level=1)[y_col]
    .reset_index(names=[Key.mat_id, Key.formula])
    .melt(
        id_vars=[Key.mat_id, Key.formula, Key.dft],
        value_vars=list(model_labels),
        var_name=color_col,
        value_name=y_col,
    )
).dropna()

fig = px.scatter(
    df_melt,
    x=Key.dft,
    y=y_col,
    hover_data=[Key.mat_id, Key.formula],
    color=color_col,
)

imaginary_tol = 0.1
# add dashed dynamic instability decision boundary separators
fig.add_vline(x=-imaginary_tol, line=dict(width=1, dash="dash"))
fig.add_hline(y=-imaginary_tol, line=dict(width=1, dash="dash"))
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
        x0=-imaginary_tol,
        y0=-imaginary_tol,
        x1=10 * sign_x,
        y1=10 * sign_y,
        fillcolor=color,
        line=dict(width=0),  # remove edge line
        layer="below",
        opacity=0.1,
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
min_xy = df_melt[[y_col, Key.dft]].min().min() - 0.4
max_xy = df_melt[[y_col, Key.dft]].max().max() + 0.4
fig.update_xaxes(range=[min_xy, max_xy], title="PBE min. freq. (THz)")
fig.update_yaxes(range=[min_xy, max_xy], title="ML min. freq. (THz)")
fig.layout.legend.update(x=0.02, y=0.9, title="")
fig.layout.margin = dict(l=10, r=10, t=10, b=10)

fig.show()
