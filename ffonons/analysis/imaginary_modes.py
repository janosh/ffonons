"""Calculate confusion matrix for whether PBE and MACE both predict imaginary modes
for each material at the Gamma point.
"""

# %%
import numpy as np
import plotly.figure_factory as ff
from pymatviz.io import save_fig
from sklearn.metrics import confusion_matrix

from ffonons import FIGS_DIR, PAPER_DIR, dft_key, pretty_label_map
from ffonons.io import load_pymatgen_phonon_docs

__author__ = "Janosh Riebesell"
__date__ = "2023-12-15"


# %% compute last phonon DOS peak for each model and MP
imaginary_freq_tol = 1e-1
ph_docs, df_summary = load_pymatgen_phonon_docs(
    which_db := "phonon_db", imaginary_freq_tol=imaginary_freq_tol
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
