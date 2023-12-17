"""Calculate confusion matrix for whether PBE and MACE both predict imaginary modes
for each material at the Gamma point.
"""

# %%
import numpy as np
import plotly.figure_factory as ff
from pymatviz.io import save_fig
from sklearn.metrics import confusion_matrix

from ffonons import PAPER_DIR, dft_key, pretty_label_map
from ffonons.io import load_pymatgen_phonon_docs

__author__ = "Janosh Riebesell"
__date__ = "2023-12-15"


# %% compute last phonon DOS peak for each model and MP
ph_docs, df_summary = load_pymatgen_phonon_docs(which_db := "phonon_db")
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
        annos = (
            ("True<br>Unstable", "False<br>Stable"),
            ("False<br>Unstable", "True<br>Stable"),
        )
        axis_labels = ("Unstable", "Stable")
    else:
        annos = (
            ("True<br>Negative", "False<br>Positive"),
            ("False<br>Negative", "True<br>Positive"),
        )
        axis_labels = ("No", "Yes")

    annotated_vals = np.array(annos, dtype=object) + "<br>" + conf_mat.astype(str) + "%"
    fig = ff.create_annotated_heatmap(
        z=np.rot90(conf_mat.T),
        x=axis_labels,
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
        x=0.5,
        y=-0.12,
        text=f"Accuracy={acc:.0%}",
        showarrow=False,
        font=dict(size=(font_size := 30)),
    )

    fig.layout.xaxis.title = pretty_label_map[model_key]
    fig.layout.yaxis.update(title=pretty_label_map[dft_key], tickangle=-90)
    fig.layout.font.size = font_size
    fig.layout.height = fig.layout.width = 425  # force same width and height
    fig.layout.margin = dict(l=0, r=0, t=0, b=40)

    fig.layout.update(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    # remove border line
    fig.layout.xaxis.update(showline=False, showgrid=False)
    fig.layout.yaxis.update(showline=False, showgrid=False)

    fig.show()

    img_name = f"{col.replace('_', '-')}confusion-matrix"
    save_fig(fig, f"{PAPER_DIR}/{img_name}.pdf")
