"""Calculate confusion matrix for whether PBE and MACE both predict imaginary modes
for each material at the Gamma point.
"""

# %%
from typing import Literal

import pandas as pd
from pymatviz.io import df_to_pdf
from sklearn.metrics import confusion_matrix, r2_score

from ffonons import FIGS_DIR, DBs, Key
from ffonons.io import get_df_summary
from ffonons.plots import pretty_labels

__author__ = "Janosh Riebesell"
__date__ = "2023-12-15"


# %% compute last phonon DOS peak for each model and MP
imaginary_freq_tol = 0.05
df_summary = get_df_summary(
    which_db := DBs.phonon_db, imaginary_freq_tol=imaginary_freq_tol
)


# %% make dataframe of metrics
df_metrics = pd.DataFrame()
df_metrics.index.name = "model"


for key, df_model in df_summary.groupby(level=1):
    if key == Key.dft:
        continue

    label = pretty_labels.get(key, key)
    df_metrics.loc[label, "N"] = len(df_model)
    df_model = df_model.droplevel(1)  # remove model key from index

    for metric in ("phdos_mae_THz", "phdos_r2"):
        pretty_metric = pretty_labels.get(metric, metric)
        df_metrics.loc[label, pretty_metric] = df_model[metric].mean()

    df_dft = df_summary.xs(Key.dft, level=1)
    for metric in (
        "imaginary_freq",
        # "imaginary_gamma_freq",
    ):
        imag_modes_pred = df_model[metric]
        imag_modes_true = df_dft[metric].loc[imag_modes_pred.index]
        normalize: Literal["true", "pred", "all", None] = "true"
        conf_mat = confusion_matrix(
            y_true=imag_modes_true, y_pred=imag_modes_pred, normalize=normalize
        )
        (tn, fp), (fn, tp) = conf_mat
        if normalize == "true":
            assert tn + fp == 1
            assert fn + tp == 1
        elif normalize == "pred":
            assert tn + fn == 1
            assert fp + tp == 1
        elif normalize == "all":
            assert tn + fp + fn + tp == 1
        acc = (tn + tp) / 2
        pretty_metric = pretty_labels.get(metric, metric)

        cols = [f"Acc. {pretty_metric}", f"FPR {pretty_metric}", f"FNR {pretty_metric}"]
        df_metrics.loc[label, cols] = acc, fp, fn

    for metric in (
        # Keys.last_dos_peak,
        Key.max_freq,
    ):
        diff = df_dft[metric] - df_model[metric]
        MAE = diff.abs().mean()
        not_nan = diff.dropna().index
        R2 = r2_score(df_dft[metric].loc[not_nan], df_model[metric].loc[not_nan])
        df_metrics.loc[label, pretty_labels[f"mae_{metric}"]] = MAE
        df_metrics.loc[label, pretty_labels[f"r2_{metric}"]] = R2

df_metrics = df_metrics.convert_dtypes().round(2)


# %% make df_metrics a styler with gradient heatmap
cmap = "RdGy"
lower_better = [
    col for col in df_metrics if any(pat in col for pat in ("MAE", "FNR", "FPR"))
]
styler = (
    df_metrics.reset_index()
    .style.background_gradient(cmap=cmap)
    .background_gradient(cmap=f"{cmap}_r", subset=lower_better)
    .format(precision=2, na_rep="-")
)

df_to_pdf(styler, f"{FIGS_DIR}/metrics-table.pdf")

styler.set_caption("Metrics for harmonic phonons from ML force fields vs PBE")
