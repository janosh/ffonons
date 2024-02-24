"""Calculate confusion matrix for whether PBE and MACE both predict imaginary modes
for each material at the Gamma point.
"""

# %%
from typing import Literal

import pandas as pd
from pymatviz.io import df_to_html_table, df_to_pdf
from sklearn.metrics import confusion_matrix, r2_score

from ffonons import DATA_DIR, PAPER_DIR, PDF_FIGS, SITE_FIGS
from ffonons.enums import DB, Key
from ffonons.io import get_df_summary
from ffonons.plots import pretty_labels

__author__ = "Janosh Riebesell"
__date__ = "2023-12-15"


# %% compute last phonon DOS peak for each model and MP
imaginary_freq_tol = 0.01
df_summary = get_df_summary(
    which_db := DB.phonon_db, imaginary_freq_tol=imaginary_freq_tol, refresh_cache=False
)


# %% save analyzed MP IDs to CSV for rendering with Typst
# material IDs where all models have results
idx_n_avail = df_summary[Key.max_freq].unstack().dropna(thresh=4).index

for folder in (PAPER_DIR, f"{DATA_DIR}/{which_db}"):
    df_summary.xs(Key.pbe, level=1).loc[idx_n_avail][
        [Key.formula, Key.supercell]
    ].sort_index(key=lambda idx: idx.str.split("-").str[1].astype(int)).to_csv(
        f"{folder}/analyzed-mp-ids.csv"
    )


# %% make dataframe with model metrics for phonon DOS and BS predictions
df_metrics = pd.DataFrame()
df_metrics.index.name = "Model"

for key, df_model in df_summary.loc[idx_n_avail].groupby(level=1):
    if key == Key.pbe:
        continue

    label = pretty_labels.get(key, key)
    # df_metrics.loc[label, "N"] = len(df_model)
    df_model = df_model.droplevel(1)  # remove model key from index

    for metric in ("phdos_mae_THz", "phdos_r2"):
        col_name = pretty_labels.get(metric, metric)
        df_metrics.loc[label, col_name] = df_model[metric].mean()

    df_dft = df_summary.xs(Key.pbe, level=1)
    for metric in (
        Key.has_imag_modes,
        # Key.imaginary_gamma_freq,
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
        col_name = pretty_labels.get(metric, metric)

        cols = [f"Acc. {col_name}", f"FPR {col_name}", f"FNR {col_name}"]
        df_metrics.loc[label, cols] = acc, fp, fn

    for metric in (
        # Key.last_dos_peak,
        Key.max_freq,
    ):
        diff = df_dft[metric] - df_model[metric]
        MAE = diff.abs().mean()
        not_nan = diff.dropna().index
        R2 = r2_score(df_dft[metric].loc[not_nan], df_model[metric].loc[not_nan])
        df_metrics.loc[label, getattr(Key, f"mae_{metric}").label] = MAE
        df_metrics.loc[label, getattr(Key, f"r2_{metric}").label] = R2

# sort by ph DOS MAE
df_metrics = df_metrics.convert_dtypes().sort_values(by=Key.dos_mae.label).round(2)


# %% --- vertical metrics table ---
cmap = "Blues"
lower_better = [
    col for col in df_metrics if any(pat in col for pat in ("MAE", "FNR", "FPR"))
]
higher_better = {*df_metrics} - set(lower_better)
styler = df_metrics.T.convert_dtypes().style.format(
    # render integers without decimal places
    lambda val: (f"{val:.0f}" if val == int(val) else f"{val:.2f}")
    if isinstance(val, float)
    else val,
    precision=2,
    na_rep="-",
)
styler = styler.background_gradient(
    cmap=f"{cmap}_r", subset=pd.IndexSlice[[*lower_better], :], axis="columns"
)
styler.background_gradient(
    cmap=cmap, subset=pd.IndexSlice[[*higher_better], :], axis="columns"
)

# add up/down arrows to indicate which metrics are better when higher/lower
arrow_suffix = dict.fromkeys(higher_better, " ↑") | dict.fromkeys(lower_better, " ↓")
styler.relabel_index(
    [f"{col}{arrow_suffix.get(col, '')}" for col in styler.data.index], axis="index"
).set_uuid("")

border = "1px solid black"
styler.set_table_styles(
    [{"selector": "tr", "props": f"border-top: {border}; border-bottom: {border};"}]
)


table_name = f"ffonon-metrics-table-tol={imaginary_freq_tol}"

df_to_pdf(styler, f"{PDF_FIGS}/{table_name}.pdf", size="landscape")
df_to_html_table(styler, f"{SITE_FIGS}/{table_name}.svelte")
styler.set_caption(
    f"Harmonic phonons from ML force fields vs Togo DB PBE<br>"
    f"(N={len(idx_n_avail)}, imaginary mode tol={imaginary_freq_tol:.2f} THz)<br><br>"
)


# %% --- horizontal metrics table ---
if False:
    lower_better = [
        col for col in df_metrics if any(pat in col for pat in ("MAE", "FNR", "FPR"))
    ]
    styler = df_metrics.reset_index().style.format(precision=2, na_rep="-")
    styler = styler.background_gradient(cmap=cmap).background_gradient(
        cmap=f"{cmap}_r", subset=lower_better
    )

    arrow_suffix = dict.fromkeys(higher_better, " ↑") | dict.fromkeys(
        lower_better, " ↓"
    )
    styler.relabel_index(
        [f"{col}{arrow_suffix.get(col, '')}" for col in styler.data], axis="columns"
    ).set_uuid("").hide(axis="index")

    df_to_pdf(styler, f"{PDF_FIGS}/{table_name}.pdf")
    df_to_html_table(styler, f"{SITE_FIGS}/{table_name}.svelte")
    styler.set_caption("Metrics for harmonic phonons from ML force fields vs PBE")
