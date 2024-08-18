"""Calculate confusion matrix for whether PBE and MACE both predict imaginary modes
for each material at the Gamma point.
"""

# %%
from typing import Literal

import pandas as pd
import pymatviz as pmv
from IPython.display import display
from pymatviz.enums import Key
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, roc_auc_score

import ffonons
from ffonons.enums import DB, Model, PhKey

__author__ = "Janosh Riebesell"
__date__ = "2023-12-15"


# %% compute last phonon DOS peak for each model and MP
imaginary_freq_tol = 0.01
df_summary = ffonons.io.get_df_summary(
    which_db := DB.phonon_db, imaginary_freq_tol=imaginary_freq_tol
)

print(f"total docs {len(df_summary)=:,}")


# %% completed phonon calcs by model
n_completed_by_model = df_summary.groupby(level=1).size().sort_values()
print(f"{n_completed_by_model=}".split("dtype: ")[0])

# get material IDs for which all models (ML + DFT) have results (filtering by
# Key.max_ph_freq but any column will)
thresh = 5
idx_n_avail = df_summary[Key.max_ph_freq].unstack().dropna(thresh=thresh).index
n_avail = len(idx_n_avail)
print(f"{n_avail:,} materials with results from at least {thresh} models (incl. DFT)")


# %% save analyzed MP IDs to CSV for rendering with Typst
for folder in (
    ffonons.PAPER_DIR,
    # f"{ffonons.DATA_DIR}/{which_db}",
):
    df_summary.xs(Key.pbe, level=1).loc[idx_n_avail][
        [Key.formula, Key.supercell, Key.n_sites]
    ].sort_index(key=lambda idx: idx.str.split("-").str[1].astype(int)).to_csv(
        f"{folder}/phonon-analysis-mp-ids.csv"
    )


# %% make dataframe with model regression metrics for phonon DOS and BS predictions
df_regr: pd.DataFrame = pd.DataFrame()
df_regr.index.name = "Model"


for model in Model:
    if model == Key.pbe or model not in df_summary.index.get_level_values(1):
        continue

    df_model = df_summary.loc[idx_n_avail].xs(model, level=1)

    for metric in (Key.ph_dos_mae, PhKey.ph_dos_r2):
        df_regr.loc[model.label, metric.label] = df_model[metric].mean()

    df_dft = df_summary.xs(Key.pbe, level=1)

    for metric in (
        # Key.last_ph_dos_peak,
        Key.max_ph_freq,
    ):
        diff = df_dft[metric] - df_model[metric]
        ph_freq_mae = diff.abs().mean()
        not_nan = diff.dropna().index
        ph_freq_r2 = r2_score(
            df_dft[metric].loc[not_nan], df_model[metric].loc[not_nan]
        )
        df_regr.loc[model.label, getattr(PhKey, f"mae_{metric}").label] = ph_freq_mae
        df_regr.loc[model.label, getattr(PhKey, f"r2_{metric}").label] = ph_freq_r2


# sort by ph DOS MAE
df_regr = df_regr.convert_dtypes().sort_values(by=Key.ph_dos_mae.label).round(2)


# %% make dataframe with model metrics for phonon DOS and BS predictions
dfs_imag: dict[str, pd.DataFrame] = {}
for col in (Key.has_imag_ph_modes, Key.has_imag_ph_gamma_modes):
    df_imag = pd.DataFrame()
    df_imag.index.name = "Model"
    dfs_imag[col] = df_imag

    for model in Model:
        if model == Key.pbe or model not in df_summary.index.get_level_values(1):
            continue

        df_model = df_summary.loc[idx_n_avail].xs(model, level=1)

        df_dft = df_summary.xs(Key.pbe, level=1)
        imag_modes_pred = df_model[col]
        imag_modes_true = df_dft[col].loc[imag_modes_pred.index]
        normalize: Literal["true", "pred", "all", None] = "true"
        conf_mat = confusion_matrix(
            y_true=imag_modes_true, y_pred=imag_modes_pred, normalize=normalize
        )
        (tn, fp), (fn, tp) = conf_mat
        err_msg = "Invalid confusion matrix"
        if normalize == "true":
            if tn + fp != 1:
                raise ValueError(f"{err_msg} {tn=} + {fp=} = {tn + fp}, should be 1")
            if fn + tp != 1:
                raise ValueError(f"{err_msg} {fn=} + {tp=} = {fn + tp}, should be 1")
        elif normalize == "pred":
            if tn + fn != 1:
                raise ValueError(f"{err_msg} {tn=} + {fn=} = {tn + fn}, should be 1")
            if fp + tp != 1:
                raise ValueError(f"{err_msg} {fp=} + {tp=} = {fp + tp}, should be 1")
        elif normalize == "all":
            if tn + fp + fn + tp != 1:
                raise ValueError(err_msg)
        acc = accuracy_score(imag_modes_true, imag_modes_pred)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        roc_auc = roc_auc_score(imag_modes_true, imag_modes_pred)
        for metric, val in {
            PhKey.prec_imag_freq: precision,
            PhKey.recall_imag_freq: recall,
            PhKey.f1_imag_freq: f1,
            PhKey.roc_auc_imag_freq: roc_auc,
            PhKey.acc_imag_freq: acc,
            PhKey.fpr_imag_freq: fp,
            PhKey.fnr_imag_freq: fn,
        }.items():
            df_imag.loc[model.label, metric.label] = val

    df_imag = df_imag.sort_values(
        by=PhKey.roc_auc_imag_freq.label, ascending=False
    ).round(2)


# %% --- display metrics table (metrics as index, models as columns) ---
def caption_factory(key: PhKey) -> str:
    """Make caption for metrics table of classifying imaginary phonon mode."""
    return (
        f"MLFF vs {which_db.label} {key.label} classification<br>"
        f"(N={len(idx_n_avail):,}, imaginary mode tol={imaginary_freq_tol:.2f} THz)<br>"
    )


cmap = "Blues"
regr_metrics_caption = (
    f"Harmonic phonons from MLFF vs PhononDB PBE (N={len(idx_n_avail):,})<br>"
)
clf_caption = caption_factory(Key.has_imag_ph_modes)
clf_gam_caption = caption_factory(Key.has_imag_ph_gamma_modes)
write_to_disk = True
lower_better, higher_better = [], []

for df_loop, caption, filename in (
    (dfs_imag[Key.has_imag_ph_modes], clf_caption, "ffonon-imag-clf-table"),
    # (
    #     dfs_imag[Key.has_imag_ph_gamma_modes],
    #     clf_gam_caption,
    #     "ffonon-imag-gamma-clf-table",
    # ),
    (df_regr, regr_metrics_caption, "ffonon-regr-metrics-table"),
):
    lower_better = [
        col for col in df_loop if any(pat in col for pat in ("MAE", "FNR", "FPR"))
    ]
    higher_better = list(set(df_loop) - set(lower_better))
    styler = df_loop.T.style.format(
        # render integers without decimal places
        lambda val: (f"{val:.0f}" if val == int(val) else f"{val:.2f}")
        if isinstance(val, float)
        else val,
        precision=2,
        na_rep="-",
    )
    styler.background_gradient(
        cmap=f"{cmap}_r", subset=pd.IndexSlice[[*lower_better], :], axis="columns"
    )
    styler.background_gradient(
        cmap=cmap, subset=pd.IndexSlice[higher_better, :], axis="columns"
    )

    # add up/down arrows to indicate which metrics are better when higher/lower
    arrow_suffix = dict.fromkeys(higher_better, " ↑") | dict.fromkeys(
        lower_better, " ↓"
    )
    styler.relabel_index(
        [f"{col}{arrow_suffix.get(col, '')}" for col in styler.data.index], axis="index"
    ).set_uuid("")

    border = "1px solid black"
    styler.set_table_styles(
        [{"selector": "tr", "props": f"border-top: {border}; border-bottom: {border};"}]
    )

    if filename and write_to_disk:
        table_name = f"{filename}-tol={imaginary_freq_tol}"
        pdf_table_path = f"{ffonons.PDF_FIGS}/{which_db}/{table_name}.pdf"
        pmv.io.df_to_pdf(styler, file_path=pdf_table_path, size="landscape")
        pmv.io.df_to_html_table(
            styler, file_path=f"{ffonons.SITE_FIGS}/{table_name}.svelte"
        )

    styler.set_caption(caption)
    display(styler)


# %% --- display transposed metrics table (models as index, metrics as columns) ---
lower_better = [
    col for col in df_regr if any(pat in col for pat in ("MAE", "FNR", "FPR"))
]
styler = df_regr.reset_index().style.format(precision=2, na_rep="-")
styler.background_gradient(cmap=cmap).background_gradient(
    cmap=f"{cmap}_r", subset=lower_better
)

arrow_suffix = dict.fromkeys(higher_better, " ↑") | dict.fromkeys(lower_better, " ↓")
styler.relabel_index(
    [f"{col}{arrow_suffix.get(col, '')}" for col in styler.data], axis="columns"
).set_uuid("").hide(axis="index")

pmv.io.df_to_pdf(styler, file_path=f"{ffonons.PDF_FIGS}/ffonon-regr-metrics-table.pdf")
pmv.io.df_to_html_table(
    styler, file_path=f"{ffonons.SITE_FIGS}/ffonon-regr-metrics-table.svelte"
)
styler.set_caption("Metrics for harmonic phonons from ML force fields vs PBE")
display(styler)
