"""Calculate confusion matrix for whether PBE and MACE both predict imaginary modes
for each material at the Gamma point.
"""

# %%
import pandas as pd
import pymatviz as pmv
from IPython.display import display
from pymatviz.enums import Key

import ffonons
from ffonons.enums import DB, PhKey
from ffonons.metrics import get_df_metrics

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
# Key.max_ph_freq but any column will do)
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


# %% Compute metrics dataframe
df_metrics = get_df_metrics(df_summary.loc[idx_n_avail])


# %% Display and save metrics tables
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

lower_better = {
    col for col in df_metrics if any(pat in col for pat in ("MAE", "FNR", "FPR"))
}
higher_better = set(df_metrics) - set(lower_better)

# Separate classification and regression metrics
regr_col_keywords = ("MAE", "R2", "R<sup>2</sup>", "RMSE", "MAPE")
regr_cols = [col for col in df_metrics if any(pat in col for pat in regr_col_keywords)]
clf_cols = list(set(df_metrics) - set(regr_cols))

for metrics_cols, table_caption, filename in (
    (clf_cols, clf_caption, "ffonon-imag-clf-table"),
    (regr_cols, regr_metrics_caption, "ffonon-regr-metrics-table"),
):
    df_subset = df_metrics[metrics_cols]

    styler = df_subset.T.style.format(
        lambda val: (f"{val:.0f}" if val == int(val) else f"{val:.2f}")
        if isinstance(val, float)
        else val,
        precision=2,
        na_rep="-",
    )
    lower_present = set(lower_better) & set(df_subset)
    higher_present = set(higher_better) & set(df_subset)
    styler.background_gradient(
        cmap=f"{cmap}_r", subset=pd.IndexSlice[[*lower_present], :], axis="index"
    )
    styler.background_gradient(
        cmap=cmap, subset=pd.IndexSlice[[*higher_present], :], axis="index"
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

    styler.set_caption(table_caption)
    display(styler)


# %% Display transposed metrics table (models as index, metrics as columns)
styler = df_metrics[regr_cols].reset_index().style.format(precision=2, na_rep="-")
styler.background_gradient(cmap=cmap).background_gradient(
    cmap=f"{cmap}_r", subset=[*set(lower_better) & set(regr_cols)]
)

arrow_suffix = dict.fromkeys(higher_better, " ↑") | dict.fromkeys(lower_better, " ↓")
styler.relabel_index(
    [f"{col}{arrow_suffix.get(col, '')}" for col in styler.data], axis="columns"
).set_uuid("").hide(axis="index")

pmv.io.df_to_pdf(styler, file_path=f"{ffonons.PDF_FIGS}/ffonon-all-metrics-table.pdf")
pmv.io.df_to_html_table(
    styler, file_path=f"{ffonons.SITE_FIGS}/ffonon-all-metrics-table.svelte"
)
styler.set_caption("Metrics for harmonic phonons from ML force fields vs PBE")
display(styler)
