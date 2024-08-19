"""Utilities for computing classification and regression metrics for ML force fields
(MLFF) to predict harmonic phonons.
"""

import numpy as np
import pandas as pd
from pymatviz.enums import Key
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, roc_auc_score

from ffonons.enums import Model, PhKey


def get_df_metrics(df_preds: pd.DataFrame) -> pd.DataFrame:
    """Compute a metrics dataframe for ML force fields predicting harmonic phonons.

    Args:
        df_preds (pd.DataFrame): A dataframe with material IDs as 1st index level and
            models as 2nd level index as well as the following columns:
            - Key.ph_dos_mae
            - Key.max_ph_freq
            - Key.has_imag_ph_modes
            - PhKey.ph_dos_r2

    Returns:
        pd.DataFrame: Regression and classification metrics for predicting various
            phonon properties.
    """
    df_metrics = pd.DataFrame()
    df_metrics.index.name = "Model"

    for model in Model:
        if model == Key.pbe or model not in df_preds.index.get_level_values(1):
            continue

        df_model = df_preds.xs(model, level=1)
        df_dft = df_preds.xs(Key.pbe, level=1)

        # Regression metrics
        for metric in (Key.ph_dos_mae, PhKey.ph_dos_r2):
            df_metrics.loc[model.label, metric.label] = df_model[metric].mean()

        for metric in (Key.max_ph_freq,):
            diff = df_dft[metric] - df_model[metric]
            ph_freq_mae = diff.abs().mean()
            not_nan = diff.dropna().index
            ph_freq_r2 = r2_score(
                df_dft[metric].loc[not_nan], df_model[metric].loc[not_nan]
            )
            df_metrics.loc[model.label, getattr(PhKey, f"mae_{metric}").label] = (
                ph_freq_mae
            )
            df_metrics.loc[model.label, getattr(PhKey, f"r2_{metric}").label] = (
                ph_freq_r2
            )

        # Classification metrics (only for Key.has_imag_ph_modes)
        has_imag_modes_pred = df_model[Key.has_imag_ph_modes]
        has_imag_modes_true = df_dft[Key.has_imag_ph_modes].loc[
            has_imag_modes_pred.index
        ]

        # warning: this returns a 1x1 matrix conf_mat=[[1]] if only one class is present
        # which will crash the unpacking of TN, FP, FN, TP
        (_true_neg, false_pos), (false_neg, true_pos) = confusion_matrix(
            y_true=has_imag_modes_true, y_pred=has_imag_modes_pred, normalize="true"
        )

        acc = accuracy_score(has_imag_modes_true, has_imag_modes_pred)
        precision = (
            true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else np.nan
        )
        recall = (
            true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else np.nan
        )
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else np.nan
        )
        roc_auc = roc_auc_score(has_imag_modes_true, has_imag_modes_pred)

        metrics = {
            PhKey.prec_imag_freq: precision,
            PhKey.recall_imag_freq: recall,
            PhKey.f1_imag_freq: f1,
            PhKey.roc_auc_imag_freq: roc_auc,
            PhKey.acc_imag_freq: acc,
            PhKey.fpr_imag_freq: false_pos,
            PhKey.fnr_imag_freq: false_neg,
        }

        for metric, val in metrics.items():
            df_metrics.loc[model.label, metric.label] = val

    if Key.ph_dos_mae.label in df_metrics.columns:
        df_metrics = df_metrics.sort_values(by=Key.ph_dos_mae.label)
    return df_metrics.round(3)
