import pandas as pd
import pytest
from pandas import DataFrame
from pymatviz.enums import Key

from ffonons.enums import Model, PhKey
from ffonons.metrics import get_df_metrics
from tests.conftest import df_preds_mock


def test_get_df_metrics() -> None:
    df_out = get_df_metrics(df_preds_mock)

    assert isinstance(df_out, DataFrame)
    assert df_out.index.name == "Model"
    assert set(df_out.index) == {
        Model.mace_mp.label,
        Model.chgnet_030.label,
        Model.m3gnet_ms.label,
        Model.sevennet_0.label,
    }
    expected_columns = [
        "Phonon DOS MAE",
        PhKey.ph_dos_r2.label,
        PhKey.mae_max_freq_thz.label,
        PhKey.r2_max_freq_thz.label,
        "Prec.",
        "Recall",
        "F1",
        "ROC AUC",
        "Acc.",
        "FPR",
        "FNR",
    ]
    assert all(col in df_out for col in expected_columns)


def test_get_df_metrics_values() -> None:
    df_out = get_df_metrics(df_preds_mock)

    assert df_out.loc[Model.mace_mp.label, "Phonon DOS MAE"] == pytest.approx(
        3.816, abs=0.001
    )
    assert df_out.loc[Model.chgnet_030.label, PhKey.ph_dos_r2.label] == pytest.approx(
        -0.096, abs=0.001
    )
    assert df_out.loc[
        Model.m3gnet_ms.label, PhKey.mae_max_freq_thz.label
    ] == pytest.approx(4.327, abs=0.001)
    assert df_out.loc[
        Model.chgnet_030.label, PhKey.r2_max_freq_thz.label
    ] == pytest.approx(0.974, abs=0.001)


def test_get_df_metrics_classification_metrics() -> None:
    df_out = get_df_metrics(df_preds_mock)

    for model in df_out.index:
        assert 0 <= df_out.loc[model, "Prec."] <= 1
        assert 0 <= df_out.loc[model, "Recall"] <= 1
        assert 0 <= df_out.loc[model, "F1"] <= 1
        assert 0 <= df_out.loc[model, "ROC AUC"] <= 1


def test_get_df_metrics_sorting() -> None:
    df_out = get_df_metrics(df_preds_mock)

    assert df_out.index.to_list() == [
        Model.sevennet_0.label,
        Model.mace_mp.label,
        Model.chgnet_030.label,
        Model.m3gnet_ms.label,
    ]


@pytest.mark.parametrize("model", [Model.mace_mp, Model.m3gnet_ms])
def test_get_df_metrics_model_exclusion(model: str) -> None:
    df_preds = df_preds_mock.drop(model, level=1)
    df_out = get_df_metrics(df_preds)
    assert model.label not in df_out.index


def test_get_df_metrics_empty_input() -> None:
    empty_df = pd.DataFrame(
        columns=[
            PhKey.ph_dos_mae,
            PhKey.ph_dos_r2,
            Key.max_ph_freq,
            Key.has_imag_ph_modes,
        ]
    )
    empty_df.index = pd.MultiIndex(
        levels=[[], []], codes=[[], []], names=[Key.mat_id, "model"]
    )
    df_out = get_df_metrics(empty_df)
    assert df_out.empty
