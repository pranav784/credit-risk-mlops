import pandas as pd

from src.fairness import group_metrics


def test_group_metrics_calculates_expected_rates():
    y_true = pd.Series([1, 1, 0, 0])
    y_pred = pd.Series([1, 0, 1, 0])

    metrics = group_metrics(y_true, y_pred)

    assert metrics["customers"] == 4
    assert metrics["recall"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["false_positive_rate"] == 0.5
    assert metrics["false_negative_rate"] == 0.5


def test_group_metrics_handles_no_positive_predictions():
    y_true = pd.Series([1, 0, 0, 0])
    y_pred = pd.Series([0, 0, 0, 0])

    metrics = group_metrics(y_true, y_pred)

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["false_negative_rate"] == 1.0