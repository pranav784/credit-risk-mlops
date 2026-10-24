import pandas as pd
import pytest

from src.config import load_config
from src.features import (
    engineer_features,
    split_features_target,
    validate_dataset,
)

CONFIG = load_config("configs/base.yaml")


def load_v1():
    return pd.read_csv("credit_card_customers_v1.csv")


def test_v1_dataset_passes_validation():
    validate_dataset(load_v1(), CONFIG)


def test_feature_engineering_adds_five_features():
    result = engineer_features(load_v1(), CONFIG)
    expected = {
        "Credit_Utilization",
        "High_Utilization_Flag",
        "Late_Payment_Flag",
        "Balance_to_Monthly_Spend",
        "Transactions_per_Tenure_Month",
    }
    assert expected.issubset(result.columns)
    assert len(result) == 800


def test_target_and_customer_id_are_not_predictors():
    df = engineer_features(load_v1(), CONFIG)
    X, y = split_features_target(df, CONFIG)
    assert "Customer_ID" not in X.columns
    assert "High_Risk_Customer" not in X.columns
    assert len(X) == len(y)


def test_duplicate_customer_ids_are_rejected():
    df = load_v1()
    df.loc[df.index[1], "Customer_ID"] = df.loc[df.index[0], "Customer_ID"]
    with pytest.raises(ValueError, match="unique"):
        validate_dataset(df, CONFIG)


def test_missing_target_values_are_rejected():
    df = load_v1()
    df.loc[df.index[0], "High_Risk_Customer"] = None
    with pytest.raises(ValueError, match="missing"):
        validate_dataset(df, CONFIG)


def test_zero_credit_limit_does_not_create_infinite_utilization():
    df = load_v1()
    df.loc[df.index[0], "Credit_Limit"] = 0
    result = engineer_features(df, CONFIG)
    assert pd.isna(result.loc[df.index[0], "Credit_Utilization"])


def test_v2_missing_values_are_preserved_for_pipeline_imputation():
    df = pd.read_csv("credit_card_customers_v2.csv")
    result = engineer_features(df, CONFIG)
    assert len(result) == 1100
    assert result["Credit_Utilization"].isna().any()


def test_demonstrating_automated_ci_test_failure():
    """Demonstrating automated CI block when unit tests fail."""
    assert False, "Deliberate test failure for CI/CD demonstration PR 3"
