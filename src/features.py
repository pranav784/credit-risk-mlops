from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config


REQUIRED_COLUMNS = {
    "Customer_ID",
    "Age",
    "Income_Bracket",
    "Credit_Limit",
    "Current_Balance",
    "Monthly_Spend",
    "Num_Transactions_Monthly",
    "Payment_Delay_Days",
    "Tenure_Months",
    "High_Risk_Customer",
}


def validate_dataset(df: pd.DataFrame, config: Config) -> None:
    """Validate the raw credit-risk dataset before modelling."""
    if df.empty:
        raise ValueError("Dataset must not be empty.")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    if df["Customer_ID"].isna().any():
        raise ValueError("Customer_ID must not contain missing values.")

    if df["Customer_ID"].duplicated().any():
        raise ValueError("Customer_ID must be unique.")

    target = config.data.target_column
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' is missing.")

    if df[target].isna().any():
        raise ValueError("Target column must not contain missing values.")

    observed_targets = set(df[target].unique())
    if not observed_targets.issubset({0, 1}):
        raise ValueError(
            f"Target must be binary (0/1); found: {sorted(observed_targets)}"
        )


def engineer_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Add interpretable features without fitting anything on the data."""
    validate_dataset(df, config)
    result = df.copy()

    credit_limit = pd.to_numeric(result["Credit_Limit"], errors="coerce")
    balance = pd.to_numeric(result["Current_Balance"], errors="coerce")
    delay = pd.to_numeric(result["Payment_Delay_Days"], errors="coerce")
    monthly_spend = pd.to_numeric(result["Monthly_Spend"], errors="coerce")
    transactions = pd.to_numeric(
        result["Num_Transactions_Monthly"], errors="coerce"
    )
    tenure = pd.to_numeric(result["Tenure_Months"], errors="coerce")

    # A zero or missing denominator produces a missing ratio, not infinity.
    result["Credit_Utilization"] = (
        balance / credit_limit.replace(0, np.nan)
    )
    result["High_Utilization_Flag"] = np.where(
        result["Credit_Utilization"].isna(),
        np.nan,
        (
            result["Credit_Utilization"]
            > config.features.utilization_threshold
        ).astype(float),
    )

    result["Late_Payment_Flag"] = np.where(
        delay.isna(),
        np.nan,
        (delay > config.features.late_payment_threshold).astype(float),
    )

    result["Balance_to_Monthly_Spend"] = (
        balance / monthly_spend.replace(0, np.nan)
    )
    result["Transactions_per_Tenure_Month"] = (
        transactions / tenure.replace(0, np.nan)
    )

    return result


def split_features_target(
    df: pd.DataFrame, config: Config
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate predictors and target; never use customer ID as a predictor."""
    target = config.data.target_column
    X = df.drop(columns=[target, config.data.id_column])
    y = df[target].astype(int)
    return X, y
