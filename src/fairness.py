
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.config import load_config
from src.features import (
    engineer_features,
    split_features_target,
    validate_dataset,
)


def group_metrics(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "customers": int(len(y_true)),
        "actual_high_risk_rate": float(y_true.mean()),
        "predicted_high_risk_rate": float(y_pred.mean()),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": (
            float(fp / (fp + tn)) if (fp + tn) else None
        ),
        "false_negative_rate": (
            float(fn / (fn + tp)) if (fn + tp) else None
        ),
    }


def evaluate_fairness(
    data_path: str,
    model_path: str,
    config_path: str,
) -> dict:
    config = load_config(config_path)
    df = pd.read_csv(data_path)
    validate_dataset(df, config)

    # Preserve the original age and income groups for analysis.
    raw_groups = df[["Age", "Income_Bracket"]].copy()

    engineered = engineer_features(df, config)
    X, y = split_features_target(engineered, config)

    _, X_test, _, y_test, _, groups_test = train_test_split(
        X,
        y,
        raw_groups,
        test_size=config.data.test_size,
        random_state=config.random_seed,
        stratify=y,
    )

    model = joblib.load(model_path)
    predictions = pd.Series(
        model.predict(X_test), index=X_test.index
    )

    results = {
        "dataset": data_path,
        "model": model_path,
        "evaluation": (
            "Held-out test split; descriptive group metrics, "
            "not proof of fairness"
        ),
        "groups": {},
    }

    age_bands = pd.cut(
        groups_test["Age"],
        bins=[-float("inf"), 29, 44, 59, float("inf")],
        labels=["18-29", "30-44", "45-59", "60+"],
    )

    group_definitions = {
        "age_group": age_bands,
        "income_bracket": groups_test["Income_Bracket"],
    }

    for dimension, group_values in group_definitions.items():
        results["groups"][dimension] = {}

        for group_name in sorted(
            group_values.dropna().astype(str).unique()
        ):
            mask = group_values.astype(str) == group_name

            results["groups"][dimension][group_name] = (
                group_metrics(
                    y_test.loc[mask],
                    predictions.loc[mask],
                )
            )

    # Calculate the max-minus-min gap for selected metrics.
    results["group_gaps"] = {}

    for dimension, groups in results["groups"].items():
        results["group_gaps"][dimension] = {}

        for metric in (
            "recall",
            "false_positive_rate",
            "false_negative_rate",
        ):
            values = [
                group[metric]
                for group in groups.values()
                if group[metric] is not None
            ]

            results["group_gaps"][dimension][metric] = (
                max(values) - min(values) if len(values) >= 2 else None
            )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate model performance across customer groups."
    )
    parser.add_argument(
        "--config", default="configs/base.yaml"
    )
    parser.add_argument(
        "--data", default="credit_card_customers_v1.csv"
    )
    parser.add_argument(
        "--model", default="artifacts/best_model.joblib"
    )
    parser.add_argument(
        "--output", default="artifacts/fairness_metrics.json"
    )
    args = parser.parse_args()

    report = evaluate_fairness(
        args.data, args.model, args.config
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
