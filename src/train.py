
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from src.config import Config, load_config
from src.features import engineer_features, split_features_target, validate_dataset


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="most_frequent", keep_empty_features=True
        )),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
    ], remainder="drop")


def create_pipeline(X: pd.DataFrame, estimator) -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor(X)),
        ("model", estimator),
    ])


def get_estimator(name: str, seed: int):
    if name == "LogisticRegression":
        return LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed
        )
    if name == "RandomForest":
        return RandomForestClassifier(
            class_weight="balanced", random_state=seed, n_jobs=1
        )
    if name == "GradientBoosting":
        return GradientBoostingClassifier(random_state=seed)
    raise ValueError(f"Unsupported model configured: {name}")


def calculate_metrics(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }


def load_baseline_metrics(baseline_path: str | Path, metric_name: str) -> dict:
    """Load reference baseline metrics from a stored JSON file."""
    path = Path(baseline_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Stored baseline file not found: {path}. "
            "A valid baseline.json is required for CI quality gate evaluation."
        )
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in baseline file {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"Baseline file {path} must contain a JSON object mapping, got {type(data).__name__}."
        )

    metrics = data.get("baseline_metrics")
    if metrics is None:
        metrics = data.get("metrics")
    if metrics is None:
        metrics = data

    if not isinstance(metrics, dict):
        raise ValueError(
            f"Metrics section in baseline file {path} must be a dictionary, got {type(metrics).__name__}."
        )

    if metric_name not in metrics:
        raise KeyError(
            f"Baseline file {path} missing required metric '{metric_name}'"
        )

    raw_val = metrics[metric_name]
    try:
        metric_val = float(raw_val)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Baseline metric '{metric_name}' in {path} must be a numeric value, got {raw_val!r}."
        ) from exc

    if not math.isfinite(metric_val):
        raise ValueError(
            f"Baseline metric '{metric_name}' in {path} must be a finite number, got {metric_val}."
        )

    if metric_name in {"f1", "precision", "recall", "accuracy", "roc_auc"}:
        if not (0.0 <= metric_val <= 1.0):
            raise ValueError(
                f"Baseline metric '{metric_name}' in {path} must be between 0.0 and 1.0 inclusive, got {metric_val}."
            )

    return {
        "model_name": data.get("model_name", "unknown") if isinstance(data, dict) else "unknown",
        "metric_value": metric_val,
        "all_metrics": metrics,
    }


def evaluate_quality_gate(
    candidate_metrics: dict,
    metric_name: str,
    threshold: float,
    baseline_path: str | Path | None = None,
) -> dict:
    """Evaluate candidate model against both an absolute floor and a stored baseline."""
    candidate_value = float(candidate_metrics.get(metric_name, -1.0))
    passed_threshold = bool(candidate_value >= threshold)

    if baseline_path:
        baseline_info = load_baseline_metrics(baseline_path, metric_name)
        baseline_value = baseline_info["metric_value"]
        delta = candidate_value - baseline_value
        passed_baseline = bool(candidate_value >= baseline_value)
        baseline_model = baseline_info["model_name"]
    else:
        baseline_value = None
        delta = None
        passed_baseline = True
        baseline_model = None

    passed = bool(passed_threshold and passed_baseline)
    return {
        "metric": metric_name,
        "threshold": threshold,
        "candidate_value": candidate_value,
        "baseline_value": baseline_value,
        "delta": delta,
        "passed_threshold": passed_threshold,
        "passed_baseline": passed_baseline,
        "passed": passed,
        "baseline_model": baseline_model,
    }


def fit_with_imbalance_weights(model, X, y):
    """Fit a model, applying balanced sample weights where supported."""
    if isinstance(model.named_steps["model"], GradientBoostingClassifier):
        weights = compute_sample_weight(class_weight="balanced", y=y)
        model.fit(X, y, model__sample_weight=weights)
    else:
        model.fit(X, y)
    return model


def train_and_evaluate(
    config: Config,
    data_path: str | Path | None = None,
    output_dir: str | Path = "artifacts",
    use_mlflow: bool = True,
) -> dict:
    data_path = Path(data_path or config.data.path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    validate_dataset(df, config)
    df = engineer_features(df, config)
    X, y = split_features_target(df, config)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.data.test_size,
        random_state=config.random_seed,
        stratify=y,
    )

    if use_mlflow:
        mlflow.set_tracking_uri(config.tracking.tracking_uri)
        mlflow.set_experiment(config.tracking.experiment_name)

    cv = StratifiedKFold(
        n_splits=config.model.cv_folds,
        shuffle=True,
        random_state=config.random_seed,
    )

    results = {}
    fitted_models = {}
    baseline_name = config.model.baseline

    # The baseline is included in cross-validation-based model selection too.
    baseline_pipeline = create_pipeline(
        X_train, get_estimator(baseline_name, config.random_seed)
    )
    baseline_search = GridSearchCV(
        estimator=baseline_pipeline,
        param_grid={},
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )

    with (
        mlflow.start_run(run_name=f"baseline-{baseline_name}")
        if use_mlflow else _no_mlflow_run()
    ):
        baseline_search.fit(X_train, y_train)
        fitted_models[baseline_name] = baseline_search.best_estimator_
        results[baseline_name] = {
            "cv_best_f1": float(baseline_search.best_score_),
            "best_params": {},
            "role": "baseline",
        }
        if use_mlflow:
            mlflow.log_param("model_name", baseline_name)
            mlflow.log_param("role", "baseline")
            mlflow.log_metric("cv_best_f1", float(baseline_search.best_score_))

    for candidate in config.model.candidates:
        name = candidate.name
        pipeline = create_pipeline(
            X_train, get_estimator(name, config.random_seed)
        )
        param_grid = {
            f"model__{parameter}": values
            for parameter, values in candidate.params.items()
        }

        # Gradient Boosting gets balanced sample weights within each CV fit.
        # This keeps weights calculated only from the corresponding training fold.
        fit_params = {}
        if name == "GradientBoosting":
            # GridSearchCV cannot automatically calculate per-fold weights
            # for this custom approach, so use a weighted estimator wrapper
            # through a dedicated manual CV loop below.
            from sklearn.model_selection import ParameterGrid, cross_val_score

            best_score = -1.0
            best_params = {}
            best_model = None
            for params in ParameterGrid(param_grid):
                fold_scores = []
                for train_idx, val_idx in cv.split(X_train, y_train):
                    fold_model = clone(pipeline).set_params(**params)
                    X_fold, y_fold = X_train.iloc[train_idx], y_train.iloc[train_idx]
                    X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
                    fit_with_imbalance_weights(fold_model, X_fold, y_fold)
                    fold_scores.append(
                        f1_score(y_val, fold_model.predict(X_val), zero_division=0)
                    )
                score = float(np.mean(fold_scores))
                if score > best_score:
                    best_score, best_params = score, params
            best_model = clone(pipeline).set_params(**best_params)
            fit_with_imbalance_weights(best_model, X_train, y_train)
        else:
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring="f1",
                cv=cv,
                n_jobs=-1,
                refit=True,
                error_score="raise",
            )
            search.fit(X_train, y_train)
            best_score = float(search.best_score_)
            best_params = search.best_params_
            best_model = search.best_estimator_

        clean_params = {
            key.replace("model__", ""): value
            for key, value in best_params.items()
        }
        fitted_models[name] = best_model
        results[name] = {
            "cv_best_f1": float(best_score),
            "best_params": clean_params,
            "role": "candidate",
        }

        if use_mlflow:
            with mlflow.start_run(run_name=f"candidate-{name}"):
                mlflow.log_param("model_name", name)
                mlflow.log_params({f"best_{k}": v for k, v in clean_params.items()})
                mlflow.log_metric("cv_best_f1", float(best_score))

    # Evaluate baseline model on the untouched holdout test set.
    baseline_model = fitted_models[baseline_name]
    baseline_metrics = calculate_metrics(baseline_model, X_test, y_test)
    results[baseline_name]["test_metrics"] = baseline_metrics

    # Select using training-only cross-validation scores.
    best_name = max(results, key=lambda name: results[name]["cv_best_f1"])
    best_model = fitted_models[best_name]

    # Evaluate the selected model once on the untouched holdout test set.
    final_metrics = calculate_metrics(best_model, X_test, y_test)
    results[best_name]["test_metrics"] = final_metrics

    # Evaluate quality gate against stored baseline and absolute threshold.
    baseline_file = config.quality_gate.baseline_file
    gate_result = evaluate_quality_gate(
        candidate_metrics=final_metrics,
        metric_name=config.quality_gate.metric,
        threshold=config.quality_gate.threshold,
        baseline_path=baseline_file,
    )

    report = {
        "data_path": str(data_path),
        "rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "target_positive_rate": float(y.mean()),
        "random_seed": config.random_seed,
        "baseline_model": baseline_name,
        "best_model": best_name,
        "selection_metric": "cross_validation_f1",
        "quality_gate_metric": config.quality_gate.metric,
        "quality_gate_threshold": config.quality_gate.threshold,
        "quality_gate_passed": gate_result["passed"],
        "quality_gate": gate_result,
        "baseline_f1": gate_result["baseline_value"],
        "candidate_f1": gate_result["candidate_value"],
        "delta_f1": gate_result["delta"],
        "passed_threshold": gate_result["passed_threshold"],
        "passed_baseline": gate_result["passed_baseline"],
        "baseline_test_metrics": baseline_metrics,
        "final_test_metrics": final_metrics,
        "models": results,
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    joblib.dump(best_model, output_dir / "best_model.joblib")

    if use_mlflow:
        with mlflow.start_run(run_name="final-selected-model"):
            mlflow.log_param("selected_model", best_name)
            mlflow.log_metrics({
                f"test_{key}": value for key, value in final_metrics.items()
            })
            mlflow.log_metric(
                "quality_gate_passed", int(report["quality_gate_passed"])
            )
            if gate_result["baseline_value"] is not None:
                mlflow.log_metric("baseline_f1", gate_result["baseline_value"])
                mlflow.log_metric("candidate_delta_f1", gate_result["delta"])
            mlflow.log_artifact(str(output_dir / "metrics.json"))
            mlflow.log_artifact(str(output_dir / "best_model.joblib"))

    return report


class _no_mlflow_run:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate credit-risk models."
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--data", default=None)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    report = train_and_evaluate(
        config=config,
        data_path=args.data,
        output_dir=args.output_dir,
        use_mlflow=not args.no_mlflow,
    )

    print(json.dumps(report, indent=2))
    gate = report["quality_gate"]
    if report["quality_gate_passed"]:
        print(
            f"\n[QUALITY GATE PASSED] Candidate {gate['metric']}={gate['candidate_value']:.4f} "
            f">= threshold {gate['threshold']:.4f} and >= baseline {gate['baseline_value']:.4f} "
            f"(delta={gate['delta']:+.4f})"
        )
    else:
        print(
            f"\n[QUALITY GATE BLOCKED] Candidate {gate['metric']}={gate['candidate_value']:.4f} "
            f"failed quality gate: passed_threshold={gate['passed_threshold']} (floor={gate['threshold']:.4f}), "
            f"passed_baseline={gate['passed_baseline']} (baseline={gate['baseline_value']:.4f}, delta={gate['delta']:+.4f})"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
