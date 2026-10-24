import json
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.linear_model import LogisticRegression
import yaml

from src.config import load_config
from src.train import build_preprocessor, create_pipeline, calculate_metrics, train_and_evaluate


def test_preprocessor_handles_missing_numeric_and_unknown_categories():
    X_train = pd.DataFrame({
        "Age": [25, None, 40, 35],
        "Income_Bracket": ["Low", "Medium", "High", "Medium"],
    })
    X_test = pd.DataFrame({
        "Age": [None, 50],
        "Income_Bracket": ["Very High", "Low"],
    })

    preprocessor = build_preprocessor(X_train)
    transformed_train = preprocessor.fit_transform(X_train)
    transformed_test = preprocessor.transform(X_test)

    assert transformed_train.shape[0] == 4
    assert transformed_test.shape[0] == 2
    assert transformed_test.shape[1] == transformed_train.shape[1]


def test_pipeline_can_fit_and_produce_metrics():
    X = pd.DataFrame({
        "Age": [22, 25, 30, 35, 40, 45, 50, 55],
        "Income_Bracket": ["Low", "Low", "Medium", "Medium", "High", "High", "Low", "High"],
    })
    y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])

    model = create_pipeline(X, LogisticRegression(max_iter=1000))
    model.fit(X, y)
    metrics = calculate_metrics(model, X, y)

    assert {"f1", "precision", "recall", "accuracy", "roc_auc"} <= set(metrics)
    assert all(0 <= value <= 1 for value in metrics.values())


def test_mlflow_model_registration_and_reload(tmp_path):
    """Verify that models can be logged in MLflow format, registered, loaded back, and lineage traced."""
    db_path = tmp_path / "test_mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("test-registry")

    X = pd.DataFrame({"Age": [25, 45, 35, 50], "Income_Bracket": ["Low", "High", "Medium", "High"]})
    y = pd.Series([0, 1, 0, 1])
    pipeline = create_pipeline(X, LogisticRegression(max_iter=500))
    pipeline.fit(X, y)

    # 1. Gate passed: registers as CreditRiskModel
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            serialization_format="cloudpickle",
            registered_model_name="CreditRiskModel",
        )
        run_id = run.info.run_id

    client = MlflowClient()
    registered = client.search_registered_models()
    assert len(registered) == 1
    assert registered[0].name == "CreditRiskModel"

    version = client.get_model_version("CreditRiskModel", "1")
    assert version.run_id == run_id

    # Verify reloading from registry and producing predictions
    loaded_model = mlflow.sklearn.load_model("models:/CreditRiskModel/1")
    predictions = loaded_model.predict(X)
    assert len(predictions) == len(X)

    # 2. Production workflow: train_and_evaluate does NOT register a model when quality gate fails
    cfg_dict = yaml.safe_load(open("configs/ci.yaml", "r", encoding="utf-8"))
    cfg_dict["tracking"]["tracking_uri"] = f"sqlite:///{db_path}"
    cfg_dict["tracking"]["experiment_name"] = "test-registry"
    cfg_dict["quality_gate"]["threshold"] = 0.99  # Guaranteed quality gate failure
    cfg_dict["model"]["candidates"] = [{"name": "RandomForest", "params": {"n_estimators": [2], "max_depth": [2]}}]
    cfg_dict["model"]["cv_folds"] = 2
    cfg_file = tmp_path / "ci_fail.yaml"
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.dump(cfg_dict, f)

    df_small = pd.read_csv("credit_card_customers_v1.csv").head(60)
    small_csv = tmp_path / "small.csv"
    df_small.to_csv(small_csv, index=False)

    config = load_config(cfg_file)
    out_dir = tmp_path / "artifacts_fail"
    report = train_and_evaluate(
        config=config,
        data_path=small_csv,
        output_dir=out_dir,
        use_mlflow=True,
    )

    assert report["quality_gate_passed"] is False
    assert report["registered_model_name"] is None
    all_versions = client.search_model_versions("name='CreditRiskModel'")
    assert len(all_versions) == 1  # Still exactly 1 version registered; gate failure blocked registration

    saved_metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert saved_metrics["quality_gate_passed"] is False
    assert saved_metrics["registered_model_name"] is None
