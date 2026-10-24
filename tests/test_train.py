import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.config import load_config
from src.train import build_preprocessor, create_pipeline, calculate_metrics


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
