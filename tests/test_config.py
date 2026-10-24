import pytest
import yaml
from pydantic import ValidationError

from src.config import Config, load_config


@pytest.mark.parametrize("path", ["configs/base.yaml", "configs/ci.yaml"])
def test_load_existing_configs(path):
    config = load_config(path)
    assert isinstance(config, Config)
    assert config.data.target_column == "High_Risk_Customer"
    assert config.model.candidates


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("configs/does_not_exist.yaml")


def test_empty_yaml(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty YAML mapping"):
        load_config(path)


def test_malformed_yaml(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("data: [unclosed", encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        load_config(path)


def test_invalid_test_size_is_rejected():
    raw = {
        "data": {
            "path": "data.csv",
            "target_column": "target",
            "id_column": "id",
            "test_size": 1.5,
            "versions": {"v1": "data.csv"},
        },
        "random_seed": 42,
        "random_seeds": [42],
        "features": {
            "utilization_threshold": 0.8,
            "late_payment_threshold": 5,
        },
        "model": {
            "baseline": "LogisticRegression",
            "candidates": [{"name": "RandomForest", "params": {"n_estimators": [10]}}],
            "cv_folds": 2,
        },
        "tracking": {"experiment_name": "test", "tracking_uri": "mlruns"},
        "quality_gate": {
            "metric": "f1",
            "threshold": 0.6,
            "baseline_file": "baseline.json",
        },
    }
    with pytest.raises(ValidationError):
        Config.model_validate(raw)


def test_unknown_field_is_rejected(tmp_path):
    path = tmp_path / "unknown.yaml"
    path.write_text(
        "unexpected: true\n", encoding="utf-8"
    )
    with pytest.raises(ValidationError):
        load_config(path)
