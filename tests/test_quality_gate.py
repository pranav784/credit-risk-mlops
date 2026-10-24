import json
from pathlib import Path
import pytest

from src.train import evaluate_quality_gate, load_baseline_metrics


@pytest.fixture
def sample_baseline_file(tmp_path: Path) -> Path:
    baseline_data = {
        "model_name": "LogisticRegression",
        "dataset_version": "v1",
        "metric": "f1",
        "baseline_metrics": {
            "f1": 0.6923076923,
            "precision": 0.5625,
            "recall": 0.9,
            "accuracy": 0.85,
            "roc_auc": 0.9233,
        },
    }
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(baseline_data), encoding="utf-8")
    return path


def test_candidate_passes_both_threshold_and_baseline(sample_baseline_file: Path):
    """Candidate beats both the 0.60 floor and the baseline F1 (0.6923)."""
    candidate_metrics = {"f1": 0.7250, "accuracy": 0.88}
    result = evaluate_quality_gate(
        candidate_metrics=candidate_metrics,
        metric_name="f1",
        threshold=0.60,
        baseline_path=sample_baseline_file,
    )

    assert result["passed"] is True
    assert result["passed_threshold"] is True
    assert result["passed_baseline"] is True
    assert result["candidate_value"] == 0.7250
    assert result["baseline_value"] == pytest.approx(0.6923076923)
    assert result["delta"] > 0


def test_candidate_meets_threshold_but_underperforms_baseline(sample_baseline_file: Path):
    """Candidate F1 >= 0.60 floor, but F1 < baseline (0.6923) -> Quality Gate must FAIL."""
    candidate_metrics = {"f1": 0.6500, "accuracy": 0.85}
    result = evaluate_quality_gate(
        candidate_metrics=candidate_metrics,
        metric_name="f1",
        threshold=0.60,
        baseline_path=sample_baseline_file,
    )

    assert result["passed"] is False
    assert result["passed_threshold"] is True
    assert result["passed_baseline"] is False
    assert result["candidate_value"] == 0.6500
    assert result["delta"] < 0


def test_candidate_below_absolute_threshold(sample_baseline_file: Path):
    """Candidate F1 < 0.60 floor -> Quality Gate must FAIL."""
    candidate_metrics = {"f1": 0.5500, "accuracy": 0.80}
    result = evaluate_quality_gate(
        candidate_metrics=candidate_metrics,
        metric_name="f1",
        threshold=0.60,
        baseline_path=sample_baseline_file,
    )

    assert result["passed"] is False
    assert result["passed_threshold"] is False
    assert result["candidate_value"] == 0.5500


def test_missing_baseline_file_raises_error(tmp_path: Path):
    """Missing baseline file must raise FileNotFoundError rather than bypassing check."""
    missing_path = tmp_path / "non_existent_baseline.json"
    with pytest.raises(FileNotFoundError, match="Stored baseline file not found"):
        evaluate_quality_gate(
            candidate_metrics={"f1": 0.75},
            metric_name="f1",
            threshold=0.60,
            baseline_path=missing_path,
        )


def test_invalid_baseline_format_raises_error(tmp_path: Path):
    """Invalid JSON in baseline file must raise ValueError."""
    bad_path = tmp_path / "invalid.json"
    bad_path.write_text("not a valid json {", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        evaluate_quality_gate(
            candidate_metrics={"f1": 0.75},
            metric_name="f1",
            threshold=0.60,
            baseline_path=bad_path,
        )


def test_baseline_not_a_json_mapping_raises_error(tmp_path: Path):
    """JSON top-level array instead of mapping must raise ValueError."""
    bad_path = tmp_path / "array.json"
    bad_path.write_text('["not", "a", "dict"]', encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object mapping"):
        load_baseline_metrics(bad_path, "f1")


def test_baseline_metrics_section_not_dict_raises_error(tmp_path: Path):
    """baseline_metrics section that is not a dictionary must raise ValueError."""
    bad_path = tmp_path / "metrics_not_dict.json"
    bad_path.write_text(json.dumps({"baseline_metrics": "invalid_string"}), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a dictionary"):
        load_baseline_metrics(bad_path, "f1")


def test_baseline_missing_metric_key_raises_key_error(tmp_path: Path):
    """Missing metric key in baseline must raise KeyError."""
    bad_path = tmp_path / "missing_f1.json"
    bad_path.write_text(json.dumps({"baseline_metrics": {"accuracy": 0.85}}), encoding="utf-8")
    with pytest.raises(KeyError, match="missing required metric 'f1'"):
        load_baseline_metrics(bad_path, "f1")


@pytest.mark.parametrize("bad_f1, match_msg", [
    ("non_numeric", "must be a numeric value"),
    (float("nan"), "must be a finite number"),
    (float("inf"), "must be a finite number"),
    (-0.25, "must be between 0.0 and 1.0"),
    (1.45, "must be between 0.0 and 1.0"),
])
def test_invalid_f1_values_raise_error(tmp_path: Path, bad_f1, match_msg):
    """Non-numeric, non-finite, and out-of-range F1 values must be rejected."""
    bad_path = tmp_path / "bad_f1.json"
    bad_path.write_text(json.dumps({"baseline_metrics": {"f1": bad_f1}}), encoding="utf-8")
    with pytest.raises(ValueError, match=match_msg):
        load_baseline_metrics(bad_path, "f1")

