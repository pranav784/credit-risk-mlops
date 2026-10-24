from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class DataConfig(StrictModel):
    path: str
    target_column: str
    id_column: str
    test_size: float = Field(gt=0, lt=1)
    versions: dict[str, str]

    @field_validator("path", "target_column", "id_column")
    @classmethod
    def non_empty_string(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value


class FeaturesConfig(StrictModel):
    utilization_threshold: float = Field(gt=0, le=1)
    late_payment_threshold: int = Field(ge=0)


class ModelCandidate(StrictModel):
    name: str
    params: dict[str, list[Any]]

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("candidate name must not be empty")
        return value

    @field_validator("params")
    @classmethod
    def parameter_lists_not_empty(
        cls, value: dict[str, list[Any]]
    ) -> dict[str, list[Any]]:
        if not value:
            raise ValueError("candidate params must not be empty")
        if any(not values for values in value.values()):
            raise ValueError("parameter lists must not be empty")
        return value


class ModelConfig(StrictModel):
    baseline: str
    candidates: list[ModelCandidate]
    cv_folds: int = Field(ge=2)

    @field_validator("baseline")
    @classmethod
    def baseline_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("baseline must not be empty")
        return value

    @field_validator("candidates")
    @classmethod
    def candidates_not_empty(
        cls, value: list[ModelCandidate]
    ) -> list[ModelCandidate]:
        if not value:
            raise ValueError("at least one candidate model is required")
        return value


class TrackingConfig(StrictModel):
    experiment_name: str
    tracking_uri: str


class QualityGateConfig(StrictModel):
    metric: str
    threshold: float = Field(ge=0, le=1)
    baseline_file: str


class Config(StrictModel):
    data: DataConfig
    features: FeaturesConfig
    model: ModelConfig
    tracking: TrackingConfig
    quality_gate: QualityGateConfig
    random_seed: int
    random_seeds: list[int]

    @field_validator("random_seeds")
    @classmethod
    def seeds_not_empty(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("random_seeds must not be empty")
        return value


def load_config(config_path: str | Path) -> Config:
    path = Path(config_path)
    with path.open("r", encoding="utf-8-sig") as file:
        raw = yaml.safe_load(file)

    if not isinstance(raw, dict):
        raise ValueError(f"Configuration must be a non-empty YAML mapping: {path}")

    return Config.model_validate(raw)
