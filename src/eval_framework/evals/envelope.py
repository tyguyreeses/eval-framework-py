from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RunGate(BaseModel):
    mode: str = Field(min_length=1)
    threshold: float = Field(ge=0.0, le=1.0)


class RunJudge(BaseModel):
    provider: str = Field(min_length=1)
    model: str | None = None


class RunMetadata(BaseModel):
    run_id: str = Field(min_length=1)
    dataset_path: str = Field(min_length=1)
    judge: RunJudge
    gate: RunGate


class ExampleRecord(BaseModel):
    example_id: str = Field(min_length=1)
    input: dict[str, Any]
    expected: dict[str, Any]
    prediction: dict[str, Any]
    scorers: dict[str, dict[str, Any]] = Field(min_length=1)
    gate: dict[str, Any] | None = None


class SummaryRecord(BaseModel):
    mean_score: float
    num_examples: int = Field(ge=0)
    median_score: float | None = None
    min_score: float | None = None
    max_score: float | None = None
    std_score: float | None = None
    pass_rate: float | None = None
    passed_examples: int | None = None
    failed_examples: int | None = None
    blocked_examples: int | None = None


class SharedEnvelope(BaseModel):
    run: RunMetadata
    task: dict[str, Any] = Field(min_length=1)
    examples: list[ExampleRecord] = Field(min_length=1)
    summary: SummaryRecord


def validate_shared_envelope(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a run payload into the shared eval envelope contract."""
    envelope = SharedEnvelope.model_validate(payload)
    return envelope.model_dump()
