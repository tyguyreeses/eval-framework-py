from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .dataset import Dataset, load_dataset
from .envelope import validate_shared_envelope
from .gates import evaluate_gate
from .grader import Grader
from .judge import resolve_judge_strategy
from .metrics import summarize_scores
from .task import TaskCallable, invoke_task, resolve_task


class EvalRunner:
    def run(
        self,
        dataset: str | Dataset,
        task: str | TaskCallable = "oracle_expected",
        grader: Grader | None = None,
        *,
        run_id: str | None = None,
        provider: str = "openai",
        model: str = "gpt-5-mini",
        deterministic_only: bool = False,
        openai_available: bool = True,
        gate_mode: str = "soft",
        gate_threshold: float = 0.8,
    ) -> dict[str, Any]:
        dataset_obj = load_dataset(dataset) if isinstance(dataset, (str, Path)) else dataset
        dataset_obj.validate()
        task_callable = resolve_task(task)
        grader_impl = grader or Grader()
        judge_strategy = resolve_judge_strategy(
            provider=provider,
            model=model,
            deterministic_only=deterministic_only,
            openai_available=openai_available,
        )

        example_records: list[dict[str, Any]] = []
        scores: list[float] = []
        blocked_examples = 0

        for example in dataset_obj.examples:
            prediction = invoke_task(task_callable, example.get("input", {}), example)
            grade = grader_impl.grade(
                prediction=prediction,
                expected=example.get("expected", {}),
                judge_strategy=judge_strategy,
            )
            score = float(grade["score"])
            scores.append(score)

            gate = evaluate_gate(score=score, threshold=gate_threshold, mode=gate_mode)
            if bool(gate["blocked"]):
                blocked_examples += 1

            example_records.append(
                {
                    "example_id": str(example["example_id"]),
                    "input": example["input"],
                    "expected": example["expected"],
                    "prediction": prediction,
                    "scorers": grade["scorers"],
                    "gate": gate,
                }
            )

        run_identifier = run_id or f"run-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:8]}"
        summary = summarize_scores(scores)
        summary["blocked_examples"] = blocked_examples

        payload = {
            "run": {
                "run_id": run_identifier,
                "dataset_path": str(dataset_obj.path),
                "judge": {
                    "provider": str(judge_strategy["provider"]),
                    "model": judge_strategy.get("model"),
                },
                "gate": {"mode": gate_mode, "threshold": gate_threshold},
            },
            "task": {
                "task_name": task if isinstance(task, str) else getattr(task, "__name__", "callable_task"),
                "executed_at": datetime.now(UTC).isoformat(),
            },
            "examples": example_records,
            "summary": summary,
        }
        return validate_shared_envelope(payload)


def run_evaluation(
    dataset: str | Dataset,
    task: str | TaskCallable = "oracle_expected",
    **kwargs: Any,
) -> dict[str, Any]:
    runner = EvalRunner()
    return runner.run(dataset=dataset, task=task, **kwargs)
