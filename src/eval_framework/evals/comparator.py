from __future__ import annotations

from pathlib import Path
from typing import Any

from .logging import load_run_log


def _extract_score(example: dict[str, Any]) -> float:
    scorers = example.get("scorers", {})
    if not isinstance(scorers, dict) or not scorers:
        return 0.0

    values: list[float] = []
    for scorer_data in scorers.values():
        if isinstance(scorer_data, dict) and "score" in scorer_data:
            values.append(float(scorer_data["score"]))
    if not values:
        return 0.0
    return sum(values) / len(values)


def _normalize_run(run: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(run, dict):
        return run
    return load_run_log(run)


def compare_runs(left_run: dict[str, Any] | str | Path, right_run: dict[str, Any] | str | Path) -> dict[str, Any]:
    left = _normalize_run(left_run)
    right = _normalize_run(right_run)

    left_examples = {str(example["example_id"]): example for example in left.get("examples", [])}
    right_examples = {str(example["example_id"]): example for example in right.get("examples", [])}
    common_ids = sorted(set(left_examples.keys()) & set(right_examples.keys()))

    example_deltas: list[dict[str, Any]] = []
    improved = 0
    regressed = 0

    for example_id in common_ids:
        left_score = _extract_score(left_examples[example_id])
        right_score = _extract_score(right_examples[example_id])
        delta = round(right_score - left_score, 6)
        if delta > 0:
            improved += 1
        elif delta < 0:
            regressed += 1

        example_deltas.append(
            {
                "example_id": example_id,
                "left_score": left_score,
                "right_score": right_score,
                "delta": delta,
            }
        )

    left_mean = float(left.get("summary", {}).get("mean_score", 0.0))
    right_mean = float(right.get("summary", {}).get("mean_score", 0.0))
    return {
        "left_run_id": left.get("run", {}).get("run_id"),
        "right_run_id": right.get("run", {}).get("run_id"),
        "shared_examples": len(common_ids),
        "summary_delta": {
            "left_mean_score": left_mean,
            "right_mean_score": right_mean,
            "mean_score_delta": round(right_mean - left_mean, 6),
            "improved_examples": improved,
            "regressed_examples": regressed,
        },
        "example_deltas": example_deltas,
    }
