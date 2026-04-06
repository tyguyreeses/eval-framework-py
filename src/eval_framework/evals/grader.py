from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _exact_match_score(prediction: Any, expected: Any) -> float:
    return 1.0 if prediction == expected else 0.0


def _label_match_score(prediction: Any, expected: Any) -> float:
    if not isinstance(prediction, dict) or not isinstance(expected, dict):
        return 0.0
    return 1.0 if prediction.get("label") == expected.get("label") else 0.0


SCORER_REGISTRY: dict[str, Callable[[Any, Any], float]] = {
    "exact_match": _exact_match_score,
    "label_match": _label_match_score,
}


def list_available_scorers() -> list[str]:
    return sorted(SCORER_REGISTRY.keys())


def resolve_scorer(scorer_name: str) -> Callable[[Any, Any], float]:
    try:
        return SCORER_REGISTRY[scorer_name]
    except KeyError as exc:
        available = ", ".join(list_available_scorers())
        raise ValueError(f"Unknown scorer `{scorer_name}`. Available scorers: {available}") from exc


class Grader:
    def __init__(self, scorer_name: str = "exact_match") -> None:
        self.scorer_name = scorer_name

    def grade(
        self,
        prediction: dict[str, Any],
        expected: dict[str, Any],
        judge_strategy: dict[str, object],
    ) -> dict[str, object]:
        scorer = resolve_scorer(self.scorer_name)
        score = float(scorer(prediction, expected))
        return {
            "score": score,
            "judge_provider": judge_strategy.get("provider"),
            "judge_model": judge_strategy.get("model"),
            "scorers": {
                self.scorer_name: {
                    "score": score,
                    "method": self.scorer_name,
                }
            },
        }
