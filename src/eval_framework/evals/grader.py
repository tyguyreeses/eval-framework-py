from __future__ import annotations

from typing import Any


def _exact_match_score(prediction: Any, expected: Any) -> float:
    return 1.0 if prediction == expected else 0.0


class Grader:
    def __init__(self, scorer_name: str = "exact_match") -> None:
        self.scorer_name = scorer_name

    def grade(
        self,
        prediction: dict[str, Any],
        expected: dict[str, Any],
        judge_strategy: dict[str, object],
    ) -> dict[str, object]:
        score = _exact_match_score(prediction, expected)
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
