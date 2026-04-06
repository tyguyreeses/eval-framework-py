from __future__ import annotations

from statistics import mean, median, pstdev


def summarize_scores(scores: list[float]) -> dict[str, float | int]:
    if not scores:
        return {
            "mean_score": 0.0,
            "median_score": 0.0,
            "min_score": 0.0,
            "max_score": 0.0,
            "std_score": 0.0,
            "num_examples": 0,
            "pass_rate": 0.0,
            "passed_examples": 0,
            "failed_examples": 0,
        }

    passed = sum(1 for score in scores if score >= 0.5)
    count = len(scores)
    return {
        "mean_score": round(mean(scores), 6),
        "median_score": round(median(scores), 6),
        "min_score": round(min(scores), 6),
        "max_score": round(max(scores), 6),
        "std_score": round(pstdev(scores), 6),
        "num_examples": count,
        "pass_rate": round(passed / count, 6),
        "passed_examples": passed,
        "failed_examples": count - passed,
    }
