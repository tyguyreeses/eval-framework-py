from __future__ import annotations


def evaluate_gate(score: float, threshold: float, mode: str = "soft") -> dict[str, object]:
    """Evaluate gate state; hard mode blocks failed runs, soft mode only records signal."""
    if mode not in {"soft", "hard"}:
        raise ValueError("mode must be either `soft` or `hard`")

    passed = score >= threshold
    return {
        "mode": mode,
        "score": score,
        "threshold": threshold,
        "passed": passed,
        "blocked": (not passed and mode == "hard"),
    }
