from __future__ import annotations


def resolve_judge_strategy(
    provider: str = "openai",
    model: str = "gpt-5-mini",
    deterministic_only: bool = False,
    openai_available: bool = True,
) -> dict[str, object]:
    """Resolve judge runtime strategy with OpenAI-first + deterministic fallback."""
    if deterministic_only:
        return {
            "provider": "deterministic",
            "model": None,
            "deterministic_only": True,
            "fallback_reason": "deterministic_only requested",
        }

    if provider == "openai" and openai_available:
        return {
            "provider": "openai",
            "model": model,
            "deterministic_only": False,
        }

    return {
        "provider": "deterministic",
        "model": None,
        "deterministic_only": True,
        "fallback_reason": "openai unavailable" if provider == "openai" else f"provider `{provider}` unsupported",
    }
