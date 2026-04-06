from __future__ import annotations

from importlib import import_module
from pathlib import Path
import re

import pytest


def _load_module(name: str):
    try:
        return import_module(name)
    except ModuleNotFoundError as exc:  # pragma: no cover - assertion path
        pytest.fail(f"Expected module `{name}` to exist for eval framework v1 contract: {exc}")


def test_shared_eval_envelope_accepts_multiple_task_schemas():
    envelope_module = _load_module("eval_framework.evals.envelope")
    validate_shared_envelope = getattr(envelope_module, "validate_shared_envelope", None)
    assert callable(validate_shared_envelope), "Expected validate_shared_envelope(payload) in eval envelope module"

    task_a_payload = {
        "run": {
            "run_id": "run-a",
            "dataset_path": "tmp/task_a.json",
            "judge": {"provider": "openai", "model": "gpt-5-mini"},
            "gate": {"mode": "soft", "threshold": 0.8},
        },
        "task": {"task_name": "classification_a"},
        "examples": [
            {
                "example_id": "ex-1",
                "input": {"text": "candidate summary"},
                "expected": {"label": "strong_hire"},
                "prediction": {"label": "strong_hire"},
                "scorers": {"exact_match": {"score": 1.0}},
            }
        ],
        "summary": {"mean_score": 1.0, "num_examples": 1},
    }
    task_b_payload = {
        "run": {
            "run_id": "run-b",
            "dataset_path": "tmp/task_b.json",
            "judge": {"provider": "openai", "model": "gpt-5-mini"},
            "gate": {"mode": "soft", "threshold": 0.8},
        },
        "task": {"task_name": "json_extraction_b"},
        "examples": [
            {
                "example_id": "ex-2",
                "input": {"job_posting": {"title": "ML Engineer", "location": "Remote"}},
                "expected": {"fields": {"title": "ML Engineer", "location": "Remote"}},
                "prediction": {"fields": {"title": "ML Engineer", "location": "Remote"}},
                "scorers": {"field_f1": {"score": 1.0}},
            }
        ],
        "summary": {"mean_score": 1.0, "num_examples": 1},
    }

    validated_a = validate_shared_envelope(task_a_payload)
    validated_b = validate_shared_envelope(task_b_payload)

    assert set(validated_a.keys()) == set(validated_b.keys()) == {"run", "task", "examples", "summary"}
    assert validated_a["examples"][0]["scorers"]
    assert validated_b["examples"][0]["scorers"]


def test_judge_strategy_defaults_to_openai_gpt_5_mini():
    judge_module = _load_module("eval_framework.evals.judge")
    resolve_judge_strategy = getattr(judge_module, "resolve_judge_strategy", None)
    assert callable(resolve_judge_strategy), "Expected resolve_judge_strategy(...) in eval judge module"

    judge = resolve_judge_strategy()
    assert judge["provider"] == "openai"
    assert judge["model"] == "gpt-5-mini"
    assert judge["deterministic_only"] is False


def test_judge_strategy_uses_deterministic_fallback_when_requested_or_openai_unavailable():
    judge_module = _load_module("eval_framework.evals.judge")
    resolve_judge_strategy = getattr(judge_module, "resolve_judge_strategy", None)
    assert callable(resolve_judge_strategy)

    explicit = resolve_judge_strategy(deterministic_only=True, openai_available=True)
    assert explicit["provider"] == "deterministic"
    assert explicit["deterministic_only"] is True

    unavailable = resolve_judge_strategy(deterministic_only=False, openai_available=False)
    assert unavailable["provider"] == "deterministic"
    assert unavailable["deterministic_only"] is True
    assert unavailable.get("fallback_reason")


def test_soft_gate_is_default_and_hard_gate_is_opt_in():
    gates_module = _load_module("eval_framework.evals.gates")
    evaluate_gate = getattr(gates_module, "evaluate_gate", None)
    assert callable(evaluate_gate), "Expected evaluate_gate(score, threshold, mode='soft') in eval gates module"

    soft_default = evaluate_gate(score=0.30, threshold=0.80)
    assert soft_default["mode"] == "soft"
    assert soft_default["passed"] is False
    assert soft_default["blocked"] is False

    hard_opt_in = evaluate_gate(score=0.30, threshold=0.80, mode="hard")
    assert hard_opt_in["mode"] == "hard"
    assert hard_opt_in["passed"] is False
    assert hard_opt_in["blocked"] is True


def test_cli_exposes_required_eval_workflow_surfaces():
    cli_module = _load_module("eval_framework.cli")
    build_parser = getattr(cli_module, "build_parser", None)
    assert callable(build_parser), "Expected build_parser() in eval CLI module"

    parser = build_parser()
    subparser_action = next(
        (action for action in parser._actions if action.__class__.__name__ == "_SubParsersAction"),
        None,
    )
    assert subparser_action is not None, "Expected top-level subcommands in eval CLI"
    top_level = set(subparser_action.choices.keys())
    assert {"run", "compare", "dataset"}.issubset(top_level)

    dataset_parser = subparser_action.choices["dataset"]
    dataset_subparser_action = next(
        (action for action in dataset_parser._actions if action.__class__.__name__ == "_SubParsersAction"),
        None,
    )
    assert dataset_subparser_action is not None
    assert "add" in dataset_subparser_action.choices


def test_sample_eval_datasets_are_redacted_and_safe_for_repo():
    datasets_dir = Path(__file__).resolve().parents[1] / "data" / "evals"
    assert datasets_dir.exists(), "Expected eval sample dataset directory at backend/data/evals"

    json_files = sorted(datasets_dir.rglob("*.json"))
    assert json_files, "Expected at least one eval sample dataset JSON file under backend/data/evals"

    pii_patterns = [
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
        re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),  # phone
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),  # email
        re.compile(r"\b\d{1,5}\s+[A-Za-z0-9.\s]+\s(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)\b", re.I),
    ]

    redaction_marker_seen = False
    for path in json_files:
        content = path.read_text(encoding="utf-8")
        lowered = content.lower()
        if "redacted" in lowered:
            redaction_marker_seen = True
        for pattern in pii_patterns:
            assert not pattern.search(content), f"Potential unredacted PII detected in {path}"

    assert redaction_marker_seen, "Expected explicit redaction markers in sample eval dataset content"
