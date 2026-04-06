from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .evals.comparator import compare_runs
from .evals.dataset import Dataset
from .evals.grader import Grader
from .evals.logging import RunLogger
from .evals.runner import run_evaluation


def _parse_json_arg(raw_json: str, arg_name: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for {arg_name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{arg_name} must decode to a JSON object")
    return payload


def _build_dataset_example(args: argparse.Namespace) -> dict[str, Any]:
    if args.example:
        return _parse_json_arg(args.example, "--example")
    if args.example_file:
        raw = Path(args.example_file).read_text(encoding="utf-8")
        return _parse_json_arg(raw, "--example-file")

    if args.example_id and args.input_json and args.expected_json:
        return {
            "example_id": args.example_id,
            "input": _parse_json_arg(args.input_json, "--input-json"),
            "expected": _parse_json_arg(args.expected_json, "--expected-json"),
        }
    raise ValueError(
        "dataset add requires either --example, --example-file, or all of --example-id, --input-json, --expected-json"
    )


def handle_dataset_add_command(args: argparse.Namespace) -> dict[str, Any]:
    dataset_path = Path(args.path)
    if dataset_path.exists():
        dataset = Dataset.load(dataset_path)
    else:
        dataset = Dataset(path=dataset_path, dataset_name=dataset_path.stem, examples=[])

    added = dataset.append(_build_dataset_example(args))
    saved_path = dataset.save()
    return {
        "command": "dataset.add",
        "dataset_path": str(saved_path),
        "total_examples": len(dataset.examples),
        "added_example_id": str(added["example_id"]),
    }


def handle_run_command(args: argparse.Namespace) -> dict[str, Any]:
    run = run_evaluation(
        dataset=str(args.dataset),
        task=args.task,
        grader=Grader(scorer_name=args.scorer),
        run_id=args.run_id,
        provider=args.provider,
        model=args.model,
        deterministic_only=args.deterministic_only,
        openai_available=not args.openai_unavailable,
        gate_mode=args.gate_mode,
        gate_threshold=args.threshold,
    )
    logger = RunLogger(base_dir=args.runs_dir)
    output_path = logger.save(run, path=args.output)
    return {
        "command": "run",
        "output_path": str(output_path),
        "run_id": run["run"]["run_id"],
        "summary": run["summary"],
    }


def handle_compare_command(args: argparse.Namespace) -> dict[str, Any]:
    result = compare_runs(args.left, args.right)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "command": "compare",
        "comparison": result,
        "output_path": str(args.output) if args.output else None,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for the eval framework workflow surfaces."""
    parser = argparse.ArgumentParser(prog="eval-framework")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run an evaluation dataset")
    run_parser.add_argument("--dataset", required=True)
    run_parser.add_argument("--task", default="oracle_expected")
    run_parser.add_argument("--run-id", required=False)
    run_parser.add_argument("--provider", default="openai")
    run_parser.add_argument("--model", default="gpt-5-mini")
    run_parser.add_argument("--scorer", default="exact_match")
    run_parser.add_argument("--deterministic-only", action="store_true")
    run_parser.add_argument("--openai-unavailable", action="store_true")
    run_parser.add_argument("--gate-mode", choices=("soft", "hard"), default="soft")
    run_parser.add_argument("--threshold", type=float, default=0.8)
    run_parser.add_argument("--runs-dir", required=False)
    run_parser.add_argument("--output", required=False)
    run_parser.set_defaults(handler=handle_run_command)

    compare_parser = subparsers.add_parser("compare", help="Compare two evaluation runs")
    compare_parser.add_argument("--left", required=True)
    compare_parser.add_argument("--right", required=True)
    compare_parser.add_argument("--output", required=False)
    compare_parser.set_defaults(handler=handle_compare_command)

    dataset_parser = subparsers.add_parser("dataset", help="Dataset management commands")
    dataset_subparsers = dataset_parser.add_subparsers(dest="dataset_command", required=True)

    dataset_add = dataset_subparsers.add_parser("add", help="Add a dataset to local registry")
    dataset_add.add_argument("path")
    dataset_add.add_argument("--example", required=False)
    dataset_add.add_argument("--example-file", required=False)
    dataset_add.add_argument("--example-id", required=False)
    dataset_add.add_argument("--input-json", required=False)
    dataset_add.add_argument("--expected-json", required=False)
    dataset_add.set_defaults(handler=handle_dataset_add_command)

    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return {"command": None}
    return handler(args)


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True))
