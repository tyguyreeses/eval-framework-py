from __future__ import annotations

import json
from pathlib import Path

from eval_framework.cli import main as cli_main
from eval_framework.evals.dataset import Dataset
from eval_framework.evals.logging import load_run_log


def _write_dataset(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "dataset_name": "integration_eval_dataset",
                "examples": [
                    {
                        "example_id": "ex-1",
                        "input": {"text": "candidate profile"},
                        "expected": {"label": "hire"},
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def test_dataset_load_append_save_roundtrip(tmp_path: Path):
    dataset_path = _write_dataset(tmp_path / "dataset.json")
    dataset = Dataset.load(dataset_path)
    dataset.append(
        {
            "example_id": "ex-2",
            "input": {"text": "follow-up sample"},
            "expected": {"label": "no_hire"},
        }
    )
    dataset.save()

    reloaded = Dataset.load(dataset_path)
    assert len(reloaded.examples) == 2
    assert {example["example_id"] for example in reloaded.examples} == {"ex-1", "ex-2"}


def test_eval_cli_run_compare_and_persisted_logs(tmp_path: Path):
    dataset_path = _write_dataset(tmp_path / "dataset.json")
    append_result = cli_main(
        [
            "dataset",
            "add",
            str(dataset_path),
            "--example",
            json.dumps(
                {
                    "example_id": "ex-2",
                    "input": {"text": "second sample"},
                    "expected": {"label": "strong_hire"},
                }
            ),
        ]
    )
    assert append_result["command"] == "dataset.add"
    assert append_result["total_examples"] == 2

    run_good_path = tmp_path / "run_good.json"
    run_bad_path = tmp_path / "run_bad.json"

    run_good = cli_main(
        [
            "run",
            "--dataset",
            str(dataset_path),
            "--task",
            "oracle_expected",
            "--run-id",
            "run-good",
            "--output",
            str(run_good_path),
        ]
    )
    run_bad = cli_main(
        [
            "run",
            "--dataset",
            str(dataset_path),
            "--task",
            "echo_input",
            "--run-id",
            "run-bad",
            "--output",
            str(run_bad_path),
        ]
    )

    assert run_good["command"] == "run"
    assert run_bad["command"] == "run"
    assert run_good["summary"]["num_examples"] == 2
    assert run_bad["summary"]["num_examples"] == 2
    assert run_good_path.exists()
    assert run_bad_path.exists()

    loaded_good = load_run_log(run_good_path)
    loaded_bad = load_run_log(run_bad_path)
    assert loaded_good["run"]["run_id"] == "run-good"
    assert loaded_bad["run"]["run_id"] == "run-bad"
    assert loaded_good["summary"]["mean_score"] == 1.0
    assert loaded_bad["summary"]["mean_score"] == 0.0

    compare_output_path = tmp_path / "compare.json"
    compare = cli_main(
        [
            "compare",
            "--left",
            str(run_good_path),
            "--right",
            str(run_bad_path),
            "--output",
            str(compare_output_path),
        ]
    )
    assert compare["command"] == "compare"
    assert compare["comparison"]["summary_delta"]["mean_score_delta"] == -1.0
    assert compare["comparison"]["shared_examples"] == 2
    assert compare_output_path.exists()
