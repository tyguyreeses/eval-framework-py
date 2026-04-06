from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .evals.comparator import compare_runs
from .evals.dataset import Dataset
from .evals.grader import Grader, list_available_scorers
from .evals.logging import RunLogger
from .evals.runner import run_evaluation
from .evals.task import list_available_tasks

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
DATASETS_DIR = PROJECT_ROOT / "data" / "evals"
RUNS_DIR = PROJECT_ROOT / "data" / "eval_runs"
TEMPLATES_DIR = PACKAGE_ROOT / "web_templates"
STATIC_DIR = PACKAGE_ROOT / "web_static"

app = FastAPI(title="Eval Framework UI")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _list_datasets() -> list[str]:
    if not DATASETS_DIR.exists():
        return []
    return [str(path) for path in sorted(DATASETS_DIR.rglob("*.json"))]


def _list_runs() -> list[str]:
    if not RUNS_DIR.exists():
        return []
    return [str(path) for path in sorted(RUNS_DIR.glob("*.json"))]


def _parse_object_json(raw: str, field_name: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for `{field_name}`: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"`{field_name}` must be a JSON object")
    return payload


def _pretty(payload: dict[str, Any] | None) -> str | None:
    if payload is None:
        return None
    return json.dumps(payload, indent=2, sort_keys=True)


def _render_home(
    request: Request,
    *,
    message: str | None = None,
    error: str | None = None,
    run_result: dict[str, Any] | None = None,
    compare_result: dict[str, Any] | None = None,
    dataset_result: dict[str, Any] | None = None,
) -> HTMLResponse:
    datasets = _list_datasets()
    runs = _list_runs()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "message": message,
            "error": error,
            "datasets": datasets,
            "runs": runs,
            "tasks": list_available_tasks(),
            "scorers": list_available_scorers(),
            "run_result": run_result,
            "compare_result": compare_result,
            "dataset_result": dataset_result,
            "run_result_text": _pretty(run_result),
            "compare_result_text": _pretty(compare_result),
            "dataset_result_text": _pretty(dataset_result),
            "default_dataset": datasets[0] if datasets else "",
            "default_left_run": runs[-2] if len(runs) > 1 else (runs[0] if runs else ""),
            "default_right_run": runs[-1] if runs else "",
        },
    )


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return _render_home(request)


@app.post("/run", response_class=HTMLResponse)
def run_eval(
    request: Request,
    dataset_path: str = Form(...),
    task_name: str = Form("oracle_expected"),
    scorer_name: str = Form("exact_match"),
    run_id: str = Form(""),
    provider: str = Form("openai"),
    model: str = Form("gpt-5-mini"),
    deterministic_only: bool = Form(False),
    openai_available: bool = Form(True),
    gate_mode: str = Form("soft"),
    threshold: float = Form(0.8),
    runs_dir: str = Form(""),
    output_path: str = Form(""),
) -> HTMLResponse:
    try:
        run = run_evaluation(
            dataset=dataset_path,
            task=task_name,
            grader=Grader(scorer_name=scorer_name),
            run_id=run_id or None,
            provider=provider,
            model=model,
            deterministic_only=deterministic_only,
            openai_available=openai_available,
            gate_mode=gate_mode,
            gate_threshold=threshold,
        )
        logger = RunLogger(base_dir=runs_dir or None)
        saved_path = logger.save(run, path=output_path or None)
        result = {
            "command": "run",
            "output_path": str(saved_path),
            "run_id": run["run"]["run_id"],
            "summary": run["summary"],
        }
        return _render_home(request, message="Evaluation run completed.", run_result=result)
    except Exception as exc:
        return _render_home(request, error=str(exc))


@app.post("/compare", response_class=HTMLResponse)
def compare(
    request: Request,
    left_path: str = Form(...),
    right_path: str = Form(...),
    output_path: str = Form(""),
) -> HTMLResponse:
    try:
        result = compare_runs(left_path, right_path)
        if output_path:
            target = Path(output_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        payload = {"command": "compare", "comparison": result, "output_path": output_path or None}
        return _render_home(request, message="Run comparison completed.", compare_result=payload)
    except Exception as exc:
        return _render_home(request, error=str(exc))


@app.post("/dataset/add", response_class=HTMLResponse)
def add_dataset_example(
    request: Request,
    dataset_path: str = Form(...),
    example_id: str = Form(...),
    input_json: str = Form(...),
    expected_json: str = Form(...),
) -> HTMLResponse:
    try:
        dataset_file = Path(dataset_path)
        if dataset_file.exists():
            dataset = Dataset.load(dataset_file)
        else:
            dataset = Dataset(path=dataset_file, dataset_name=dataset_file.stem, examples=[])

        example = {
            "example_id": example_id,
            "input": _parse_object_json(input_json, "input_json"),
            "expected": _parse_object_json(expected_json, "expected_json"),
        }
        added = dataset.append(example)
        saved_path = dataset.save()
        payload = {
            "command": "dataset.add",
            "dataset_path": str(saved_path),
            "total_examples": len(dataset.examples),
            "added_example_id": str(added["example_id"]),
        }
        return _render_home(request, message="Dataset example added.", dataset_result=payload)
    except Exception as exc:
        return _render_home(request, error=str(exc))


def main() -> None:
    import uvicorn

    uvicorn.run("eval_framework.webapp:app", host="127.0.0.1", port=8000, reload=True)
