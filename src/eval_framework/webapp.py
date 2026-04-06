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
PRESETS_PATH = PROJECT_ROOT / "data" / "web_presets.json"
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


def _load_presets() -> list[dict[str, Any]]:
    if not PRESETS_PATH.exists():
        return []
    payload = json.loads(PRESETS_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in payload:
        if isinstance(item, dict) and "name" in item and "values" in item:
            normalized.append(item)
    return sorted(normalized, key=lambda preset: str(preset["name"]).lower())


def _save_presets(presets: list[dict[str, Any]]) -> None:
    PRESETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PRESETS_PATH.write_text(json.dumps(presets, indent=2, sort_keys=True), encoding="utf-8")


def _upsert_preset(name: str, values: dict[str, Any]) -> None:
    if not name.strip():
        raise ValueError("Preset name is required.")
    presets = _load_presets()
    updated: list[dict[str, Any]] = []
    replaced = False
    for preset in presets:
        if str(preset.get("name")) == name:
            updated.append({"name": name, "values": values})
            replaced = True
        else:
            updated.append(preset)
    if not replaced:
        updated.append({"name": name, "values": values})
    _save_presets(updated)


def _find_preset(name: str) -> dict[str, Any] | None:
    for preset in _load_presets():
        if str(preset.get("name")) == name:
            values = preset.get("values")
            return values if isinstance(values, dict) else None
    return None


def _delete_preset(name: str) -> bool:
    presets = _load_presets()
    kept = [preset for preset in presets if str(preset.get("name")) != name]
    changed = len(kept) != len(presets)
    if changed:
        _save_presets(kept)
    return changed


def _build_trend_series(limit: int = 40) -> list[dict[str, Any]]:
    points: list[tuple[float, dict[str, Any]]] = []
    for run_path in RUNS_DIR.glob("*.json"):
        try:
            payload = json.loads(run_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        run = payload.get("run", {})
        summary = payload.get("summary", {})
        run_id = str(run.get("run_id", run_path.stem))
        score = float(summary.get("mean_score", 0.0))
        pass_rate = float(summary.get("pass_rate", 0.0))
        dataset_path = str(run.get("dataset_path", ""))
        points.append(
            (
                run_path.stat().st_mtime,
                {
                    "run_id": run_id,
                    "label": run_id[:14] + ("..." if len(run_id) > 14 else ""),
                    "mean_score": score,
                    "pass_rate": pass_rate,
                    "dataset_path": dataset_path,
                    "path": str(run_path),
                },
            )
        )
    points.sort(key=lambda item: item[0])
    return [point for _, point in points[-limit:]]


def _parse_object_json(raw: str, field_name: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for `{field_name}`: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"`{field_name}` must be a JSON object")
    return payload


def _parse_checkbox(raw: str | None) -> bool:
    return raw is not None


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
    form_defaults: dict[str, Any] | None = None,
) -> HTMLResponse:
    datasets = _list_datasets()
    runs = _list_runs()
    presets = _load_presets()
    trend = _build_trend_series()
    defaults = {
        "dataset_path": datasets[0] if datasets else "",
        "task_name": "oracle_expected",
        "scorer_name": "exact_match",
        "run_id": "",
        "provider": "openai",
        "model": "gpt-5-mini",
        "deterministic_only": False,
        "openai_available": True,
        "gate_mode": "soft",
        "threshold": 0.8,
        "runs_dir": "",
        "output_path": "",
        "left_path": runs[-2] if len(runs) > 1 else (runs[0] if runs else ""),
        "right_path": runs[-1] if runs else "",
        "compare_output_path": "",
        "dataset_add_path": datasets[0] if datasets else "",
        "input_json": '{"text": "sample input"}',
        "expected_json": '{"label": "hire"}',
    }
    if form_defaults:
        defaults.update(form_defaults)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "message": message,
            "error": error,
            "datasets": datasets,
            "runs": runs,
            "presets": presets,
            "tasks": list_available_tasks(),
            "scorers": list_available_scorers(),
            "run_result": run_result,
            "compare_result": compare_result,
            "dataset_result": dataset_result,
            "run_result_text": _pretty(run_result),
            "compare_result_text": _pretty(compare_result),
            "dataset_result_text": _pretty(dataset_result),
            "form_defaults": defaults,
            "trend": trend,
            "trend_json": json.dumps(trend),
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
    deterministic_only: str | None = Form(None),
    openai_available: str | None = Form(None),
    gate_mode: str = Form("soft"),
    threshold: float = Form(0.8),
    runs_dir: str = Form(""),
    output_path: str = Form(""),
) -> HTMLResponse:
    deterministic_only_flag = _parse_checkbox(deterministic_only)
    openai_available_flag = _parse_checkbox(openai_available)
    form_defaults = {
        "dataset_path": dataset_path,
        "task_name": task_name,
        "scorer_name": scorer_name,
        "run_id": run_id,
        "provider": provider,
        "model": model,
        "deterministic_only": deterministic_only_flag,
        "openai_available": openai_available_flag,
        "gate_mode": gate_mode,
        "threshold": threshold,
        "runs_dir": runs_dir,
        "output_path": output_path,
    }
    try:
        run = run_evaluation(
            dataset=dataset_path,
            task=task_name,
            grader=Grader(scorer_name=scorer_name),
            run_id=run_id or None,
            provider=provider,
            model=model,
            deterministic_only=deterministic_only_flag,
            openai_available=openai_available_flag,
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
        form_defaults["run_id"] = run["run"]["run_id"]
        form_defaults["output_path"] = str(saved_path)
        return _render_home(request, message="Evaluation run completed.", run_result=result, form_defaults=form_defaults)
    except Exception as exc:
        return _render_home(request, error=str(exc), form_defaults=form_defaults)


@app.post("/compare", response_class=HTMLResponse)
def compare(
    request: Request,
    left_path: str = Form(...),
    right_path: str = Form(...),
    output_path: str = Form(""),
) -> HTMLResponse:
    form_defaults = {
        "left_path": left_path,
        "right_path": right_path,
        "compare_output_path": output_path,
    }
    try:
        result = compare_runs(left_path, right_path)
        if output_path:
            target = Path(output_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        payload = {"command": "compare", "comparison": result, "output_path": output_path or None}
        return _render_home(request, message="Run comparison completed.", compare_result=payload, form_defaults=form_defaults)
    except Exception as exc:
        return _render_home(request, error=str(exc), form_defaults=form_defaults)


@app.post("/dataset/add", response_class=HTMLResponse)
def add_dataset_example(
    request: Request,
    dataset_path: str = Form(...),
    example_id: str = Form(...),
    input_json: str = Form(...),
    expected_json: str = Form(...),
) -> HTMLResponse:
    form_defaults = {
        "dataset_add_path": dataset_path,
        "input_json": input_json,
        "expected_json": expected_json,
    }
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
        form_defaults["dataset_add_path"] = str(saved_path)
        return _render_home(request, message="Dataset example added.", dataset_result=payload, form_defaults=form_defaults)
    except Exception as exc:
        return _render_home(request, error=str(exc), form_defaults=form_defaults)


@app.post("/presets/save", response_class=HTMLResponse)
def save_preset(
    request: Request,
    preset_name: str = Form(...),
    dataset_path: str = Form(""),
    task_name: str = Form("oracle_expected"),
    scorer_name: str = Form("exact_match"),
    provider: str = Form("openai"),
    model: str = Form("gpt-5-mini"),
    deterministic_only: str | None = Form(None),
    openai_available: str | None = Form(None),
    gate_mode: str = Form("soft"),
    threshold: float = Form(0.8),
    runs_dir: str = Form(""),
) -> HTMLResponse:
    deterministic_only_flag = _parse_checkbox(deterministic_only)
    openai_available_flag = _parse_checkbox(openai_available)
    values = {
        "dataset_path": dataset_path,
        "task_name": task_name,
        "scorer_name": scorer_name,
        "provider": provider,
        "model": model,
        "deterministic_only": deterministic_only_flag,
        "openai_available": openai_available_flag,
        "gate_mode": gate_mode,
        "threshold": threshold,
        "runs_dir": runs_dir,
    }
    try:
        _upsert_preset(preset_name.strip(), values)
        return _render_home(request, message=f"Preset `{preset_name}` saved.", form_defaults=values)
    except Exception as exc:
        return _render_home(request, error=str(exc), form_defaults=values)


@app.post("/presets/apply", response_class=HTMLResponse)
def apply_preset(
    request: Request,
    preset_name: str = Form(...),
) -> HTMLResponse:
    values = _find_preset(preset_name)
    if values is None:
        return _render_home(request, error=f"Preset `{preset_name}` not found.")
    return _render_home(request, message=f"Preset `{preset_name}` applied.", form_defaults=values)


@app.post("/presets/delete", response_class=HTMLResponse)
def delete_preset(
    request: Request,
    preset_name: str = Form(...),
) -> HTMLResponse:
    if _delete_preset(preset_name):
        return _render_home(request, message=f"Preset `{preset_name}` deleted.")
    return _render_home(request, error=f"Preset `{preset_name}` not found.")


def main() -> None:
    import uvicorn

    uvicorn.run("eval_framework.webapp:app", host="127.0.0.1", port=8000, reload=True)
