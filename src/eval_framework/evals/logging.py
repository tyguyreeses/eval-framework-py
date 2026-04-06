from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .envelope import validate_shared_envelope

DEFAULT_RUNS_DIR = Path(__file__).resolve().parents[2] / "data" / "eval_runs"


class RunLogger:
    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = Path(base_dir) if base_dir else DEFAULT_RUNS_DIR

    def save(self, run: dict[str, Any], path: str | Path | None = None) -> Path:
        validated = validate_shared_envelope(run)
        target_path = Path(path) if path else self.base_dir / f"{validated['run']['run_id']}.json"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(json.dumps(validated, indent=2, sort_keys=True), encoding="utf-8")
        return target_path

    def load(self, path: str | Path) -> dict[str, Any]:
        run_path = Path(path)
        if not run_path.exists():
            raise FileNotFoundError(f"Run log file not found: {run_path}")
        payload = json.loads(run_path.read_text(encoding="utf-8"))
        return validate_shared_envelope(payload)

    def list_runs(self) -> list[Path]:
        if not self.base_dir.exists():
            return []
        return sorted(self.base_dir.glob("*.json"))


def save_run_log(run: dict[str, Any], path: str | Path | None = None) -> Path:
    logger = RunLogger()
    return logger.save(run, path)


def load_run_log(path: str | Path) -> dict[str, Any]:
    logger = RunLogger()
    return logger.load(path)
