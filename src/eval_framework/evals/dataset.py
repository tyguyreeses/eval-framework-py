from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


def _validate_example(example: dict[str, Any]) -> None:
    required_fields = {"example_id", "input", "expected"}
    missing = required_fields - set(example.keys())
    if missing:
        missing_csv = ", ".join(sorted(missing))
        raise ValueError(f"Dataset example missing required fields: {missing_csv}")


@dataclass
class Dataset:
    path: Path
    dataset_name: str = "unnamed_dataset"
    examples: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "Dataset":
        dataset_path = Path(path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
        examples = payload.get("examples", [])
        if not isinstance(examples, list):
            raise ValueError("Dataset `examples` must be a list")

        for example in examples:
            if not isinstance(example, dict):
                raise ValueError("Dataset examples must be JSON objects")
            _validate_example(example)

        metadata = {
            key: value
            for key, value in payload.items()
            if key not in {"dataset_name", "examples"}
        }
        dataset_name = str(payload.get("dataset_name") or dataset_path.stem)
        return cls(path=dataset_path, dataset_name=dataset_name, examples=examples, metadata=metadata)

    def validate(self) -> None:
        for example in self.examples:
            _validate_example(example)

    def append(self, example: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(example, dict):
            raise ValueError("Dataset example must be a JSON object")
        _validate_example(example)

        example_id = str(example["example_id"])
        if any(str(existing.get("example_id")) == example_id for existing in self.examples):
            raise ValueError(f"Duplicate example_id: {example_id}")

        self.examples.append(example)
        return example

    def save(self, path: str | Path | None = None) -> Path:
        target_path = Path(path) if path else self.path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset_name": self.dataset_name,
            **self.metadata,
            "examples": self.examples,
        }
        target_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        self.path = target_path
        return target_path


def load_dataset(path: str | Path) -> Dataset:
    return Dataset.load(path)


def append_dataset_example(path: str | Path, example: dict[str, Any]) -> dict[str, Any]:
    dataset = Dataset.load(path)
    added = dataset.append(example)
    dataset.save()
    return added
