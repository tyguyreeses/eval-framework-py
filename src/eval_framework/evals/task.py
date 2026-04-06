from __future__ import annotations

from collections.abc import Callable
import inspect
from typing import Any


TaskCallable = Callable[..., dict[str, Any]]


def task_oracle_expected(_: dict[str, Any], *, example: dict[str, Any] | None = None) -> dict[str, Any]:
    if example is None:
        return {}
    expected = example.get("expected")
    return expected if isinstance(expected, dict) else {"expected": expected}


def task_echo_input(example_input: dict[str, Any], **_: Any) -> dict[str, Any]:
    return example_input


DEFAULT_TASK_REGISTRY: dict[str, TaskCallable] = {
    "oracle_expected": task_oracle_expected,
    "echo_input": task_echo_input,
}


def list_available_tasks() -> list[str]:
    return sorted(DEFAULT_TASK_REGISTRY.keys())


def resolve_task(task: str | TaskCallable) -> TaskCallable:
    if callable(task):
        return task
    try:
        return DEFAULT_TASK_REGISTRY[task]
    except KeyError as exc:
        available = ", ".join(sorted(DEFAULT_TASK_REGISTRY.keys()))
        raise ValueError(f"Unknown task `{task}`. Available tasks: {available}") from exc


def invoke_task(task: TaskCallable, example_input: dict[str, Any], example: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(task)
    if "example" in signature.parameters:
        prediction = task(example_input, example=example)
    else:
        prediction = task(example_input)

    if isinstance(prediction, dict):
        return prediction
    return {"output": prediction}
