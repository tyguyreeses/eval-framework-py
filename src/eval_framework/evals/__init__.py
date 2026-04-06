"""Core evaluation modules."""

from .comparator import compare_runs
from .dataset import Dataset, append_dataset_example, load_dataset
from .envelope import validate_shared_envelope
from .gates import evaluate_gate
from .judge import resolve_judge_strategy
from .logging import RunLogger, load_run_log, save_run_log
from .runner import EvalRunner, run_evaluation

__all__ = [
    "Dataset",
    "EvalRunner",
    "RunLogger",
    "append_dataset_example",
    "compare_runs",
    "evaluate_gate",
    "load_dataset",
    "load_run_log",
    "resolve_judge_strategy",
    "run_evaluation",
    "save_run_log",
    "validate_shared_envelope",
]
