"""Workflow trace loading and deterministic dataset splitting.

Workflow loading is deliberately independent of candidate evolution.  This
keeps data preparation usable by profiling, evaluation, and final testing
without importing the search loop.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from .candidate import EvaluationTrace
from .openai_compatible import plan_from_dict


@dataclass(frozen=True)
class ToolCallPlanDatasetSplits:
    train: tuple[EvaluationTrace, ...]
    validation: tuple[EvaluationTrace, ...]
    test: tuple[EvaluationTrace, ...]


@dataclass(frozen=True)
class ToolCallPlanDataset(Sequence[EvaluationTrace]):
    """Replayable JSONL collection of device-independent workflow traces."""

    traces: tuple[EvaluationTrace, ...]

    @classmethod
    def load(cls, path: Path | str) -> "ToolCallPlanDataset":
        dataset_path = Path(path)
        traces: list[EvaluationTrace] = []
        with dataset_path.open(encoding="utf-8") as source:
            for line_number, raw_line in enumerate(source, start=1):
                if not raw_line.strip():
                    raise ValueError(f"{dataset_path}: line {line_number} is empty")
                try:
                    row = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{dataset_path}: invalid JSON on line {line_number}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{dataset_path}: line {line_number} must contain a JSON object")
                try:
                    dags = _plans_from_row(row)
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"{dataset_path}: invalid ToolCallPlan candidates on line {line_number}: {exc}") from exc
                task_input = row.get("task_input", row.get("inputs", {}))
                if not isinstance(task_input, dict):
                    raise ValueError(f"{dataset_path}: line {line_number} task_input must be an object")
                trace_id = row.get("trace_id", f"mnms-{line_number - 1:06d}")
                if not isinstance(trace_id, str) or not trace_id:
                    raise ValueError(f"{dataset_path}: line {line_number} trace_id must be a non-empty string")
                system_state = row.get("system_state", {})
                if not isinstance(system_state, dict):
                    raise ValueError(f"{dataset_path}: line {line_number} system_state must be an object")
                traces.append(EvaluationTrace(trace_id, task_input, dags[0], dags, system_state))
        return cls(tuple(traces))

    def __len__(self) -> int:
        return len(self.traces)

    def __getitem__(self, index: int | slice) -> EvaluationTrace | tuple[EvaluationTrace, ...]:
        return self.traces[index]

    def __iter__(self) -> Iterator[EvaluationTrace]:
        return iter(self.traces)

    def split(self) -> ToolCallPlanDatasetSplits:
        count = len(self)
        counts = split_counts(count)
        train_end = counts[0]
        validation_end = train_end + counts[1]
        return ToolCallPlanDatasetSplits(self.traces[:train_end], self.traces[train_end:validation_end], self.traces[validation_end:])


def split_counts(count: int) -> tuple[int, int, int]:
    """Return deterministic 4:3:3 counts, assigning remainders early."""
    if count < 0:
        raise ValueError("dataset count must not be negative")
    quotients, remainder = divmod(count, 10)
    counts = [4 * quotients, 3 * quotients, 3 * quotients]
    for index in range(remainder):
        counts[index % 3] += 1
    return counts[0], counts[1], counts[2]


def _plans_from_row(row: dict[str, object]) -> tuple:
    """Read the versioned three-DAG row, with a legacy single-DAG fallback.

    ``dags`` is the canonical key.  ``candidate_dags`` and ``plans`` are
    accepted as import aliases because early offline exports used both names.
    A legacy ToolCallPlan row is expanded to three identical alternatives so
    all Scheduler baselines see the same interface without rewriting the
    checked-in MnMS reference export.
    """
    raw_candidates = next(
        (row[key] for key in ("dags", "candidate_dags", "plans") if key in row),
        None,
    )
    if raw_candidates is None:
        dag = plan_from_dict(row)  # legacy device-independent DAG row
        return (dag, dag, dag)
    if not isinstance(raw_candidates, list):
        raise TypeError("dags must be an array")
    if len(raw_candidates) != 3:
        raise ValueError("dags must contain exactly three ToolCallPlans")
    if any(not isinstance(candidate, dict) for candidate in raw_candidates):
        raise TypeError("each DAG candidate must be an object")
    return tuple(plan_from_dict(candidate) for candidate in raw_candidates)


__all__ = ["ToolCallPlanDataset", "ToolCallPlanDatasetSplits", "split_counts"]
