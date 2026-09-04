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
                    dag = plan_from_dict(row)
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"{dataset_path}: invalid ToolCallPlan on line {line_number}") from exc
                traces.append(EvaluationTrace(f"mnms-{line_number - 1:06d}", {}, dag))
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


__all__ = ["ToolCallPlanDataset", "ToolCallPlanDatasetSplits", "split_counts"]
