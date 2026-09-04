"""Read-only tools over trusted Candidate Evaluation Evidence."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json
import math
import statistics
from typing import Any, TypeGuard


_MISSING = object()


@dataclass(frozen=True)
class TraceAnalysisTools:
    """Expose bounded slices and summaries without evaluator mutation authority."""

    document: Mapping[str, object]
    max_output_chars: int = 16_000

    def inspect_node_dimension(
        self, dimension: str, trace_id: str | None = None, limit: int = 100
    ) -> str:
        if not 1 <= limit <= 1_000:
            raise ValueError("limit must be between 1 and 1000")
        rows: list[dict[str, object]] = []
        for current_trace_id, node in self._nodes():
            if trace_id is not None and current_trace_id != trace_id:
                continue
            value = _lookup(node, dimension)
            if value is _MISSING:
                continue
            rows.append(
                {
                    "trace_id": current_trace_id,
                    "node_id": node.get("node_id"),
                    "tool_id": node.get("tool_id"),
                    "device_id": node.get("device_id", node.get("device")),
                    "configuration_id": node.get("configuration_id"),
                    dimension: value,
                }
            )
        return self._render({"dimension": dimension, "row_count": len(rows), "rows": rows[:limit]})

    def summarize_node_dimension(
        self, dimension: str, group_by: str = "device_id", trace_id: str | None = None
    ) -> str:
        values: defaultdict[str, list[float]] = defaultdict(list)
        skipped = 0
        for current_trace_id, node in self._nodes():
            if trace_id is not None and current_trace_id != trace_id:
                continue
            value, group = _lookup(node, dimension), _lookup(node, group_by)
            if not _finite(value) or group is _MISSING:
                skipped += 1
                continue
            values[str(group)].append(float(value))
        groups = {name: _summary(items) for name, items in sorted(values.items())}
        return self._render(
            {"dimension": dimension, "group_by": group_by, "groups": groups, "skipped": skipped}
        )

    def compare_assignments(self, left_version: int, right_version: int) -> str:
        """Show node choices that differ between two Scheduler Candidates."""
        by_version: defaultdict[int, dict[tuple[str | None, str], tuple[object, object]]] = defaultdict(dict)
        for trace_id, node in self._nodes():
            version = node.get("scheduler_version")
            node_id = node.get("node_id")
            if isinstance(version, int) and isinstance(node_id, str):
                by_version[version][(trace_id, node_id)] = (
                    node.get("configuration_id"),
                    node.get("device_id", node.get("device")),
                )
        keys = sorted(set(by_version[left_version]) | set(by_version[right_version]))
        changes = [
            {"trace_id": key[0], "node_id": key[1], "left": by_version[left_version].get(key), "right": by_version[right_version].get(key)}
            for key in keys
            if by_version[left_version].get(key) != by_version[right_version].get(key)
        ]
        return self._render({"left_version": left_version, "right_version": right_version, "changes": changes})

    def _nodes(self) -> Iterable[tuple[str | None, Mapping[str, object]]]:
        yield from _walk(self.document, self.document.get("trace_id"))

    def _render(self, value: object) -> str:
        rendered = json.dumps(value, ensure_ascii=False, default=str, indent=2)
        if len(rendered) <= self.max_output_chars:
            return rendered
        return rendered[: self.max_output_chars] + "\n... tool output truncated"


def _walk(value: object, trace_id: object) -> Iterable[tuple[str | None, Mapping[str, object]]]:
    if isinstance(value, Mapping):
        current_trace_id = value.get("trace_id", trace_id)
        for key, child in value.items():
            if key in {"nodes", "node_timings", "simulated_nodes"} and isinstance(child, list):
                for node in child:
                    if isinstance(node, Mapping):
                        yield str(current_trace_id) if current_trace_id is not None else None, node
            yield from _walk(child, current_trace_id)
    elif isinstance(value, list | tuple):
        for child in value:
            yield from _walk(child, trace_id)


def _lookup(value: Mapping[str, object], path: str) -> object:
    current: object = value
    for part in path.split("."):
        if not part or not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _finite(value: object) -> TypeGuard[int | float]:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(float(value))


def _summary(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "count": len(ordered),
        "mean": round(statistics.fmean(ordered), 4),
        "p95": round(ordered[p95_index], 4),
        "min": round(ordered[0], 4),
        "max": round(ordered[-1], 4),
    }
