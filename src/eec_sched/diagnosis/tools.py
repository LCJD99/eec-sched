"""Read-only tools over trusted Candidate Evaluation Evidence."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json
import math
import statistics
from typing import Literal, Sequence, TypeGuard, cast

from ..domain import ToolCallPlan
from ..evaluation.evaluator import evaluate_assignments
from ..evaluation.models import NodeAssignment
from ..evaluation.scoring import ScoringContext
from ..openai_compatible import plan_from_dict
from ..profiling.snapshot import ProfilingDatabaseSnapshot


_MISSING = object()
_PROFILE_TRACE_FOCI = ("latency", "resource", "quality", "communication", "placement")


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
        return self._render(
            {"dimension": dimension, "row_count": len(rows), "rows": rows[:limit]}
        )

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
            {
                "dimension": dimension,
                "group_by": group_by,
                "groups": groups,
                "skipped": skipped,
            }
        )

    def compare_assignments(self, left_version: int, right_version: int) -> str:
        """Show node choices that differ between two Scheduler Candidates."""
        by_version: defaultdict[
            int, dict[tuple[str | None, str], tuple[object, object]]
        ] = defaultdict(dict)
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
            {
                "trace_id": key[0],
                "node_id": key[1],
                "left": by_version[left_version].get(key),
                "right": by_version[right_version].get(key),
            }
            for key in keys
            if by_version[left_version].get(key) != by_version[right_version].get(key)
        ]
        return self._render(
            {
                "left_version": left_version,
                "right_version": right_version,
                "changes": changes,
            }
        )

    def _nodes(self) -> Iterable[tuple[str | None, Mapping[str, object]]]:
        yield from _walk(self.document, self.document.get("trace_id"))

    def _render(self, value: object) -> str:
        rendered = json.dumps(value, ensure_ascii=False, default=str, indent=2)
        if len(rendered) <= self.max_output_chars:
            return rendered
        return rendered[: self.max_output_chars] + "\n... tool output truncated"


def _walk(
    value: object, trace_id: object
) -> Iterable[tuple[str | None, Mapping[str, object]]]:
    if isinstance(value, Mapping):
        current_trace_id = value.get("trace_id", trace_id)
        for key, child in value.items():
            if key in {"nodes", "node_timings", "simulated_nodes"} and isinstance(
                child, list
            ):
                for node in child:
                    if isinstance(node, Mapping):
                        yield (
                            str(current_trace_id)
                            if current_trace_id is not None
                            else None,
                            node,
                        )
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
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


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


class BottleneckMetaTools:
    """Small, JSON-facing profiling and counterfactual evaluation tools.

    The class deliberately has only three callable entry points.  It receives
    the source-free trace evidence produced by the evolution loop and, for
    interventions, the immutable profiling snapshot used by the trusted
    evaluator.  None of the methods mutate either input.
    """

    def __init__(
        self,
        document: Mapping[str, object] | Sequence[object],
        snapshot: ProfilingDatabaseSnapshot | None = None,
        *,
        scoring_context: ScoringContext | None = None,
        max_output_chars: int = 16_000,
    ) -> None:
        if (
            not isinstance(max_output_chars, int)
            or isinstance(max_output_chars, bool)
            or max_output_chars < 1
        ):
            raise ValueError("max_output_chars must be a positive integer")
        self.document = document
        self.snapshot = snapshot
        self.scoring_context = scoring_context or ScoringContext()
        self.max_output_chars = max_output_chars

    def profile_trace(
        self,
        trace_id: str,
        focus: Literal[
            "latency", "resource", "quality", "communication", "placement"
        ]
        | None = None,
        top_k: int = 5,
    ) -> str:
        """Return a deterministic, bounded analysis of one complete Trace.

        The model-facing contract is deliberately analysis-oriented: the
        immutable Candidate Evaluation Evidence, Profiling Database Snapshot,
        and Scoring Context are joined here and returned as one stable
        ``TraceProfile``. ``section`` and raw storage-layout filtering are not
        part of the contract.
        """
        if not isinstance(trace_id, str) or not trace_id:
            return self._error("trace_id must be a non-empty string")
        if focus is not None and focus not in _PROFILE_TRACE_FOCI:
            return self._error(
                "unknown profile focus",
                focus=focus,
                allowed_focus=list(_PROFILE_TRACE_FOCI),
            )
        if (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or not 1 <= top_k <= 100
        ):
            return self._error("top_k must be an integer between 1 and 100", top_k=top_k)
        trace = self._find_trace(trace_id)
        if trace is None:
            return self._error("unknown trace_id", trace_id=trace_id)
        return self._render(self._build_trace_profile(trace_id, trace, focus, top_k))

    def _build_trace_profile(
        self, trace_id: str, trace: object, focus: str | None, top_k: int
    ) -> dict[str, object]:
        dag = _as_dag(_trace_value(trace, "dag", {}))
        safe_dag = _json_safe(_trace_value(trace, "dag", {}))
        dag_nodes = _mapping_list(
            safe_dag.get("nodes", ()) if isinstance(safe_dag, Mapping) else ()
        )
        dag_by_id = {
            str(node.get("node_id")): node
            for node in dag_nodes
            if isinstance(node.get("node_id"), str)
        }
        assignments = _json_safe(_trace_value(trace, "assignments", {}))
        assignments_map = assignments if isinstance(assignments, Mapping) else {}
        observed_nodes = _mapping_list(_trace_value(trace, "nodes", ()))
        observed_by_id = {
            str(node.get("node_id")): node
            for node in observed_nodes
            if isinstance(node.get("node_id"), str)
        }
        transfers = _mapping_list(_trace_value(trace, "transfers", ()))
        metrics = _trace_metrics(trace)
        node_profiles: list[dict[str, object]] = []
        omitted: dict[str, int] = {}
        for node_id, dag_node in sorted(dag_by_id.items()):
            raw_assignment = assignments_map.get(node_id, {})
            assignment = _as_assignment(raw_assignment)
            observed = observed_by_id.get(node_id, {})
            row: dict[str, object] = {
                "node_id": node_id,
                "tool_id": dag_node.get("tool_id"),
                "configuration_id": assignment.configuration_id if assignment else raw_assignment.get("configuration_id") if isinstance(raw_assignment, Mapping) else None,
                "device_id": assignment.device_id if assignment else raw_assignment.get("device_id") if isinstance(raw_assignment, Mapping) else None,
            }
            start, finish = observed.get("start_ms"), observed.get("finish_ms")
            if _finite(start) and _finite(finish):
                row["observed_latency_ms"] = round(float(finish) - float(start), 6)
                row["start_ms"] = start
                row["finish_ms"] = finish
            tool_id = row.get("tool_id")
            config_id, device_id = row.get("configuration_id"), row.get("device_id")
            if self.snapshot is not None and isinstance(tool_id, str) and isinstance(config_id, str) and isinstance(device_id, str):
                try:
                    execution = self.snapshot.execution_profile(tool_id, config_id, device_id)
                    quality = self.snapshot.quality_profile(tool_id, config_id)
                    row.update({
                        "profile_latency_ms": execution.warm_latency_p95_ms,
                        "gpu_memory_mib": execution.gpu_memory_mib,
                        "normalized_quality_lcb": quality.normalized_quality_lcb,
                        "representative_output_bytes": quality.representative_output_bytes,
                    })
                except KeyError:
                    row["profile_status"] = "unavailable"
            node_profiles.append(row)

        edge_pairs = _dag_edges(dag, dag_nodes)
        critical = _critical_path(node_profiles, transfers, edge_pairs)
        hotspots = _ranked_hotspots(node_profiles, transfers, focus, top_k)
        placement, alternatives = self._placement_analysis(node_profiles, focus, top_k)
        hypotheses = _hypotheses(hotspots, placement, alternatives, focus, top_k)
        evidence_refs = _evidence_refs(trace_id, critical, hotspots, placement, self.snapshot)
        for name, count in (
            (
                "critical_path",
                max(
                    0,
                    len(cast(Sequence[object], critical.get("activities", ())))
                    - top_k,
                ),
            ),
            ("ranked_hotspots", max(0, len(hotspots) - top_k)),
            ("placement_findings", max(0, len(placement) - top_k)),
            ("configuration_alternatives", max(0, len(alternatives) - top_k)),
            ("bottleneck_hypotheses", max(0, len(hypotheses) - top_k)),
            ("evidence_refs", max(0, len(evidence_refs) - top_k)),
        ):
            if count:
                omitted[name] = count
        value: dict[str, object] = {
            "trace_id": trace_id,
            "focus": focus,
            "scoring_context": {
                "accuracy_weight": self.scoring_context.accuracy_weight,
                "latency_weight": self.scoring_context.latency_weight,
                "resource_weight": self.scoring_context.resource_weight,
                "latency_scale_ms": self.scoring_context.latency_scale_ms,
                "resource_scale_mib": self.scoring_context.resource_scale_mib,
            },
            "metrics": metrics,
            "critical_path": {
                "nodes": list(cast(Sequence[object], critical.get("nodes", ())))[:top_k],
                "activities": list(
                    cast(Sequence[object], critical.get("activities", ()))
                )[:top_k],
                "total_makespan_ms": critical.get("total_makespan_ms"),
                "contributions": list(
                    cast(Sequence[object], critical.get("contributions", ()))
                )[:top_k],
                "critical_timeline": critical.get("critical_timeline", True),
                "approximation": critical.get("approximation", True),
                "approximation_reasons": critical.get("approximation_reasons", ()),
            },
            "ranked_hotspots": hotspots[:top_k],
            "placement_findings": placement[:top_k],
            "configuration_alternatives": alternatives[:top_k],
            "bottleneck_hypotheses": hypotheses[:top_k],
            "evidence_refs": evidence_refs[:top_k],
            "omitted": omitted,
        }
        return value

    def _placement_analysis(
        self, node_profiles: Sequence[Mapping[str, object]], focus: str | None, top_k: int
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        findings: list[dict[str, object]] = []
        alternatives: list[dict[str, object]] = []
        if self.snapshot is None:
            return findings, alternatives
        for node in node_profiles:
            tool_id = node.get("tool_id")
            current_config, current_device = node.get("configuration_id"), node.get("device_id")
            if not isinstance(tool_id, str) or not isinstance(current_config, str) or not isinstance(current_device, str):
                continue
            current_latency = _number(node.get("profile_latency_ms"))
            current_resource = _number(node.get("gpu_memory_mib"))
            current_quality = _number(node.get("normalized_quality_lcb"))
            rows: list[dict[str, object]] = []
            try:
                configurations = self.snapshot.configurations(tool_id)
            except KeyError:
                continue
            for config in configurations:
                for device_id in self.snapshot.compatible_devices(tool_id, config.configuration_id):
                    if config.configuration_id == current_config and device_id == current_device:
                        continue
                    try:
                        execution = self.snapshot.execution_profile(tool_id, config.configuration_id, device_id)
                        quality = self.snapshot.quality_profile(tool_id, config.configuration_id)
                    except KeyError:
                        continue
                    row: dict[str, object] = {
                        "node_id": node["node_id"],
                        "configuration_id": config.configuration_id,
                        "device_id": device_id,
                        "profile_latency_ms": execution.warm_latency_p95_ms,
                        "gpu_memory_mib": execution.gpu_memory_mib,
                        "normalized_quality_lcb": quality.normalized_quality_lcb,
                    }
                    if current_latency is not None:
                        row["latency_delta_ms"] = round(execution.warm_latency_p95_ms - current_latency, 6)
                    if current_resource is not None:
                        row["resource_delta_mib"] = round(execution.gpu_memory_mib - current_resource, 6)
                    if current_quality is not None:
                        row["quality_delta"] = round(quality.normalized_quality_lcb - current_quality, 6)
                    row["dominates_current"] = bool(
                        current_latency is not None and current_resource is not None and current_quality is not None
                        and execution.warm_latency_p95_ms <= current_latency
                        and execution.gpu_memory_mib <= current_resource
                        and quality.normalized_quality_lcb >= current_quality
                        and (
                            execution.warm_latency_p95_ms < current_latency
                            or execution.gpu_memory_mib < current_resource
                            or quality.normalized_quality_lcb > current_quality
                        )
                    )
                    rows.append(row)
            rows.sort(key=lambda row: _alternative_sort_key(row, focus))
            alternatives.extend(rows)
            if rows:
                findings.append({
                    "node_id": node["node_id"],
                    "tool_id": tool_id,
                    "current": {
                        "configuration_id": current_config,
                        "device_id": current_device,
                        "profile_latency_ms": current_latency,
                        "gpu_memory_mib": current_resource,
                        "normalized_quality_lcb": current_quality,
                    },
                    "best_alternative": rows[0],
                    "dominated": any(bool(row.get("dominates_current")) for row in rows),
                })
        findings.sort(key=lambda row: (not bool(row.get("dominated")), str(row.get("node_id"))))
        alternatives.sort(key=lambda row: _alternative_sort_key(row, focus))
        # Keep the global ranking, but reserve the first-ranked legal
        # alternative for each node before filling the remaining budget.  This
        # avoids letting a node with many configurations consume every visible
        # slot while preserving deterministic quality/latency ordering.
        by_node: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
        for row in alternatives:
            by_node[str(row.get("node_id"))].append(row)
        covered = [rows[0] for _, rows in sorted(by_node.items()) if rows]
        covered_ids = {id(row) for row in covered}
        remainder = [row for row in alternatives if id(row) not in covered_ids]
        covered.sort(key=lambda row: _alternative_sort_key(row, focus))
        remainder.sort(key=lambda row: _alternative_sort_key(row, focus))
        return findings, [*covered, *remainder]

    def intervene_assignment(
        self,
        trace_id: str | None = None,
        assignment_patch: Mapping[str, object] | str | None = None,
    ) -> str:
        """Evaluate a temporary node-assignment patch against one trace."""
        trace = self._find_trace(trace_id) if isinstance(trace_id, str) else None
        if trace is None:
            return self._error("unknown trace_id", trace_id=trace_id)
        assert isinstance(trace_id, str)
        patch = self._mapping(assignment_patch, "assignment_patch")
        if patch is None:
            return self._error("assignment_patch must be a mapping")
        # Also accept one compact patch record: {node_id, configuration_id,
        # device_id}.  The canonical form remains {node_id: { ... }}.
        if {"node_id", "configuration_id", "device_id"} <= set(patch):
            patch = {str(patch["node_id"]): patch}
        return self._evaluate_patch(trace_id, trace, patch)

    def validate_intervention(
        self,
        interventions: Mapping[str, object] | str | None = None,
    ) -> str:
        """Run explicit patches across the supplied traces and summarize deltas."""
        requested = self._mapping(interventions, "interventions")
        if requested is None:
            return self._error("interventions must map trace_id to assignment patches")
        results: list[object] = []
        deltas: dict[str, list[float]] = defaultdict(list)
        for trace_id, patch in requested.items():
            if not isinstance(trace_id, str) or not isinstance(patch, Mapping | str):
                result: object = {
                    "trace_id": str(trace_id),
                    "error": "invalid trace intervention",
                }
            else:
                trace = self._find_trace(trace_id)
                if trace is None:
                    result = {"trace_id": trace_id, "error": "unknown trace_id"}
                else:
                    normalized_patch = self._mapping(patch, "assignment_patch")
                    if normalized_patch is None:
                        result = {
                            "trace_id": trace_id,
                            "error": "assignment_patch must be a mapping",
                        }
                    else:
                        if {"node_id", "configuration_id", "device_id"} <= set(
                            normalized_patch
                        ):
                            normalized_patch = {
                                str(normalized_patch["node_id"]): normalized_patch
                            }
                        rendered = self._evaluate_patch(
                            trace_id, trace, normalized_patch
                        )
                        result = json.loads(rendered)
            if isinstance(result, Mapping):
                for name, value in _mapping_value(result, "delta").items():
                    if _finite(value):
                        deltas[name].append(float(value))
            results.append(result)
        summary = {
            "trace_count": len(results),
            "evaluated_count": sum(
                1
                for result in results
                if isinstance(result, Mapping) and "delta" in result
            ),
            "delta": {
                name: round(statistics.fmean(values), 6)
                for name, values in sorted(deltas.items())
            },
        }
        return self._render({"results": results, "summary": summary})

    def _evaluate_patch(
        self,
        trace_id: str,
        trace: object,
        patch: Mapping[str, object],
    ) -> str:
        if self.snapshot is None:
            return self._error(
                "snapshot is required for intervention", trace_id=trace_id
            )
        dag_value = _trace_value(trace, "dag", {})
        try:
            dag = _as_dag(dag_value)
        except (KeyError, TypeError, ValueError) as exc:
            return self._error("invalid trace DAG", trace_id=trace_id, detail=str(exc))
        if dag is None:
            return self._error("trace DAG is required", trace_id=trace_id)
        baseline_raw = _trace_value(trace, "assignments", {})
        if not isinstance(baseline_raw, Mapping):
            return self._error("trace assignments must be a mapping", trace_id=trace_id)
        baseline: dict[str, NodeAssignment] = {}
        for node_id, value in baseline_raw.items():
            assignment = _as_assignment(value)
            if not isinstance(node_id, str) or assignment is None:
                return self._error(
                    "invalid baseline assignment",
                    trace_id=trace_id,
                    node_id=str(node_id),
                )
            baseline[node_id] = assignment
        node_ids = {node.node_id for node in dag.nodes}
        if set(baseline) != node_ids:
            return self._error(
                "baseline assignments must cover the DAG", trace_id=trace_id
            )
        new_assignments = dict(baseline)
        for node_id, value in patch.items():
            if not isinstance(node_id, str) or node_id not in node_ids:
                return self._error(
                    "unknown node assignment", trace_id=trace_id, node_id=str(node_id)
                )
            assignment = _as_assignment(value)
            if assignment is None:
                return self._error(
                    "assignment must contain configuration_id and device_id",
                    trace_id=trace_id,
                    node_id=node_id,
                )
            new_assignments[node_id] = assignment
        try:
            baseline_report = evaluate_assignments(
                self.snapshot,
                dag,
                baseline,
                scoring_context=self.scoring_context,
            )
            new_report = evaluate_assignments(
                self.snapshot,
                dag,
                new_assignments,
                scoring_context=self.scoring_context,
            )
        except (KeyError, TypeError, ValueError) as exc:
            return self._error(
                "intervention evaluation failed", trace_id=trace_id, detail=str(exc)
            )
        baseline_metrics = _report_metrics(baseline_report)
        new_metrics = _report_metrics(new_report)
        delta: dict[str, float] = {}
        for name in baseline_metrics.keys() & new_metrics.keys():
            baseline_value = baseline_metrics[name]
            new_value = new_metrics[name]
            if _finite(baseline_value) and _finite(new_value):
                delta[name] = round(float(new_value) - float(baseline_value), 6)
        return self._render(
            {
                "trace_id": trace_id,
                "baseline": {
                    "status": baseline_report.scheduler_status,
                    "metrics": baseline_metrics,
                    "validation_errors": list(baseline_report.validation_errors),
                },
                "new": {
                    "status": new_report.scheduler_status,
                    "metrics": new_metrics,
                    "validation_errors": list(new_report.validation_errors),
                },
                "delta": delta,
            }
        )

    def _find_trace(self, trace_id: str) -> object | None:
        for trace in _trace_items(self.document):
            if str(_trace_value(trace, "trace_id", "")) == trace_id:
                return trace
        return None

    def _mapping(
        self, value: Mapping[str, object] | Sequence[object] | str | None, name: str
    ) -> Mapping[str, object] | None:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return None
        return value if isinstance(value, Mapping) else None

    def _render(self, value: object) -> str:
        safe_value = _json_safe(value)
        rendered = json.dumps(
            safe_value, ensure_ascii=False, separators=(",", ":"), default=str
        )
        if len(rendered) <= self.max_output_chars:
            return rendered
        if isinstance(safe_value, Mapping) and _is_trace_profile(safe_value):
            # Preserve the stable tool contract even when the transport budget
            # is smaller than the complete profile.  The detailed arrays are
            # compacted and the omitted counter remains explicit; callers do
            # not have to special-case a ``{"truncated": ...}`` response.
            omitted = dict(safe_value.get("omitted", {})) if isinstance(safe_value.get("omitted"), Mapping) else {}
            omitted["render_budget"] = len(rendered)
            critical = safe_value.get("critical_path", {})
            critical = critical if isinstance(critical, Mapping) else {}
            compact = {
                "trace_id": safe_value.get("trace_id"),
                "focus": safe_value.get("focus"),
                "scoring_context": safe_value.get("scoring_context", {}),
                "metrics": safe_value.get("metrics", {}),
                "critical_path": {
                    "nodes": _first_items(critical.get("nodes")),
                    "activities": _first_items(critical.get("activities")),
                    "total_makespan_ms": critical.get("total_makespan_ms"),
                    "contributions": _first_items(critical.get("contributions")),
                    "critical_timeline": critical.get("critical_timeline", True),
                    "approximation": critical.get("approximation", True),
                    "approximation_reasons": _first_items(critical.get("approximation_reasons")),
                },
                "ranked_hotspots": _first_items(safe_value.get("ranked_hotspots")),
                "placement_findings": _first_items(safe_value.get("placement_findings")),
                "configuration_alternatives": _first_items(safe_value.get("configuration_alternatives")),
                "bottleneck_hypotheses": _first_items(safe_value.get("bottleneck_hypotheses")),
                "evidence_refs": _first_items(safe_value.get("evidence_refs")),
                "omitted": omitted,
            }
            return json.dumps(compact, ensure_ascii=False, separators=(",", ":"), default=str)
        preview_size = max(0, self.max_output_chars - 96)
        return json.dumps(
            {
                "truncated": True,
                "output_chars": len(rendered),
                "preview": rendered[:preview_size],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def _error(self, message: str, **details: object) -> str:
        return self._render({"error": message, **details})


def _is_trace_profile(value: Mapping[str, object]) -> bool:
    return {
        "trace_id",
        "focus",
        "metrics",
        "critical_path",
        "ranked_hotspots",
        "placement_findings",
        "configuration_alternatives",
        "bottleneck_hypotheses",
        "evidence_refs",
        "omitted",
    }.issubset(value)


def _first_items(value: object) -> list[object]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return list(value[:1])
    return []


def _trace_items(document: object) -> tuple[object, ...]:
    if isinstance(document, Mapping):
        traces = document.get("traces")
        if isinstance(traces, Sequence) and not isinstance(traces, (str, bytes)):
            return tuple(traces)
        return (document,) if "trace_id" in document else ()
    if isinstance(document, Sequence) and not isinstance(document, (str, bytes)):
        return tuple(document)
    return ()


def _trace_value(trace: object, name: str, default: object = None) -> object:
    if isinstance(trace, Mapping):
        return trace.get(name, default)
    return getattr(trace, name, default)


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, NodeAssignment):
        return {
            "configuration_id": value.configuration_id,
            "device_id": value.device_id,
        }
    if isinstance(value, ToolCallPlan):
        return {
            "nodes": _json_safe(value.nodes),
            "final_outputs": _json_safe(value.final_outputs),
        }
    if hasattr(value, "__dataclass_fields__"):
        return {
            name: _json_safe(getattr(value, name))
            for name in value.__dataclass_fields__  # type: ignore[attr-defined]
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _mapping_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    rows: list[dict[str, object]] = []
    for item in value:
        safe = _json_safe(item)
        if isinstance(safe, Mapping):
            rows.append({str(key): child for key, child in safe.items()})
    return rows


def _as_dag(value: object) -> ToolCallPlan | None:
    if isinstance(value, ToolCallPlan):
        return value
    safe = _json_safe(value)
    if isinstance(safe, Mapping) and "nodes" in safe and "final_outputs" in safe:
        return plan_from_dict(dict(safe))
    return None


def _as_assignment(value: object) -> NodeAssignment | None:
    if isinstance(value, NodeAssignment):
        return value
    safe = _json_safe(value)
    if not isinstance(safe, Mapping):
        return None
    configuration_id = safe.get("configuration_id")
    device_id = safe.get("device_id")
    if isinstance(configuration_id, str) and isinstance(device_id, str):
        return NodeAssignment(configuration_id, device_id)
    return None


def _matches(value: Mapping[str, object], filters: Mapping[str, object]) -> bool:
    return all(value.get(str(key)) == expected for key, expected in filters.items())


def _mapping_value(value: Mapping[object, object], key: str) -> Mapping[str, object]:
    nested = value.get(key, {})
    return nested if isinstance(nested, Mapping) else {}


def _report_metrics(report: object) -> dict[str, object]:
    names = (
        "accuracy",
        "latency",
        "resource",
        "simulated_makespan_ms",
        "scheduler_solving_time_ms",
        "composite_score",
    )
    return {
        name: value
        for name in names
        if (value := getattr(report, name, None)) is not None
    }


def _number(value: object) -> float | None:
    return float(value) if _finite(value) else None


def _trace_metrics(trace: object) -> dict[str, object]:
    raw = _trace_value(trace, "metrics", {})
    if not isinstance(raw, Mapping):
        return {}
    names = (
        "accuracy",
        "latency",
        "resource",
        "simulated_makespan_ms",
        "scheduler_solving_time_ms",
        "composite_score",
    )
    result: dict[str, object] = {}
    for name in names:
        value = raw.get(name)
        if _finite(value):
            result[name] = value
    return result


def _dag_edges(
    dag: ToolCallPlan | None, dag_nodes: Sequence[Mapping[str, object]]
) -> tuple[tuple[str, str], ...]:
    edges: set[tuple[str, str]] = set()
    if dag is not None:
        for node in dag.nodes:
            for source in node.inputs.values():
                if source.kind == "node":
                    edges.add((source.name, node.node_id))
    else:
        for node in dag_nodes:
            destination = node.get("node_id")
            inputs = node.get("inputs")
            if not isinstance(destination, str) or not isinstance(inputs, Mapping):
                continue
            for source in inputs.values():
                if not isinstance(source, Mapping):
                    continue
                if source.get("kind") == "node" and isinstance(source.get("name"), str):
                    edges.add((str(source["name"]), destination))
    return tuple(sorted(edges))


def _transfer_latency_for(
    transfers: Sequence[Mapping[str, object]], source: str, destination: str
) -> float:
    values = [
        _number(item.get("latency_ms"))
        for item in transfers
        if item.get("source_node_id") == source and item.get("destination_node_id") == destination
    ]
    return max((value for value in values if value is not None), default=0.0)


def _critical_path(
    nodes: Sequence[Mapping[str, object]],
    transfers: Sequence[Mapping[str, object]],
    edges: Sequence[tuple[str, str]],
) -> dict[str, object]:
    """Reconstruct the observed critical timeline from activity timestamps.

    The simulator serializes more than the DAG itself: request/final transfers,
    per-device execution queues, and directional transfer links can all delay
    an activity.  We therefore build the observed activity graph first and
    walk backwards from the latest-finishing activity, selecting the latest
    predecessor that could have constrained its start.  This is intentionally
    a timeline reconstruction, not a claim about unobserved causal effects.
    """
    node_by_id = {
        str(node.get("node_id")): node
        for node in nodes
        if isinstance(node.get("node_id"), str)
    }
    activities: dict[str, dict[str, object]] = {}
    by_node_activity: dict[str, str] = {}
    approximation_reasons: list[str] = []
    for node_id, node in sorted(node_by_id.items()):
        start = _number(node.get("start_ms"))
        finish = _number(node.get("finish_ms"))
        if start is None or finish is None or finish < start:
            approximation_reasons.append(f"node:{node_id} has incomplete timestamps")
            continue
        activity_id = f"node:{node_id}"
        by_node_activity[node_id] = activity_id
        activities[activity_id] = {
            "activity_id": activity_id,
            "kind": "execution",
            "node_id": node_id,
            "start_ms": start,
            "finish_ms": finish,
            "contribution_ms": round(finish - start, 6),
            "device_id": node.get("device_id"),
        }

    transfer_activity_ids: list[tuple[str, Mapping[str, object]]] = []
    for index, transfer in enumerate(transfers):
        start = _number(transfer.get("start_ms"))
        finish = _number(transfer.get("finish_ms"))
        if start is None or finish is None or finish < start:
            approximation_reasons.append(f"transfer:{index} has incomplete timestamps")
            continue
        source = str(transfer.get("source_node_id"))
        destination = str(transfer.get("destination_node_id"))
        activity_id = f"transfer:{index}:{source}->{destination}"
        activity = {
            "activity_id": activity_id,
            "kind": "communication",
            "edge": [transfer.get("source_node_id"), transfer.get("destination_node_id")],
            "source_node_id": transfer.get("source_node_id"),
            "destination_node_id": transfer.get("destination_node_id"),
            "source_device_id": transfer.get("source_device_id"),
            "destination_device_id": transfer.get("destination_device_id"),
            "start_ms": start,
            "finish_ms": finish,
            "contribution_ms": round(finish - start, 6),
        }
        activities[activity_id] = activity
        transfer_activity_ids.append((activity_id, transfer))

    incoming: defaultdict[str, set[str]] = defaultdict(set)

    def add_edge(source: str | None, destination: str) -> None:
        if source is not None and source in activities and destination in activities:
            incoming[destination].add(source)

    # A transfer is constrained by the producing execution, and request or
    # inter-node transfers constrain the receiving execution.
    for activity_id, transfer in transfer_activity_ids:
        source_node = transfer.get("source_node_id")
        destination_node = transfer.get("destination_node_id")
        if isinstance(source_node, str) and source_node in by_node_activity:
            add_edge(by_node_activity[source_node], activity_id)
        if isinstance(destination_node, str) and destination_node in by_node_activity:
            add_edge(activity_id, by_node_activity[destination_node])

    # DAG dependencies are either represented by an observed transfer or,
    # for same-device execution, directly by the predecessor execution.
    transfer_by_edge: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for activity_id, transfer in transfer_activity_ids:
        source_node = transfer.get("source_node_id")
        destination_node = transfer.get("destination_node_id")
        if isinstance(source_node, str) and isinstance(destination_node, str):
            transfer_by_edge[(source_node, destination_node)].append(activity_id)
    for source, destination in edges:
        source_activity = by_node_activity.get(source)
        destination_activity = by_node_activity.get(destination)
        if source_activity is None or destination_activity is None:
            approximation_reasons.append(f"DAG edge {source}->{destination} lacks activity")
            continue
        matching_transfers = transfer_by_edge.get((source, destination), ())
        if matching_transfers:
            for transfer_id in matching_transfers:
                add_edge(transfer_id, destination_activity)
        elif node_by_id[source].get("device_id") == node_by_id[destination].get("device_id"):
            add_edge(source_activity, destination_activity)
        else:
            approximation_reasons.append(
                f"cross-device DAG edge {source}->{destination} has no observed transfer"
            )
            add_edge(source_activity, destination_activity)

    # Infer the immediate predecessor in each serialized device queue from the
    # observed execution timestamps.
    by_device: defaultdict[str, list[str]] = defaultdict(list)
    for activity_id, activity in activities.items():
        if activity.get("kind") == "execution" and isinstance(activity.get("device_id"), str):
            by_device[str(activity["device_id"])].append(activity_id)
    for activity_ids in by_device.values():
        activity_ids.sort(
            key=lambda item: (
                _number(activities[item].get("start_ms")) or 0.0,
                _number(activities[item].get("finish_ms")) or 0.0,
                item,
            )
        )
        for previous, current in zip(activity_ids, activity_ids[1:]):
            if (_number(activities[previous].get("finish_ms")) or 0.0) <= (
                _number(activities[current].get("start_ms")) or 0.0
            ) + 1e-6:
                add_edge(previous, current)

    # Infer the immediate predecessor in each serialized directional link.
    by_link: defaultdict[tuple[object, object], list[str]] = defaultdict(list)
    for activity_id, transfer in transfer_activity_ids:
        link = (transfer.get("source_device_id"), transfer.get("destination_device_id"))
        by_link[link].append(activity_id)
    for activity_ids in by_link.values():
        activity_ids.sort(
            key=lambda item: (
                _number(activities[item].get("start_ms")) or 0.0,
                _number(activities[item].get("finish_ms")) or 0.0,
                item,
            )
        )
        for previous, current in zip(activity_ids, activity_ids[1:]):
            if (_number(activities[previous].get("finish_ms")) or 0.0) <= (
                _number(activities[current].get("start_ms")) or 0.0
            ) + 1e-6:
                add_edge(previous, current)

    finish_values = [
        _number(activity.get("finish_ms")) or 0.0 for activity in activities.values()
    ]
    total = max(finish_values, default=0.0)
    if not activities:
        approximation_reasons.append("no complete execution or transfer activity")
        return {
            "nodes": [],
            "activities": [],
            "total_makespan_ms": 0.0,
            "contributions": [],
            "critical_timeline": True,
            "approximation": True,
            "approximation_reasons": approximation_reasons,
        }

    # If an activity starts after all known predecessors finish, its idle time
    # is observable but its cause is not fully represented in the evidence.
    for activity_id, activity in activities.items():
        start = _number(activity.get("start_ms")) or 0.0
        known_finish = max(
            (_number(activities[pred].get("finish_ms")) or 0.0 for pred in incoming.get(activity_id, ())),
            default=0.0,
        )
        if not incoming.get(activity_id) and start > 1e-6:
            approximation_reasons.append(f"{activity_id} has an unexplained initial wait")
        elif incoming.get(activity_id) and start > known_finish + 1e-6:
            approximation_reasons.append(f"{activity_id} includes unexplained idle time")

    current_id = max(
        activities,
        key=lambda item: (
            _number(activities[item].get("finish_ms")) or 0.0,
            item,
        ),
    )
    reverse_chain: list[str] = []
    visited: set[str] = set()
    while current_id not in visited:
        visited.add(current_id)
        reverse_chain.append(current_id)
        start = _number(activities[current_id].get("start_ms")) or 0.0
        feasible = [
            predecessor
            for predecessor in incoming.get(current_id, ())
            if (_number(activities[predecessor].get("finish_ms")) or 0.0) <= start + 1e-6
        ]
        if not feasible:
            break
        current_id = max(
            feasible,
            key=lambda item: (
                _number(activities[item].get("finish_ms")) or 0.0,
                item,
            ),
        )
    chain_ids = list(reversed(reverse_chain))
    chain_activities = [activities[item] for item in chain_ids]
    contributions: list[dict[str, object]] = []
    for activity in chain_activities:
        contribution = _number(activity.get("contribution_ms")) or 0.0
        item = {
            "activity_id": activity["activity_id"],
            "kind": activity["kind"],
            "contribution_ms": contribution,
        }
        if activity.get("kind") == "execution":
            item["node_id"] = activity.get("node_id")
        else:
            item["edge"] = activity.get("edge")
        item["share"] = round(contribution / total if total > 0.0 else 0.0, 6)
        contributions.append(item)

    return {
        "nodes": [
            activity.get("node_id")
            for activity in chain_activities
            if activity.get("kind") == "execution"
        ],
        "activities": chain_activities,
        "total_makespan_ms": round(total, 6),
        "contributions": contributions,
        "critical_timeline": True,
        "approximation": bool(approximation_reasons),
        "approximation_reasons": list(dict.fromkeys(approximation_reasons)),
    }


def _ranked_hotspots(
    nodes: Sequence[Mapping[str, object]],
    transfers: Sequence[Mapping[str, object]],
    focus: str | None,
    top_k: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for node in nodes:
        latency = _number(node.get("observed_latency_ms"))
        if latency is None:
            latency = _number(node.get("profile_latency_ms"))
        resource = _number(node.get("gpu_memory_mib"))
        quality = _number(node.get("normalized_quality_lcb"))
        if latency is None and resource is None and quality is None:
            continue
        rows.append({
            "kind": "execution",
            "node_id": node.get("node_id"),
            "tool_id": node.get("tool_id"),
            "latency_ms": latency,
            "gpu_memory_mib": resource,
            "normalized_quality_lcb": quality,
        })
    for transfer in transfers:
        latency = _number(transfer.get("latency_ms"))
        if latency is None:
            continue
        rows.append({
            "kind": "communication",
            "edge": [transfer.get("source_node_id"), transfer.get("destination_node_id")],
            "latency_ms": latency,
            "source_device_id": transfer.get("source_device_id"),
            "destination_device_id": transfer.get("destination_device_id"),
        })

    def key(row: Mapping[str, object]) -> tuple[float, float, str]:
        kind = row.get("kind")
        if focus == "communication":
            priority = 0.0 if kind == "communication" else 1.0
            value = _number(row.get("latency_ms")) or 0.0
        elif focus == "resource":
            priority = 0.0 if kind == "execution" else 1.0
            value = _number(row.get("gpu_memory_mib")) or 0.0
        elif focus == "quality":
            priority = 0.0 if kind == "execution" else 1.0
            value = -(_number(row.get("normalized_quality_lcb")) or 0.0)
        else:
            priority = 0.0
            value = _number(row.get("latency_ms")) or 0.0
        return (priority, -value, str(row.get("node_id", row.get("edge", ""))))

    rows.sort(key=key)
    return rows


def _alternative_sort_key(row: Mapping[str, object], focus: str | None) -> tuple[float, float, float, str]:
    if focus == "quality":
        primary = -(_number(row.get("normalized_quality_lcb")) or 0.0)
        secondary = _number(row.get("profile_latency_ms")) or 0.0
    elif focus == "resource":
        primary = _number(row.get("gpu_memory_mib")) or 0.0
        secondary = _number(row.get("profile_latency_ms")) or 0.0
    elif focus == "latency" or focus == "communication" or focus == "placement":
        primary = _number(row.get("profile_latency_ms")) or 0.0
        secondary = -(_number(row.get("normalized_quality_lcb")) or 0.0)
    else:
        primary = _number(row.get("profile_latency_ms")) or 0.0
        secondary = _number(row.get("gpu_memory_mib")) or 0.0
    return (0.0 if bool(row.get("dominates_current")) else 1.0, primary, secondary, f"{row.get('configuration_id')}:{row.get('device_id')}")


def _hypotheses(
    hotspots: Sequence[Mapping[str, object]],
    placement: Sequence[Mapping[str, object]],
    alternatives: Sequence[Mapping[str, object]],
    focus: str | None,
    top_k: int,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for row in hotspots:
        if row.get("kind") == "communication":
            edge = row.get("edge", [])
            result.append({
                "kind": "communication",
                "hypothesis": "A cross-device transfer is a measured latency hotspot; placement may be worth testing.",
                "evidence_refs": [f"transfer:{edge[0]}->{edge[1]}" if isinstance(edge, list) and len(edge) == 2 else "transfer"],
                "requires_intervention": True,
            })
        elif row.get("node_id") is not None:
            result.append({
                "kind": "execution",
                "hypothesis": "This node has a measured execution hotspot; an assignment change is a candidate to test.",
                "evidence_refs": [f"node:{row['node_id']}"],
                "requires_intervention": True,
            })
    for row in placement:
        if row.get("dominated"):
            current = row.get("current", {})
            result.append({
                "kind": "placement",
                "hypothesis": "The current assignment is dominated by a profiled alternative on latency, Resource, and Quality Profile; validate it with intervention.",
                "evidence_refs": [f"node:{row.get('node_id')}", f"alternative:{row.get('node_id')}"],
                "requires_intervention": True,
                "node_id": row.get("node_id"),
                "current": current,
            })
    # Preserve deterministic ordering while preventing duplicate node spam.
    seen: set[tuple[object, object]] = set()
    unique: list[dict[str, object]] = []
    for item in result:
        raw_refs = item.get("evidence_refs", ())
        refs = tuple(raw_refs) if isinstance(raw_refs, Sequence) and not isinstance(raw_refs, str | bytes) else ()
        key = (item.get("kind"), refs)
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _evidence_refs(
    trace_id: str,
    critical: Mapping[str, object],
    hotspots: Sequence[Mapping[str, object]],
    placement: Sequence[Mapping[str, object]],
    snapshot: ProfilingDatabaseSnapshot | None,
) -> list[str]:
    refs = [f"trace:{trace_id}"]
    raw_nodes = critical.get("nodes", ())
    if isinstance(raw_nodes, Sequence) and not isinstance(raw_nodes, str | bytes):
        for node_id in raw_nodes:
            refs.append(f"node:{node_id}")
    for row in hotspots:
        if row.get("kind") == "communication":
            edge = row.get("edge")
            if isinstance(edge, list) and len(edge) == 2:
                refs.append(f"transfer:{edge[0]}->{edge[1]}")
        elif row.get("node_id") is not None:
            refs.append(f"node:{row['node_id']}")
    for row in placement:
        if row.get("node_id") is not None:
            refs.append(f"placement:{row['node_id']}")
    if snapshot is not None:
        refs.append(f"snapshot:{snapshot.snapshot_digest}")
    return list(dict.fromkeys(refs))
