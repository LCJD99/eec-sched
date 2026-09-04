"""Canonical trusted evaluation orchestration.

The evaluator owns proposal validation, timing, and score construction.  A
Scheduler is untrusted: it can select only values present in the snapshot and
cannot supply measurements or timing values itself.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Callable, Mapping, Protocol, Sequence, cast

from ..domain import ToolCallPlan
from ..profiling.snapshot import ProfilingDatabaseSnapshot
from .models import EvaluationReport, NodeAssignment
from .scoring import ScoringContext, accuracy, composite_score, raw_accuracy_metrics, resource
from .simulator import simulate


class Scheduler(Protocol):
    def schedule(self, dag: ToolCallPlan) -> Mapping[str, Any] | Sequence[Any]: ...


def parse_assignments(result: Any, node_ids: tuple[str, ...], errors: list[str]) -> dict[str, NodeAssignment]:
    parsed: dict[str, NodeAssignment] = {}
    if isinstance(result, Mapping):
        items = list(result.items())
    elif isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
        items = []
        for item in result:
            if isinstance(item, Mapping):
                items.append((item.get("node_id"), item))
            else:
                items.append((getattr(item, "node_id", None), item))
    else:
        errors.append("scheduler returned a non-collection")
        return parsed
    for node_id, value in items:
        if not isinstance(node_id, str) or node_id not in node_ids:
            errors.append(f"unknown node assignment {node_id!r}")
            continue
        if node_id in parsed:
            errors.append(f"duplicate assignment for node {node_id!r}")
            continue
        try:
            if isinstance(value, Mapping):
                config, device = value["configuration_id"], value["device_id"]
                if set(value) - {"configuration_id", "device_id", "node_id"}:
                    errors.append(f"node {node_id}: illegal extra assignment fields")
            else:
                config, device = value.configuration_id, value.device_id
            if not isinstance(config, str) or not isinstance(device, str):
                raise TypeError
            parsed[node_id] = NodeAssignment(config, device)
        except (KeyError, AttributeError, TypeError):
            errors.append(f"node {node_id}: assignment must contain configuration_id and device_id")
    missing = [node_id for node_id in node_ids if node_id not in parsed]
    if missing:
        errors.append(f"missing node assignments: {missing}")
    return parsed


def _invalid(
    snapshot: ProfilingDatabaseSnapshot,
    *errors: str,
    assignments: Mapping[str, NodeAssignment] | None = None,
    scheduler_solving_time_ms: float | None = None,
) -> EvaluationReport:
    return EvaluationReport(
        snapshot.snapshot_digest,
        "rejected",
        tuple(errors),
        assignments or {},
        scheduler_solving_time_ms=scheduler_solving_time_ms,
    )


def evaluate_scheduler_instance(
    snapshot: ProfilingDatabaseSnapshot,
    dag: ToolCallPlan,
    scheduler: Scheduler | Any,
    *,
    scoring_context: ScoringContext | None = None,
    _clock=None,
    _simulator=None,
) -> EvaluationReport:
    """Run and independently evaluate one scheduler output against a snapshot."""
    if _clock is None:
        _clock = perf_counter
    if _simulator is None:
        _simulator = simulate
    scoring_context = scoring_context or ScoringContext()
    errors: list[str] = []
    node_ids = tuple(node.node_id for node in dag.nodes)
    if len(set(node_ids)) != len(node_ids):
        errors.append("duplicate node_id in DAG")
    started = _clock()
    try:
        result = (
            scheduler.schedule(dag)
            if hasattr(scheduler, "schedule")
            else cast(Callable[[ToolCallPlan], object], scheduler)(dag)
        )
    except Exception as exc:
        return _invalid(snapshot, f"scheduler_exception: {type(exc).__name__}: {exc}", scheduler_solving_time_ms=(_clock() - started) * 1000.0)
    measured_time = (_clock() - started) * 1000.0
    assignments = parse_assignments(result, node_ids, errors)
    if errors:
        return _invalid(snapshot, *errors, assignments=assignments, scheduler_solving_time_ms=measured_time)
    for node in dag.nodes:
        choice = assignments[node.node_id]
        try:
            snapshot.configuration(node.tool_id, choice.configuration_id)
            if choice.device_id not in snapshot.compatible_devices(node.tool_id, choice.configuration_id):
                errors.append(f"node {node.node_id}: incompatible device {choice.device_id!r}")
            snapshot.execution_profile(node.tool_id, choice.configuration_id, choice.device_id)
        except KeyError as exc:
            errors.append(f"node {node.node_id}: unknown profile or choice ({exc.args[0]})")
    if errors:
        return _invalid(snapshot, *errors, assignments=assignments, scheduler_solving_time_ms=measured_time)
    from .models import TrustedEvaluationError
    try:
        nodes, transfers = _simulator(snapshot, dag, assignments)
        accuracy_value = accuracy(snapshot, dag, assignments)
    except (KeyError, TrustedEvaluationError) as exc:
        raise TrustedEvaluationError(f"trusted simulation failed: {exc}") from exc
    makespan = max((value for value in (*[node.finish_ms for node in nodes], *[transfer.finish_ms for transfer in transfers])), default=0.0)
    latency_value = measured_time + makespan
    resource_value = resource(snapshot, dag, assignments)
    composite = composite_score(
        accuracy_value,
        latency_value,
        resource_value,
        scoring_context,
    )
    return EvaluationReport(
        snapshot_digest=snapshot.snapshot_digest,
        scheduler_status="scheduled",
        validation_errors=(),
        assignments=assignments,
        nodes=nodes,
        transfers=transfers,
        accuracy=accuracy_value,
        raw_accuracy_metrics=raw_accuracy_metrics(snapshot, dag, assignments),
        simulated_makespan_ms=makespan,
        scheduler_solving_time_ms=measured_time,
        latency=latency_value,
        resource=resource_value,
        composite_score=composite,
    )


evaluate = evaluate_scheduler_instance
_parse_assignments = parse_assignments
