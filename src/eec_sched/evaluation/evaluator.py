"""Canonical trusted evaluation orchestration.

The evaluator owns proposal validation, timing, and score construction.  A
Scheduler is untrusted: it can select only values present in the snapshot and
cannot supply measurements or timing values itself.
"""

from __future__ import annotations

from math import isfinite
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


def _validate_assignments(
    snapshot: ProfilingDatabaseSnapshot,
    dag: ToolCallPlan,
    assignments: Mapping[str, NodeAssignment] | Any,
    errors: list[str],
) -> dict[str, NodeAssignment]:
    """Validate and copy a canonical assignment mapping.

    Scheduler output is normalized by :func:`parse_assignments` before it
    reaches this helper.  The direct assignment seam remains strict: callers
    must provide ``NodeAssignment`` values, so no caller-controlled mapping is
    interpreted as trusted configuration data.
    """
    if not isinstance(assignments, Mapping):
        errors.append("assignments must be a mapping")
        return {}

    node_ids = tuple(node.node_id for node in dag.nodes)
    known_node_ids = set(node_ids)
    validated: dict[str, NodeAssignment] = {}
    for node_id, value in assignments.items():
        if not isinstance(node_id, str) or node_id not in known_node_ids:
            errors.append(f"unknown node assignment {node_id!r}")
            continue
        if node_id in validated:
            errors.append(f"duplicate assignment for node {node_id!r}")
            continue
        if not isinstance(value, NodeAssignment):
            errors.append(f"node {node_id}: assignment must be a NodeAssignment")
            continue
        if not isinstance(value.configuration_id, str) or not isinstance(value.device_id, str):
            errors.append(f"node {node_id}: assignment must contain configuration_id and device_id")
            continue
        validated[node_id] = value

    missing = [node_id for node_id in node_ids if node_id not in validated]
    if missing:
        errors.append(f"missing node assignments: {missing}")

    by_id = {node.node_id: node for node in dag.nodes}
    for node_id, assignment in validated.items():
        node = by_id[node_id]
        try:
            snapshot.configuration(node.tool_id, assignment.configuration_id)
            if assignment.device_id not in snapshot.compatible_devices(node.tool_id, assignment.configuration_id):
                errors.append(f"node {node_id}: incompatible device {assignment.device_id!r}")
            snapshot.execution_profile(node.tool_id, assignment.configuration_id, assignment.device_id)
        except KeyError as exc:
            errors.append(f"node {node_id}: unknown profile or choice ({exc.args[0]})")
    return validated


def _evaluate_assignments(
    snapshot: ProfilingDatabaseSnapshot,
    dag: ToolCallPlan,
    assignments: Mapping[str, NodeAssignment],
    *,
    scoring_context: ScoringContext,
    scheduler_computation_time_ms: float,
    simulator,
) -> EvaluationReport:
    """Implementation shared by scheduler and counterfactual evaluation."""
    errors: list[str] = []
    node_ids = tuple(node.node_id for node in dag.nodes)
    if len(set(node_ids)) != len(node_ids):
        errors.append("duplicate node_id in DAG")
    if not isfinite(scheduler_computation_time_ms) or scheduler_computation_time_ms < 0.0:
        errors.append("scheduler_computation_time_ms must be finite and non-negative")

    validated = _validate_assignments(snapshot, dag, assignments, errors)
    if errors:
        return _invalid(
            snapshot,
            *errors,
            assignments=validated,
            scheduler_solving_time_ms=scheduler_computation_time_ms,
        )

    from .models import TrustedEvaluationError

    try:
        nodes, transfers = simulator(snapshot, dag, validated)
        accuracy_value = accuracy(snapshot, dag, validated)
    except (KeyError, TrustedEvaluationError) as exc:
        raise TrustedEvaluationError(f"trusted simulation failed: {exc}") from exc
    makespan = max(
        (
            value
            for value in (
                *[node.finish_ms for node in nodes],
                *[transfer.finish_ms for transfer in transfers],
            )
        ),
        default=0.0,
    )
    latency_value = scheduler_computation_time_ms + makespan
    resource_value = resource(snapshot, dag, validated)
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
        assignments=validated,
        nodes=nodes,
        transfers=transfers,
        accuracy=accuracy_value,
        raw_accuracy_metrics=raw_accuracy_metrics(snapshot, dag, validated),
        simulated_makespan_ms=makespan,
        scheduler_solving_time_ms=scheduler_computation_time_ms,
        latency=latency_value,
        resource=resource_value,
        composite_score=composite,
    )


def evaluate_assignments(
    snapshot: ProfilingDatabaseSnapshot,
    dag: ToolCallPlan,
    assignments: Mapping[str, NodeAssignment],
    *,
    scoring_context: ScoringContext | None = None,
    scheduler_computation_time_ms: float = 0.0,
) -> EvaluationReport:
    """Evaluate a complete assignment using only trusted snapshot evidence.

    This is the replay seam for local assignment interventions: it does not
    execute a Scheduler Candidate.  Every assignment must cover the DAG and
    refer to a compatible, profiled Configuration/Device pair.  Invalid input
    returns a rejected report; trusted simulator failures remain errors.
    """
    return _evaluate_assignments(
        snapshot,
        dag,
        assignments,
        scoring_context=scoring_context or ScoringContext(),
        scheduler_computation_time_ms=scheduler_computation_time_ms,
        simulator=simulate,
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
    return _evaluate_assignments(
        snapshot,
        dag,
        assignments,
        scoring_context=scoring_context,
        scheduler_computation_time_ms=measured_time,
        simulator=_simulator,
    )


evaluate = evaluate_scheduler_instance
_parse_assignments = parse_assignments
