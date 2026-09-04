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
from .scoring import UTILITY_EPSILON, accuracy, execution_energy, normalized_performance, utility
from .simulator import END_DEVICE_ID, FIXED_DATA_SIZE_BYTES, simulate


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
    minimum_accuracy: float,
    maximum_latency_ms: float,
    gamma: float,
    _clock=None,
    _simulator=None,
) -> EvaluationReport:
    """Run and independently evaluate one scheduler output against a snapshot."""
    if _clock is None:
        _clock = perf_counter
    if _simulator is None:
        _simulator = simulate
    errors: list[str] = []
    if not 0 <= minimum_accuracy <= 1:
        errors.append("minimum_accuracy must be in [0, 1]")
    if maximum_latency_ms <= 0:
        errors.append("maximum_latency_ms must be positive")
    if not 0 < gamma < 1:
        errors.append("gamma must be in (0, 1)")
    if errors:
        return _invalid(snapshot, *errors)
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
        accuracy_lcb = accuracy(snapshot, dag, assignments)
    except (KeyError, TrustedEvaluationError) as exc:
        raise TrustedEvaluationError(f"trusted simulation failed: {exc}") from exc
    makespan = max((value for value in (*[node.finish_ms for node in nodes], *[transfer.finish_ms for transfer in transfers])), default=0.0)
    compute_energy = sum(execution_energy(snapshot, node, assignments[node.node_id]) for node in dag.nodes)
    communication_energy = sum(transfer.energy_j for transfer in transfers)
    total_energy = compute_energy + communication_energy
    latency_proxy = measured_time + makespan
    accuracy_ok = accuracy_lcb >= minimum_accuracy
    latency_ok = latency_proxy <= maximum_latency_ms
    performance = normalized_performance(accuracy_lcb, latency_proxy, minimum_accuracy=minimum_accuracy, maximum_latency_ms=maximum_latency_ms, gamma=gamma)
    return EvaluationReport(
        snapshot.snapshot_digest, "scheduled", (), assignments, nodes, transfers, accuracy_lcb,
        makespan, measured_time, latency_proxy, compute_energy, communication_energy,
        total_energy, accuracy_ok, latency_ok, accuracy_ok and latency_ok, performance,
        utility(performance, total_energy),
    )


evaluate = evaluate_scheduler_instance
_parse_assignments = parse_assignments
