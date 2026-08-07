"""Independent, deterministic Issue #4 scheduler evaluation seam.

The scheduler is deliberately treated as an untrusted provider of opaque
``(configuration_id, device_id)`` choices.  All measurements and all timing
decisions in this module come from the validated profiling snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from time import perf_counter
from typing import Mapping, Protocol, Sequence, Any

from .domain import ToolCallPlan
from .profiling_database import ProfilingDatabaseSnapshot

UTILITY_EPSILON = 1e-12


class Scheduler(Protocol):
    def schedule(self, dag: ToolCallPlan) -> Mapping[str, Any] | Sequence[Any]: ...


@dataclass(frozen=True)
class NodeAssignment:
    configuration_id: str
    device_id: str


@dataclass(frozen=True)
class SimulatedNode:
    node_id: str
    configuration_id: str
    device_id: str
    start_ms: float
    finish_ms: float


@dataclass(frozen=True)
class SimulatedTransfer:
    source_node_id: str
    destination_node_id: str
    source_device_id: str
    destination_device_id: str
    start_ms: float
    finish_ms: float
    latency_ms: float
    energy_j: float


@dataclass(frozen=True)
class EvaluationReport:
    """Immutable result, including validation failures instead of trusting a scheduler."""

    snapshot_digest: str
    scheduler_status: str
    validation_errors: tuple[str, ...]
    assignments: Mapping[str, NodeAssignment]
    nodes: tuple[SimulatedNode, ...] = ()
    transfers: tuple[SimulatedTransfer, ...] = ()
    accuracy_lcb: float | None = None
    simulated_makespan_ms: float | None = None
    scheduler_solving_time_ms: float | None = None
    latency_proxy_ms: float | None = None
    compute_energy_j: float | None = None
    communication_energy_j: float | None = None
    incremental_execution_energy_j: float | None = None
    accuracy_feasible: bool | None = None
    latency_feasible: bool | None = None
    feasible: bool = False
    normalized_performance: float | None = None
    utility: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "assignments", MappingProxyType(dict(self.assignments)))


class TrustedEvaluationError(ValueError):
    pass


def evaluate_scheduler_instance(
    snapshot: ProfilingDatabaseSnapshot,
    dag: ToolCallPlan,
    scheduler: Scheduler | Any,
    *,
    minimum_accuracy: float,
    maximum_latency_ms: float,
    gamma: float,
    scheduler_solving_time_ms: float | None = None,
) -> EvaluationReport:
    """Run and independently evaluate one scheduler output against one snapshot."""
    errors: list[str] = []
    if not 0 <= minimum_accuracy <= 1:
        errors.append("minimum_accuracy must be in [0, 1]")
    if maximum_latency_ms <= 0:
        errors.append("maximum_latency_ms must be positive")
    if not 0 < gamma < 1:
        errors.append("gamma must be in (0, 1)")
    if scheduler_solving_time_ms is not None and scheduler_solving_time_ms < 0:
        errors.append("scheduler_solving_time_ms must be non-negative")
    if errors:
        return _invalid(snapshot, *errors)
    node_ids = tuple(node.node_id for node in dag.nodes)
    if len(set(node_ids)) != len(node_ids):
        errors.append("duplicate node_id in DAG")
    started = perf_counter()
    try:
        result = scheduler.schedule(dag) if hasattr(scheduler, "schedule") else scheduler(dag)
    except Exception as exc:  # scheduler is intentionally untrusted
        return _invalid(snapshot, f"scheduler_exception: {type(exc).__name__}: {exc}")
    measured_scheduler_time_ms = (perf_counter() - started) * 1000.0
    effective_scheduler_time_ms = measured_scheduler_time_ms if scheduler_solving_time_ms is None else scheduler_solving_time_ms
    if effective_scheduler_time_ms < 0:
        return _invalid(snapshot, "scheduler_solving_time_ms must be non-negative")
    assignments = _parse_assignments(result, node_ids, errors)
    if errors:
        return _invalid(snapshot, *errors, assignments=assignments)
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
        return _invalid(snapshot, *errors, assignments=assignments)

    try:
        nodes, transfers = _simulate(snapshot, dag, assignments)
        accuracy = _accuracy(snapshot, dag, assignments)
    except (KeyError, TrustedEvaluationError) as exc:
        return _invalid(snapshot, f"simulation_error: {exc}", assignments=assignments)
    makespan = max((n.finish_ms for n in nodes), default=0.0)
    compute_energy = sum(_execution_energy(snapshot, node, assignments[node.node_id]) for node in dag.nodes)
    communication_energy = sum(t.energy_j for t in transfers)
    total_energy = compute_energy + communication_energy
    accuracy_ok = accuracy >= minimum_accuracy
    latency_proxy = effective_scheduler_time_ms + makespan
    latency_ok = latency_proxy <= maximum_latency_ms
    feasible = accuracy_ok and latency_ok
    performance = None
    utility = None
    if feasible:
        accuracy_surplus = 0.0 if minimum_accuracy == 1 else (accuracy - minimum_accuracy) / (1 - minimum_accuracy)
        latency_surplus = (maximum_latency_ms - latency_proxy) / maximum_latency_ms
        performance = gamma * accuracy_surplus + (1 - gamma) * latency_surplus
        utility = performance / (total_energy + UTILITY_EPSILON)
    status = "scheduled" if feasible else "infeasible"
    return EvaluationReport(
        snapshot.snapshot_digest, status, (), assignments, nodes, transfers, accuracy,
        makespan, effective_scheduler_time_ms, latency_proxy, compute_energy, communication_energy,
        total_energy, accuracy_ok, latency_ok, feasible, performance, utility,
    )


def _invalid(snapshot: ProfilingDatabaseSnapshot, *errors: str, assignments: Mapping[str, NodeAssignment] = ()) -> EvaluationReport:
    return EvaluationReport(snapshot.snapshot_digest, "rejected", tuple(errors), assignments)


def _parse_assignments(result: Any, node_ids: tuple[str, ...], errors: list[str]) -> dict[str, NodeAssignment]:
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


def _predecessors(dag: ToolCallPlan) -> dict[str, tuple[str, ...]]:
    result = {node.node_id: [] for node in dag.nodes}
    for node in dag.nodes:
        for source in node.inputs.values():
            if source.kind == "node":
                result[node.node_id].append(source.name)
    return {key: tuple(value) for key, value in result.items()}


def _simulate(snapshot: ProfilingDatabaseSnapshot, dag: ToolCallPlan, assignments: Mapping[str, NodeAssignment]) -> tuple[tuple[SimulatedNode, ...], tuple[SimulatedTransfer, ...]]:
    predecessors = _predecessors(dag)
    by_id = {node.node_id: node for node in dag.nodes}
    device_free: dict[str, float] = {device.device_id: 0.0 for device in snapshot.devices()}
    link_free: dict[tuple[str, str], float] = {}
    finish: dict[str, float] = {}
    transfer_ready: dict[str, float] = {node.node_id: 0.0 for node in dag.nodes}
    simulated: dict[str, SimulatedNode] = {}
    transfers: list[SimulatedTransfer] = []
    remaining = set(by_id)
    while remaining:
        ready = [node_id for node_id in remaining if all(pred in finish for pred in predecessors[node_id])]
        if not ready:
            raise TrustedEvaluationError("DAG contains a cycle or unknown predecessor")
        node_id = min(ready)
        node = by_id[node_id]
        assignment = assignments[node_id]
        start = max(device_free[assignment.device_id], transfer_ready[node_id])
        profile = snapshot.execution_profile(node.tool_id, assignment.configuration_id, assignment.device_id)
        finish_time = start + profile.warm_latency_p95_ms
        simulated[node_id] = SimulatedNode(node_id, assignment.configuration_id, assignment.device_id, start, finish_time)
        finish[node_id] = finish_time
        device_free[assignment.device_id] = finish_time
        for child_id, preds in predecessors.items():
            if node_id not in preds:
                continue
            child_assignment = assignments[child_id]
            if assignment.device_id == child_assignment.device_id:
                transfer_ready[child_id] = max(transfer_ready[child_id], finish_time)
                continue
            transfer = snapshot.transfer_profile(assignment.device_id, child_assignment.device_id)
            size = snapshot.representative_output_bytes(node.tool_id, assignment.configuration_id)
            latency = transfer.propagation_delay_ms + 1000.0 * size / transfer.bandwidth_bytes_per_second
            energy = transfer.setup_energy_j + size * transfer.energy_per_byte_j
            link = (assignment.device_id, child_assignment.device_id)
            transfer_start = max(finish_time, link_free.get(link, 0.0))
            transfer_finish = transfer_start + latency
            link_free[link] = transfer_finish
            transfer_ready[child_id] = max(transfer_ready[child_id], transfer_finish)
            transfers.append(SimulatedTransfer(node_id, child_id, assignment.device_id, child_assignment.device_id, transfer_start, transfer_finish, latency, energy))
        remaining.remove(node_id)
    return tuple(simulated[node_id] for node_id in sorted(simulated)), tuple(transfers)


def _execution_energy(snapshot: ProfilingDatabaseSnapshot, node: Any, assignment: NodeAssignment) -> float:
    return snapshot.execution_profile(node.tool_id, assignment.configuration_id, assignment.device_id).mean_incremental_execution_energy_j


def _accuracy(snapshot: ProfilingDatabaseSnapshot, dag: ToolCallPlan, assignments: Mapping[str, NodeAssignment]) -> float:
    parents = _predecessors(dag)
    required: set[str] = set(output.node_id for output in dag.final_outputs)
    stack = list(required)
    while stack:
        current = stack.pop()
        for parent in parents[current]:
            if parent not in required:
                required.add(parent)
                stack.append(parent)
    result = 1.0
    by_id = {node.node_id: node for node in dag.nodes}
    for node_id in sorted(required):
        choice = assignments[node_id]
        result *= snapshot.quality_profile(by_id[node_id].tool_id, choice.configuration_id).normalized_quality_lcb
    return result


# Short alias for callers that prefer the noun used in the issue.
evaluate = evaluate_scheduler_instance
