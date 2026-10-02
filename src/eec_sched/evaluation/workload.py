"""Shared-clock evaluation for a fixed sequence of arriving requests.

This module deliberately keeps workload replay separate from the existing
single-request evaluator.  A candidate is called once per arrival, while the
replay owns all resource queues and advances one deterministic event clock.
"""

from __future__ import annotations

import json
import signal
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from math import isfinite
from time import perf_counter
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence, cast

from ..domain import ToolCallPlan, ToolCallPlanCandidates, ToolNode
from ..profiling.snapshot import ProfilingDatabaseSnapshot
from .evaluator import (
    _validate_assignments,
    parse_assignments,
    parse_scheduler_output,
)
from .models import NodeAssignment, TrustedEvaluationError
from .scoring import ScoringContext
from .simulator import END_DEVICE_ID, FIXED_DATA_SIZE_BYTES, transfer_latency

if TYPE_CHECKING:
    from ..candidate import EvaluationTrace, SchedulerCandidate


_CANDIDATE_TIMEOUT_SECONDS = 5.0


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _operation_key(*parts: str) -> str:
    """Encode identity without collisions from delimiters in request or node IDs."""
    return json.dumps(parts, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class WorkloadNode:
    """Timeline evidence for one node, identified by request and node IDs."""

    request_id: str
    node_id: str
    configuration_id: str
    device_id: str
    start_ms: float | None
    finish_ms: float | None
    gpu_memory_mib: float

    @property
    def trace_id(self) -> str:
        return self.request_id


@dataclass(frozen=True)
class WorkloadTransfer:
    """Timeline evidence for one directional transfer."""

    request_id: str
    source_node_id: str
    destination_node_id: str
    source_device_id: str
    destination_device_id: str
    start_ms: float | None
    finish_ms: float | None
    latency_ms: float

    @property
    def trace_id(self) -> str:
        return self.request_id


@dataclass(frozen=True)
class WorkloadRequest:
    """Result for one arrived request."""

    trace_id: str
    arrival_time_ms: float
    status: Literal["completed", "rejected", "failed"]
    completion_time_ms: float | None
    latency_ms: float | None
    selected_path_index: int | None
    assignments: Mapping[str, NodeAssignment]
    nodes: tuple[WorkloadNode, ...]
    transfers: tuple[WorkloadTransfer, ...]
    scheduler_computation_time_ms: float | None
    reason: str | None = None

    @property
    def request_id(self) -> str:
        return self.trace_id

    def __post_init__(self) -> None:
        object.__setattr__(self, "assignments", MappingProxyType(dict(self.assignments)))


@dataclass(frozen=True)
class WorkloadEvaluation:
    """Raw evidence and scenario metrics for one observation window."""

    requests: tuple[WorkloadRequest, ...]
    observation_window_ms: float
    completed_within_window: int
    backlog_at_window_end: int
    throughput_per_second: float
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    snapshot_digest: str
    scheduler_version: int


@dataclass
class _Operation:
    key: str
    request_id: str
    kind: Literal["node", "transfer"]
    resource_key: tuple[str, ...]
    duration_ms: float
    ready_ms: float | None = None
    start_ms: float | None = None
    finish_ms: float | None = None
    ready: bool = False
    completed: bool = False
    node_state: "_NodeState | None" = None
    child_state: "_NodeState | None" = None
    output_transfer: bool = False
    source_node_state: "_NodeState | None" = None
    source_node_id: str = END_DEVICE_ID
    destination_node_id: str = END_DEVICE_ID
    gpu_memory_mib: float = 0.0


@dataclass
class _NodeState:
    request: "_RequestState"
    node: ToolNode
    assignment: NodeAssignment
    operation: _Operation
    parent_ids: set[str] = field(default_factory=set)
    transfer_ids: set[str] = field(default_factory=set)
    completed: bool = False
    transfer_done: set[str] = field(default_factory=set)


@dataclass
class _RequestState:
    trace: EvaluationTrace
    arrival_time_ms: float
    selected_dag: ToolCallPlan | None = None
    selected_path_index: int | None = None
    assignments: dict[str, NodeAssignment] = field(default_factory=dict)
    scheduler_time_ms: float | None = None
    status: Literal["completed", "rejected", "failed"] | None = None
    completion_time_ms: float | None = None
    reason: str | None = None
    nodes: dict[str, _NodeState] = field(default_factory=dict)
    operations: dict[str, _Operation] = field(default_factory=dict)
    final_transfer_ids: set[str] = field(default_factory=set)
    final_transfer_done: set[str] = field(default_factory=set)


class _Resource:
    """One serialized device or directed-link resource."""

    def __init__(self, key: tuple[str, ...]) -> None:
        self.key = key
        self.pending: dict[str, _Operation] = {}
        self.queue: list[_Operation] = []
        self.running: _Operation | None = None

    def add(self, operation: _Operation, *, ready: bool, now: float) -> None:
        self.pending[operation.key] = operation
        if ready:
            self.mark_ready(operation, now)

    def mark_ready(self, operation: _Operation, now: float) -> None:
        if operation.completed or operation.ready:
            return
        operation.ready = True
        operation.ready_ms = now
        self.queue.append(operation)

    def start(self, now: float) -> None:
        if self.running is not None:
            return
        if self.queue:
            self.queue.sort(key=lambda operation: (operation.ready_ms or 0.0, operation.key))
            operation = self.queue.pop(0)
            operation.start_ms = now
            operation.finish_ms = now + operation.duration_ms
            self.running = operation

    def complete_if_due(self, now: float) -> _Operation | None:
        operation = self.running
        if operation is None or operation.finish_ms is None or operation.finish_ms > now:
            return None
        self.running = None
        self.pending.pop(operation.key, None)
        operation.completed = True
        operation.finish_ms = now
        return operation

    def state(self, now: float) -> Mapping[str, object]:
        running = self.running
        remaining = 0.0
        if running is not None and running.finish_ms is not None:
            remaining = max(0.0, running.finish_ms - now)
        committed = remaining
        for operation in self.pending.values():
            if operation is not running:
                committed += operation.duration_ms
        return {
            "busy": running is not None,
            "queue_depth": len(self.queue),
            "remaining_ms": remaining,
            "committed_work_ms": committed,
        }


class _Replay:
    def __init__(self, snapshot: ProfilingDatabaseSnapshot) -> None:
        self.snapshot = snapshot
        self.now = 0.0
        self.devices = {
            device.device_id: _Resource(("device", device.device_id))
            for device in snapshot.devices()
        }
        self.links: dict[tuple[str, str], _Resource] = {}
        for profile in snapshot.transfer_profiles():
            self.links[(profile.source_device_id, profile.destination_device_id)] = _Resource(
                ("link", profile.source_device_id, profile.destination_device_id)
            )
        self.requests: dict[str, _RequestState] = {}

    def resource(self, resource_key: tuple[str, ...]) -> _Resource:
        if resource_key[0] == "device":
            return self.devices[resource_key[1]]
        return self.links.setdefault(
            (resource_key[1], resource_key[2]), _Resource(resource_key)
        )

    def system_state(self) -> Mapping[str, object]:
        return _freeze(
            {
                "time_ms": self.now,
                "devices": {
                    device_id: resource.state(self.now)
                    for device_id, resource in sorted(self.devices.items())
                },
                "links": {
                    f"{source}->{destination}": resource.state(self.now)
                    for (source, destination), resource in sorted(self.links.items())
                },
            }
        )

    def _add_operation(self, operation: _Operation, *, ready: bool) -> None:
        self.resource(operation.resource_key).add(operation, ready=ready, now=self.now)

    def _mark_ready(self, operation: _Operation) -> None:
        self.resource(operation.resource_key).mark_ready(operation, self.now)

    def _maybe_ready_node(self, state: _NodeState) -> None:
        if state.completed or state.operation.ready or state.operation.completed:
            return
        if not all(state.request.nodes[parent].completed for parent in state.parent_ids):
            return
        if state.transfer_ids - state.transfer_done:
            return
        self._mark_ready(state.operation)

    def _try_complete_request(self, request: _RequestState) -> None:
        if request.status != "completed" or request.completion_time_ms is not None:
            return
        if not all(state.completed for state in request.nodes.values()):
            return
        if request.final_transfer_ids - request.final_transfer_done:
            return
        request.completion_time_ms = self.now

    def _complete_operation(self, operation: _Operation) -> None:
        request = self.requests[operation.request_id]
        if operation.kind == "transfer":
            if operation.output_transfer:
                request.final_transfer_done.add(operation.key)
            elif operation.child_state is not None:
                operation.child_state.transfer_done.add(operation.key)
            return

        assert operation.node_state is not None
        state = operation.node_state
        state.completed = True
        for operation_candidate in request.operations.values():
            if operation_candidate.output_transfer and operation_candidate.source_node_state is state:
                self._mark_ready(operation_candidate)
        for child in request.nodes.values():
            if state.node.node_id not in child.parent_ids:
                continue
            for operation_candidate in request.operations.values():
                if (
                    operation_candidate.child_state is child
                    and operation_candidate.kind == "transfer"
                    and operation_candidate.source_node_state is state
                ):
                    self._mark_ready(operation_candidate)

    def settle(self, now: float) -> None:
        self.now = now
        completed: list[_Operation] = []
        for resource in (*self.devices.values(), *self.links.values()):
            operation = resource.complete_if_due(now)
            if operation is not None:
                completed.append(operation)
        # Mark every completion before looking for newly-ready nodes.  This
        # makes simultaneous parent and transfer completions deterministic.
        for operation in completed:
            self._complete_operation(operation)
        for request in self.requests.values():
            for state in request.nodes.values():
                self._maybe_ready_node(state)
            self._try_complete_request(request)
        self.start_ready()

    def start_ready(self) -> None:
        for resource in (*self.devices.values(), *self.links.values()):
            resource.start(self.now)

    def next_completion(self) -> float | None:
        times = [
            resource.running.finish_ms
            for resource in (*self.devices.values(), *self.links.values())
            if resource.running is not None and resource.running.finish_ms is not None
        ]
        return min(times) if times else None

    def add_request(self, request: _RequestState) -> None:
        """Build and commit a complete request graph atomically."""
        dag = request.selected_dag
        assert dag is not None
        by_id = {node.node_id: node for node in dag.nodes}
        if len(by_id) != len(dag.nodes):
            raise ValueError("duplicate node_id in selected DAG")
        for node in dag.nodes:
            assignment = request.assignments[node.node_id]
            profile = self.snapshot.execution_profile(
                node.tool_id, assignment.configuration_id, assignment.device_id
            )
            operation = _Operation(
                _operation_key(request.trace.trace_id, "node", node.node_id),
                request.trace.trace_id,
                "node",
                ("device", assignment.device_id),
                profile.warm_latency_p95_ms,
                gpu_memory_mib=profile.gpu_memory_mib,
            )
            state = _NodeState(request, node, assignment, operation)
            operation.node_state = state
            request.nodes[node.node_id] = state
            request.operations[operation.key] = operation

        # Every input occurrence is a separate transfer, matching simulator.py.
        for node in dag.nodes:
            child = request.nodes[node.node_id]
            for index, (input_name, source) in enumerate(node.inputs.items()):
                if source.kind == "request":
                    if child.assignment.device_id == END_DEVICE_ID:
                        continue
                    size = FIXED_DATA_SIZE_BYTES[source.data_type or "text"]
                    profile = self.snapshot.transfer_profile(END_DEVICE_ID, child.assignment.device_id)
                    operation = _Operation(
                        _operation_key(request.trace.trace_id, "ingress", node.node_id, input_name, str(index)),
                        request.trace.trace_id,
                        "transfer",
                        ("link", END_DEVICE_ID, child.assignment.device_id),
                        transfer_latency(profile, size),
                        child_state=child,
                        source_node_id=END_DEVICE_ID,
                        destination_node_id=node.node_id,
                    )
                    request.operations[operation.key] = operation
                    child.transfer_ids.add(operation.key)
                elif source.kind == "node":
                    if source.name not in request.nodes:
                        raise ValueError(f"unknown predecessor {source.name!r}")
                    parent = request.nodes[source.name]
                    child.parent_ids.add(parent.node.node_id)
                    if parent.assignment.device_id == child.assignment.device_id:
                        continue
                    size = self.snapshot.representative_output_bytes(
                        parent.node.tool_id, parent.assignment.configuration_id
                    )
                    profile = self.snapshot.transfer_profile(
                        parent.assignment.device_id, child.assignment.device_id
                    )
                    operation = _Operation(
                        _operation_key(request.trace.trace_id, "transfer", source.name, node.node_id, input_name, str(index)),
                        request.trace.trace_id,
                        "transfer",
                        ("link", parent.assignment.device_id, child.assignment.device_id),
                        transfer_latency(profile, size),
                        child_state=child,
                        source_node_state=parent,
                        source_node_id=source.name,
                        destination_node_id=node.node_id,
                    )
                    request.operations[operation.key] = operation
                    child.transfer_ids.add(operation.key)

        for index, output in enumerate(dag.final_outputs):
            if output.node_id not in request.nodes:
                raise ValueError(f"unknown final output node {output.node_id!r}")
            node_state = request.nodes[output.node_id]
            if node_state.assignment.device_id == END_DEVICE_ID:
                continue
            size = self.snapshot.representative_output_bytes(
                node_state.node.tool_id, node_state.assignment.configuration_id
            )
            profile = self.snapshot.transfer_profile(node_state.assignment.device_id, END_DEVICE_ID)
            operation = _Operation(
                _operation_key(request.trace.trace_id, "egress", output.node_id, output.port, str(index)),
                request.trace.trace_id,
                "transfer",
                ("link", node_state.assignment.device_id, END_DEVICE_ID),
                transfer_latency(profile, size),
                output_transfer=True,
                source_node_state=node_state,
                source_node_id=output.node_id,
                destination_node_id=END_DEVICE_ID,
            )
            request.operations[operation.key] = operation
            request.final_transfer_ids.add(operation.key)

        self.requests[request.trace.trace_id] = request
        for operation in request.operations.values():
            # Ingress transfers are ready at arrival.  Other transfers become
            # ready after their source node completes.
            ready = (
                operation.kind == "transfer"
                and operation.source_node_state is None
                and operation.child_state is not None
            )
            self._add_operation(operation, ready=ready)
        for state in request.nodes.values():
            self._maybe_ready_node(state)
        if not request.nodes:
            request.completion_time_ms = self.now
        self.start_ready()

    def unfinished_requests(self) -> list[_RequestState]:
        return [
            request
            for request in self.requests.values()
            if request.status == "completed" and request.completion_time_ms is None
        ]


def _scheduler_evidence(snapshot: ProfilingDatabaseSnapshot, dags: Sequence[ToolCallPlan]) -> Mapping[str, object]:
    tool_ids = {node.tool_id for dag in dags for node in dag.nodes}
    tools = tuple(tool for tool in snapshot.data["tools"] if tool["tool_id"] in tool_ids)
    return _freeze(
        {
            "devices": snapshot.data["devices"],
            "tools": tools,
            "transfer_profiles": snapshot.data["transfer_profiles"],
        }
    )


@contextmanager
def _candidate_timeout() -> Any:
    """Keep the existing five-second trusted candidate boundary."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, _CANDIDATE_TIMEOUT_SECONDS)

    def raise_timeout(signum: int, frame: Any) -> None:
        raise TimeoutError("Scheduler Candidate did not return before the trusted time limit")

    signal.signal(signal.SIGALRM, raise_timeout)
    try:
        yield
    finally:
        signal.signal(signal.SIGALRM, previous_handler)
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _make_request_record(request: _RequestState) -> WorkloadRequest:
    nodes = tuple(
        WorkloadNode(
            request.trace.trace_id,
            state.node.node_id,
            state.assignment.configuration_id,
            state.assignment.device_id,
            state.operation.start_ms,
            state.operation.finish_ms,
            # This profile lookup happened while building the operation.  A
            # missing profile would have failed before any resource was added.
            state.operation.gpu_memory_mib,
        )
        for state in sorted(request.nodes.values(), key=lambda item: item.node.node_id)
        if state.operation.start_ms is not None
    )
    transfers = tuple(
        WorkloadTransfer(
            request.trace.trace_id,
            operation.source_node_id,
            operation.destination_node_id,
            operation.resource_key[1],
            operation.resource_key[2],
            operation.start_ms,
            operation.finish_ms,
            operation.duration_ms,
        )
        for operation in sorted(request.operations.values(), key=lambda item: item.key)
        if operation.kind == "transfer" and operation.start_ms is not None
    )
    completion = request.completion_time_ms
    latency = None if completion is None else completion - request.arrival_time_ms
    status = request.status or "failed"
    return WorkloadRequest(
        request.trace.trace_id,
        request.arrival_time_ms,
        status,
        completion,
        latency,
        request.selected_path_index,
        request.assignments,
        nodes,
        transfers,
        request.scheduler_time_ms,
        request.reason,
    )


def _set_failed(request: _RequestState, reason: str) -> None:
    request.status = "failed"
    request.reason = reason
    request.completion_time_ms = None


def evaluate_workload(
    snapshot: ProfilingDatabaseSnapshot,
    arrivals: Sequence[tuple[float, EvaluationTrace]],
    candidate: SchedulerCandidate,
    *,
    observation_window_ms: float,
    scoring_context: ScoringContext | None = None,
) -> WorkloadEvaluation:
    """Replay arrivals through one candidate using shared non-preemptive queues."""
    # Imported lazily because candidate.py imports evaluation.models through
    # the evaluation package during package initialization.
    from ..candidate import EvaluationTrace, SchedulerView

    if not isfinite(observation_window_ms) or observation_window_ms <= 0.0:
        raise ValueError("observation_window_ms must be finite and positive")
    normalized: list[tuple[float, EvaluationTrace]] = []
    identifiers: set[str] = set()
    for item in arrivals:
        try:
            arrival, trace = item
        except (TypeError, ValueError) as exc:
            raise ValueError("arrivals must contain (arrival_time_ms, EvaluationTrace) pairs") from exc
        if isinstance(arrival, bool):
            raise ValueError("arrival_time_ms must be finite and within the observation window")
        arrival = float(arrival)
        if not isfinite(arrival) or not 0.0 <= arrival < observation_window_ms:
            raise ValueError("arrival_time_ms must satisfy 0 <= arrival_time_ms < observation_window_ms")
        if not isinstance(trace, EvaluationTrace):
            raise TypeError("arrivals must contain EvaluationTrace values")
        if trace.trace_id in identifiers:
            raise ValueError(f"Trace identifiers must be unique: {trace.trace_id!r}")
        identifiers.add(trace.trace_id)
        normalized.append((arrival, trace))
    normalized.sort(key=lambda item: (item[0], item[1].trace_id))

    context = scoring_context or ScoringContext()
    replay = _Replay(snapshot)
    index = 0
    while index < len(normalized) or replay.next_completion() is not None:
        next_arrival = normalized[index][0] if index < len(normalized) else None
        next_completion = replay.next_completion()
        if next_completion is not None and (next_arrival is None or next_completion <= next_arrival):
            replay.settle(next_completion)
            continue
        if next_arrival is None:
            break
        replay.now = next_arrival
        _, trace = normalized[index]
        request = _RequestState(trace, next_arrival)
        replay.requests[trace.trace_id] = request
        view = SchedulerView(
            trace.dags[0],
            context,
            snapshot.snapshot_digest,
            _scheduler_evidence(snapshot, trace.dags),
            tuple(trace.dags),
            replay.system_state(),
        )
        started = perf_counter()
        try:
            with _candidate_timeout():
                result = candidate.propose(view)
        except Exception as exc:
            request.scheduler_time_ms = (perf_counter() - started) * 1000.0
            _set_failed(request, f"scheduler_exception: {type(exc).__name__}: {exc}")
            index += 1
            continue
        request.scheduler_time_ms = (perf_counter() - started) * 1000.0
        errors: list[str] = []
        candidates = ToolCallPlanCandidates(trace.dags)
        selected_dag, raw_assignments, path_index = parse_scheduler_output(result, candidates, errors)
        request.selected_dag = selected_dag
        request.selected_path_index = path_index
        node_ids = tuple(node.node_id for node in selected_dag.nodes)
        if len(set(node_ids)) != len(node_ids):
            errors.append("duplicate node_id in selected DAG")
        request.assignments.update(parse_assignments(raw_assignments, node_ids, errors))
        if not errors:
            request.assignments.update(_validate_assignments(snapshot, selected_dag, request.assignments, errors))
        if errors:
            request.status = "rejected"
            request.reason = "; ".join(errors)
            index += 1
            continue
        request.status = "completed"
        try:
            replay.add_request(request)
        except (KeyError, ValueError) as exc:
            raise TrustedEvaluationError(f"trusted workload simulation failed: {exc}") from exc
        replay.start_ready()
        index += 1

    # Complete every accepted request after the observation window as well.
    while replay.next_completion() is not None:
        replay.settle(cast(float, replay.next_completion()))
    unfinished = replay.unfinished_requests()
    if unfinished:
        raise TrustedEvaluationError(
            "trusted workload simulation stalled for requests: "
            + ", ".join(request.trace.trace_id for request in unfinished)
        )

    records = tuple(_make_request_record(replay.requests[trace.trace_id]) for _, trace in normalized)
    completed_within_window = sum(
        1
        for record in records
        if record.status == "completed"
        and record.completion_time_ms is not None
        and record.completion_time_ms <= observation_window_ms
    )
    backlog_at_window_end = sum(
        1
        for record in records
        if record.status == "completed"
        and (record.completion_time_ms is None or record.completion_time_ms > observation_window_ms)
    )
    latencies = [
        cast(float, record.latency_ms)
        for record in records
        if record.status == "completed" and record.latency_ms is not None
    ]
    return WorkloadEvaluation(
        records,
        observation_window_ms,
        completed_within_window,
        backlog_at_window_end,
        completed_within_window / (observation_window_ms / 1000.0),
        _percentile(latencies, 0.50),
        _percentile(latencies, 0.95),
        snapshot.snapshot_digest,
        candidate.scheduler_version,
    )


__all__ = [
    "WorkloadEvaluation",
    "WorkloadNode",
    "WorkloadRequest",
    "WorkloadTransfer",
    "evaluate_workload",
]
