"""Deterministic schedule simulation against a validated profiling snapshot."""

from __future__ import annotations

from typing import Any, Mapping

from ..domain import ToolCallPlan
from ..profiling.snapshot import ProfilingDatabaseSnapshot
from .models import NodeAssignment, SimulatedNode, SimulatedTransfer, TrustedEvaluationError

END_DEVICE_ID = "device"
FIXED_DATA_SIZE_BYTES = {"text": 256, "image": 262_144, "audio": 1_048_576}


def transfer_latency(transfer: Any, size: int) -> float:
    return transfer.propagation_delay_ms + 1000.0 * size / transfer.bandwidth_bytes_per_second


def _request_data_type(source: Any) -> str:
    return source.data_type or "text"


def predecessors(dag: ToolCallPlan) -> dict[str, tuple[str, ...]]:
    result = {node.node_id: [] for node in dag.nodes}
    for node in dag.nodes:
        for source in node.inputs.values():
            if source.kind == "node":
                result[node.node_id].append(source.name)
    return {key: tuple(value) for key, value in result.items()}


def simulate(
    snapshot: ProfilingDatabaseSnapshot,
    dag: ToolCallPlan,
    assignments: Mapping[str, NodeAssignment],
) -> tuple[tuple[SimulatedNode, ...], tuple[SimulatedTransfer, ...]]:
    """Simulate execution in canonical node-id order.

    Device queues and directional links are serialized.  This deterministic
    policy is part of the evaluation contract, not a scheduler-provided fact.
    """
    graph = predecessors(dag)
    by_id = {node.node_id: node for node in dag.nodes}
    device_free = {device.device_id: 0.0 for device in snapshot.devices()}
    link_free: dict[tuple[str, str], float] = {}
    finish: dict[str, float] = {}
    transfer_ready = {node.node_id: 0.0 for node in dag.nodes}
    simulated: dict[str, SimulatedNode] = {}
    transfers: list[SimulatedTransfer] = []
    remaining = set(graph)
    while remaining:
        ready = [node_id for node_id in remaining if all(pred in finish for pred in graph[node_id])]
        if not ready:
            raise TrustedEvaluationError("DAG contains a cycle or unknown predecessor")
        node_id = min(ready)
        node = by_id[node_id]
        assignment = assignments[node_id]
        for source in node.inputs.values():
            if source.kind != "request" or assignment.device_id == END_DEVICE_ID:
                continue
            size = FIXED_DATA_SIZE_BYTES[_request_data_type(source)]
            transfer = snapshot.transfer_profile(END_DEVICE_ID, assignment.device_id)
            latency = transfer_latency(transfer, size)
            link = (END_DEVICE_ID, assignment.device_id)
            start = link_free.get(link, 0.0)
            finish_time = start + latency
            link_free[link] = finish_time
            transfer_ready[node_id] = max(transfer_ready[node_id], finish_time)
            transfers.append(SimulatedTransfer(END_DEVICE_ID, node_id, END_DEVICE_ID, assignment.device_id, start, finish_time, latency))
        start = max(device_free[assignment.device_id], transfer_ready[node_id])
        profile = snapshot.execution_profile(node.tool_id, assignment.configuration_id, assignment.device_id)
        finish_time = start + profile.warm_latency_p95_ms
        simulated[node_id] = SimulatedNode(
            node_id,
            assignment.configuration_id,
            assignment.device_id,
            start,
            finish_time,
            profile.gpu_memory_mib,
        )
        finish[node_id] = finish_time
        device_free[assignment.device_id] = finish_time
        for child_id, child_predecessors in graph.items():
            if node_id not in child_predecessors:
                continue
            child_assignment = assignments[child_id]
            if assignment.device_id == child_assignment.device_id:
                transfer_ready[child_id] = max(transfer_ready[child_id], finish_time)
                continue
            transfer = snapshot.transfer_profile(assignment.device_id, child_assignment.device_id)
            size = snapshot.representative_output_bytes(node.tool_id, assignment.configuration_id)
            latency = transfer_latency(transfer, size)
            link = (assignment.device_id, child_assignment.device_id)
            start = max(finish_time, link_free.get(link, 0.0))
            transfer_finish = start + latency
            link_free[link] = transfer_finish
            transfer_ready[child_id] = max(transfer_ready[child_id], transfer_finish)
            transfers.append(SimulatedTransfer(node_id, child_id, assignment.device_id, child_assignment.device_id, start, transfer_finish, latency))
        remaining.remove(node_id)
    for output in dag.final_outputs:
        assignment = assignments[output.node_id]
        if assignment.device_id == END_DEVICE_ID:
            continue
        transfer = snapshot.transfer_profile(assignment.device_id, END_DEVICE_ID)
        size = snapshot.representative_output_bytes(by_id[output.node_id].tool_id, assignment.configuration_id)
        latency = transfer_latency(transfer, size)
        link = (assignment.device_id, END_DEVICE_ID)
        start = max(finish[output.node_id], link_free.get(link, 0.0))
        transfer_finish = start + latency
        link_free[link] = transfer_finish
        transfers.append(SimulatedTransfer(output.node_id, END_DEVICE_ID, assignment.device_id, END_DEVICE_ID, start, transfer_finish, latency))
    return tuple(simulated[node_id] for node_id in sorted(simulated)), tuple(transfers)


# Explicit aliases make migration from the former private seam painless.
_simulate = simulate
_predecessors = predecessors
_transfer_latency = transfer_latency
