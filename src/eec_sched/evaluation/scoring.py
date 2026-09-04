"""Trusted, deterministic scoring policies for evaluation reports."""

from __future__ import annotations

from typing import Mapping

from ..domain import ToolCallPlan
from ..profiling.snapshot import ProfilingDatabaseSnapshot
from .models import NodeAssignment
from .simulator import predecessors

UTILITY_EPSILON = 1e-12


def accuracy(
    snapshot: ProfilingDatabaseSnapshot,
    dag: ToolCallPlan,
    assignments: Mapping[str, NodeAssignment],
) -> float:
    """Multiply quality lower confidence bounds on the output dependency cone."""
    graph = predecessors(dag)
    required = {output.node_id for output in dag.final_outputs}
    stack = list(required)
    while stack:
        current = stack.pop()
        for parent in graph[current]:
            if parent not in required:
                required.add(parent)
                stack.append(parent)
    by_id = {node.node_id: node for node in dag.nodes}
    result = 1.0
    for node_id in sorted(required):
        assignment = assignments[node_id]
        result *= snapshot.quality_profile(by_id[node_id].tool_id, assignment.configuration_id).normalized_quality_lcb
    return result


def execution_energy(
    snapshot: ProfilingDatabaseSnapshot,
    node: object,
    assignment: NodeAssignment,
) -> float:
    return snapshot.execution_profile(node.tool_id, assignment.configuration_id, assignment.device_id).mean_incremental_execution_energy_j  # type: ignore[attr-defined]


def normalized_performance(
    accuracy_lcb: float,
    latency_proxy_ms: float,
    *,
    minimum_accuracy: float,
    maximum_latency_ms: float,
    gamma: float,
) -> float:
    accuracy_surplus = 0.0 if minimum_accuracy == 1 else (accuracy_lcb - minimum_accuracy) / (1 - minimum_accuracy)
    latency_surplus = (maximum_latency_ms - latency_proxy_ms) / maximum_latency_ms
    return gamma * accuracy_surplus + (1 - gamma) * latency_surplus


def utility(performance: float, total_energy_j: float) -> float:
    return performance / (total_energy_j + UTILITY_EPSILON)


# Names used by early prototypes and useful to external adapters.
calculate_accuracy = accuracy
calculate_normalized_performance = normalized_performance
calculate_utility = utility
