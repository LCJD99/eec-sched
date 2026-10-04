"""Trusted, deterministic scoring policies for evaluation reports."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

from ..domain import ToolCallPlan, ToolNode
from ..profiling.snapshot import ProfilingDatabaseSnapshot
from .models import NodeAssignment
from .simulator import predecessors


@dataclass(frozen=True)
class ScoringContext:
    """Weights and fixed scales for the three-metric composite score.

    The context contains no workload SLA or feasibility gate. Accuracy is
    maximized, while latency and resource (GPU memory) are minimized. The
    scales only make the latter two raw measurements dimensionless; they do
    not describe limits that a proposal must satisfy. ``beta`` is the
    provisional bonus per additional node on the longest selected DAG path.
    """

    accuracy_weight: float = 1.0 / 3.0
    latency_weight: float = 1.0 / 3.0
    resource_weight: float = 1.0 / 3.0
    latency_scale_ms: float = 100.0
    resource_scale_mib: float = 8192.0
    beta: float = 0.01

    def __post_init__(self) -> None:
        weights = (
            self.accuracy_weight,
            self.latency_weight,
            self.resource_weight,
        )
        if any(not isfinite(weight) or weight < 0.0 for weight in weights):
            raise ValueError("composite score weights must be finite and non-negative")
        if sum(weights) <= 0.0:
            raise ValueError("at least one composite score weight must be positive")
        if not isfinite(self.latency_scale_ms) or self.latency_scale_ms <= 0.0:
            raise ValueError("latency_scale_ms must be finite and positive")
        if not isfinite(self.resource_scale_mib) or self.resource_scale_mib <= 0.0:
            raise ValueError("resource_scale_mib must be finite and positive")
        if not isfinite(self.beta) or self.beta < 0.0:
            raise ValueError("beta must be finite and non-negative")

    @property
    def normalized_weights(self) -> tuple[float, float, float]:
        total = self.accuracy_weight + self.latency_weight + self.resource_weight
        return (
            self.accuracy_weight / total,
            self.latency_weight / total,
            self.resource_weight / total,
        )


def accuracy(
    snapshot: ProfilingDatabaseSnapshot,
    dag: ToolCallPlan,
    assignments: Mapping[str, NodeAssignment],
    scoring_context: ScoringContext | None = None,
) -> float:
    """Score all selected DAG nodes with a mean quality and path bonus.

    The provisional policy is ``min(1, mean(q_v) + beta * (L - 1))`` where
    ``L`` is the node count on the longest dependency path in the selected
    DAG.  It deliberately uses every selected node, including disconnected
    nodes, as the quality evidence for this evaluation stage.
    """
    context = scoring_context or ScoringContext()
    graph = predecessors(dag)
    by_id = {node.node_id: node for node in dag.nodes}
    qualities = []
    for node_id in sorted(by_id):
        assignment = assignments[node_id]
        qualities.append(
            snapshot.quality_profile(
                by_id[node_id].tool_id, assignment.configuration_id
            ).normalized_quality_lcb
        )
    if not qualities:
        return 0.0
    path_lengths: dict[str, int] = {}

    def longest_path(node_id: str) -> int:
        if node_id in path_lengths:
            return path_lengths[node_id]
        path_lengths[node_id] = 1 + max(
            (longest_path(parent) for parent in graph[node_id]),
            default=0,
        )
        return path_lengths[node_id]

    longest = max(longest_path(node_id) for node_id in by_id)
    return min(1.0, sum(qualities) / len(qualities) + context.beta * (longest - 1))


def raw_accuracy_metrics(
    snapshot: ProfilingDatabaseSnapshot,
    dag: ToolCallPlan,
    assignments: Mapping[str, NodeAssignment],
) -> Mapping[str, float]:
    """Return raw quality metrics for every selected DAG node."""
    by_id = {node.node_id: node for node in dag.nodes}
    return {
        node_id: float(
            snapshot.quality_profile(
                by_id[node_id].tool_id, assignments[node_id].configuration_id
            ).raw_metric["point_estimate"]
        )
        for node_id in sorted(by_id)
    }


def execution_resource(
    snapshot: ProfilingDatabaseSnapshot,
    node: ToolNode,
    assignment: NodeAssignment,
) -> float:
    """Return one node's profiled GPU-memory resource value in MiB."""
    return snapshot.execution_profile(
        node.tool_id, assignment.configuration_id, assignment.device_id
    ).gpu_memory_mib


def resource(
    snapshot: ProfilingDatabaseSnapshot,
    dag: ToolCallPlan,
    assignments: Mapping[str, NodeAssignment],
) -> float:
    """Return aggregate GPU-memory resource in MiB for a Trace."""
    return sum(
        execution_resource(snapshot, node, assignments[node.node_id])
        for node in dag.nodes
    )


def composite_score(
    accuracy_value: float,
    latency_ms: float,
    resource_value: float,
    scoring_context: ScoringContext | None = None,
) -> float:
    """Combine accuracy, latency, and GPU-memory resource into one score.

    The score is a weighted arithmetic mean of three dimensionless
    indicators. Accuracy is already the trusted normalized DAG quality. The
    lower-is-better metrics use a reciprocal response relative to fixed
    scales, so a candidate is never accepted or rejected by a threshold.
    """
    context = scoring_context or ScoringContext()
    values = (accuracy_value, latency_ms, resource_value)
    if any(not isfinite(value) for value in values):
        raise ValueError("composite score metrics must be finite")
    if latency_ms < 0.0:
        raise ValueError("latency_ms must not be negative")
    if resource_value < 0.0:
        raise ValueError("resource_value must not be negative")
    accuracy_indicator = max(0.0, min(1.0, accuracy_value))
    latency_indicator = 1.0 / (1.0 + latency_ms / context.latency_scale_ms)
    resource_indicator = 1.0 / (1.0 + resource_value / context.resource_scale_mib)
    accuracy_weight, latency_weight, resource_weight = context.normalized_weights
    return (
        accuracy_weight * accuracy_indicator
        + latency_weight * latency_indicator
        + resource_weight * resource_indicator
    )


# Names used by early adapters, retained where they already match the current
# metric vocabulary.
calculate_accuracy = accuracy
calculate_resource = resource
calculate_composite_score = composite_score
