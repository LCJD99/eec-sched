"""Value objects returned by the trusted evaluation module.

These types intentionally contain no evaluation policy.  Keeping the report
types separate from the evaluator makes reports useful to alternative
simulators and scorers while preserving one stable serialization shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


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
    """Immutable trusted evidence for one scheduler proposal.

    A rejected proposal retains validation errors and any measured candidate
    runtime, but never receives a score.  ``assignments`` is copied and
    frozen so callers cannot mutate the evidence after evaluation.
    """

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
    """An evaluator or profiling snapshot invariant was violated."""

