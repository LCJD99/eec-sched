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
    gpu_memory_mib: float


@dataclass(frozen=True)
class SimulatedTransfer:
    source_node_id: str
    destination_node_id: str
    source_device_id: str
    destination_device_id: str
    start_ms: float
    finish_ms: float
    latency_ms: float


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
    accuracy: float | None = None
    raw_accuracy_metrics: Mapping[str, float] = MappingProxyType({})
    simulated_makespan_ms: float | None = None
    scheduler_solving_time_ms: float | None = None
    latency: float | None = None
    resource: float | None = None
    composite_score: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "assignments", MappingProxyType(dict(self.assignments)))
        object.__setattr__(self, "raw_accuracy_metrics", MappingProxyType(dict(self.raw_accuracy_metrics)))

    @property
    def raw_metrics(self) -> Mapping[str, object]:
        """The metric breakdown used to construct the composite score.

        ``accuracy`` is the normalized aggregate quality indicator used by the
        evaluator; the original per-node quality points are retained in
        ``raw_accuracy_metrics``. ``resource`` is measured from GPU memory.
        """
        values: dict[str, object] = {}
        if self.accuracy is not None:
            values["accuracy"] = self.accuracy
        if self.latency is not None:
            values["latency"] = self.latency
        if self.resource is not None:
            values["resource"] = self.resource
        if self.raw_accuracy_metrics:
            values["raw_accuracy_metrics"] = self.raw_accuracy_metrics
        return MappingProxyType(values)

    @property
    def gpu_memory(self) -> float | None:
        """The aggregate resource value, measured from GPU memory profiles."""
        return self.resource


class TrustedEvaluationError(ValueError):
    """An evaluator or profiling snapshot invariant was violated."""
