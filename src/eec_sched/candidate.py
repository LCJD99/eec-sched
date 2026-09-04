"""Scheduler Candidate and Trace contracts.

Candidate records are data contracts, not the evolution engine.  Keeping them
here lets workflow loading and evaluation users depend on a small stable
module without importing the (larger) search implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping, Sequence

from .domain import FinalOutput, ToolCallPlan, ToolNode
from .evaluation.models import EvaluationReport, NodeAssignment
from .evaluation.scoring import ScoringContext

SCHEMA_VERSION = "v1"
SchedulerProposal = Mapping[str, Any] | Sequence[Any]


def _readonly(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _readonly(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_readonly(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_readonly(item) for item in value)
    return value


def readonly_dag(dag: ToolCallPlan) -> ToolCallPlan:
    """Copy a plan so candidate code cannot mutate trusted workflow input."""
    return ToolCallPlan(
        tuple(ToolNode(node.node_id, node.tool_id, MappingProxyType(dict(node.inputs))) for node in dag.nodes),
        tuple(FinalOutput(output.node_id, output.port) for output in dag.final_outputs),
    )


@dataclass(frozen=True)
class SchedulerView:
    dag: ToolCallPlan
    scoring_context: ScoringContext
    snapshot_digest: str
    snapshot_evidence: Mapping[str, Any]


@dataclass(frozen=True)
class SchedulerCandidate:
    scheduler_version: int
    propose: Callable[[SchedulerView], SchedulerProposal]
    source_code: str = "def propose(view):\n    raise NotImplementedError\n"
    parent_scheduler_versions: tuple[int, ...] = ()
    strategy_description: str = "legacy scheduler strategy"
    reflection_context: str | None = None
    reflection_feedback: str | None = None
    coding_context: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.scheduler_version, int) or isinstance(self.scheduler_version, bool) or self.scheduler_version <= 0:
            raise ValueError("Scheduler Candidate version must be a positive integer")
        if not self.source_code.strip():
            raise ValueError("Scheduler Candidate source code must not be empty")
        try:
            compile(self.source_code, f"scheduler-candidate-{self.scheduler_version}", "exec")
        except SyntaxError as exc:
            raise ValueError("Scheduler Candidate source code must be executable Python") from exc
        if not self.strategy_description.strip():
            raise ValueError("Scheduler Strategy Description must not be empty")
        if len(set(self.parent_scheduler_versions)) != len(self.parent_scheduler_versions):
            raise ValueError("Parent Scheduler Versions must be unique")
        if any(not isinstance(version, int) or isinstance(version, bool) or version <= 0 for version in self.parent_scheduler_versions):
            raise ValueError("Parent Scheduler Versions must be positive integers")


@dataclass(frozen=True)
class SchedulerCandidateDraft:
    propose: Callable[[SchedulerView], SchedulerProposal]
    source_code: str
    strategy_description: str
    reflection_context: str | None = None
    reflection_feedback: str | None = None
    coding_context: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not self.source_code.strip():
            raise ValueError("Scheduler Candidate source code must not be empty")
        try:
            compile(self.source_code, "scheduler-candidate-draft", "exec")
        except SyntaxError as exc:
            raise ValueError("Scheduler Candidate source code must be executable Python") from exc
        if not self.strategy_description.strip():
            raise ValueError("Scheduler Strategy Description must not be empty")


@dataclass(frozen=True)
class SchedulerCandidateRegistry:
    candidates: Mapping[int, SchedulerCandidate]

    def __post_init__(self) -> None:
        candidates = dict(self.candidates)
        if any(version != candidate.scheduler_version for version, candidate in candidates.items()):
            raise ValueError("Scheduler Candidate registry keys must match Scheduler Versions")
        object.__setattr__(self, "candidates", MappingProxyType(candidates))

    def resolve(self, scheduler_version: int) -> SchedulerCandidate:
        try:
            return self.candidates[scheduler_version]
        except KeyError as exc:
            raise ValueError(f"unknown Scheduler Version: {scheduler_version}") from exc


@dataclass(frozen=True)
class EvaluationTrace:
    trace_id: str
    task_input: Mapping[str, object]
    dag: ToolCallPlan

    def __post_init__(self) -> None:
        if not self.trace_id:
            raise ValueError("Trace identifier must not be empty")
        object.__setattr__(self, "task_input", MappingProxyType(dict(self.task_input)))
        object.__setattr__(self, "dag", readonly_dag(self.dag))


@dataclass(frozen=True)
class TraceEvaluation:
    trace: EvaluationTrace
    assignments: Mapping[str, NodeAssignment]
    status: Literal["scored", "rejected", "failed"]
    score: float | None
    report: EvaluationReport
    reason: str | None = None
    scheduler_view: SchedulerView | None = None
    scheduler_computation_time_ms: float | None = None
    scheduler_version: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "assignments", MappingProxyType(dict(self.assignments)))

    @property
    def scheduler_proposal(self) -> Mapping[str, NodeAssignment]:
        return self.assignments

    @property
    def score_contribution(self) -> float:
        return self.score if self.score is not None else 0.0

    @property
    def raw_metrics(self) -> Mapping[str, object]:
        """Raw accuracy, latency, and GPU-memory values for this Trace."""
        return self.report.raw_metrics


@dataclass(frozen=True)
class CandidateEvaluation:
    schema_version: str
    scheduler_version: int
    traces: tuple[TraceEvaluation, ...]
    candidate_score: float | None

    def concise_projection(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "scheduler_version": self.scheduler_version,
            "traces": [_concise_trace(record) for record in self.traces],
            "candidate_score": self.candidate_score,
        }


@dataclass(frozen=True)
class FinalTraceComparison:
    trace_evaluation: TraceEvaluation
    oracle_reference_score: float | None
    difference: float | None


@dataclass(frozen=True)
class FinalEvaluation:
    candidate_evaluation: CandidateEvaluation
    traces: tuple[FinalTraceComparison, ...]
    average_candidate_score: float | None
    average_oracle_reference_score: float | None
    average_difference: float | None

    def concise_projection(self) -> dict[str, object]:
        return {
            "schema_version": self.candidate_evaluation.schema_version,
            "scheduler_version": self.candidate_evaluation.scheduler_version,
            "traces": [{**_concise_trace(item.trace_evaluation), "oracle_reference_score": item.oracle_reference_score, "difference": item.difference} for item in self.traces],
            "candidate_score": self.candidate_evaluation.candidate_score,
            "average_candidate_score": self.average_candidate_score,
            "average_oracle_reference_score": self.average_oracle_reference_score,
            "average_difference": self.average_difference,
        }


def _concise_trace(record: TraceEvaluation) -> dict[str, object]:
    metrics = dict(record.raw_metrics)
    if record.report.raw_accuracy_metrics:
        metrics["raw_accuracy_metrics"] = dict(record.report.raw_accuracy_metrics)
    result: dict[str, object] = {
        "trace_id": record.trace.trace_id,
        "status": record.status,
        "score": record.score_contribution,
        "metrics": metrics,
    }
    if record.reason is not None:
        result["reason"] = record.reason
    return result


__all__ = [
    "CandidateEvaluation", "EvaluationTrace", "FinalEvaluation", "FinalTraceComparison", "SCHEMA_VERSION",
    "ScoringContext", "SchedulerCandidate", "SchedulerCandidateDraft", "SchedulerCandidateRegistry",
    "SchedulerProposal", "SchedulerView", "TraceEvaluation", "readonly_dag",
]
