"""Trusted evaluation flow for versioned Scheduler Candidates.

This module deliberately records planning, scheduling, and evaluation as
separate Trace stages.  It never executes tools or exposes a mutable snapshot
to untrusted Scheduler Candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence

from .domain import FinalOutput, ToolCallPlan, ToolNode
from .profiling_database import ProfilingDatabaseSnapshot
from .trusted_evaluation import EvaluationReport, NodeAssignment, evaluate_scheduler_instance

SCHEMA_VERSION = "v1"


def _readonly(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _readonly(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_readonly(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_readonly(item) for item in value)
    return value


def _readonly_dag(dag: ToolCallPlan) -> ToolCallPlan:
    """Copy the Planner DAG so Candidate code cannot alter trusted input."""
    return ToolCallPlan(
        tuple(ToolNode(node.node_id, node.tool_id, MappingProxyType(dict(node.inputs))) for node in dag.nodes),
        tuple(FinalOutput(output.node_id, output.port) for output in dag.final_outputs),
    )


@dataclass(frozen=True)
class SchedulerView:
    """Read-only scheduling evidence supplied to one Scheduler Candidate."""

    dag: ToolCallPlan
    minimum_accuracy: float
    maximum_latency_ms: float
    gamma: float
    snapshot_digest: str
    snapshot_evidence: Mapping[str, Any]


SchedulerProposal = Mapping[str, Any] | Sequence[Any]


@dataclass(frozen=True)
class SchedulerCandidate:
    """A strictly versioned, untrusted producer of Scheduler Proposals."""

    version: int
    propose: Callable[[SchedulerView], SchedulerProposal]

    def __post_init__(self) -> None:
        if not isinstance(self.version, int) or isinstance(self.version, bool) or self.version <= 0:
            raise ValueError("Scheduler Candidate version must be a positive integer")


@dataclass(frozen=True)
class EvaluationTrace:
    """One independent planning occurrence and its device-independent DAG."""

    trace_id: str
    task_input: Mapping[str, object]
    dag: ToolCallPlan
    minimum_accuracy: float
    maximum_latency_ms: float
    gamma: float
    scheduler_solving_time_ms: float = 0.0

    def __post_init__(self) -> None:
        if not self.trace_id:
            raise ValueError("Trace identifier must not be empty")
        object.__setattr__(self, "task_input", MappingProxyType(dict(self.task_input)))


@dataclass(frozen=True)
class TraceEvaluation:
    """Scheduling proposal and trusted evaluation facts for one Trace."""

    trace: EvaluationTrace
    assignments: Mapping[str, NodeAssignment]
    status: Literal["scored", "rejected", "failed"]
    score: float | None
    report: EvaluationReport
    reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "assignments", MappingProxyType(dict(self.assignments)))


@dataclass(frozen=True)
class CandidateEvaluation:
    schema_version: str
    candidate_version: int
    traces: tuple[TraceEvaluation, ...]
    candidate_score: float | None

    def concise_projection(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "scheduler_candidate_version": self.candidate_version,
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
            "scheduler_candidate_version": self.candidate_evaluation.candidate_version,
            "traces": [
                {
                    **_concise_trace(comparison.trace_evaluation),
                    "oracle_reference_score": comparison.oracle_reference_score,
                    "difference": comparison.difference,
                }
                for comparison in self.traces
            ],
            "candidate_score": self.candidate_evaluation.candidate_score,
            "average_candidate_score": self.average_candidate_score,
            "average_oracle_reference_score": self.average_oracle_reference_score,
            "average_difference": self.average_difference,
        }


def _concise_trace(record: TraceEvaluation) -> dict[str, object]:
    result: dict[str, object] = {"trace_id": record.trace.trace_id, "status": record.status}
    if record.score is not None:
        result["score"] = record.score
    elif record.reason is not None:
        result["reason"] = record.reason
    return result


def evaluate_scheduler_candidate(
    snapshot: ProfilingDatabaseSnapshot,
    traces: Sequence[EvaluationTrace],
    candidate: SchedulerCandidate,
    *,
    mode: Literal["candidate", "final"] = "candidate",
    oracle_reference: SchedulerCandidate | None = None,
) -> CandidateEvaluation | FinalEvaluation:
    """Evaluate a Candidate over a fixed Trace set, optionally against an Oracle.

    The same trusted evaluator produces every score.  A valid but infeasible
    proposal is still a scored Trace with score zero; invalid proposals and
    Candidate runtime failures have no score and therefore no Candidate Score.
    """
    _validate_trace_set(traces)
    if mode == "final" and oracle_reference is None:
        raise ValueError("Final Evaluation requires an Oracle Reference")
    if mode not in {"candidate", "final"}:
        raise ValueError(f"unknown evaluation mode: {mode}")

    candidate_result = _evaluate_candidate(snapshot, traces, candidate)
    if mode == "candidate":
        return candidate_result
    assert oracle_reference is not None
    oracle_result = _evaluate_candidate(snapshot, traces, oracle_reference)
    comparisons = tuple(
        FinalTraceComparison(
            candidate_record,
            oracle_record.score,
            None if candidate_record.score is None or oracle_record.score is None else candidate_record.score - oracle_record.score,
        )
        for candidate_record, oracle_record in zip(candidate_result.traces, oracle_result.traces, strict=True)
    )
    candidate_scores = [item.trace_evaluation.score for item in comparisons if item.trace_evaluation.score is not None]
    oracle_scores = [item.oracle_reference_score for item in comparisons if item.oracle_reference_score is not None]
    differences = [item.difference for item in comparisons if item.difference is not None]
    return FinalEvaluation(
        candidate_result,
        comparisons,
        _mean(candidate_scores),
        _mean(oracle_scores),
        _mean(differences),
    )


def _validate_trace_set(traces: Sequence[EvaluationTrace]) -> None:
    identifiers = [trace.trace_id for trace in traces]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Trace identifiers must be unique within a Trace set")


def _evaluate_candidate(
    snapshot: ProfilingDatabaseSnapshot,
    traces: Sequence[EvaluationTrace],
    candidate: SchedulerCandidate,
) -> CandidateEvaluation:
    records = tuple(_evaluate_trace(snapshot, trace, candidate) for trace in traces)
    scores = [record.score for record in records]
    return CandidateEvaluation(SCHEMA_VERSION, candidate.version, records, _mean(scores) if all(score is not None for score in scores) else None)


def _evaluate_trace(snapshot: ProfilingDatabaseSnapshot, trace: EvaluationTrace, candidate: SchedulerCandidate) -> TraceEvaluation:
    view = SchedulerView(
        _readonly_dag(trace.dag),
        trace.minimum_accuracy,
        trace.maximum_latency_ms,
        trace.gamma,
        snapshot.snapshot_digest,
        _readonly(snapshot.data),
    )

    class CandidateAdapter:
        def schedule(self, dag: ToolCallPlan) -> SchedulerProposal:
            if dag is not trace.dag:
                raise RuntimeError("trusted evaluator passed an unexpected DAG")
            return candidate.propose(view)

    report = evaluate_scheduler_instance(
        snapshot,
        trace.dag,
        CandidateAdapter(),
        minimum_accuracy=trace.minimum_accuracy,
        maximum_latency_ms=trace.maximum_latency_ms,
        gamma=trace.gamma,
        scheduler_solving_time_ms=trace.scheduler_solving_time_ms,
    )
    reason = "; ".join(report.validation_errors) or None
    if report.scheduler_status == "rejected" and reason and reason.startswith("scheduler_exception:"):
        return TraceEvaluation(trace, report.assignments, "failed", None, report, reason)
    if report.scheduler_status == "rejected":
        return TraceEvaluation(trace, report.assignments, "rejected", None, report, reason)
    return TraceEvaluation(trace, report.assignments, "scored", report.utility if report.utility is not None else 0.0, report)


def _mean(values: Sequence[float | None]) -> float | None:
    numeric = [value for value in values if value is not None]
    return None if not numeric else sum(numeric) / len(numeric)


class TraceSelectionStrategy(Protocol):
    def __call__(self, traces: tuple[TraceEvaluation, ...]) -> Sequence[TraceEvaluation]: ...


@dataclass(frozen=True)
class ModelContext:
    traces: tuple[TraceEvaluation, ...]


CandidateProposer = Callable[[ModelContext, SchedulerCandidate], SchedulerCandidate]


@dataclass(frozen=True)
class EvolutionResult:
    observed_evaluations: tuple[CandidateEvaluation, ...]
    selected_candidate: SchedulerCandidate
    selected_evaluation: CandidateEvaluation


@dataclass
class EvolutionModule:
    """Fixed-round orchestration; it delegates all score calculation above."""

    snapshot: ProfilingDatabaseSnapshot
    evolution_traces: Sequence[EvaluationTrace]
    trace_selection_strategy: TraceSelectionStrategy
    candidate_proposer: CandidateProposer

    def run(self, initial_candidate: SchedulerCandidate, *, rounds: int) -> EvolutionResult:
        if rounds < 0:
            raise ValueError("rounds must not be negative")
        observed = [_evaluate_candidate(self.snapshot, self.evolution_traces, initial_candidate)]
        candidates = {initial_candidate.version: initial_candidate}
        current = initial_candidate
        for _ in range(rounds):
            context = ModelContext(tuple(self.trace_selection_strategy(observed[-1].traces)))
            proposed = self.candidate_proposer(context, current)
            if proposed.version in candidates:
                raise ValueError(f"duplicate Scheduler Candidate version: {proposed.version}")
            candidates[proposed.version] = proposed
            current = proposed
            observed.append(_evaluate_candidate(self.snapshot, self.evolution_traces, current))
        scorable = [evaluation for evaluation in observed if evaluation.candidate_score is not None]
        if not scorable:
            raise ValueError("no observed Scheduler Candidate received a Candidate Score")
        selected = max(scorable, key=lambda evaluation: evaluation.candidate_score)
        return EvolutionResult(tuple(observed), candidates[selected.candidate_version], selected)

    def final_evaluate(
        self,
        evolution: EvolutionResult,
        final_evaluation_traces: Sequence[EvaluationTrace],
        oracle_reference: SchedulerCandidate,
    ) -> FinalEvaluation:
        """Evaluate the selected Candidate only on an independent Trace set."""
        evolution_ids = {trace.trace_id for trace in self.evolution_traces}
        final_ids = {trace.trace_id for trace in final_evaluation_traces}
        if evolution_ids & final_ids:
            raise ValueError("Final Evaluation Trace Set must be independent from the Evolution Trace Set")
        result = evaluate_scheduler_candidate(
            self.snapshot,
            final_evaluation_traces,
            evolution.selected_candidate,
            mode="final",
            oracle_reference=oracle_reference,
        )
        assert isinstance(result, FinalEvaluation)
        return result
