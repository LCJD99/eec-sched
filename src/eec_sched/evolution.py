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
        object.__setattr__(self, "dag", _readonly_dag(self.dag))


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
            "scheduler_version": self.candidate_version,
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
            "scheduler_version": self.candidate_evaluation.candidate_version,
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
    scores = [record.score if record.score is not None else 0.0 for record in records]
    return CandidateEvaluation(SCHEMA_VERSION, candidate.version, records, _mean(scores) if records else 0.0)


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


TrustedTraceEvaluator = Callable[
    [ProfilingDatabaseSnapshot, EvaluationTrace, SchedulerCandidate], TraceEvaluation
]
OracleDependency = Callable[
    [ProfilingDatabaseSnapshot, tuple[TraceEvaluation, ...], SchedulerCandidate], Sequence[float | None]
]


@dataclass(frozen=True)
class EvolutionLoopResult:
    """The complete, bounded application-level result for one evolution run."""

    observed_evaluations: tuple[CandidateEvaluation, ...]
    selected_candidate: SchedulerCandidate
    selected_evaluation: CandidateEvaluation
    final_evaluation: FinalEvaluation


@dataclass(frozen=True)
class EvolutionLoop:
    """Canonical application boundary for candidate evolution and final scoring.

    The injected evaluator is the sole authority for a Trace status, assignment,
    report, and score.  The loop only aggregates those trusted facts and never
    exposes its mutable control state to Candidate or model code.
    """

    snapshot: ProfilingDatabaseSnapshot
    evolution_traces: Sequence[EvaluationTrace]
    final_evaluation_traces: Sequence[EvaluationTrace]
    initial_candidate: SchedulerCandidate
    rounds: int
    candidate_proposer: CandidateProposer
    trace_selection_strategy: TraceSelectionStrategy
    trusted_evaluator: TrustedTraceEvaluator
    oracle: OracleDependency

    def __post_init__(self) -> None:
        if self.rounds < 0:
            raise ValueError("rounds must not be negative")
        evolution_traces = tuple(self.evolution_traces)
        final_traces = tuple(self.final_evaluation_traces)
        _validate_trace_set(evolution_traces)
        _validate_trace_set(final_traces)
        if {trace.trace_id for trace in evolution_traces} & {trace.trace_id for trace in final_traces}:
            raise ValueError("Final Evaluation Trace Set must be independent from the Evolution Trace Set")
        object.__setattr__(self, "evolution_traces", evolution_traces)
        object.__setattr__(self, "final_evaluation_traces", final_traces)

    def run(self) -> EvolutionLoopResult:
        candidates = {self.initial_candidate.version: self.initial_candidate}
        current = self.initial_candidate
        observed = [self._candidate_evaluation(current, self.evolution_traces)]

        for _ in range(self.rounds):
            selected = tuple(self.trace_selection_strategy(observed[-1].traces))
            _validate_selected_contexts(selected, observed[-1].traces)
            proposed = self.candidate_proposer(ModelContext(selected), current)
            if proposed.version in candidates:
                raise ValueError(f"duplicate Scheduler Candidate version: {proposed.version}")
            candidates[proposed.version] = proposed
            current = proposed
            observed.append(self._candidate_evaluation(current, self.evolution_traces))

        selected_evaluation = _highest_scoring(observed)
        selected_candidate = candidates[selected_evaluation.candidate_version]
        final_candidate_evaluation = self._candidate_evaluation(selected_candidate, self.final_evaluation_traces)
        oracle_scores = tuple(self.oracle(self.snapshot, final_candidate_evaluation.traces, selected_candidate))
        if len(oracle_scores) != len(final_candidate_evaluation.traces):
            raise ValueError("Oracle dependency must return one score per Final Evaluation Trace")
        comparisons = tuple(
            FinalTraceComparison(
                record,
                oracle_score,
                None if record.score is None or oracle_score is None else record.score - oracle_score,
            )
            for record, oracle_score in zip(final_candidate_evaluation.traces, oracle_scores, strict=True)
        )
        final = FinalEvaluation(
            final_candidate_evaluation,
            comparisons,
            final_candidate_evaluation.candidate_score,
            _mean(oracle_scores),
            _mean([comparison.difference for comparison in comparisons]),
        )
        return EvolutionLoopResult(tuple(observed), selected_candidate, selected_evaluation, final)

    def _candidate_evaluation(
        self, candidate: SchedulerCandidate, traces: Sequence[EvaluationTrace]
    ) -> CandidateEvaluation:
        records = tuple(self.trusted_evaluator(self.snapshot, trace, candidate) for trace in traces)
        for record, trace in zip(records, traces, strict=True):
            if record.trace != trace:
                raise ValueError("trusted evaluator returned a result for a different Trace")
            if record.status not in {"scored", "rejected", "failed"}:
                raise ValueError("trusted evaluator returned an unknown Evaluation Status")
            if record.status == "scored" and record.score is None:
                raise ValueError("a scored Trace must include a score")
            if record.status != "scored" and record.score is not None:
                raise ValueError("a rejected or failed Trace cannot include a score")
        # Failed and rejected Traces remain visible but contribute zero, per the
        # Candidate Score glossary definition.
        score = sum(record.score if record.score is not None else 0.0 for record in records) / len(records) if records else 0.0
        return CandidateEvaluation(SCHEMA_VERSION, candidate.version, records, score)


def _validate_selected_contexts(
    selected: Sequence[TraceEvaluation], available: Sequence[TraceEvaluation]
) -> None:
    available_ids = {id(record) for record in available}
    if any(id(record) not in available_ids for record in selected):
        raise ValueError("Trace Selection Strategy must select from the supplied Trace results")


def _highest_scoring(observed: Sequence[CandidateEvaluation]) -> CandidateEvaluation:
    if not observed:
        raise ValueError("Evolution Loop requires an initial Candidate Evaluation")
    # Strict comparison deliberately keeps the earliest observed candidate on a tie.
    best = observed[0]
    for evaluation in observed[1:]:
        if evaluation.candidate_score is not None and (
            best.candidate_score is None or evaluation.candidate_score > best.candidate_score
        ):
            best = evaluation
    return best
