"""Trusted evaluation flow for versioned Scheduler Candidates.

This module deliberately records planning, scheduling, and evaluation as
separate Trace stages.  It never executes tools or exposes a mutable snapshot
to untrusted Scheduler Candidates.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from math import sqrt
from random import choice as random_choice, random
import signal
import sys
from types import MappingProxyType
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

from tqdm import tqdm

from .domain import FinalOutput, ToolCallPlan, ToolNode
from .profiling_database import ProfilingDatabaseSnapshot
from .trusted_evaluation import EvaluationReport, NodeAssignment, evaluate_scheduler_instance

SCHEMA_VERSION = "v1"
_LEGACY_MINIMUM_ACCURACY = 0.0
_LEGACY_MAXIMUM_LATENCY_MS = 100.0
_LEGACY_GAMMA = 0.5
_CANDIDATE_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class ScoringContext:
    """Trusted scoring inputs visible to a Scheduler Candidate."""

    minimum_accuracy: float
    maximum_latency_ms: float
    gamma: float


_SCORING_CONTEXT = ScoringContext(
    minimum_accuracy=_LEGACY_MINIMUM_ACCURACY,
    maximum_latency_ms=_LEGACY_MAXIMUM_LATENCY_MS,
    gamma=_LEGACY_GAMMA,
)


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
    scoring_context: ScoringContext
    snapshot_digest: str
    snapshot_evidence: Mapping[str, Any]


SchedulerProposal = Mapping[str, Any] | Sequence[Any]


@dataclass(frozen=True)
class SchedulerCandidate:
    """A strictly versioned, untrusted producer of Scheduler Proposals."""

    scheduler_version: int
    propose: Callable[[SchedulerView], SchedulerProposal]
    source_code: str = "def propose(view):\n    raise NotImplementedError\n"
    parent_scheduler_versions: tuple[int, ...] = ()
    strategy_description: str = "legacy scheduler strategy"
    reflection_context: str | None = None
    reflection_feedback: str | None = None
    coding_context: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.scheduler_version, int)
            or isinstance(self.scheduler_version, bool)
            or self.scheduler_version <= 0
        ):
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
        if any(version <= 0 for version in self.parent_scheduler_versions):
            raise ValueError("Parent Scheduler Versions must be positive integers")


@dataclass(frozen=True)
class SchedulerCandidateDraft:
    """A generated Scheduler strategy before trusted graph metadata is assigned."""

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
    """Trusted mapping from Scheduler Versions to untrusted Candidate code."""

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
    """One independent planning occurrence and its device-independent DAG."""

    trace_id: str
    task_input: Mapping[str, object]
    dag: ToolCallPlan

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
    scheduler_view: SchedulerView | None = None
    scheduler_computation_time_ms: float | None = None
    scheduler_version: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "assignments", MappingProxyType(dict(self.assignments)))

    @property
    def scheduler_proposal(self) -> Mapping[str, NodeAssignment]:
        """The trusted, normalized proposal returned for this Trace."""
        return self.assignments

    @property
    def score_contribution(self) -> float:
        """The numerical Candidate Score contribution for this Trace."""
        return self.score if self.score is not None else 0.0


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
    result: dict[str, object] = {
        "trace_id": record.trace.trace_id,
        "status": record.status,
        "score": record.score_contribution,
    }
    if record.reason is not None:
        result["reason"] = record.reason
    return result


def evaluate_scheduler_candidate(
    snapshot: ProfilingDatabaseSnapshot,
    traces: Sequence[EvaluationTrace],
    scheduler_version: int,
    candidate_registry: SchedulerCandidateRegistry,
    *,
    mode: Literal["candidate", "final"] = "candidate",
    oracle_reference: SchedulerCandidate | None = None,
) -> CandidateEvaluation | FinalEvaluation:
    """Evaluate a Candidate over a fixed Trace set, optionally against an Oracle.

    The same trusted evaluator produces every score. Every valid proposal is
    scored by the continuous utility formula; invalid proposals and Candidate
    runtime failures have no score and therefore contribute zero.
    """
    _validate_trace_set(traces)
    if mode == "final" and oracle_reference is None:
        raise ValueError("Final Evaluation requires an Oracle Reference")
    if mode not in {"candidate", "final"}:
        raise ValueError(f"unknown evaluation mode: {mode}")

    candidate = candidate_registry.resolve(scheduler_version)
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
    oracle_scores = [item.oracle_reference_score for item in comparisons if item.oracle_reference_score is not None]
    differences = [item.difference for item in comparisons if item.difference is not None]
    return FinalEvaluation(
        candidate_result,
        comparisons,
        candidate_result.candidate_score,
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
    scores = [record.score_contribution for record in records]
    return CandidateEvaluation(SCHEMA_VERSION, candidate.scheduler_version, records, _mean(scores) if records else 0.0)


def _evaluate_trace(snapshot: ProfilingDatabaseSnapshot, trace: EvaluationTrace, candidate: SchedulerCandidate) -> TraceEvaluation:
    view = SchedulerView(
        _readonly_dag(trace.dag),
        _SCORING_CONTEXT,
        snapshot.snapshot_digest,
        _scheduler_evidence(snapshot, trace.dag),
    )

    class CandidateAdapter:
        def schedule(self, dag: ToolCallPlan) -> SchedulerProposal:
            if dag is not trace.dag:
                raise RuntimeError("trusted evaluator passed an unexpected DAG")
            with _candidate_timeout():
                return candidate.propose(view)

    report = evaluate_scheduler_instance(
        snapshot,
        trace.dag,
        CandidateAdapter(),
        minimum_accuracy=view.scoring_context.minimum_accuracy,
        maximum_latency_ms=view.scoring_context.maximum_latency_ms,
        gamma=view.scoring_context.gamma,
    )
    reason = "; ".join(report.validation_errors) or None
    computation_time = report.scheduler_solving_time_ms
    if computation_time is not None and computation_time >= _CANDIDATE_TIMEOUT_SECONDS * 1000.0:
        return TraceEvaluation(
            trace,
            report.assignments,
            "failed",
            None,
            report,
            "Scheduler Candidate exceeded the trusted computation time limit",
            view,
            computation_time,
            candidate.scheduler_version,
        )
    if report.scheduler_status == "rejected" and reason and reason.startswith("scheduler_exception:"):
        return TraceEvaluation(trace, report.assignments, "failed", None, report, reason, view, computation_time, candidate.scheduler_version)
    if report.scheduler_status == "rejected":
        return TraceEvaluation(trace, report.assignments, "rejected", None, report, reason, view, computation_time, candidate.scheduler_version)
    return TraceEvaluation(
        trace,
        report.assignments,
        "scored",
        report.utility if report.utility is not None else 0.0,
        report,
        scheduler_view=view,
        scheduler_computation_time_ms=computation_time,
        scheduler_version=candidate.scheduler_version,
    )


def _scheduler_evidence(snapshot: ProfilingDatabaseSnapshot, dag: ToolCallPlan) -> Mapping[str, Any]:
    """Expose only evidence relevant to this Trace, never the snapshot handle."""
    tool_ids = {node.tool_id for node in dag.nodes}
    tools = tuple(tool for tool in snapshot.data["tools"] if tool["tool_id"] in tool_ids)
    return _readonly(
        {
            "devices": snapshot.data["devices"],
            "tools": tools,
            "transfer_profiles": snapshot.data["transfer_profiles"],
        }
    )


@contextmanager
def _candidate_timeout() -> Any:
    """Interrupt an untrusted Candidate that does not return a proposal."""
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


def _mean(values: Sequence[float | None]) -> float | None:
    numeric = [value for value in values if value is not None]
    return None if not numeric else sum(numeric) / len(numeric)


TrustedTraceEvaluator = Callable[
    [ProfilingDatabaseSnapshot, EvaluationTrace, SchedulerCandidate], TraceEvaluation
]
OracleDependency = Callable[
    [ProfilingDatabaseSnapshot, tuple[TraceEvaluation, ...], SchedulerCandidate], Sequence[float | None]
]


@dataclass(frozen=True)
class EvolutionGraph:
    """All Candidates and their trusted evidence for one Evolution Loop run."""

    candidates: Mapping[int, SchedulerCandidate]
    evaluations: Mapping[int, CandidateEvaluation]

    def __post_init__(self) -> None:
        candidates = dict(self.candidates)
        evaluations = dict(self.evaluations)
        if set(candidates) != set(evaluations):
            raise ValueError("Evolution Graph Candidates and Evaluations must have matching versions")
        if any(candidate.scheduler_version != version for version, candidate in candidates.items()):
            raise ValueError("Evolution Graph keys must match Scheduler Versions")
        if any(evaluation.scheduler_version != version for version, evaluation in evaluations.items()):
            raise ValueError("Evolution Graph evidence must match Scheduler Versions")
        object.__setattr__(self, "candidates", MappingProxyType(candidates))
        object.__setattr__(self, "evaluations", MappingProxyType(evaluations))


@dataclass(frozen=True)
class ReflectionTraceEvidence:
    """Small, source-free projection of trusted evidence for Reflection."""

    trace_id: str
    candidate_scheduler_version: int
    candidate_status: Literal["scored", "rejected", "failed"]
    candidate_score: float
    candidate_reason: str | None
    dag: tuple[str, ...]
    task_input: str
    assignments: tuple[str, ...]
    metrics: Mapping[str, object]
    compared_scheduler_version: int | None = None
    compared_status: Literal["scored", "rejected", "failed"] | None = None
    compared_score: float | None = None
    compared_assignments: tuple[str, ...] = ()
    score_delta: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))


@dataclass(frozen=True)
class ReflectionInput:
    """Bounded evidence and parent source code used to build a Reflection prompt."""

    operator: Literal["mutation", "crossover"]
    operator_description: str
    strategy_descriptions: Mapping[int, str]
    parent_source_codes: tuple[str, ...]
    trace_evidence: tuple[ReflectionTraceEvidence, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "strategy_descriptions", MappingProxyType(dict(self.strategy_descriptions)))


@dataclass(frozen=True)
class CandidateGenerationRequest:
    """Advice and parent source code supplied to the Coding Agent."""

    operator: Literal["mutation", "crossover"]
    parent_scheduler_versions: tuple[int, ...]
    parent_source_codes: tuple[str, ...]
    reflection_advice: str


ReflectionAgent = Callable[[ReflectionInput], str]
CodingAgent = Callable[[CandidateGenerationRequest], SchedulerCandidateDraft]


@dataclass(frozen=True)
class EvolutionLoopResult:
    """The complete, bounded application-level result for one evolution run."""

    evolution_graph: EvolutionGraph
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
    initial_candidates: Sequence[SchedulerCandidate]
    rounds: int
    mutation_probability: float
    reflection_agent: ReflectionAgent
    coding_agent: CodingAgent
    trusted_evaluator: TrustedTraceEvaluator
    oracle: OracleDependency
    random_float: Callable[[], float] = random
    random_choice: Callable[[Sequence[SchedulerCandidate]], SchedulerCandidate] = random_choice
    show_progress: bool = False

    def __post_init__(self) -> None:
        if self.rounds < 0:
            raise ValueError("rounds must not be negative")
        if not 0.0 <= self.mutation_probability <= 1.0:
            raise ValueError("mutation_probability must be between zero and one")
        roots = tuple(self.initial_candidates)
        if len(roots) != 3:
            raise ValueError("Evolution Graph requires exactly three root Scheduler Candidates")
        if len({candidate.scheduler_version for candidate in roots}) != 3:
            raise ValueError("root Scheduler Candidate versions must be distinct")
        if any(candidate.parent_scheduler_versions for candidate in roots):
            raise ValueError("root Scheduler Candidates must not have Parent Scheduler Versions")
        evolution_traces = tuple(self.evolution_traces)
        final_traces = tuple(self.final_evaluation_traces)
        _validate_trace_set(evolution_traces)
        _validate_trace_set(final_traces)
        if {trace.trace_id for trace in evolution_traces} & {trace.trace_id for trace in final_traces}:
            raise ValueError("Final Evaluation Trace Set must be independent from the Evolution Trace Set")
        object.__setattr__(self, "evolution_traces", evolution_traces)
        object.__setattr__(self, "final_evaluation_traces", final_traces)
        object.__setattr__(self, "initial_candidates", roots)

    def run(self) -> EvolutionLoopResult:
        candidates = {candidate.scheduler_version: candidate for candidate in self.initial_candidates}
        evaluations = {
            candidate.scheduler_version: self._candidate_evaluation(
                candidate, self.evolution_traces, progress_label=f"initial candidate {candidate.scheduler_version}"
            )
            for candidate in self.initial_candidates
        }

        for round_number in range(1, self.rounds + 1):
            self._report(f"Evolution round {round_number}/{self.rounds}: starting")
            operator: Literal["mutation", "crossover"] = (
                "mutation" if self.random_float() < self.mutation_probability else "crossover"
            )
            parents = self._select_mutation_parent(candidates, evaluations) if operator == "mutation" else self._select_crossover_parents(candidates, evaluations)
            reflection = self._reflection_input(operator, parents, candidates, evaluations)
            advice = self.reflection_agent(reflection)
            request = CandidateGenerationRequest(
                operator,
                tuple(parent.scheduler_version for parent in parents),
                tuple(parent.source_code for parent in parents),
                advice,
            )
            draft = self.coding_agent(request)
            next_scheduler_version = max(candidates) + 1
            proposed = SchedulerCandidate(
                scheduler_version=next_scheduler_version,
                propose=draft.propose,
                source_code=draft.source_code,
                parent_scheduler_versions=request.parent_scheduler_versions,
                strategy_description=draft.strategy_description,
                reflection_context=draft.reflection_context,
                reflection_feedback=draft.reflection_feedback,
                coding_context=draft.coding_context,
            )
            candidates[proposed.scheduler_version] = proposed
            evaluations[proposed.scheduler_version] = self._candidate_evaluation(
                proposed,
                self.evolution_traces,
                progress_label=f"round {round_number}/{self.rounds} candidate {proposed.scheduler_version}",
            )

        graph = EvolutionGraph(candidates, evaluations)
        selected_evaluation = _highest_scoring(tuple(graph.evaluations.values()))
        selected_candidate = candidates[selected_evaluation.scheduler_version]
        final_candidate_evaluation = self._candidate_evaluation(
            selected_candidate, self.final_evaluation_traces, progress_label="final evaluation"
        )
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
        return EvolutionLoopResult(graph, selected_candidate, selected_evaluation, final)

    def _select_mutation_parent(
        self, candidates: Mapping[int, SchedulerCandidate], evaluations: Mapping[int, CandidateEvaluation]
    ) -> tuple[SchedulerCandidate, ...]:
        frontier = _pareto_frontier(tuple(evaluations.values()))
        return (self.random_choice(tuple(candidates[item.scheduler_version] for item in frontier)),)

    def _select_crossover_parents(
        self, candidates: Mapping[int, SchedulerCandidate], evaluations: Mapping[int, CandidateEvaluation]
    ) -> tuple[SchedulerCandidate, ...]:
        ranked = sorted(evaluations.values(), key=lambda item: (-item.candidate_score, item.scheduler_version))[:5]
        pairs = ((left, right) for index, left in enumerate(ranked) for right in ranked[index + 1 :])
        left, right = min(pairs, key=lambda pair: (_cosine_similarity(pair[0], pair[1]), pair[0].scheduler_version, pair[1].scheduler_version))
        return candidates[left.scheduler_version], candidates[right.scheduler_version]

    def _reflection_input(
        self,
        operator: Literal["mutation", "crossover"],
        parents: tuple[SchedulerCandidate, ...],
        candidates: Mapping[int, SchedulerCandidate],
        evaluations: Mapping[int, CandidateEvaluation],
    ) -> ReflectionInput:
        if operator == "crossover":
            selected_evidence = _crossover_evidence(evaluations[parents[0].scheduler_version], evaluations[parents[1].scheduler_version])
        else:
            parent = parents[0]
            evaluation = evaluations[parent.scheduler_version]
            selected_evidence = _root_evidence(evaluation) if not parent.parent_scheduler_versions else _lineage_evidence(
                evaluation, tuple(evaluations[version] for version in parent.parent_scheduler_versions)
            )
        involved = parents if operator == "crossover" else (
            parents[0], *(candidates[version] for version in parents[0].parent_scheduler_versions)
        )
        descriptions = {candidate.scheduler_version: candidate.strategy_description for candidate in involved}
        return ReflectionInput(
            operator,
            _operator_description(operator),
            descriptions,
            tuple(parent.source_code for parent in parents),
            tuple(_reflection_trace_evidence(records) for records in selected_evidence),
        )

    def _candidate_evaluation(
        self,
        candidate: SchedulerCandidate,
        traces: Sequence[EvaluationTrace],
        *,
        progress_label: str | None = None,
    ) -> CandidateEvaluation:
        records = tuple(
            self.trusted_evaluator(self.snapshot, trace, candidate)
            for trace in _progress_traces(traces, progress_label, enabled=self.show_progress)
        )
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
        score = sum(record.score_contribution for record in records) / len(records) if records else 0.0
        return CandidateEvaluation(SCHEMA_VERSION, candidate.scheduler_version, records, score)

    def _report(self, message: str) -> None:
        if self.show_progress:
            print(message, file=sys.stderr, flush=True)


def _progress_traces(
    traces: Sequence[EvaluationTrace], label: str | None, *, enabled: bool
) -> Iterable[EvaluationTrace]:
    """Yield dataset traces with an optional tqdm progress bar."""
    if not enabled or label is None:
        return traces
    return tqdm(traces, desc=label, unit="trace", file=sys.stderr)


def _score_vector(evaluation: CandidateEvaluation) -> tuple[float, ...]:
    return tuple(record.score_contribution for record in evaluation.traces)


def _pareto_frontier(evaluations: Sequence[CandidateEvaluation]) -> tuple[CandidateEvaluation, ...]:
    def dominated(candidate: CandidateEvaluation) -> bool:
        vector = _score_vector(candidate)
        return any(
            all(other >= current for other, current in zip(_score_vector(comparator), vector, strict=True))
            and any(other > current for other, current in zip(_score_vector(comparator), vector, strict=True))
            for comparator in evaluations
            if comparator is not candidate
        )

    return tuple(item for item in evaluations if not dominated(item))


def _cosine_similarity(left: CandidateEvaluation, right: CandidateEvaluation) -> float:
    left_vector, right_vector = _score_vector(left), _score_vector(right)
    left_norm = sqrt(sum(value * value for value in left_vector))
    right_norm = sqrt(sum(value * value for value in right_vector))
    if not left_norm and not right_norm:
        return 1.0
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left_vector, right_vector, strict=True)) / (left_norm * right_norm)


def _operator_description(operator: Literal["mutation", "crossover"]) -> str:
    if operator == "mutation":
        return "Modify an existing strategy, preserve what works, and make a verifiable improvement based on the evaluation evidence."
    return "Combine complementary ideas from two strategies while avoiding their respective weaknesses."


def _root_evidence(evaluation: CandidateEvaluation) -> tuple[tuple[TraceEvaluation, ...], ...]:
    records = evaluation.traces
    if not records:
        return ()
    return (
        (max(records, key=lambda item: item.score_contribution),),
        (min(records, key=lambda item: item.score_contribution),),
    )


def _lineage_evidence(
    candidate: CandidateEvaluation, direct_parents: Sequence[CandidateEvaluation]
) -> tuple[tuple[TraceEvaluation, ...], ...]:
    evidence: list[tuple[TraceEvaluation, ...]] = []
    for parent in direct_parents:
        comparisons = tuple(zip(candidate.traces, parent.traces, strict=True))
        if comparisons:
            improvement = max(comparisons, key=lambda pair: pair[0].score_contribution - pair[1].score_contribution)
            regression = min(comparisons, key=lambda pair: pair[0].score_contribution - pair[1].score_contribution)
            evidence.extend((improvement, regression))
    return tuple(evidence)


def _crossover_evidence(left: CandidateEvaluation, right: CandidateEvaluation) -> tuple[tuple[TraceEvaluation, ...], ...]:
    comparisons = tuple(zip(left.traces, right.traces, strict=True))
    if not comparisons:
        return ()
    return (
        max(comparisons, key=lambda pair: pair[0].score_contribution - pair[1].score_contribution),
        max(comparisons, key=lambda pair: pair[1].score_contribution - pair[0].score_contribution)[::-1],
    )


def _reflection_trace_evidence(records: tuple[TraceEvaluation, ...]) -> ReflectionTraceEvidence:
    candidate = records[0]
    compared = records[1] if len(records) > 1 else None
    report = candidate.report
    return ReflectionTraceEvidence(
        trace_id=candidate.trace.trace_id,
        candidate_scheduler_version=candidate.scheduler_version or 0,
        candidate_status=candidate.status,
        candidate_score=candidate.score_contribution,
        candidate_reason=candidate.reason,
        dag=tuple(
            f"{node.node_id} uses {node.tool_id} with inputs {', '.join(sorted(node.inputs)) or 'none'}"
            for node in candidate.trace.dag.nodes
        ),
        task_input=_compact_task_input(candidate.trace.task_input),
        assignments=tuple(
            f"{node_id} -> configuration={assignment.configuration_id}, device={assignment.device_id}"
            for node_id, assignment in sorted(candidate.assignments.items())
        ),
        metrics={
            "accuracy_lcb": report.accuracy_lcb,
            "simulated_makespan_ms": report.simulated_makespan_ms,
            "scheduler_solving_time_ms": candidate.scheduler_computation_time_ms,
            "latency_proxy_ms": report.latency_proxy_ms,
            "compute_energy_j": report.compute_energy_j,
            "communication_energy_j": report.communication_energy_j,
            "utility": report.utility,
        },
        compared_scheduler_version=compared.scheduler_version if compared is not None else None,
        compared_status=compared.status if compared is not None else None,
        compared_score=compared.score_contribution if compared is not None else None,
        compared_assignments=(
            tuple(
                f"{node_id} -> configuration={assignment.configuration_id}, device={assignment.device_id}"
                for node_id, assignment in sorted(compared.assignments.items())
            )
            if compared is not None
            else ()
        ),
        score_delta=(
            candidate.score_contribution - compared.score_contribution if compared is not None else None
        ),
    )


def _compact_task_input(task_input: Mapping[str, object]) -> str:
    """Keep task identity useful without allowing raw inputs to fill the prompt."""
    rendered = ", ".join(
        f"{key}={(f'{value:.3f}' if isinstance(value, float) else str(value))[:120]}"
        for key, value in sorted(task_input.items())
    )
    return rendered[:400]


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
