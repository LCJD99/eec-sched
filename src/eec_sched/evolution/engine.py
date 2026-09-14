"""Trusted evaluation and evolution orchestration.

The engine owns control flow and validation.  Selection, behavior description,
memory, and generation are replaceable adapters in sibling modules.
"""

from __future__ import annotations

import signal
import sys
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from random import choice as random_choice
from random import random
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence, cast

from tqdm import tqdm

from ..evaluation import evaluate_scheduler_instance
from ..diagnosis.models import DiagnosisResult
from ..profiling.snapshot import ProfilingDatabaseSnapshot
from .memory import EmptyMemory, Experience, Memory
from .models import (
    SCHEMA_VERSION,
    CandidateEvaluation,
    CandidateGenerationRequest,
    EvaluationTrace,
    EvolutionGraph,
    EvolutionLoopResult,
    FinalEvaluation,
    FinalTraceComparison,
    ReflectionInput,
    ReflectionTraceEvidence,
    SchedulerCandidate,
    SchedulerCandidateDraft,
    SchedulerCandidateRegistry,
    SchedulerView,
    ScoringContext,
    TraceEvaluation,
    _mean,
    readonly_dag,
)
from .strategies import (
    BehaviorDescriptor,
    CrossoverParentSelectorProtocol,
    InMemoryRepertoire,
    MutationParentSelector,
    MutationParentSelectorProtocol,
    OperatorSelector,
    ParetoMutationParentSelector,
    Repertoire,
    TopKCosineCrossoverParentSelector,
    TraceScoreBehaviorDescriptor,
)

_CANDIDATE_TIMEOUT_SECONDS = 5.0


def evaluate_scheduler_candidate(
    snapshot: ProfilingDatabaseSnapshot,
    traces: Sequence[EvaluationTrace],
    scheduler_version: int,
    candidate_registry: SchedulerCandidateRegistry,
    *,
    mode: Literal["candidate", "final"] = "candidate",
    oracle_reference: SchedulerCandidate | None = None,
    scoring_context: ScoringContext | None = None,
) -> CandidateEvaluation | FinalEvaluation:
    """Evaluate a Candidate over a fixed Trace set, optionally against an Oracle."""
    _validate_trace_set(traces)
    if mode == "final" and oracle_reference is None:
        raise ValueError("Final Evaluation requires an Oracle Reference")
    if mode not in {"candidate", "final"}:
        raise ValueError(f"unknown evaluation mode: {mode}")
    candidate = candidate_registry.resolve(scheduler_version)
    context = scoring_context or ScoringContext()
    candidate_result = _evaluate_candidate(snapshot, traces, candidate, context)
    if mode == "candidate":
        return candidate_result
    assert oracle_reference is not None
    oracle_result = _evaluate_candidate(snapshot, traces, oracle_reference, context)
    comparisons = tuple(
        FinalTraceComparison(
            candidate_record,
            oracle_record.score,
            None
            if candidate_record.score is None or oracle_record.score is None
            else candidate_record.score - oracle_record.score,
        )
        for candidate_record, oracle_record in zip(
            candidate_result.traces, oracle_result.traces, strict=True
        )
    )
    return FinalEvaluation(
        candidate_result,
        comparisons,
        candidate_result.candidate_score,
        _mean([item.oracle_reference_score for item in comparisons]),
        _mean([item.difference for item in comparisons]),
    )


def _validate_trace_set(traces: Sequence[EvaluationTrace]) -> None:
    identifiers = [trace.trace_id for trace in traces]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Trace identifiers must be unique within a Trace set")


def _evaluate_candidate(
    snapshot: ProfilingDatabaseSnapshot,
    traces: Sequence[EvaluationTrace],
    candidate: SchedulerCandidate,
    scoring_context: ScoringContext,
) -> CandidateEvaluation:
    records = tuple(
        _evaluate_trace(snapshot, trace, candidate, scoring_context) for trace in traces
    )
    return CandidateEvaluation(
        SCHEMA_VERSION,
        candidate.scheduler_version,
        records,
        _mean([record.score_contribution for record in records]),
    )


def _evaluate_trace(
    snapshot: ProfilingDatabaseSnapshot,
    trace: EvaluationTrace,
    candidate: SchedulerCandidate,
    scoring_context: ScoringContext,
) -> TraceEvaluation:
    view = SchedulerView(
        readonly_dag(trace.dag),
        scoring_context,
        snapshot.snapshot_digest,
        _scheduler_evidence(snapshot, trace.dag),
    )

    class CandidateAdapter:
        def schedule(self, dag):
            if dag is not trace.dag:
                raise RuntimeError("trusted evaluator passed an unexpected DAG")
            with _candidate_timeout():
                return candidate.propose(view)

    report = evaluate_scheduler_instance(
        snapshot,
        trace.dag,
        CandidateAdapter(),
        scoring_context=view.scoring_context,
    )
    reason = "; ".join(report.validation_errors) or None
    computation_time = report.scheduler_solving_time_ms
    if computation_time is not None and computation_time >= _timeout_seconds() * 1000.0:
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
    if (
        report.scheduler_status == "rejected"
        and reason
        and reason.startswith("scheduler_exception:")
    ):
        return TraceEvaluation(
            trace,
            report.assignments,
            "failed",
            None,
            report,
            reason,
            view,
            computation_time,
            candidate.scheduler_version,
        )
    if report.scheduler_status == "rejected":
        return TraceEvaluation(
            trace,
            report.assignments,
            "rejected",
            None,
            report,
            reason,
            view,
            computation_time,
            candidate.scheduler_version,
        )
    return TraceEvaluation(
        trace,
        report.assignments,
        "scored",
        report.composite_score if report.composite_score is not None else 0.0,
        report,
        scheduler_view=view,
        scheduler_computation_time_ms=computation_time,
        scheduler_version=candidate.scheduler_version,
    )


def _timeout_seconds() -> float:
    """Read the public compatibility knob, including test monkeypatches."""
    import sys

    package = sys.modules.get("eec_sched.evolution")
    return float(
        getattr(package, "_CANDIDATE_TIMEOUT_SECONDS", _CANDIDATE_TIMEOUT_SECONDS)
    )


def _scheduler_evidence(snapshot: ProfilingDatabaseSnapshot, dag) -> Mapping[str, Any]:
    tool_ids = {node.tool_id for node in dag.nodes}
    tools = tuple(
        tool for tool in snapshot.data["tools"] if tool["tool_id"] in tool_ids
    )
    from .models import _readonly

    return _readonly(
        {
            "devices": snapshot.data["devices"],
            "tools": tools,
            "transfer_profiles": snapshot.data["transfer_profiles"],
        }
    )


@contextmanager
def _candidate_timeout() -> Any:
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, _timeout_seconds())

    def raise_timeout(signum: int, frame: Any) -> None:
        raise TimeoutError(
            "Scheduler Candidate did not return before the trusted time limit"
        )

    signal.signal(signal.SIGALRM, raise_timeout)
    try:
        yield
    finally:
        signal.signal(signal.SIGALRM, previous_handler)
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)


TrustedTraceEvaluator = Callable[
    [ProfilingDatabaseSnapshot, EvaluationTrace, SchedulerCandidate], TraceEvaluation
]
OracleDependency = Callable[
    [ProfilingDatabaseSnapshot, tuple[TraceEvaluation, ...], SchedulerCandidate],
    Sequence[float | None],
]
ReflectionAgent = Callable[[ReflectionInput], str]
CodingAgent = Callable[[CandidateGenerationRequest], SchedulerCandidateDraft]
DiagnosisCallable = Callable[[Mapping[str, object]], DiagnosisResult]
TraceRecorder = Callable[[Mapping[str, object]], None]


def _json_safe(value: object) -> object:
    """Convert trusted value objects into an artifact-safe JSON projection."""
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _json_safe(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"cannot serialize trusted trace value {type(value).__name__}")


def _complete_trace_event(record: TraceEvaluation, scheduler_version: int | None = None) -> dict[str, object]:
    """Return the complete, source-free trusted evidence for one Trace."""
    report = record.report
    return {
        "type": "evaluation_trace",
        "schema_version": SCHEMA_VERSION,
        "trace_id": record.trace.trace_id,
        "scheduler_version": record.scheduler_version or scheduler_version,
        "snapshot_digest": report.snapshot_digest,
        "task_input": _json_safe(record.trace.task_input),
        "dag": _json_safe(record.trace.dag),
        "assignments": _json_safe(record.assignments),
        "status": record.status,
        "reason": record.reason,
        "nodes": _json_safe(report.nodes),
        "transfers": _json_safe(report.transfers),
        "raw_metrics": _json_safe(report.raw_metrics),
        "metrics": {
            "accuracy": report.accuracy,
            "raw_accuracy_metrics": _json_safe(report.raw_accuracy_metrics),
            "simulated_makespan_ms": report.simulated_makespan_ms,
            "scheduler_solving_time_ms": report.scheduler_solving_time_ms,
            "latency": report.latency,
            "resource": report.resource,
            "composite_score": report.composite_score,
        },
    }


def _diagnosis_evidence(candidate: SchedulerCandidate, evaluation: CandidateEvaluation) -> dict[str, object]:
    """Build the diagnosis input without exposing Candidate source code."""
    return {
        "schema_version": SCHEMA_VERSION,
        "scheduler_version": candidate.scheduler_version,
        "snapshot_digest": evaluation.traces[0].report.snapshot_digest if evaluation.traces else None,
        "traces": [
            {key: value for key, value in _complete_trace_event(record, candidate.scheduler_version).items() if key != "type"}
            for record in evaluation.traces
        ],
    }


@dataclass(frozen=True)
class EvolutionLoop:
    """Canonical application boundary for Candidate evolution and final scoring."""

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
    random_choice: Callable[[Sequence[SchedulerCandidate]], SchedulerCandidate] = (
        random_choice
    )
    show_progress: bool = False
    mutation_parent_selector: MutationParentSelectorProtocol | None = None
    crossover_parent_selector: CrossoverParentSelectorProtocol | None = None
    behavior_descriptor: BehaviorDescriptor | None = None
    operator_selector: OperatorSelector | None = None
    repertoire: Repertoire | None = None
    memory: Memory | None = None

    def __post_init__(self) -> None:
        if self.rounds < 0:
            raise ValueError("rounds must not be negative")
        if not 0.0 <= self.mutation_probability <= 1.0:
            raise ValueError("mutation_probability must be between zero and one")
        roots = tuple(self.initial_candidates)
        if len(roots) != 3:
            raise ValueError(
                "Evolution Graph requires exactly three root Scheduler Candidates"
            )
        if len({candidate.scheduler_version for candidate in roots}) != 3:
            raise ValueError("root Scheduler Candidate versions must be distinct")
        if any(candidate.parent_scheduler_versions for candidate in roots):
            raise ValueError(
                "root Scheduler Candidates must not have Parent Scheduler Versions"
            )
        evolution_traces, final_traces = (
            tuple(self.evolution_traces),
            tuple(self.final_evaluation_traces),
        )
        _validate_trace_set(evolution_traces)
        _validate_trace_set(final_traces)
        if {trace.trace_id for trace in evolution_traces} & {
            trace.trace_id for trace in final_traces
        }:
            raise ValueError(
                "Final Evaluation Trace Set must be independent from the Evolution Trace Set"
            )
        object.__setattr__(self, "evolution_traces", evolution_traces)
        object.__setattr__(self, "final_evaluation_traces", final_traces)
        object.__setattr__(self, "initial_candidates", roots)
        object.__setattr__(
            self,
            "mutation_parent_selector",
            self.mutation_parent_selector or ParetoMutationParentSelector(),
        )
        object.__setattr__(
            self,
            "crossover_parent_selector",
            self.crossover_parent_selector or TopKCosineCrossoverParentSelector(),
        )
        descriptor = self.behavior_descriptor or TraceScoreBehaviorDescriptor()
        object.__setattr__(self, "behavior_descriptor", descriptor)
        object.__setattr__(
            self, "repertoire", self.repertoire or InMemoryRepertoire(descriptor)
        )
        object.__setattr__(self, "memory", self.memory or EmptyMemory())

    def run(self) -> EvolutionLoopResult:
        assert self.repertoire is not None
        assert self.memory is not None
        assert self.mutation_parent_selector is not None
        assert self.crossover_parent_selector is not None
        candidates = {
            candidate.scheduler_version: candidate
            for candidate in self.initial_candidates
        }
        evaluations = {
            candidate.scheduler_version: self._candidate_evaluation(
                candidate,
                self.evolution_traces,
                f"initial candidate {candidate.scheduler_version}",
            )
            for candidate in self.initial_candidates
        }
        for candidate in self.initial_candidates:
            self.repertoire.add(candidate, evaluations[candidate.scheduler_version])
            self.memory.record(
                Experience(
                    f"candidate {candidate.scheduler_version}: {evaluations[candidate.scheduler_version].candidate_score}",
                    candidate.scheduler_version,
                )
            )
        for round_number in range(1, self.rounds + 1):
            self._report(f"Evolution round {round_number}/{self.rounds}: starting")
            operator: Literal["mutation", "crossover"] = (
                self.operator_selector.select(self.random_float())
                if self.operator_selector is not None
                else (
                    "mutation"
                    if self.random_float() < self.mutation_probability
                    else "crossover"
                )
            )  # type: ignore[assignment]
            parents = (
                self.mutation_parent_selector.select(
                    candidates, evaluations, random_choice=self.random_choice
                )
                if operator == "mutation"
                else self.crossover_parent_selector.select(candidates, evaluations)
            )
            reflection = self._reflection_input(
                operator, parents, candidates, evaluations
            )
            advice = self.reflection_agent(reflection)
            memories = self.memory.retrieve(operator, limit=10)
            if memories:
                advice = (
                    advice
                    + "\n\nRelevant prior experience:\n"
                    + "\n".join(item.summary for item in memories)
                )
            request = CandidateGenerationRequest(
                operator,
                tuple(parent.scheduler_version for parent in parents),
                tuple(parent.source_code for parent in parents),
                advice,
            )
            draft = self.coding_agent(request)
            next_version = max(candidates) + 1
            proposed = SchedulerCandidate(
                next_version,
                draft.propose,
                draft.source_code,
                request.parent_scheduler_versions,
                draft.strategy_description,
                draft.reflection_context,
                draft.reflection_feedback,
                draft.coding_context,
            )
            candidates[next_version] = proposed
            evaluations[next_version] = self._candidate_evaluation(
                proposed,
                self.evolution_traces,
                f"round {round_number}/{self.rounds} candidate {next_version}",
            )
            self.repertoire.add(proposed, evaluations[next_version])
            self.memory.record(
                Experience(
                    f"{operator} candidate {next_version}: {evaluations[next_version].candidate_score}",
                    next_version,
                    {"operator": operator, "round": round_number},
                )
            )
        graph = EvolutionGraph(candidates, evaluations)
        selected_evaluation = _highest_scoring(tuple(graph.evaluations.values()))
        selected_candidate = candidates[selected_evaluation.scheduler_version]
        final_candidate_evaluation = self._candidate_evaluation(
            selected_candidate, self.final_evaluation_traces, "final evaluation"
        )
        oracle_scores = tuple(
            self.oracle(
                self.snapshot, final_candidate_evaluation.traces, selected_candidate
            )
        )
        if len(oracle_scores) != len(final_candidate_evaluation.traces):
            raise ValueError(
                "Oracle dependency must return one score per Final Evaluation Trace"
            )
        comparisons = tuple(
            FinalTraceComparison(
                record,
                score,
                None if record.score is None or score is None else record.score - score,
            )
            for record, score in zip(
                final_candidate_evaluation.traces, oracle_scores, strict=True
            )
        )
        final = FinalEvaluation(
            final_candidate_evaluation,
            comparisons,
            final_candidate_evaluation.candidate_score,
            _mean(oracle_scores),
            _mean([item.difference for item in comparisons]),
        )
        return EvolutionLoopResult(
            graph, selected_candidate, selected_evaluation, final
        )

    def _select_mutation_parent(self, candidates, evaluations):
        assert self.mutation_parent_selector is not None
        return self.mutation_parent_selector.select(
            candidates, evaluations, random_choice=self.random_choice
        )

    def _select_crossover_parents(self, candidates, evaluations):
        assert self.crossover_parent_selector is not None
        return self.crossover_parent_selector.select(candidates, evaluations)

    def _reflection_input(self, operator, parents, candidates, evaluations):
        if operator == "crossover":
            selected_evidence = _crossover_evidence(
                evaluations[parents[0].scheduler_version],
                evaluations[parents[1].scheduler_version],
            )
        else:
            parent = parents[0]
            evaluation = evaluations[parent.scheduler_version]
            selected_evidence = (
                _root_evidence(evaluation)
                if not parent.parent_scheduler_versions
                else _lineage_evidence(
                    evaluation,
                    tuple(
                        evaluations[version]
                        for version in parent.parent_scheduler_versions
                    ),
                )
            )
        involved = (
            parents
            if operator == "crossover"
            else (
                parents[0],
                *(
                    candidates[version]
                    for version in parents[0].parent_scheduler_versions
                ),
            )
        )
        return ReflectionInput(
            operator,
            _operator_description(operator),
            {
                candidate.scheduler_version: candidate.strategy_description
                for candidate in involved
            },
            tuple(parent.source_code for parent in parents),
            tuple(_reflection_trace_evidence(records) for records in selected_evidence),
        )

    def _candidate_evaluation(self, candidate, traces, progress_label=None):
        records = tuple(
            self.trusted_evaluator(self.snapshot, trace, candidate)
            for trace in _progress_traces(
                traces, progress_label, enabled=self.show_progress
            )
        )
        for record, trace in zip(records, traces, strict=True):
            if record.trace != trace:
                raise ValueError(
                    "trusted evaluator returned a result for a different Trace"
                )
            if record.status not in {"scored", "rejected", "failed"}:
                raise ValueError(
                    "trusted evaluator returned an unknown Evaluation Status"
                )
            if record.status == "scored" and record.score is None:
                raise ValueError("a scored Trace must include a score")
            if record.status != "scored" and record.score is not None:
                raise ValueError("a rejected or failed Trace cannot include a score")
        return CandidateEvaluation(
            SCHEMA_VERSION,
            candidate.scheduler_version,
            records,
            sum(item.score_contribution for item in records) / len(records)
            if records
            else 0.0,
        )

    def _report(self, message: str) -> None:
        if self.show_progress:
            print(message, file=sys.stderr, flush=True)


def _noop_trace_recorder(event: Mapping[str, object]) -> None:
    return None


@dataclass(frozen=True)
class PostEvaluationDiagnosisEvolutionLoop:
    """Evolution loop whose diagnosis follows trusted evaluation.

    This is intentionally a separate implementation from :class:`EvolutionLoop`.
    The legacy loop remains the reflection-before-coding comparison baseline;
    this loop makes the ordering an invariant instead of a runtime mode flag.
    """

    snapshot: ProfilingDatabaseSnapshot
    evolution_traces: Sequence[EvaluationTrace]
    final_evaluation_traces: Sequence[EvaluationTrace]
    initial_candidates: Sequence[SchedulerCandidate]
    rounds: int
    mutation_probability: float
    diagnosis_agent: DiagnosisCallable
    coding_agent: CodingAgent
    trusted_evaluator: TrustedTraceEvaluator
    oracle: OracleDependency
    random_float: Callable[[], float] = random
    random_choice: Callable[[Sequence[SchedulerCandidate]], SchedulerCandidate] = random_choice
    show_progress: bool = False
    mutation_parent_selector: MutationParentSelectorProtocol | None = None
    crossover_parent_selector: CrossoverParentSelectorProtocol | None = None
    behavior_descriptor: BehaviorDescriptor | None = None
    operator_selector: OperatorSelector | None = None
    repertoire: Repertoire | None = None
    memory: Memory | None = None
    trace_recorder: TraceRecorder = _noop_trace_recorder

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
        evolution_traces, final_traces = tuple(self.evolution_traces), tuple(self.final_evaluation_traces)
        _validate_trace_set(evolution_traces)
        _validate_trace_set(final_traces)
        if {trace.trace_id for trace in evolution_traces} & {trace.trace_id for trace in final_traces}:
            raise ValueError("Final Evaluation Trace Set must be independent from the Evolution Trace Set")
        object.__setattr__(self, "evolution_traces", evolution_traces)
        object.__setattr__(self, "final_evaluation_traces", final_traces)
        object.__setattr__(self, "initial_candidates", roots)
        object.__setattr__(self, "mutation_parent_selector", self.mutation_parent_selector or ParetoMutationParentSelector())
        object.__setattr__(self, "crossover_parent_selector", self.crossover_parent_selector or TopKCosineCrossoverParentSelector())
        descriptor = self.behavior_descriptor or TraceScoreBehaviorDescriptor()
        object.__setattr__(self, "behavior_descriptor", descriptor)
        object.__setattr__(self, "repertoire", self.repertoire or InMemoryRepertoire(descriptor))
        object.__setattr__(self, "memory", self.memory or EmptyMemory())

    def run(self) -> EvolutionLoopResult:
        assert self.repertoire is not None and self.memory is not None
        assert self.mutation_parent_selector is not None and self.crossover_parent_selector is not None
        candidates = {candidate.scheduler_version: candidate for candidate in self.initial_candidates}
        evaluations: dict[int, CandidateEvaluation] = {}
        diagnoses: dict[int, DiagnosisResult] = {}
        trace_events: list[Mapping[str, object]] = []

        # The roots establish a fully diagnosed initial pool before selection.
        for candidate in self.initial_candidates:
            evaluation = self._candidate_evaluation(candidate, self.evolution_traces, f"initial candidate {candidate.scheduler_version}")
            evaluations[candidate.scheduler_version] = evaluation
            self._record_evaluation(candidate, evaluation, trace_events)
            diagnosis = self._diagnose_and_record(candidate, evaluation, trace_events)
            diagnoses[candidate.scheduler_version] = diagnosis
            self.repertoire.add(candidate, evaluation)
            self.memory.record(Experience(
                f"candidate {candidate.scheduler_version}: {diagnosis.advice}",
                candidate.scheduler_version,
                {"diagnosis": diagnosis.advice, "bottlenecks": diagnosis.bottlenecks},
            ))

        for round_number in range(1, self.rounds + 1):
            self._report(f"Evolution round {round_number}/{self.rounds}: starting")
            operator: Literal["mutation", "crossover"] = (
                self.operator_selector.select(self.random_float())
                if self.operator_selector is not None
                else ("mutation" if self.random_float() < self.mutation_probability else "crossover")
            )  # type: ignore[assignment]
            parents = (
                self.mutation_parent_selector.select(candidates, evaluations, random_choice=self.random_choice)
                if operator == "mutation"
                else self.crossover_parent_selector.select(candidates, evaluations)
            )
            advice = self._diagnosis_brief(operator, parents, diagnoses)
            memories = self.memory.retrieve(operator, limit=10)
            if memories:
                advice += "\n\nRelevant prior experience:\n" + "\n".join(item.summary for item in memories)
            request = CandidateGenerationRequest(
                operator,
                tuple(parent.scheduler_version for parent in parents),
                tuple(parent.source_code for parent in parents),
                advice,
            )
            draft = self.coding_agent(request)
            next_version = max(candidates) + 1
            proposed = SchedulerCandidate(
                next_version, draft.propose, draft.source_code,
                request.parent_scheduler_versions, draft.strategy_description,
                draft.reflection_context, draft.reflection_feedback, draft.coding_context,
            )
            candidates[next_version] = proposed
            evaluation = self._candidate_evaluation(proposed, self.evolution_traces, f"round {round_number}/{self.rounds} candidate {next_version}")
            evaluations[next_version] = evaluation
            self._record_evaluation(proposed, evaluation, trace_events)
            diagnosis = self._diagnose_and_record(proposed, evaluation, trace_events)
            diagnoses[next_version] = diagnosis
            self.repertoire.add(proposed, evaluation)
            self.memory.record(Experience(
                f"{operator} candidate {next_version}: {diagnosis.advice}", next_version,
                {"operator": operator, "round": round_number, "diagnosis": diagnosis.advice, "bottlenecks": diagnosis.bottlenecks},
            ))

        graph = EvolutionGraph(candidates, evaluations)
        selected_evaluation = _highest_scoring(tuple(graph.evaluations.values()))
        selected_candidate = candidates[selected_evaluation.scheduler_version]
        final_candidate_evaluation = self._candidate_evaluation(selected_candidate, self.final_evaluation_traces, "final evaluation")
        oracle_scores = tuple(self.oracle(self.snapshot, final_candidate_evaluation.traces, selected_candidate))
        if len(oracle_scores) != len(final_candidate_evaluation.traces):
            raise ValueError("Oracle dependency must return one score per Final Evaluation Trace")
        comparisons = tuple(
            FinalTraceComparison(record, score, None if record.score is None or score is None else record.score - score)
            for record, score in zip(final_candidate_evaluation.traces, oracle_scores, strict=True)
        )
        final = FinalEvaluation(final_candidate_evaluation, comparisons, final_candidate_evaluation.candidate_score, _mean(oracle_scores), _mean([item.difference for item in comparisons]))
        return EvolutionLoopResult(graph, selected_candidate, selected_evaluation, final, diagnoses, tuple(trace_events))

    def _record_evaluation(self, candidate: SchedulerCandidate, evaluation: CandidateEvaluation, events: list[Mapping[str, object]]) -> None:
        for record in evaluation.traces:
            event = _complete_trace_event(record, candidate.scheduler_version)
            events.append(event)
            self._record_event(event)

    def _diagnose_and_record(self, candidate: SchedulerCandidate, evaluation: CandidateEvaluation, events: list[Mapping[str, object]]) -> DiagnosisResult:
        evidence = _diagnosis_evidence(candidate, evaluation)
        # Accept either the small callable seam or an existing ``Diagnosis``
        # adapter, keeping composition convenient for tests and ablations.
        if callable(self.diagnosis_agent):
            diagnosis = self.diagnosis_agent(evidence)
        else:  # pragma: no cover - retained for protocol-style adapters
            diagnosis = self.diagnosis_agent.diagnose(evidence)  # type: ignore[union-attr]
        if not isinstance(diagnosis, DiagnosisResult):
            raise TypeError("Diagnosis Agent must return DiagnosisResult")
        event: dict[str, object] = {
            "type": "diagnosis",
            "schema_version": SCHEMA_VERSION,
            "scheduler_version": candidate.scheduler_version,
            "snapshot_digest": evidence["snapshot_digest"],
            "diagnosis": _json_safe(diagnosis),
        }
        events.append(event)
        self._record_event(event)
        return diagnosis

    def _record_event(self, event: Mapping[str, object]) -> None:
        if callable(self.trace_recorder):
            self.trace_recorder(event)
        else:  # pragma: no cover - protocol-style recorder compatibility
            recorder = self.trace_recorder
            if hasattr(recorder, "record"):
                recorder.record(event)  # type: ignore[union-attr]
            else:
                recorder.record_event(event)  # type: ignore[union-attr]

    def _diagnosis_brief(self, operator: str, parents: tuple[SchedulerCandidate, ...], diagnoses: Mapping[int, DiagnosisResult]) -> str:
        if operator == "mutation":
            parent = parents[0]
            diagnosis = diagnoses[parent.scheduler_version]
            return (
                f"Diagnosis Brief (mutation; scheduler_version={parent.scheduler_version})\n"
                f"bottlenecks: {', '.join(diagnosis.bottlenecks) or 'none'}\n"
                f"evidence: {'; '.join(diagnosis.evidence) or 'none'}\n"
                f"advice: {diagnosis.advice}"
            )
        left, right = parents
        left_diagnosis, right_diagnosis = diagnoses[left.scheduler_version], diagnoses[right.scheduler_version]
        return (
            f"Diagnosis Brief (crossover; scheduler_versions={left.scheduler_version},{right.scheduler_version})\n"
            f"parent {left.scheduler_version}: {left_diagnosis.advice}\n"
            f"parent {right.scheduler_version}: {right_diagnosis.advice}"
        )

    def _candidate_evaluation(self, candidate, traces, progress_label=None):
        # The implementation only relies on the shared evaluation fields and
        # progress hook; keep the seam local while satisfying the static type
        # of the legacy method.
        return EvolutionLoop._candidate_evaluation(cast(Any, self), candidate, traces, progress_label)

    def _report(self, message: str) -> None:
        if self.show_progress:
            print(message, file=sys.stderr, flush=True)


def _progress_traces(traces, label, *, enabled):
    return (
        tqdm(traces, desc=label, unit="trace", file=sys.stderr)
        if enabled and label is not None
        else traces
    )


def _operator_description(operator):
    return (
        "Modify an existing strategy, preserve what works, and make a verifiable improvement based on the evaluation evidence."
        if operator == "mutation"
        else "Combine complementary ideas from two strategies while avoiding their respective weaknesses."
    )


def _root_evidence(evaluation):
    if not evaluation.traces:
        return ()
    return (
        (max(evaluation.traces, key=lambda item: item.score_contribution),),
        (min(evaluation.traces, key=lambda item: item.score_contribution),),
    )


def _lineage_evidence(candidate, direct_parents):
    evidence = []
    for parent in direct_parents:
        comparisons = tuple(zip(candidate.traces, parent.traces, strict=True))
        if comparisons:
            improvement = max(
                comparisons,
                key=lambda pair: (
                    pair[0].score_contribution - pair[1].score_contribution
                ),
            )
            regression = min(
                comparisons,
                key=lambda pair: (
                    pair[0].score_contribution - pair[1].score_contribution
                ),
            )
            evidence.extend((improvement, regression))
    return tuple(evidence)


def _crossover_evidence(left, right):
    comparisons = tuple(zip(left.traces, right.traces, strict=True))
    if not comparisons:
        return ()
    return (
        max(
            comparisons,
            key=lambda pair: pair[0].score_contribution - pair[1].score_contribution,
        ),
        max(
            comparisons,
            key=lambda pair: pair[1].score_contribution - pair[0].score_contribution,
        )[::-1],
    )


def _reflection_trace_evidence(records):
    candidate = records[0]
    compared = records[1] if len(records) > 1 else None
    report = candidate.report
    return ReflectionTraceEvidence(
        candidate.trace.trace_id,
        candidate.scheduler_version or 0,
        candidate.status,
        candidate.score_contribution,
        candidate.reason,
        tuple(
            f"{node.node_id} uses {node.tool_id} with inputs {', '.join(sorted(node.inputs)) or 'none'}"
            for node in candidate.trace.dag.nodes
        ),
        _compact_task_input(candidate.trace.task_input),
        tuple(
            f"{node_id} -> configuration={assignment.configuration_id}, device={assignment.device_id}"
            for node_id, assignment in sorted(candidate.assignments.items())
        ),
        {
            "accuracy": report.accuracy,
            "raw_accuracy_metrics": report.raw_accuracy_metrics,
            "simulated_makespan_ms": report.simulated_makespan_ms,
            "scheduler_solving_time_ms": candidate.scheduler_computation_time_ms,
            "latency": report.latency,
            "resource": report.resource,
            "composite_score": report.composite_score,
        },
        compared.scheduler_version if compared is not None else None,
        compared.status if compared is not None else None,
        compared.score_contribution if compared is not None else None,
        tuple(
            f"{node_id} -> configuration={assignment.configuration_id}, device={assignment.device_id}"
            for node_id, assignment in sorted(compared.assignments.items())
        )
        if compared is not None
        else (),
        candidate.score_contribution - compared.score_contribution
        if compared is not None
        else None,
    )


def _compact_task_input(task_input):
    rendered = ", ".join(
        f"{key}={(f'{value:.3f}' if isinstance(value, float) else str(value))[:120]}"
        for key, value in sorted(task_input.items())
    )
    return rendered[:400]


def _highest_scoring(observed):
    if not observed:
        raise ValueError("Evolution Loop requires an initial Candidate Evaluation")
    best = observed[0]
    for evaluation in observed[1:]:
        if evaluation.candidate_score is not None and (
            best.candidate_score is None
            or evaluation.candidate_score > best.candidate_score
        ):
            best = evaluation
    return best
