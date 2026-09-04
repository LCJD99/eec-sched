"""Evolution-specific contracts and compatibility re-exports.

Candidate and evaluation value objects live in :mod:`eec_sched.candidate` so
workflow/evaluation code can use them without importing the search engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping, Sequence

from ..candidate import (
    SCHEMA_VERSION,
    CandidateEvaluation,
    EvaluationTrace,
    FinalEvaluation,
    FinalTraceComparison,
    SchedulerCandidate,
    SchedulerCandidateDraft,
    SchedulerCandidateRegistry,
    SchedulerProposal,
    SchedulerView,
    ScoringContext,
    TraceEvaluation,
    _readonly,
    readonly_dag,
)


@dataclass(frozen=True)
class EvolutionGraph:
    """All candidates and trusted evidence for one evolution run."""

    candidates: Mapping[int, SchedulerCandidate]
    evaluations: Mapping[int, CandidateEvaluation]

    def __post_init__(self) -> None:
        candidates, evaluations = dict(self.candidates), dict(self.evaluations)
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
    operator: Literal["mutation", "crossover"]
    operator_description: str
    strategy_descriptions: Mapping[int, str]
    parent_source_codes: tuple[str, ...]
    trace_evidence: tuple[ReflectionTraceEvidence, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "strategy_descriptions", MappingProxyType(dict(self.strategy_descriptions)))


@dataclass(frozen=True)
class CandidateGenerationRequest:
    operator: Literal["mutation", "crossover"]
    parent_scheduler_versions: tuple[int, ...]
    parent_source_codes: tuple[str, ...]
    reflection_advice: str


@dataclass(frozen=True)
class EvolutionDiagnosisContext:
    """Optional operator context independent of the diagnosis package."""

    summary: str = ""
    bottlenecks: tuple[str, ...] = ()
    recommendations: tuple[str, ...] = ()
    evidence: Mapping[str, object] = MappingProxyType({})

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))


@dataclass(frozen=True)
class EvolutionLoopResult:
    evolution_graph: EvolutionGraph
    selected_candidate: SchedulerCandidate
    selected_evaluation: CandidateEvaluation
    final_evaluation: FinalEvaluation


def _mean(values: Sequence[float | None]) -> float | None:
    numeric = [value for value in values if value is not None]
    return None if not numeric else sum(numeric) / len(numeric)


__all__ = [
    "SCHEMA_VERSION", "CandidateEvaluation", "CandidateGenerationRequest", "EvaluationTrace",
    "EvolutionDiagnosisContext", "EvolutionGraph", "EvolutionLoopResult", "FinalEvaluation",
    "FinalTraceComparison", "ReflectionInput", "ReflectionTraceEvidence", "SchedulerCandidate",
    "SchedulerCandidateDraft", "SchedulerCandidateRegistry", "SchedulerProposal", "SchedulerView",
    "ScoringContext", "TraceEvaluation", "readonly_dag",
]
