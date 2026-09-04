"""Evolution package with backwards-compatible public names.

The package replaces the former monolithic ``evolution.py`` module while
keeping its import surface stable for existing scripts and tests.
"""

from .engine import (
    CodingAgent,
    EvolutionLoop,
    OracleDependency,
    ReflectionAgent,
    TrustedTraceEvaluator,
    evaluate_scheduler_candidate,
)
from .engine import _CANDIDATE_TIMEOUT_SECONDS
from .memory import EmptyMemory, Experience, InMemoryMemory, JsonlMemory, Memory
from .models import (
    SCHEMA_VERSION,
    CandidateEvaluation,
    CandidateGenerationRequest,
    EvolutionDiagnosisContext,
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
    SchedulerProposal,
    SchedulerView,
    ScoringContext,
    TraceEvaluation,
)
from .operators import CrossoverOperator, EvolutionOperator, MutationOperator
from .strategies import (
    BehaviorDescriptor,
    ComplementaryBehaviorCrossoverParentSelector,
    CrossoverParentSelectorProtocol,
    InMemoryRepertoire,
    FeatureDiverseMutationParentSelector,
    OperatorSelector,
    ParetoMutationParentSelector,
    PerformanceBehaviorDescriptor,
    MutationParentSelector,
    CrossoverParentSelector,
    Repertoire,
    ProbabilisticOperatorSelector,
    TopKCosineCrossoverParentSelector,
    TraceScoreBehaviorDescriptor,
    UnboundedRepertoire,
    cosine_similarity,
    pareto_frontier,
    score_vector,
)

# Private helper aliases retained for notebooks that imported the old module's
# implementation details while the public seam remains the strategy classes.
_pareto_frontier = pareto_frontier
_cosine_similarity = cosine_similarity

__all__ = [
    "SCHEMA_VERSION",
    "BehaviorDescriptor",
    "ComplementaryBehaviorCrossoverParentSelector",
    "CandidateEvaluation",
    "CandidateGenerationRequest",
    "CodingAgent",
    "CrossoverOperator",
    "CrossoverParentSelectorProtocol",
    "EvolutionDiagnosisContext",
    "EmptyMemory",
    "EvaluationTrace",
    "EvolutionGraph",
    "EvolutionLoop",
    "EvolutionLoopResult",
    "EvolutionOperator",
    "Experience",
    "FinalEvaluation",
    "FinalTraceComparison",
    "InMemoryMemory",
    "JsonlMemory",
    "InMemoryRepertoire",
    "FeatureDiverseMutationParentSelector",
    "Memory",
    "MutationOperator",
    "MutationParentSelector",
    "CrossoverParentSelector",
    "OperatorSelector",
    "OracleDependency",
    "ParetoMutationParentSelector",
    "PerformanceBehaviorDescriptor",
    "ReflectionAgent",
    "ReflectionInput",
    "ReflectionTraceEvidence",
    "Repertoire",
    "ProbabilisticOperatorSelector",
    "SchedulerCandidate",
    "SchedulerCandidateDraft",
    "SchedulerCandidateRegistry",
    "SchedulerProposal",
    "SchedulerView",
    "ScoringContext",
    "TopKCosineCrossoverParentSelector",
    "TraceEvaluation",
    "TraceScoreBehaviorDescriptor",
    "UnboundedRepertoire",
    "TrustedTraceEvaluator",
    "cosine_similarity",
    "evaluate_scheduler_candidate",
    "pareto_frontier",
    "score_vector",
]
