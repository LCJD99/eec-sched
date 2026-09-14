"""Public contracts for End–Edge–Cloud Scheduler evolution."""

from .candidate import (
    CandidateEvaluation,
    EvaluationTrace,
    FinalEvaluation,
    FinalTraceComparison,
    SchedulerCandidate,
    SchedulerCandidateDraft,
    SchedulerCandidateRegistry,
    SchedulerView,
    ScoringContext,
    TraceEvaluation,
)
from .domain import (
    Configuration,
    FinalOutput,
    InputSource,
    PlanningRequest,
    Port,
    ToolCallPlan,
    ToolNode,
    ToolRegistry,
    ToolSpec,
)
from .evaluation import (
    EvaluationReport,
    NodeAssignment,
    SimulatedNode,
    SimulatedTransfer,
    TrustedEvaluationError,
    composite_score,
    evaluate_assignments,
    evaluate_scheduler_instance,
)
from .evolution import (
    CandidateGenerationRequest,
    EvolutionGraph,
    EvolutionLoop,
    EvolutionLoopResult,
    PostEvaluationDiagnosisEvolutionLoop,
    ReflectionInput,
    ReflectionTraceEvidence,
    evaluate_scheduler_candidate,
)
from .mnms_tools import HUGGINGFACE_MODEL_IDS, MNMS_MODELS, MnmsToolRunner, mnms_tool_specs, register_mnms_tools
from .openai_compatible import OpenAICompatiblePlannerClient
from .profiling import (
    AccuracyProfile,
    EvaluationSample,
    EvaluationSuite,
    InMemoryProfileRepository,
    LatencyProfile,
    ProfileArtifact,
    evaluate_configuration,
    load_huggingface_samples,
    load_profile_artifact,
    profile_suite,
    save_profile_artifact,
    select_fixed_samples,
)
from .profiling.snapshot import (
    Configuration as ProfilingConfiguration,
    Device,
    ExecutionProfile,
    MeasurementScope,
    ProfilingDatabaseSnapshot,
    ProfilingDatabaseValidationError,
    QualityProfile,
    SnapshotMetadata,
    Tool as ProfiledTool,
    TransferProfile,
    load_profiling_database,
    snapshot_digest,
    validate_profiling_database,
)
from .workflow import ToolCallPlanDataset, ToolCallPlanDatasetSplits
