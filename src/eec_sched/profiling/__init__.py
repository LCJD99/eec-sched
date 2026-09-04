"""Profiling modules.

The package is organized by implementation depth rather than by every
individual experiment knob.  Public imports are grouped here so callers can
switch adapters through configuration without depending on internal files.
"""

from .adapters import (
    evaluate_asr,
    evaluate_captioning,
    evaluate_coco_detection,
    evaluate_coco_segmentation,
    evaluate_image_classification,
    evaluate_ocr,
    evaluate_question_answering,
    evaluate_summarization,
    evaluate_text_classification,
    evaluate_vqa,
)
from .campaign import (
    ACTIVE_MODELS,
    DISABLED_MODELS,
    Candidate,
    Evaluator,
    Measurement,
    canonicalize_candidate,
    continuous_initial_candidates,
    default_evaluator,
    legal_candidates,
    load_json,
    load_samples,
    propose,
    propose_continuous,
    run,
    run_model,
    sobol_candidates,
    utc_now,
    validate_config,
    write_unavailable,
)
from .export import export_campaign_database
from .profiles import (
    AccuracyProfile,
    EvaluationSample,
    EvaluationSuite,
    InMemoryProfileRepository,
    LatencyProfile,
    ProfileArtifact,
    evaluate_configuration,
    load_huggingface_samples,
    load_persisted_samples,
    load_profile_artifact,
    profile_suite,
    profile_warm_latency,
    profile_warm_latency_samples,
    runtime_metadata,
    save_profile_artifact,
    select_fixed_samples,
)
from .snapshot import (
    Configuration as ProfiledConfiguration,
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

__all__ = [
    "ACTIVE_MODELS", "DISABLED_MODELS", "AccuracyProfile", "Candidate",
    "EvaluationSample", "EvaluationSuite",
    "Device", "Evaluator", "ExecutionProfile", "InMemoryProfileRepository",
    "LatencyProfile", "Measurement", "MeasurementScope", "ProfileArtifact",
    "ProfiledConfiguration", "ProfiledTool", "ProfilingDatabaseSnapshot",
    "ProfilingDatabaseValidationError", "QualityProfile", "SnapshotMetadata", "TransferProfile",
    "canonicalize_candidate", "continuous_initial_candidates",
    "default_evaluator", "evaluate_asr", "evaluate_captioning",
    "evaluate_coco_detection", "evaluate_coco_segmentation",
    "evaluate_configuration", "evaluate_image_classification", "evaluate_ocr",
    "evaluate_question_answering", "evaluate_summarization",
    "evaluate_text_classification", "evaluate_vqa", "export_campaign_database",
    "legal_candidates", "load_huggingface_samples", "load_json", "load_samples",
    "load_profiling_database",
    "load_persisted_samples", "load_profile_artifact",
    "profile_suite", "profile_warm_latency", "profile_warm_latency_samples",
    "propose", "propose_continuous", "run", "run_model", "runtime_metadata",
    "save_profile_artifact", "select_fixed_samples", "sobol_candidates",
    "snapshot_digest", "utc_now", "validate_config", "validate_profiling_database", "write_unavailable",
]
