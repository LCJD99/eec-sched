"""Trusted scheduler evaluation package.

The public seam is :func:`evaluate_scheduler_instance`; the submodules expose
the report model, deterministic simulator, and scoring policies separately so
Hydra adapters can replace them without changing callers.
"""

from .evaluator import Scheduler, evaluate, evaluate_assignments, evaluate_scheduler_instance, parse_assignments
from .models import EvaluationReport, NodeAssignment, SimulatedNode, SimulatedTransfer, TrustedEvaluationError
from .scoring import ScoringContext, accuracy, calculate_accuracy, calculate_composite_score, calculate_resource, composite_score, execution_resource, raw_accuracy_metrics, resource
from .simulator import END_DEVICE_ID, FIXED_DATA_SIZE_BYTES, predecessors, simulate

__all__ = [
    "END_DEVICE_ID",
    "EvaluationReport",
    "FIXED_DATA_SIZE_BYTES",
    "NodeAssignment",
    "Scheduler",
    "SimulatedNode",
    "SimulatedTransfer",
    "TrustedEvaluationError",
    "ScoringContext",
    "accuracy",
    "calculate_accuracy",
    "calculate_composite_score",
    "calculate_resource",
    "composite_score",
    "execution_resource",
    "evaluate",
    "evaluate_assignments",
    "evaluate_scheduler_instance",
    "parse_assignments",
    "predecessors",
    "raw_accuracy_metrics",
    "resource",
    "simulate",
]
