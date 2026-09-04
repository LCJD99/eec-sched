"""Trusted scheduler evaluation package.

The public seam is :func:`evaluate_scheduler_instance`; the submodules expose
the report model, deterministic simulator, and scoring policies separately so
Hydra adapters can replace them without changing callers.
"""

from .evaluator import Scheduler, evaluate, evaluate_scheduler_instance, parse_assignments
from .models import EvaluationReport, NodeAssignment, SimulatedNode, SimulatedTransfer, TrustedEvaluationError
from .scoring import UTILITY_EPSILON, accuracy, calculate_accuracy, calculate_normalized_performance, calculate_utility, normalized_performance, utility
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
    "UTILITY_EPSILON",
    "accuracy",
    "calculate_accuracy",
    "calculate_normalized_performance",
    "calculate_utility",
    "evaluate",
    "evaluate_scheduler_instance",
    "normalized_performance",
    "parse_assignments",
    "predecessors",
    "simulate",
    "utility",
]
