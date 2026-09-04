"""Replaceable search strategies used by :mod:`eec_sched.evolution.engine`."""

from __future__ import annotations

from math import sqrt
from statistics import fmean
from random import choice as random_choice
from typing import Callable, Mapping, Protocol, Sequence

from .models import CandidateEvaluation, SchedulerCandidate


class BehaviorDescriptor(Protocol):
    """Map trusted evaluation evidence to a behavior-space point."""

    def describe(self, evaluation: CandidateEvaluation) -> tuple[float, ...]: ...


class TraceScoreBehaviorDescriptor:
    """Baseline descriptor: one coordinate per trace score contribution."""

    def describe(self, evaluation: CandidateEvaluation) -> tuple[float, ...]:
        return tuple(record.score_contribution for record in evaluation.traces)


class PerformanceBehaviorDescriptor:
    """Describe quality, latency, resource, and device-placement behavior.

    The coordinates are aggregate behavior observations rather than evaluator
    objectives, allowing similarly scored Candidates to remain distinguishable.
    """

    def describe(self, evaluation: CandidateEvaluation) -> tuple[float, ...]:
        scored = [record for record in evaluation.traces if record.status == "scored"]
        if not scored:
            return (0.0,) * 6
        accuracies = [record.report.accuracy or 0.0 for record in scored]
        latencies = [record.report.latency or 0.0 for record in scored]
        resources = [record.report.resource or 0.0 for record in scored]
        assignments = [assignment for record in scored for assignment in record.assignments.values()]
        total = max(1, len(assignments))
        placement = tuple(
            sum(item.device_id == device_id for item in assignments) / total
            for device_id in ("device", "edge", "cloud")
        )
        return (
            fmean(accuracies),
            fmean(latencies),
            fmean(resources),
            *placement,
        )


class Repertoire(Protocol):
    """Archive candidates by quality and behavior-space coverage."""

    def add(self, candidate: SchedulerCandidate, evaluation: CandidateEvaluation) -> None: ...

    def items(self) -> tuple[tuple[SchedulerCandidate, CandidateEvaluation], ...]: ...


class InMemoryRepertoire:
    """Small reference archive retaining the best candidate per behavior point."""

    def __init__(self, descriptor: BehaviorDescriptor | None = None) -> None:
        self.descriptor = descriptor or TraceScoreBehaviorDescriptor()
        self._items: dict[tuple[float, ...], tuple[SchedulerCandidate, CandidateEvaluation]] = {}

    def add(self, candidate: SchedulerCandidate, evaluation: CandidateEvaluation) -> None:
        point = self.descriptor.describe(evaluation)
        previous = self._items.get(point)
        if previous is None or (evaluation.candidate_score or 0.0) > (previous[1].candidate_score or 0.0):
            self._items[point] = (candidate, evaluation)

    def items(self) -> tuple[tuple[SchedulerCandidate, CandidateEvaluation], ...]:
        return tuple(self._items.values())


class UnboundedRepertoire:
    """Baseline archive retaining every evaluated Candidate by version."""

    def __init__(self) -> None:
        self._items: dict[int, tuple[SchedulerCandidate, CandidateEvaluation]] = {}

    def add(self, candidate: SchedulerCandidate, evaluation: CandidateEvaluation) -> None:
        self._items[candidate.scheduler_version] = (candidate, evaluation)

    def items(self) -> tuple[tuple[SchedulerCandidate, CandidateEvaluation], ...]:
        return tuple(self._items.values())


class MutationParentSelectorProtocol(Protocol):
    def select(
        self,
        candidates: Mapping[int, SchedulerCandidate],
        evaluations: Mapping[int, CandidateEvaluation],
        *,
        random_choice: Callable[[Sequence[SchedulerCandidate]], SchedulerCandidate] = random_choice,
    ) -> tuple[SchedulerCandidate, ...]: ...


class CrossoverParentSelectorProtocol(Protocol):
    def select(
        self, candidates: Mapping[int, SchedulerCandidate], evaluations: Mapping[int, CandidateEvaluation]
    ) -> tuple[SchedulerCandidate, ...]: ...


# Short names are convenient for dependency injection while the explicit
# ``*Protocol`` names remain available to callers that want the contract.
MutationParentSelector = MutationParentSelectorProtocol
CrossoverParentSelector = CrossoverParentSelectorProtocol


def score_vector(evaluation: CandidateEvaluation) -> tuple[float, ...]:
    return tuple(record.score_contribution for record in evaluation.traces)


def pareto_frontier(evaluations: Sequence[CandidateEvaluation]) -> tuple[CandidateEvaluation, ...]:
    """Return non-dominated evaluations, maximizing every trace coordinate."""

    def dominated(candidate: CandidateEvaluation) -> bool:
        vector = score_vector(candidate)
        return any(
            all(other >= current for other, current in zip(score_vector(comparator), vector, strict=True))
            and any(other > current for other, current in zip(score_vector(comparator), vector, strict=True))
            for comparator in evaluations
            if comparator is not candidate
        )

    return tuple(item for item in evaluations if not dominated(item))


class ParetoMutationParentSelector:
    """Choose one parent uniformly from the trace-score Pareto frontier."""

    def select(self, candidates, evaluations, *, random_choice=random_choice):
        frontier = pareto_frontier(tuple(evaluations.values()))
        if not frontier:
            raise ValueError("cannot select a mutation parent from an empty population")
        return (random_choice(tuple(candidates[item.scheduler_version] for item in frontier)),)


def cosine_similarity(left: CandidateEvaluation, right: CandidateEvaluation) -> float:
    left_vector, right_vector = score_vector(left), score_vector(right)
    left_norm = sqrt(sum(value * value for value in left_vector))
    right_norm = sqrt(sum(value * value for value in right_vector))
    if not left_norm and not right_norm:
        return 1.0
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left_vector, right_vector, strict=True)) / (left_norm * right_norm)


class TopKCosineCrossoverParentSelector:
    """Select the least-similar pair among the top-k candidates by score."""

    def __init__(self, k: int = 5) -> None:
        if k < 2:
            raise ValueError("crossover selector k must be at least two")
        self.k = k

    def select(self, candidates, evaluations):
        ranked = sorted(evaluations.values(), key=lambda item: (-(item.candidate_score or 0.0), item.scheduler_version))[: self.k]
        pairs = [(left, right) for index, left in enumerate(ranked) for right in ranked[index + 1 :]]
        if not pairs:
            raise ValueError("crossover selection requires at least two candidates")
        left, right = min(pairs, key=lambda pair: (cosine_similarity(pair[0], pair[1]), pair[0].scheduler_version, pair[1].scheduler_version))
        return candidates[left.scheduler_version], candidates[right.scheduler_version]


def _euclidean(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise ValueError("behavior descriptors must have a stable dimension")
    return sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


class FeatureDiverseMutationParentSelector:
    """Sample from Candidates that are isolated in the behavior feature space."""

    def __init__(self, descriptor: BehaviorDescriptor | None = None, pool_size: int = 5) -> None:
        if pool_size < 1:
            raise ValueError("feature-diverse pool_size must be positive")
        self.descriptor = descriptor or PerformanceBehaviorDescriptor()
        self.pool_size = pool_size

    def select(self, candidates, evaluations, *, random_choice=random_choice):
        observed = tuple(evaluations.values())
        if not observed:
            raise ValueError("cannot select a mutation parent from an empty population")
        points = {item.scheduler_version: self.descriptor.describe(item) for item in observed}
        isolation = {
            item.scheduler_version: min(
                (_euclidean(points[item.scheduler_version], points[other.scheduler_version]) for other in observed if other is not item),
                default=float("inf"),
            )
            for item in observed
        }
        ranked = sorted(
            observed,
            key=lambda item: (-isolation[item.scheduler_version], -(item.candidate_score or 0.0), item.scheduler_version),
        )[: self.pool_size]
        return (random_choice(tuple(candidates[item.scheduler_version] for item in ranked)),)


class ComplementaryBehaviorCrossoverParentSelector:
    """Select the highest-quality pair with maximally different behavior."""

    def __init__(self, descriptor: BehaviorDescriptor | None = None, top_k: int = 5) -> None:
        if top_k < 2:
            raise ValueError("complementary crossover top_k must be at least two")
        self.descriptor = descriptor or PerformanceBehaviorDescriptor()
        self.top_k = top_k

    def select(self, candidates, evaluations):
        ranked = sorted(
            evaluations.values(),
            key=lambda item: (-(item.candidate_score or 0.0), item.scheduler_version),
        )[: self.top_k]
        pairs = [(left, right) for index, left in enumerate(ranked) for right in ranked[index + 1 :]]
        if not pairs:
            raise ValueError("crossover selection requires at least two candidates")
        left, right = max(
            pairs,
            key=lambda pair: (
                _euclidean(self.descriptor.describe(pair[0]), self.descriptor.describe(pair[1])),
                -pair[0].scheduler_version,
                -pair[1].scheduler_version,
            ),
        )
        return candidates[left.scheduler_version], candidates[right.scheduler_version]


class OperatorSelector(Protocol):
    def select(self, random_value: float) -> str: ...


class ProbabilisticOperatorSelector:
    """Choose mutation below ``mutation_probability`` and crossover otherwise."""

    def __init__(self, mutation_probability: float = 0.5) -> None:
        if not 0.0 <= mutation_probability <= 1.0:
            raise ValueError("mutation_probability must be between zero and one")
        self.mutation_probability = mutation_probability

    def select(self, random_value: float) -> str:
        if not 0.0 <= random_value <= 1.0:
            raise ValueError("random_value must be between zero and one")
        return "mutation" if random_value < self.mutation_probability else "crossover"
