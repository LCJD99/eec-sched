"""Replaceable scheduler interface and deterministic naive baseline."""

from __future__ import annotations

from dataclasses import dataclass
from math import log

from .domain import Configuration, PlanningRequest, ToolCallPlan, ToolRegistry
from .profiles import InMemoryProfileRepository


@dataclass(frozen=True)
class ScheduledNode:
    node_id: str
    tool_id: str
    configuration_id: str
    quality: float
    latency_ms: float


@dataclass(frozen=True)
class SchedulingPlan:
    status: str
    nodes: tuple[ScheduledNode, ...] = ()
    predicted_accuracy: float | None = None
    predicted_latency_ms: float | None = None
    utility: float | None = None


def _utility(accuracy: float, latency: float, request: PlanningRequest) -> float:
    accuracy_term = 0.0 if request.minimum_accuracy == 1 else (accuracy - request.minimum_accuracy) / (1 - request.minimum_accuracy)
    return request.gamma * accuracy_term + (1 - request.gamma) * (request.maximum_latency_ms - latency) / request.maximum_latency_ms


class NaiveScheduler:
    def schedule(self, plan: ToolCallPlan, request: PlanningRequest, registry: ToolRegistry, profiles: InMemoryProfileRepository, device: str) -> SchedulingPlan:
        if not 0 <= request.minimum_accuracy <= 1 or request.maximum_latency_ms <= 0 or not 0 < request.gamma < 1:
            return SchedulingPlan("invalid_constraints")
        frontiers: dict[str, tuple[ScheduledNode, ...]] = {}
        for node in plan.nodes:
            spec = registry.spec(node.tool_id)
            if spec.metric_higher_is_better is None or spec.metric_floor is None:
                return SchedulingPlan("accuracy_profile_required")
            measured = []
            raw_profiles = []
            for config in spec.configurations:
                accuracy = profiles.accuracy(spec.tool_id, config.configuration_id)
                if accuracy is None:
                    return SchedulingPlan("accuracy_profile_required")
                raw_profiles.append((config, accuracy.raw_metric))
            oriented = [(config, score if spec.metric_higher_is_better else -score) for config, score in raw_profiles]
            floor = spec.metric_floor if spec.metric_higher_is_better else -spec.metric_floor
            reference = max(score for _, score in oriented)
            for config, score in oriented:
                latency = profiles.latency(spec.tool_id, config.configuration_id, device)
                if latency is None:
                    continue
                quality = 1.0 if reference == floor else max(0.0, min(1.0, (score - floor) / (reference - floor)))
                measured.append(ScheduledNode(node.node_id, node.tool_id, config.configuration_id, quality, latency.p95_ms))
            if not measured:
                return SchedulingPlan("profiling_required")
            frontiers[node.node_id] = self._pareto(measured)
        fastest = sum(min(option.latency_ms for option in options) for options in frontiers.values())
        highest = self._accuracy([max(options, key=lambda o: (o.quality, o.configuration_id)) for options in frontiers.values()])
        if fastest > request.maximum_latency_ms:
            return SchedulingPlan("infeasible")
        if highest < request.minimum_accuracy:
            return SchedulingPlan("infeasible")
        selected = {node_id: max(options, key=lambda o: (o.quality, o.configuration_id)) for node_id, options in frontiers.items()}
        while self._latency(selected.values()) > request.maximum_latency_ms:
            moves: list[tuple[float, str, ScheduledNode]] = []
            current_accuracy = self._accuracy(selected.values())
            for node_id, current in selected.items():
                for candidate in frontiers[node_id]:
                    if candidate.latency_ms >= current.latency_ms or candidate.quality >= current.quality or candidate.quality <= 0:
                        continue
                    new_accuracy = current_accuracy / current.quality * candidate.quality
                    if new_accuracy < request.minimum_accuracy:
                        continue
                    ratio = (current.latency_ms - candidate.latency_ms) / -log(candidate.quality / current.quality)
                    moves.append((ratio, node_id, candidate))
            if not moves:
                return SchedulingPlan("no_feasible_solution_found")
            _, node_id, candidate = max(moves, key=lambda item: (item[0], item[1], item[2].configuration_id))
            selected[node_id] = candidate
        improved = True
        while improved:
            improved = False
            current_accuracy = self._accuracy(selected.values())
            current_latency = self._latency(selected.values())
            current_utility = _utility(current_accuracy, current_latency, request)
            alternatives: list[tuple[float, str, ScheduledNode]] = []
            for node_id, current in selected.items():
                for candidate in frontiers[node_id]:
                    if candidate == current or current.quality == 0:
                        continue
                    accuracy = current_accuracy / current.quality * candidate.quality
                    latency = current_latency - current.latency_ms + candidate.latency_ms
                    if accuracy >= request.minimum_accuracy and latency <= request.maximum_latency_ms:
                        utility = _utility(accuracy, latency, request)
                        if utility > current_utility:
                            alternatives.append((utility, node_id, candidate))
            if alternatives:
                _, node_id, candidate = max(alternatives, key=lambda item: (item[0], item[1], item[2].configuration_id))
                selected[node_id] = candidate
                improved = True
        nodes = tuple(selected[node.node_id] for node in plan.nodes)
        return SchedulingPlan("scheduled", nodes, self._accuracy(nodes), self._latency(nodes), _utility(self._accuracy(nodes), self._latency(nodes), request))

    @staticmethod
    def _pareto(options: list[ScheduledNode]) -> tuple[ScheduledNode, ...]:
        return tuple(sorted((option for option in options if not any(
            other != option and other.quality >= option.quality and other.latency_ms <= option.latency_ms and (other.quality > option.quality or other.latency_ms < option.latency_ms)
            for other in options
        )), key=lambda option: option.configuration_id))

    @staticmethod
    def _accuracy(nodes: object) -> float:
        result = 1.0
        for node in nodes:  # type: ignore[union-attr]
            result *= node.quality
        return result

    @staticmethod
    def _latency(nodes: object) -> float:
        return sum(node.latency_ms for node in nodes)  # type: ignore[union-attr]
