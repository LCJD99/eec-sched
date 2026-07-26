"""In-memory profile storage and accuracy normalization."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from time import perf_counter
from typing import Iterable

from .domain import Configuration, ToolRunner


@dataclass(frozen=True)
class AccuracyProfile:
    tool_id: str
    configuration_id: str
    raw_metric: float
    model_name: str = ""
    configuration_parameters: dict[str, object] | None = None


@dataclass(frozen=True)
class LatencyProfile:
    tool_id: str
    configuration_id: str
    device: str
    input_bucket: str
    p50_ms: float
    p95_ms: float
    sample_count: int
    model_name: str = ""
    configuration_parameters: dict[str, object] | None = None


class InMemoryProfileRepository:
    def __init__(self, accuracy: Iterable[AccuracyProfile] = (), latency: Iterable[LatencyProfile] = ()) -> None:
        self._accuracy = {(p.tool_id, p.configuration_id): p for p in accuracy}
        self._latency = {(p.tool_id, p.configuration_id, p.device, p.input_bucket): p for p in latency}

    def accuracy(self, tool_id: str, configuration_id: str) -> AccuracyProfile | None:
        return self._accuracy.get((tool_id, configuration_id))

    def latency(self, tool_id: str, configuration_id: str, device: str, input_bucket: str = "default") -> LatencyProfile | None:
        return self._latency.get((tool_id, configuration_id, device, input_bucket))


def profile_warm_latency(tool_id: str, model_name: str, configuration: Configuration, device: str, runner: ToolRunner, inputs: dict[str, object], samples: int = 10, input_bucket: str = "default") -> LatencyProfile:
    """Measure warm end-to-end node latency; preparation/loading is excluded."""
    if samples < 1:
        raise ValueError("samples must be positive")
    runner.prepare(configuration)
    timings: list[float] = []
    for _ in range(samples):
        started = perf_counter()
        runner.run(inputs, configuration)
        timings.append((perf_counter() - started) * 1000)
    ordered = sorted(timings)
    p95_index = min(len(ordered) - 1, max(0, round(0.95 * len(ordered)) - 1))
    return LatencyProfile(tool_id, configuration.configuration_id, device, input_bucket, median(timings), ordered[p95_index], samples, model_name, dict(configuration.parameters))
