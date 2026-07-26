"""Stable domain types for planning tool-call DAGs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Protocol

Modality = Literal["image", "text"]


@dataclass(frozen=True)
class Port:
    name: str
    modality: Modality
    required: bool = True


@dataclass(frozen=True)
class Configuration:
    """A finite, opaque tool configuration selected only from profiles."""

    configuration_id: str
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AccuracyEvaluation:
    """Tool-owned evaluation metadata; raw measurements remain in profiles."""

    dataset_loader: Callable[[], object]
    metric: Callable[[object, object], float]
    reference_selection: str = "best_measured"


@dataclass(frozen=True)
class ToolSpec:
    tool_id: str
    description: str
    inputs: Mapping[str, Port]
    outputs: Mapping[str, Port]
    configurations: tuple[Configuration, ...]
    metric_higher_is_better: bool | None = None
    metric_floor: float | None = None
    accuracy_evaluation: AccuracyEvaluation | None = None
    profile_key: str | None = None


class ToolRunner(Protocol):
    def prepare(self, configuration: Configuration) -> None: ...

    def run(self, inputs: Mapping[str, object], configuration: Configuration) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class InputSource:
    kind: Literal["request", "node"]
    name: str
    port: str | None = None

    @classmethod
    def request(cls, name: str) -> "InputSource":
        return cls("request", name)

    @classmethod
    def node(cls, node_id: str, port: str) -> "InputSource":
        return cls("node", node_id, port)


@dataclass(frozen=True)
class ToolNode:
    node_id: str
    tool_id: str
    inputs: Mapping[str, InputSource]


@dataclass(frozen=True)
class FinalOutput:
    node_id: str
    port: str


@dataclass(frozen=True)
class ToolCallPlan:
    nodes: tuple[ToolNode, ...]
    final_outputs: tuple[FinalOutput, ...]


@dataclass(frozen=True)
class PlanningRequest:
    prompt: str
    inputs: Mapping[str, object]
    minimum_accuracy: float
    maximum_latency_ms: float
    gamma: float


@dataclass(frozen=True)
class ValidationError:
    code: str
    message: str


class PlannerClient(Protocol):
    def plan(
        self,
        request: PlanningRequest,
        catalog: tuple[ToolSpec, ...],
        repair_errors: tuple[ValidationError, ...] = (),
    ) -> ToolCallPlan: ...


class FakePlannerClient:
    """Deterministic planner for application tests and offline demonstrations."""

    def __init__(self, plans: list[ToolCallPlan]) -> None:
        self._plans = iter(plans)

    def plan(self, request: PlanningRequest, catalog: tuple[ToolSpec, ...], repair_errors: tuple[ValidationError, ...] = ()) -> ToolCallPlan:
        return next(self._plans)


RunnerFactory = Callable[[], ToolRunner]


class ToolRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, tuple[ToolSpec, RunnerFactory]] = {}

    def register(self, spec: ToolSpec, runner_factory: RunnerFactory) -> None:
        if spec.tool_id in self._entries:
            raise ValueError(f"duplicate tool: {spec.tool_id}")
        self._entries[spec.tool_id] = (spec, runner_factory)

    def spec(self, tool_id: str) -> ToolSpec:
        return self._entries[tool_id][0]

    def runner(self, tool_id: str) -> ToolRunner:
        return self._entries[tool_id][1]()

    def catalog(self) -> tuple[ToolSpec, ...]:
        return tuple(entry[0] for entry in self._entries.values())

    def contains(self, tool_id: str) -> bool:
        return tool_id in self._entries
