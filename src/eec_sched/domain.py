"""Stable domain types for planning tool-call DAGs."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping, Protocol

# Audio is intentionally a first-class port only because MnMS speech recognition
# consumes an audio file.  Structured values still cross the DAG as text, keeping
# the original image/text-oriented composition contract small.
Modality = Literal["audio", "image", "text"]


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
    data_type: Modality | None = None

    @classmethod
    def request(cls, name: str, data_type: Modality = "text") -> "InputSource":
        return cls("request", name, None, data_type)

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
class ToolCallPlanCandidates:
    """The frozen Planner output consumed by a Scheduler.

    A Planner invocation produces exactly three device-independent plans.  A
    plan may be repeated when the model has fewer than three useful options;
    keeping the cardinality fixed makes the workload comparable across
    Scheduler baselines and gives the Evaluator a small, explicit path
    selection contract.
    """

    dags: tuple[ToolCallPlan, ...]

    def __post_init__(self) -> None:
        dags = tuple(self.dags)
        if len(dags) != 3:
            raise ValueError("Planner output must contain exactly three DAGs")
        if any(not isinstance(dag, ToolCallPlan) for dag in dags):
            raise TypeError("Planner candidates must be ToolCallPlan instances")
        object.__setattr__(
            self,
            "dags",
            tuple(
                ToolCallPlan(
                    tuple(ToolNode(node.node_id, node.tool_id, MappingProxyType(dict(node.inputs))) for node in dag.nodes),
                    tuple(FinalOutput(output.node_id, output.port) for output in dag.final_outputs),
                )
                for dag in dags
            ),
        )

    @property
    def plans(self) -> tuple[ToolCallPlan, ...]:
        """Compatibility spelling for callers that use ``plans``."""
        return self.dags

    @property
    def candidate_dags(self) -> tuple[ToolCallPlan, ...]:
        return self.dags

    @property
    def nodes(self) -> tuple[ToolNode, ...]:
        """Expose the first DAG's shape for legacy single-DAG callers."""
        return self.dags[0].nodes

    @property
    def final_outputs(self) -> tuple[FinalOutput, ...]:
        """Expose the first DAG for legacy single-DAG callers."""
        return self.dags[0].final_outputs

    def __len__(self) -> int:
        return len(self.dags)

    def __iter__(self):
        return iter(self.dags)

    def __getitem__(self, index: int) -> ToolCallPlan:
        return self.dags[index]


@dataclass(frozen=True)
class SchedulerOutput:
    """A Scheduler's selected path and trusted input to placement decisions.

    ``assignments`` is intentionally opaque at this boundary.  The trusted
    Evaluator parses it into :class:`NodeAssignment` values and checks every
    configuration/device against the profiling snapshot before simulation.
    """

    dag: ToolCallPlan
    assignments: Mapping[str, object]
    path_index: int | None = None
    resource_configuration: Mapping[str, object] = field(default_factory=dict)
    selected_path_index: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.dag, ToolCallPlan):
            raise TypeError("Scheduler output dag must be a ToolCallPlan")
        if self.path_index is not None and self.selected_path_index is not None and self.path_index != self.selected_path_index:
            raise ValueError("Scheduler output path_index aliases disagree")
        selected_path_index = self.path_index if self.path_index is not None else self.selected_path_index
        if selected_path_index is not None and (
            not isinstance(selected_path_index, int) or isinstance(selected_path_index, bool)
        ):
            raise TypeError("Scheduler output path_index must be an integer or None")
        object.__setattr__(self, "path_index", selected_path_index)
        object.__setattr__(self, "selected_path_index", selected_path_index)
        object.__setattr__(self, "assignments", MappingProxyType(dict(self.assignments)))
        object.__setattr__(self, "resource_configuration", MappingProxyType(dict(self.resource_configuration)))


# Names used by early adapters and experiments.  The canonical public name is
# SchedulerOutput; aliases make the migration from a mapping-only proposal
# explicit without forcing all existing strategies to change at once.
ScheduledPlan = SchedulerOutput
SchedulerDecision = SchedulerOutput
SchedulerResult = SchedulerOutput
ScheduledDAG = SchedulerOutput
PlannerOutput = ToolCallPlanCandidates
PlannerCandidates = ToolCallPlanCandidates


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

    def plan_candidates(
        self,
        request: PlanningRequest,
        catalog: tuple[ToolSpec, ...],
        repair_errors: tuple[ValidationError, ...] = (),
    ) -> ToolCallPlanCandidates: ...


class FakePlannerClient:
    """Deterministic planner for application tests and offline demonstrations."""

    def __init__(self, plans: list[ToolCallPlan]) -> None:
        self._plans = iter(plans)

    def plan(self, request: PlanningRequest, catalog: tuple[ToolSpec, ...], repair_errors: tuple[ValidationError, ...] = ()) -> ToolCallPlan:
        return next(self._plans)

    def plan_candidates(self, request: PlanningRequest, catalog: tuple[ToolSpec, ...], repair_errors: tuple[ValidationError, ...] = ()) -> ToolCallPlanCandidates:
        plan = self.plan(request, catalog, repair_errors)
        return ToolCallPlanCandidates((plan, plan, plan))


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
