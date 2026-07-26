"""Serial, single-device execution of a validated scheduled DAG."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from .domain import Configuration, ToolCallPlan, ToolRegistry
from .planning import topological_nodes
from .scheduling import SchedulingPlan


@dataclass(frozen=True)
class NodeObservation:
    node_id: str
    configuration_id: str
    elapsed_ms: float


@dataclass(frozen=True)
class ExecutionResult:
    status: str
    outputs: dict[str, object]
    observations: tuple[NodeObservation, ...]
    setup_latency_ms: float = 0.0
    failed_configuration_id: str | None = None
    failed_node_id: str | None = None
    error: str | None = None


def execute(plan: ToolCallPlan, schedule: SchedulingPlan, request_inputs: dict[str, object], registry: ToolRegistry) -> ExecutionResult:
    configurations = {
        scheduled.node_id: next(config for config in registry.spec(scheduled.tool_id).configurations if config.configuration_id == scheduled.configuration_id)
        for scheduled in schedule.nodes
    }
    runners = {node.node_id: registry.runner(node.tool_id) for node in plan.nodes}
    setup_started = perf_counter()
    try:
        for node in plan.nodes:
            runners[node.node_id].prepare(configurations[node.node_id])
    except Exception as error:
        return ExecutionResult("execution_failed", {}, (), (perf_counter() - setup_started) * 1000, configurations[node.node_id].configuration_id, node.node_id, f"{type(error).__name__}: {error}")
    setup_latency_ms = (perf_counter() - setup_started) * 1000
    nodes = {node.node_id: node for node in plan.nodes}
    values: dict[tuple[str, str], object] = {}
    observations: list[NodeObservation] = []
    for node_id in topological_nodes(plan):
        node = nodes[node_id]
        inputs = {name: request_inputs[source.name] if source.kind == "request" else values[(source.name, source.port or "")] for name, source in node.inputs.items()}
        started = perf_counter()
        try:
            outputs = runners[node_id].run(inputs, configurations[node_id])
        except Exception as error:
            return ExecutionResult("execution_failed", {}, tuple(observations), setup_latency_ms, configurations[node_id].configuration_id, node_id, f"{type(error).__name__}: {error}")
        observations.append(NodeObservation(node_id, configurations[node_id].configuration_id, (perf_counter() - started) * 1000))
        values.update({(node_id, port): value for port, value in outputs.items()})
    return ExecutionResult("completed", {final.port: values[(final.node_id, final.port)] for final in plan.final_outputs}, tuple(observations), setup_latency_ms)
