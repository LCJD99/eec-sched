"""Deterministic validation of untrusted planner output."""

from __future__ import annotations

from importlib import import_module

from .domain import PlanningRequest, ToolCallPlan, ToolRegistry, ValidationError


def _request_modality(value: object) -> str | None:
    if isinstance(value, str):
        return "text"
    try:
        image_module = import_module("PIL.Image")
        if isinstance(value, image_module.Image):
            return "image"
    except ImportError:
        pass
    return None


def validate_plan(plan: ToolCallPlan, request: PlanningRequest, registry: ToolRegistry) -> tuple[ValidationError, ...]:
    errors: list[ValidationError] = []
    nodes = {node.node_id: node for node in plan.nodes}
    if len(nodes) != len(plan.nodes):
        errors.append(ValidationError("duplicate_node", "node identifiers must be unique"))
    dependencies: dict[str, set[str]] = {node.node_id: set() for node in plan.nodes}
    for node in plan.nodes:
        if not registry.contains(node.tool_id):
            errors.append(ValidationError("unknown_tool", f"{node.tool_id} is not registered"))
            continue
        spec = registry.spec(node.tool_id)
        for port_name, port in spec.inputs.items():
            if port.required and port_name not in node.inputs:
                errors.append(ValidationError("missing_input", f"{node.node_id}.{port_name} is required"))
        for input_name, source in node.inputs.items():
            destination = spec.inputs.get(input_name)
            if destination is None:
                errors.append(ValidationError("unknown_port", f"{node.node_id}.{input_name} is not an input port"))
                continue
            if source.kind == "request":
                if source.name not in request.inputs:
                    errors.append(ValidationError("missing_request_input", f"{source.name} is not supplied"))
                elif _request_modality(request.inputs[source.name]) != destination.modality:
                    errors.append(ValidationError("modality_mismatch", f"request input {source.name} cannot feed {node.node_id}.{input_name}"))
                continue
            upstream = nodes.get(source.name)
            if upstream is None:
                errors.append(ValidationError("unknown_node", f"{source.name} does not exist"))
                continue
            if not registry.contains(upstream.tool_id):
                continue
            output = registry.spec(upstream.tool_id).outputs.get(source.port or "")
            if output is None:
                errors.append(ValidationError("unknown_port", f"{source.name}.{source.port} is not an output port"))
            elif output.modality != destination.modality:
                errors.append(ValidationError("modality_mismatch", f"{source.name}.{source.port} cannot feed {node.node_id}.{input_name}"))
            dependencies[node.node_id].add(source.name)
    temporary: set[str] = set()
    permanent: set[str] = set()
    def visit(node_id: str) -> None:
        if node_id in temporary:
            errors.append(ValidationError("cycle", "tool-call graph contains a cycle"))
        elif node_id not in permanent:
            temporary.add(node_id)
            for parent in dependencies.get(node_id, set()):
                visit(parent)
            temporary.remove(node_id)
            permanent.add(node_id)
    for node_id in dependencies:
        visit(node_id)
    if not plan.final_outputs:
        errors.append(ValidationError("missing_final_output", "a final output is required"))
    for final in plan.final_outputs:
        node = nodes.get(final.node_id)
        if node is None or not registry.contains(node.tool_id) or final.port not in registry.spec(node.tool_id).outputs:
            errors.append(ValidationError("invalid_final_output", f"{final.node_id}.{final.port} is not an output"))
    return tuple(errors)


def topological_nodes(plan: ToolCallPlan) -> tuple[str, ...]:
    pending = {node.node_id: {s.name for s in node.inputs.values() if s.kind == "node"} for node in plan.nodes}
    ordered: list[str] = []
    while pending:
        ready = sorted(node_id for node_id, parents in pending.items() if not parents)
        if not ready:
            raise ValueError("plan must be validated before execution")
        for node_id in ready:
            ordered.append(node_id)
            pending.pop(node_id)
        for parents in pending.values():
            parents.difference_update(ready)
    return tuple(ordered)
