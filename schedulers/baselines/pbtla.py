"""PBTLA-inspired dependency-aware placement baseline.

This is the first heuristic scheduling baseline adapted from the paper's
PriAssign (Algorithm 1) and Greedy Batch-driven Scheduling (Algorithm 2).
The paper ranks ready subtasks by the delay their successors may add to the
critical path, then greedily places each subtask on the device with the
earliest estimated completion time.  This module keeps that placement idea
while adapting it to this repository's ``propose(view)`` contract:

* among the three Planner candidates, select the DAG with the longest
  dependency chain (number of nodes; the lowest candidate index breaks ties);
* select the highest ``normalized_quality_lcb`` configuration for each tool,
  considering only configurations with a compatible execution profile; and
* process ready nodes in descending remaining critical-path order, estimating
  device queue, predecessor transfers, request ingress, and final-output
  egress before committing a device choice.

The profiling evidence exposed by ``SchedulerView`` has no image-layer or
virtual-core state, so this baseline does not reproduce the paper's layer
loading/deletion or core allocation decisions.  It also does not invent layer
identifiers or verify snapshot hashes.  The evaluator owns the final
node-order simulation; the priority here guides placement estimates.
"""

from __future__ import annotations

from typing import Any, Mapping

from eec_sched import SchedulerOutput


_END_DEVICE = "device"
_REQUEST_BYTES = {"text": 256, "image": 262_144, "audio": 1_048_576}


def _parents(dag: Any) -> dict[str, tuple[str, ...]]:
    """Return direct predecessors in the DAG's declared input order."""

    result: dict[str, list[str]] = {node.node_id: [] for node in dag.nodes}
    for node in dag.nodes:
        for source in node.inputs.values():
            if source.kind == "node":
                result[node.node_id].append(source.name)
    return {node_id: tuple(parents) for node_id, parents in result.items()}


def _children(parents: Mapping[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    result: dict[str, list[str]] = {node_id: [] for node_id in parents}
    for child, direct_parents in parents.items():
        for parent in direct_parents:
            result[parent].append(child)
    return {node_id: tuple(children) for node_id, children in result.items()}


def _longest_chain(dag: Any) -> int:
    """Return the node count of the longest dependency chain."""

    parents = _parents(dag)
    memo: dict[str, int] = {}

    def depth(node_id: str) -> int:
        if node_id in memo:
            return memo[node_id]
        direct_parents = parents[node_id]
        value = 1 if not direct_parents else 1 + max(depth(parent) for parent in direct_parents)
        memo[node_id] = value
        return value

    return max((depth(node_id) for node_id in parents), default=0)


def _evidence(view: Any) -> tuple[dict[str, Mapping[str, Any]], set[str], dict[tuple[str, str], Mapping[str, Any]]]:
    snapshot = view.snapshot_evidence
    tools = {tool["tool_id"]: tool for tool in snapshot["tools"]}
    devices = {device["device_id"] for device in snapshot["devices"]}
    transfers = {
        (profile["source_device_id"], profile["destination_device_id"]): profile
        for profile in snapshot["transfer_profiles"]
    }
    return tools, devices, transfers


def _compatible_profiles(
    tool: Mapping[str, Any],
    configuration_id: str,
    devices: set[str],
) -> dict[str, Mapping[str, Any]]:
    """Index execution profiles that agree with configuration compatibility."""

    configuration = next(
        item for item in tool["configurations"] if item["configuration_id"] == configuration_id
    )
    compatible = {
        item["device_id"]
        for item in configuration["device_compatibility"]
        if item["status"] == "compatible" and item["device_id"] in devices
    }
    profiles: dict[str, Mapping[str, Any]] = {}
    for profile in tool["execution_profiles"]:
        if profile["configuration_id"] != configuration_id or profile["device_id"] not in compatible:
            continue
        current = profiles.get(profile["device_id"])
        if current is None or profile["warm_latency_p95_ms"] < current["warm_latency_p95_ms"]:
            profiles[profile["device_id"]] = profile
    return profiles


def _select_configurations(
    dag: Any,
    tools: Mapping[str, Mapping[str, Any]],
    devices: set[str],
) -> tuple[dict[str, str], dict[str, dict[str, Mapping[str, Any]]], dict[str, float], dict[str, int]]:
    """Choose one quality-first configuration and its usable profiles per tool."""

    selected: dict[str, str] = {}
    profiles_by_tool: dict[str, dict[str, Mapping[str, Any]]] = {}
    latency_by_tool: dict[str, float] = {}
    output_bytes_by_tool: dict[str, int] = {}
    tool_ids = {node.tool_id for node in dag.nodes}

    for tool_id in sorted(tool_ids):
        tool = tools[tool_id]
        quality = {
            profile["configuration_id"]: profile
            for profile in tool["quality_profiles"]
        }
        candidates: list[tuple[float, str, dict[str, Mapping[str, Any]], Mapping[str, Any]]] = []
        for configuration in tool["configurations"]:
            configuration_id = configuration["configuration_id"]
            quality_profile = quality.get(configuration_id)
            if quality_profile is None:
                continue
            profiles = _compatible_profiles(tool, configuration_id, devices)
            if profiles:
                candidates.append(
                    (
                        quality_profile["normalized_quality_lcb"],
                        configuration_id,
                        profiles,
                        quality_profile,
                    )
                )
        if not candidates:
            raise ValueError(f"no compatible quality configuration for tool {tool_id}")
        quality_value = max(item[0] for item in candidates)
        tied = [item for item in candidates if item[0] == quality_value]
        _, configuration_id, profiles, quality_profile = min(tied, key=lambda item: item[1])
        selected[tool_id] = configuration_id
        profiles_by_tool[tool_id] = profiles
        latency_by_tool[tool_id] = min(profile["warm_latency_p95_ms"] for profile in profiles.values())
        output_bytes_by_tool[tool_id] = quality_profile["representative_output_bytes"]

    return selected, profiles_by_tool, latency_by_tool, output_bytes_by_tool


def _transfer_ms(
    transfers: Mapping[tuple[str, str], Mapping[str, Any]],
    source: str,
    destination: str,
    size_bytes: int,
) -> float:
    if source == destination:
        return 0.0
    profile = transfers[(source, destination)]
    return profile["propagation_delay_ms"] + 1000.0 * size_bytes / profile["bandwidth_bytes_per_second"]


def _critical_ranks(
    dag: Any,
    profiles_by_tool: Mapping[str, Mapping[str, Mapping[str, Any]]],
    latency_by_tool: Mapping[str, float],
    output_bytes_by_tool: Mapping[str, int],
    transfers: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, float]:
    """Estimate remaining critical-path work with optimistic device choices."""

    parents = _parents(dag)
    children = _children(parents)
    by_id = {node.node_id: node for node in dag.nodes}
    memo: dict[str, float] = {}

    def rank(node_id: str) -> float:
        if node_id in memo:
            return memo[node_id]
        node = by_id[node_id]
        own = latency_by_tool[node.tool_id]
        successor_work = 0.0
        for child_id in children[node_id]:
            child = by_id[child_id]
            min_transfer = min(
                _transfer_ms(transfers, source_device, destination_device, output_bytes_by_tool[node.tool_id])
                for source_device in profiles_by_tool[node.tool_id].keys()
                for destination_device in profiles_by_tool[child.tool_id].keys()
            )
            successor_work = max(successor_work, min_transfer + rank(child_id))
        # Final output transfer is part of the request's completion time.
        if any(output.node_id == node_id for output in dag.final_outputs):
            min_egress = min(
                _transfer_ms(transfers, device_id, _END_DEVICE, output_bytes_by_tool[node.tool_id])
                for device_id in profiles_by_tool[node.tool_id]
            )
            successor_work = max(successor_work, min_egress)
        memo[node_id] = own + successor_work
        return memo[node_id]

    for node_id in by_id:
        rank(node_id)
    return memo


def _resource_state(view: Any, devices: set[str], transfers: Mapping[tuple[str, str], Mapping[str, Any]]) -> tuple[float, dict[str, float], dict[tuple[str, str], float]]:
    """Convert current replay state into absolute availability estimates."""

    state = view.system_state
    now = float(state.get("time_ms", 0.0))
    device_state = state.get("devices", {})
    device_available = {device_id: now for device_id in devices}
    for device_id in devices:
        details = device_state.get(device_id, {})
        backlog = details.get("committed_work_ms", details.get("remaining_ms", 0.0))
        device_available[device_id] = now + float(backlog)

    link_state = state.get("links", {})
    link_available: dict[tuple[str, str], float] = {}
    for source_device, destination_device in transfers:
        details = link_state.get(f"{source_device}->{destination_device}", {})
        backlog = details.get("committed_work_ms", details.get("remaining_ms", 0.0))
        link_available[(source_device, destination_device)] = now + float(backlog)
    return now, device_available, link_available


def _estimate_node(
    node: Any,
    device_id: str,
    profile: Mapping[str, Any],
    parents: Mapping[str, tuple[str, ...]],
    by_id: Mapping[str, Any],
    output_bytes_by_tool: Mapping[str, int],
    transfers: Mapping[tuple[str, str], Mapping[str, Any]],
    device_available: Mapping[str, float],
    link_available: Mapping[tuple[str, str], float],
    node_finish: Mapping[str, float],
    node_device: Mapping[str, str],
    request_output_count: int,
) -> tuple[float, float, dict[tuple[str, str], float]]:
    """Estimate execution and completion, returning non-egress reservations."""

    projected_links = dict(link_available)

    def reserve(source: str, destination: str, ready_time: float, size_bytes: int) -> float:
        if source == destination:
            return ready_time
        key = (source, destination)
        start = max(ready_time, projected_links[key])
        finish = start + _transfer_ms(transfers, source, destination, size_bytes)
        projected_links[key] = finish
        return finish

    transfer_ready = device_available[device_id]
    for source in node.inputs.values():
        if source.kind == "request":
            if device_id == _END_DEVICE:
                continue
            input_type = source.data_type or "text"
            transfer_ready = max(
                transfer_ready,
                reserve(_END_DEVICE, device_id, 0.0, _REQUEST_BYTES[input_type]),
            )
            continue
        parent_id = source.name
        parent_device = node_device[parent_id]
        parent_finish = node_finish[parent_id]
        if parent_device == device_id:
            transfer_ready = max(transfer_ready, parent_finish)
        else:
            parent = by_id[parent_id]
            transfer_ready = max(
                transfer_ready,
                reserve(
                    parent_device,
                    device_id,
                    parent_finish,
                    output_bytes_by_tool[parent.tool_id],
                ),
            )

    execution_finish = max(transfer_ready, device_available[device_id]) + profile["warm_latency_p95_ms"]
    # The trusted simulator schedules final-output transfers only after all
    # DAG nodes finish.  Include egress in this candidate's completion
    # estimate, but keep its temporary link occupancy out of reservations
    # carried into later node-placement estimates.
    execution_links = dict(projected_links)
    completion = execution_finish
    for _ in range(request_output_count):
        completion = max(
            completion,
            reserve(
                device_id,
                _END_DEVICE,
                execution_finish,
                output_bytes_by_tool[node.tool_id],
            ),
        )
    return execution_finish, completion, execution_links


def propose(view: Any) -> SchedulerOutput:
    """Choose a dependency-critical DAG and greedily place its nodes."""

    candidates = tuple(view.candidate_dags)
    path_index = max(range(len(candidates)), key=lambda index: (_longest_chain(candidates[index]), -index))
    dag = candidates[path_index]
    tools, devices, transfers = _evidence(view)
    selected_configs, profiles_by_tool, latency_by_tool, output_bytes_by_tool = _select_configurations(
        dag, tools, devices
    )
    ranks = _critical_ranks(
        dag,
        profiles_by_tool,
        latency_by_tool,
        output_bytes_by_tool,
        transfers,
    )
    parents = _parents(dag)
    by_id = {node.node_id: node for node in dag.nodes}
    output_counts: dict[str, int] = {}
    for output in dag.final_outputs:
        output_counts[output.node_id] = output_counts.get(output.node_id, 0) + 1

    _, device_available, link_available = _resource_state(view, devices, transfers)
    assignments: dict[str, dict[str, str]] = {}
    node_finish: dict[str, float] = {}
    node_device: dict[str, str] = {}
    remaining = set(by_id)

    while remaining:
        ready = [
            node_id
            for node_id in remaining
            if all(parent in node_finish for parent in parents[node_id])
        ]
        if not ready:
            raise ValueError("DAG contains a cycle or unknown predecessor")
        node_id = min(ready, key=lambda candidate_id: (-ranks[candidate_id], candidate_id))
        node = by_id[node_id]
        tool_profiles = profiles_by_tool[node.tool_id]
        choices: list[tuple[float, float, str, Mapping[str, Any], dict[tuple[str, str], float]]] = []
        for device_id, profile in tool_profiles.items():
            execution_finish, completion, projected_links = _estimate_node(
                node,
                device_id,
                profile,
                parents,
                by_id,
                output_bytes_by_tool,
                transfers,
                device_available,
                link_available,
                node_finish,
                node_device,
                output_counts.get(node_id, 0),
            )
            choices.append((completion, execution_finish, device_id, profile, projected_links))
        completion, execution_finish, device_id, profile, projected_links = min(
            choices,
            key=lambda choice: (choice[0], choice[1], choice[2]),
        )
        del completion
        assignments[node_id] = {
            "configuration_id": selected_configs[node.tool_id],
            "device_id": device_id,
        }
        node_finish[node_id] = execution_finish
        node_device[node_id] = device_id
        device_available[device_id] = execution_finish
        link_available.update(projected_links)
        remaining.remove(node_id)

    return SchedulerOutput(dag, assignments, path_index=path_index)
