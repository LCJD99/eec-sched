"""Startup-aware dependent task scheduling (SDTS) baseline.

The paper's SDTS heuristic gives ready tasks an upward-rank priority and
places each task on the compatible device with the earliest estimated finish.
The profiling snapshot used by this repository does not contain model image
sizes, environment download times, or device core counts.  Consequently the
environment-download term in Eq. (9) is zero here; no startup number is
invented.  The observable part of the estimate still includes request
ingress, predecessor transfers, device queues, warm execution, and final
output egress.

The paper defines averages over its edge-server set.  This adaptation uses
the devices compatible with the selected configurations and the directed
transfer profiles available in the snapshot, since this repository exposes
``device``, ``edge``, and ``cloud`` as scheduler candidates.
"""

from __future__ import annotations

from typing import Any, Mapping

from eec_sched import SchedulerOutput


_END_DEVICE = "device"
_REQUEST_BYTES = {"text": 256, "image": 262_144, "audio": 1_048_576}


def _parents(dag: Any) -> dict[str, tuple[str, ...]]:
    """Return direct predecessors in each node's declared input order."""

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


def _evidence(
    view: Any,
) -> tuple[
    dict[str, Mapping[str, Any]],
    set[str],
    dict[tuple[str, str], Mapping[str, Any]],
]:
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
    """Index warm execution profiles for compatible devices."""

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
        if profile["configuration_id"] != configuration_id:
            continue
        if profile["device_id"] not in compatible:
            continue
        current = profiles.get(profile["device_id"])
        if current is None or profile["warm_latency_p95_ms"] < current["warm_latency_p95_ms"]:
            profiles[profile["device_id"]] = profile
    return profiles


def _select_configurations(
    dag: Any,
    tools: Mapping[str, Mapping[str, Any]],
    devices: set[str],
) -> tuple[
    dict[str, str],
    dict[str, dict[str, Mapping[str, Any]]],
    dict[str, int],
]:
    """Select the highest quality usable configuration for every tool."""

    selected: dict[str, str] = {}
    profiles_by_tool: dict[str, dict[str, Mapping[str, Any]]] = {}
    output_bytes_by_tool: dict[str, int] = {}

    for tool_id in sorted({node.tool_id for node in dag.nodes}):
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
                candidates.append((
                    quality_profile["normalized_quality_lcb"],
                    configuration_id,
                    profiles,
                    quality_profile,
                ))
        if not candidates:
            raise ValueError(f"no compatible quality configuration for tool {tool_id}")
        best_quality = max(item[0] for item in candidates)
        tied = [item for item in candidates if item[0] == best_quality]
        _, configuration_id, profiles, quality_profile = min(tied, key=lambda item: item[1])
        selected[tool_id] = configuration_id
        profiles_by_tool[tool_id] = profiles
        output_bytes_by_tool[tool_id] = quality_profile["representative_output_bytes"]

    return selected, profiles_by_tool, output_bytes_by_tool


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


def _average_transfer_ms(
    transfers: Mapping[tuple[str, str], Mapping[str, Any]],
    source_devices: Mapping[str, Mapping[str, Any]],
    destination_devices: Mapping[str, Mapping[str, Any]],
    size_bytes: int,
) -> float:
    """Average transfer cost over compatible source/destination devices.

    Same-device pairs contribute zero.  Missing directed links are simply not
    part of the average; a valid snapshot normally contains every directed
    link, while this keeps the helper usable with a reduced test snapshot.
    """

    values = [
        _transfer_ms(transfers, source, destination, size_bytes)
        for source in source_devices
        for destination in destination_devices
        if source == destination or (source, destination) in transfers
    ]
    if not values:
        raise ValueError("no transfer profile between compatible devices")
    return sum(values) / len(values)


def _upward_ranks(
    dag: Any,
    profiles_by_tool: Mapping[str, Mapping[str, Mapping[str, Any]]],
    output_bytes_by_tool: Mapping[str, int],
    transfers: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, float]:
    """Compute Eq. (10) with mean execution and transfer times."""

    parents = _parents(dag)
    children = _children(parents)
    by_id = {node.node_id: node for node in dag.nodes}
    memo: dict[str, float] = {}

    def rank(node_id: str) -> float:
        if node_id in memo:
            return memo[node_id]
        node = by_id[node_id]
        profiles = profiles_by_tool[node.tool_id]
        own = sum(profile["warm_latency_p95_ms"] for profile in profiles.values()) / len(profiles)
        successor_work = 0.0
        for child_id in children[node_id]:
            child = by_id[child_id]
            transfer = _average_transfer_ms(
                transfers,
                profiles,
                profiles_by_tool[child.tool_id],
                output_bytes_by_tool[node.tool_id],
            )
            successor_work = max(successor_work, transfer + rank(child_id))
        # The environment-download term in Eq. (9) is zero because no image
        # size or environment preparation profile is exposed by SchedulerView.
        memo[node_id] = own + successor_work
        return memo[node_id]

    for node_id in by_id:
        rank(node_id)
    return memo


def _resource_state(
    view: Any,
    devices: set[str],
    transfers: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[float, dict[str, float], dict[tuple[str, str], float]]:
    """Convert current replay state into absolute resource availability."""

    state = view.system_state
    now = float(state.get("time_ms", 0.0))
    device_available = {device_id: now for device_id in devices}
    device_state = state.get("devices", {})
    for device_id in devices:
        details = device_state.get(device_id, {})
        backlog = details.get("committed_work_ms", details.get("remaining_ms", 0.0))
        device_available[device_id] = now + float(backlog)

    link_available: dict[tuple[str, str], float] = {}
    link_state = state.get("links", {})
    for source_device, destination_device in transfers:
        details = link_state.get(f"{source_device}->{destination_device}", {})
        backlog = details.get("committed_work_ms", details.get("remaining_ms", 0.0))
        link_available[(source_device, destination_device)] = now + float(backlog)
    return now, device_available, link_available


def _estimate_node(
    node: Any,
    device_id: str,
    profile: Mapping[str, Any],
    by_id: Mapping[str, Any],
    output_bytes_by_tool: Mapping[str, int],
    transfers: Mapping[tuple[str, str], Mapping[str, Any]],
    device_available: Mapping[str, float],
    link_available: Mapping[tuple[str, str], float],
    node_finish: Mapping[str, float],
    node_device: Mapping[str, str],
    final_output_count: int,
) -> tuple[float, float, dict[tuple[str, str], float]]:
    """Return execution finish, completion estimate, and non-egress queues."""

    projected_links = dict(link_available)

    def reserve(source: str, destination: str, ready_time: float, size_bytes: int) -> float:
        if source == destination:
            return ready_time
        key = (source, destination)
        start = max(ready_time, projected_links[key])
        finish = start + _transfer_ms(transfers, source, destination, size_bytes)
        projected_links[key] = finish
        return finish

    input_ready = device_available[device_id]
    for source in node.inputs.values():
        if source.kind == "request":
            if device_id == _END_DEVICE:
                continue
            input_ready = max(
                input_ready,
                reserve(_END_DEVICE, device_id, 0.0, _REQUEST_BYTES[source.data_type or "text"]),
            )
            continue
        parent_id = source.name
        parent_device = node_device[parent_id]
        parent_finish = node_finish[parent_id]
        if parent_device == device_id:
            input_ready = max(input_ready, parent_finish)
        else:
            parent = by_id[parent_id]
            input_ready = max(
                input_ready,
                reserve(
                    parent_device,
                    device_id,
                    parent_finish,
                    output_bytes_by_tool[parent.tool_id],
                ),
            )

    execution_finish = max(input_ready, device_available[device_id]) + profile["warm_latency_p95_ms"]

    # Final output transfers are used to compare this node's completion time,
    # but are deliberately not committed to projected_links.  The evaluator
    # schedules egress after all DAG nodes complete, so this temporary estimate
    # must not delay placement estimates for later nodes.
    completion = execution_finish
    egress_links = dict(projected_links)
    for _ in range(final_output_count):
        completion = max(
            completion,
            _reserve_on_links(
                egress_links,
                transfers,
                device_id,
                _END_DEVICE,
                execution_finish,
                output_bytes_by_tool[node.tool_id],
            ),
        )
    return execution_finish, completion, projected_links


def _reserve_on_links(
    link_available: dict[tuple[str, str], float],
    transfers: Mapping[tuple[str, str], Mapping[str, Any]],
    source: str,
    destination: str,
    ready_time: float,
    size_bytes: int,
) -> float:
    if source == destination:
        return ready_time
    key = (source, destination)
    start = max(ready_time, link_available[key])
    finish = start + _transfer_ms(transfers, source, destination, size_bytes)
    link_available[key] = finish
    return finish


def propose(view: Any) -> SchedulerOutput:
    """Select the longest candidate DAG and greedily place its ready nodes."""

    candidates = tuple(view.candidate_dags)
    path_index = max(
        range(len(candidates)),
        key=lambda index: (_longest_chain(candidates[index]), -index),
    )
    dag = candidates[path_index]
    tools, devices, transfers = _evidence(view)
    selected_configs, profiles_by_tool, output_bytes_by_tool = _select_configurations(
        dag, tools, devices
    )
    ranks = _upward_ranks(dag, profiles_by_tool, output_bytes_by_tool, transfers)
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
        choices: list[tuple[float, float, str, Mapping[str, Any], dict[tuple[str, str], float]]] = []
        for device_id, profile in profiles_by_tool[node.tool_id].items():
            execution_finish, completion, projected_links = _estimate_node(
                node,
                device_id,
                profile,
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
        if not choices:
            raise ValueError(f"no compatible execution profile for node {node_id}")
        completion, execution_finish, device_id, _profile, projected_links = min(
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
