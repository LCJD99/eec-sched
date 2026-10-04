"""Profile-guided Murakkab placement baseline.

Murakkab's paper solves a profile-based MILP over workflow configurations,
model profiles, demand, SLOs, and resource budgets.  The scheduler interface
in this repository exposes one request DAG, a small device set, execution and
transfer profiles, and current queue state instead.  This baseline therefore
uses the part of the design that is observable here: choose a high-quality
profiled configuration and place the whole DAG using a deterministic
profile-guided heuristic.

The heuristic first chooses the candidate DAG with the longest dependency
chain (the lowest candidate index breaks ties).  For each tool it chooses the
usable configuration with the largest ``normalized_quality_lcb``.  A
configuration is usable only when a compatible device also has an execution
profile.  It then starts from a stable placement and performs coordinate
descent over nodes.  Each trial schedules the complete DAG, including request
ingress, predecessor transfers, device/link committed work, execution, and
final-output egress.  The placement objective is a weighted sum of latency and
GPU-memory penalties normalized by ``ScoringContext`` scales.

The snapshot does not expose Murakkab's demand traces, SLO filters, energy,
cost, instance counts, or model-loading state, so this file does not claim to
reproduce the paper's MILP or its energy/cost optimization.  Those fields are
not fabricated here.
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


def _longest_chain(dag: Any) -> int:
    """Return the node count of the longest dependency chain."""

    parents = _parents(dag)
    memo: dict[str, int] = {}
    visiting: set[str] = set()

    def depth(node_id: str) -> int:
        if node_id in memo:
            return memo[node_id]
        if node_id in visiting:
            raise ValueError("DAG contains a cycle")
        visiting.add(node_id)
        value = 1 + max((depth(parent) for parent in parents[node_id]), default=0)
        visiting.remove(node_id)
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
    """Index the best execution profile per compatible device."""

    configuration = next(
        item
        for item in tool["configurations"]
        if item["configuration_id"] == configuration_id
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
    return {device_id: profiles[device_id] for device_id in sorted(profiles)}


def _select_configurations(
    dag: Any,
    tools: Mapping[str, Mapping[str, Any]],
    devices: set[str],
) -> tuple[
    dict[str, str],
    dict[str, dict[str, Mapping[str, Any]]],
    dict[str, int],
]:
    """Choose the highest-quality usable configuration for every tool."""

    selected: dict[str, str] = {}
    profiles_by_tool: dict[str, dict[str, Mapping[str, Any]]] = {}
    output_bytes_by_tool: dict[str, int] = {}

    for tool_id in sorted({node.tool_id for node in dag.nodes}):
        tool = tools[tool_id]
        quality_by_configuration = {
            profile["configuration_id"]: profile
            for profile in tool["quality_profiles"]
        }
        candidates: list[
            tuple[float, str, dict[str, Mapping[str, Any]], Mapping[str, Any]]
        ] = []
        for configuration in tool["configurations"]:
            configuration_id = configuration["configuration_id"]
            quality_profile = quality_by_configuration.get(configuration_id)
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
        best_quality = max(item[0] for item in candidates)
        tied = [item for item in candidates if item[0] == best_quality]
        _, configuration_id, profiles, quality_profile = min(
            tied, key=lambda item: item[1]
        )
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


def _resource_state(
    view: Any,
    devices: set[str],
    transfers: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[float, dict[str, float], dict[tuple[str, str], float]]:
    """Convert current replay state into absolute resource availability."""

    state = view.system_state
    now = float(state.get("time_ms", 0.0))
    device_available = {device_id: now for device_id in devices}
    for device_id in devices:
        details = state.get("devices", {}).get(device_id, {})
        backlog = details.get("committed_work_ms", details.get("remaining_ms", 0.0))
        device_available[device_id] = now + float(backlog)

    link_available: dict[tuple[str, str], float] = {}
    for source_device, destination_device in transfers:
        details = state.get("links", {}).get(
            f"{source_device}->{destination_device}", {}
        )
        backlog = details.get("committed_work_ms", details.get("remaining_ms", 0.0))
        link_available[(source_device, destination_device)] = now + float(backlog)
    return now, device_available, link_available


def _schedule(
    view: Any,
    dag: Any,
    assignments: Mapping[str, Mapping[str, str]],
    profiles_by_tool: Mapping[str, Mapping[str, Mapping[str, Any]]],
    output_bytes_by_tool: Mapping[str, int],
    transfers: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[float, float]:
    """Estimate absolute completion latency and aggregate GPU memory.

    Nodes use the evaluator's deterministic ready-node order.  A trial starts
    from the same observed device and link queues, so coordinate descent never
    mutates the live replay state.
    """

    parents = _parents(dag)
    by_id = {node.node_id: node for node in dag.nodes}
    profiled_devices = {
        device_id
        for tool_profiles in profiles_by_tool.values()
        for device_id in tool_profiles
    }
    now, device_available, link_available = _resource_state(
        view, profiled_devices, transfers
    )
    # The assignment set is the authoritative set of devices for this trial.
    device_available = {
        device_id: device_available[device_id]
        for device_id in {choice["device_id"] for choice in assignments.values()}
    }

    node_finish: dict[str, float] = {}
    transfer_ready = {node_id: now for node_id in by_id}
    remaining = set(by_id)

    def reserve(
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

    while remaining:
        ready = sorted(
            node_id
            for node_id in remaining
            if all(parent in node_finish for parent in parents[node_id])
        )
        if not ready:
            raise ValueError("DAG contains a cycle or unknown predecessor")
        node_id = ready[0]
        node = by_id[node_id]
        assignment = assignments[node_id]
        device_id = assignment["device_id"]
        profile = profiles_by_tool[node.tool_id][device_id]
        input_ready = transfer_ready[node_id]

        for source in node.inputs.values():
            if source.kind == "request":
                input_ready = max(
                    input_ready,
                    reserve(
                        _END_DEVICE,
                        device_id,
                        now,
                        _REQUEST_BYTES[source.data_type or "text"],
                    ),
                )
                continue

        execution_start = max(input_ready, device_available[device_id])
        finish = execution_start + profile["warm_latency_p95_ms"]
        node_finish[node_id] = finish
        device_available[device_id] = finish

        # The trusted simulator reserves every outgoing transfer as soon as
        # the parent finishes.  This matters when fan-out transfers share a
        # directional link with another child or with a later parent.
        for child_id, child in by_id.items():
            occurrences = sum(
                1
                for source in child.inputs.values()
                if source.kind == "node" and source.name == node_id
            )
            if occurrences == 0:
                continue
            child_device = assignments[child_id]["device_id"]
            for _ in range(occurrences):
                if device_id == child_device:
                    transfer_ready[child_id] = max(transfer_ready[child_id], finish)
                    continue
                transfer_ready[child_id] = max(
                    transfer_ready[child_id],
                    reserve(
                        device_id,
                        child_device,
                        finish,
                        output_bytes_by_tool[node.tool_id],
                    ),
                )
        remaining.remove(node_id)

    completion = max(node_finish.values(), default=now)
    for output in dag.final_outputs:
        node = by_id[output.node_id]
        device_id = assignments[node.node_id]["device_id"]
        completion = max(
            completion,
            reserve(
                device_id,
                _END_DEVICE,
                node_finish[node.node_id],
                output_bytes_by_tool[node.tool_id],
            ),
        )

    resource = sum(
        profiles_by_tool[by_id[node_id].tool_id][assignment["device_id"]][
            "gpu_memory_mib"
        ]
        for node_id, assignment in assignments.items()
    )
    return completion - now, resource


def _objective(
    view: Any,
    latency_ms: float,
    resource_mib: float,
) -> float:
    """Return a lower-is-better normalized placement objective."""

    context = getattr(view, "scoring_context", None)
    latency_weight = float(getattr(context, "latency_weight", 1.0 / 3.0))
    resource_weight = float(getattr(context, "resource_weight", 1.0 / 3.0))
    latency_scale = float(getattr(context, "latency_scale_ms", 100.0))
    resource_scale = float(getattr(context, "resource_scale_mib", 8192.0))
    return (
        latency_weight * latency_ms / latency_scale
        + resource_weight * resource_mib / resource_scale
    )


def propose(view: Any) -> SchedulerOutput:
    """Select the longest candidate DAG and profile-guided place its nodes."""

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

    by_id = {node.node_id: node for node in dag.nodes}
    assignments: dict[str, dict[str, str]] = {
        node_id: {
            "configuration_id": selected_configs[node.tool_id],
            "device_id": min(profiles_by_tool[node.tool_id]),
        }
        for node_id, node in sorted(by_id.items())
    }

    # Coordinate descent evaluates the entire DAG for each local alternative.
    # The finite profile set and strict deterministic key keep this bounded in
    # practice without enumerating the Cartesian product of placements.
    for _ in range(max(1, len(by_id) * max(1, len(devices)))):
        changed = False
        for node_id in sorted(by_id):
            node = by_id[node_id]
            choices: list[tuple[float, float, float, str]] = []
            for device_id in sorted(profiles_by_tool[node.tool_id]):
                trial = dict(assignments)
                trial[node_id] = {
                    "configuration_id": selected_configs[node.tool_id],
                    "device_id": device_id,
                }
                latency_ms, resource_mib = _schedule(
                    view,
                    dag,
                    trial,
                    profiles_by_tool,
                    output_bytes_by_tool,
                    transfers,
                )
                choices.append(
                    (
                        _objective(view, latency_ms, resource_mib),
                        latency_ms,
                        resource_mib,
                        device_id,
                    )
                )
            best = min(choices)
            current_latency, current_resource = _schedule(
                view,
                dag,
                assignments,
                profiles_by_tool,
                output_bytes_by_tool,
                transfers,
            )
            current_key = (
                _objective(view, current_latency, current_resource),
                current_latency,
                current_resource,
            )
            if best[:3] < current_key:
                assignments[node_id] = {
                    "configuration_id": selected_configs[node.tool_id],
                    "device_id": best[3],
                }
                changed = True
        if not changed:
            break

    return SchedulerOutput(dag, assignments, path_index=path_index)
