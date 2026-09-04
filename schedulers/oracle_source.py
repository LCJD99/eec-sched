"""Exhaustive reference Scheduler Candidate.

This source is intentionally self-contained because Candidate source is loaded
without importing project modules.  It enumerates every compatible
configuration/device assignment and scores each assignment using the same
latency, energy, transfer, and quality formula as the trusted evaluator.
"""


_EPSILON = 1e-12


def _predecessors(dag):
    predecessors = {node.node_id: [] for node in dag.nodes}
    for node in dag.nodes:
        for source in node.inputs.values():
            if source.kind == "node":
                predecessors[node.node_id].append(source.name)
    return predecessors


def _relevant_nodes(view, predecessors):
    required = {output.node_id for output in view.dag.final_outputs}
    pending = list(required)
    while pending:
        node_id = pending.pop()
        for parent in predecessors[node_id]:
            if parent not in required:
                required.add(parent)
                pending.append(parent)
    return required


def _tool_index(view):
    return {tool["tool_id"]: tool for tool in view.snapshot_evidence["tools"]}


def _device_ids(view):
    return tuple(device["device_id"] for device in view.snapshot_evidence["devices"])


def _options(view, node, tools, devices):
    tool = tools[node.tool_id]
    execution = {
        (profile["configuration_id"], profile["device_id"]): profile
        for profile in tool["execution_profiles"]
    }
    choices = []
    for configuration in tool["configurations"]:
        configuration_id = configuration["configuration_id"]
        compatible = {
            item["device_id"]
            for item in configuration["device_compatibility"]
            if item["status"] == "compatible"
        }
        for device_id in devices:
            if device_id in compatible and (configuration_id, device_id) in execution:
                choices.append((configuration_id, device_id))
    if not choices:
        raise ValueError("no compatible configuration/device choice for " + node.node_id)
    return tuple(choices)


def _profile(tool, configuration_id):
    for profile in tool["quality_profiles"]:
        if profile["configuration_id"] == configuration_id:
            return profile
    raise KeyError(configuration_id)


def _execution(tool, configuration_id, device_id):
    for profile in tool["execution_profiles"]:
        if profile["configuration_id"] == configuration_id and profile["device_id"] == device_id:
            return profile
    raise KeyError((configuration_id, device_id))


def _transfer(transfers, source, destination):
    for profile in transfers:
        if profile["source_device_id"] == source and profile["destination_device_id"] == destination:
            return profile
    raise KeyError((source, destination))


def _simulate(view, assignments, tools, predecessors, transfers, devices):
    device_free = {device_id: 0.0 for device_id in devices}
    link_free = {}
    finish = {}
    transfer_ready = {node.node_id: 0.0 for node in view.dag.nodes}
    remaining = {node.node_id for node in view.dag.nodes}
    by_id = {node.node_id: node for node in view.dag.nodes}
    total_energy = 0.0

    while remaining:
        ready = [
            node_id
            for node_id in remaining
            if all(parent in finish for parent in predecessors[node_id])
        ]
        if not ready:
            raise ValueError("DAG contains a cycle or unknown predecessor")
        node_id = min(ready)
        node = by_id[node_id]
        configuration_id, device_id = assignments[node_id]
        profile = _execution(tools[node.tool_id], configuration_id, device_id)
        start = max(device_free[device_id], transfer_ready[node_id])
        finish_time = start + profile["warm_latency_p95_ms"]
        finish[node_id] = finish_time
        device_free[device_id] = finish_time
        total_energy += profile["mean_incremental_execution_energy_j"]

        for child_id, parents in predecessors.items():
            if node_id not in parents:
                continue
            child_configuration, child_device = assignments[child_id]
            if device_id == child_device:
                transfer_ready[child_id] = max(transfer_ready[child_id], finish_time)
                continue
            transfer = _transfer(transfers, device_id, child_device)
            output_bytes = _profile(tools[node.tool_id], configuration_id)["representative_output_bytes"]
            latency = transfer["propagation_delay_ms"] + 1000.0 * output_bytes / transfer["bandwidth_bytes_per_second"]
            energy = transfer["setup_energy_j"] + output_bytes * transfer["energy_per_byte_j"]
            link = (device_id, child_device)
            transfer_start = max(finish_time, link_free.get(link, 0.0))
            transfer_finish = transfer_start + latency
            link_free[link] = transfer_finish
            transfer_ready[child_id] = max(transfer_ready[child_id], transfer_finish)
            total_energy += energy
        remaining.remove(node_id)

    return max(finish.values(), default=0.0), total_energy


def _accuracy(view, assignments, tools, relevant, by_id):
    result = 1.0
    for node_id in sorted(relevant):
        node = by_id[node_id]
        configuration_id, _ = assignments[node_id]
        result *= _profile(tools[node.tool_id], configuration_id)["normalized_quality_lcb"]
    return result


def _score(view, assignments, tools, relevant, by_id, predecessors, transfers, devices):
    makespan, energy = _simulate(view, assignments, tools, predecessors, transfers, devices)
    accuracy = _accuracy(view, assignments, tools, relevant, by_id)
    context = view.scoring_context
    accuracy_surplus = 0.0 if context.minimum_accuracy == 1 else (
        (accuracy - context.minimum_accuracy) / (1 - context.minimum_accuracy)
    )
    latency_surplus = (context.maximum_latency_ms - makespan) / context.maximum_latency_ms
    performance = context.gamma * accuracy_surplus + (1 - context.gamma) * latency_surplus
    return performance / (energy + _EPSILON)


def propose(view):
    """Return the globally best assignment under the trusted scoring model."""
    tools = _tool_index(view)
    devices = _device_ids(view)
    transfers = view.snapshot_evidence["transfer_profiles"]
    predecessors = _predecessors(view.dag)
    relevant = _relevant_nodes(view, predecessors)
    by_id = {node.node_id: node for node in view.dag.nodes}
    nodes = tuple(view.dag.nodes)
    choices = tuple(_options(view, node, tools, devices) for node in nodes)

    best_score = None
    best_assignment = None

    def search(index, current):
        nonlocal best_score, best_assignment
        if index == len(nodes):
            score = _score(view, current, tools, relevant, by_id, predecessors, transfers, devices)
            if best_score is None or score > best_score:
                best_score = score
                best_assignment = dict(current)
            return
        node_id = nodes[index].node_id
        for configuration_id, device_id in choices[index]:
            current[node_id] = (configuration_id, device_id)
            search(index + 1, current)
        current.pop(node_id, None)

    search(0, {})
    if best_assignment is None:
        raise ValueError("DAG has no schedulable assignment")
    return {
        node_id: {"configuration_id": configuration_id, "device_id": device_id}
        for node_id, (configuration_id, device_id) in best_assignment.items()
    }
