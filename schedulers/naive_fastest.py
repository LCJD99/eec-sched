"""Naive latency-first Scheduler Candidate.

For each node, choose the compatible execution profile with the smallest
warm p95 latency.  This is intentionally local: it does not reason about
downstream transfers or the global utility score.
"""


def propose(view):
    tools = {tool["tool_id"]: tool for tool in view.snapshot_evidence["tools"]}
    assignments = {}
    device_ids = {device["device_id"] for device in view.snapshot_evidence["devices"]}
    for node in view.dag.nodes:
        tool = tools[node.tool_id]
        compatible = {}
        for configuration in tool["configurations"]:
            configuration_id = configuration["configuration_id"]
            compatible[configuration_id] = {
                item["device_id"]
                for item in configuration["device_compatibility"]
                if item["status"] == "compatible" and item["device_id"] in device_ids
            }
        candidates = [
            profile
            for profile in tool["execution_profiles"]
            if profile["device_id"] in compatible.get(profile["configuration_id"], set())
        ]
        if not candidates:
            raise ValueError("no compatible execution profile for " + node.node_id)
        selected = min(
            candidates,
            key=lambda profile: (
                profile["warm_latency_p95_ms"],
                profile["mean_incremental_execution_energy_j"],
                profile["configuration_id"],
                profile["device_id"],
            ),
        )
        assignments[node.node_id] = {
            "configuration_id": selected["configuration_id"],
            "device_id": selected["device_id"],
        }
    return assignments
