"""Naive resource-first Scheduler Candidate.

For each node, choose the compatible execution profile with the lowest
profiled GPU-memory resource. Latency and quality are deterministic tie
breakers; DAG-wide transfer effects are intentionally ignored.
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
        quality = {
            profile["configuration_id"]: profile["normalized_quality_lcb"]
            for profile in tool["quality_profiles"]
        }
        candidates = [
            profile
            for profile in tool["execution_profiles"]
            if profile["device_id"] in compatible.get(profile["configuration_id"], set())
            and profile["configuration_id"] in quality
        ]
        if not candidates:
            raise ValueError("no compatible execution profile for " + node.node_id)
        selected = min(
            candidates,
            key=lambda profile: (
                profile["gpu_memory_mib"],
                profile["warm_latency_p95_ms"],
                -quality[profile["configuration_id"]],
                profile["configuration_id"],
                profile["device_id"],
            ),
        )
        assignments[node.node_id] = {
            "configuration_id": selected["configuration_id"],
            "device_id": selected["device_id"],
        }
    return assignments
