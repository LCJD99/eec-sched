"""Initial latency-first Scheduler Candidate used for evaluation smoke tests."""


def propose(view):
    """Assign every node to its fastest profiled Configuration/device pair.

    The trusted evaluator still validates the complete proposal and calculates
    its score.  This candidate only reads the per-Trace Scheduler View.
    """
    tools = {tool["tool_id"]: tool for tool in view.snapshot_evidence["tools"]}
    assignments = {}
    for node in view.dag.nodes:
        profiles = tools[node.tool_id]["execution_profiles"]
        fastest = min(
            profiles,
            key=lambda profile: (
                profile["warm_latency_p95_ms"],
                profile["mean_incremental_execution_energy_j"],
                profile["configuration_id"],
                profile["device_id"],
            ),
        )
        assignments[node.node_id] = {
            "configuration_id": fastest["configuration_id"],
            "device_id": fastest["device_id"],
        }
    return assignments
