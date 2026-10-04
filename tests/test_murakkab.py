from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from eec_sched import FinalOutput, InputSource, ToolCallPlan, ToolNode
from eec_sched.evaluation.scoring import ScoringContext


SOURCE = Path(__file__).parents[1] / "schedulers/baselines/murakkab.py"
SPEC = importlib.util.spec_from_file_location("murakkab_baseline", SOURCE)
assert SPEC and SPEC.loader
murakkab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(murakkab)


def _node(node_id: str, tool_id: str, parent: str | None = None) -> ToolNode:
    source = InputSource.request("text") if parent is None else InputSource.node(parent, "text")
    return ToolNode(node_id, tool_id, {"text": source})


def _dag(*nodes: ToolNode) -> ToolCallPlan:
    return ToolCallPlan(tuple(nodes), (FinalOutput(nodes[-1].node_id, "text"),))


def _tool(
    tool_id: str,
    profiles: dict[str, dict[str, tuple[float, float]]],
    quality: dict[str, float] | None = None,
    output_bytes: int = 100,
) -> dict[str, object]:
    quality = quality or {configuration_id: 1.0 for configuration_id in profiles}
    return {
        "tool_id": tool_id,
        "configurations": [
            {
                "configuration_id": configuration_id,
                "device_compatibility": [
                    {"device_id": device_id, "status": "compatible"}
                    for device_id in devices
                ],
            }
            for configuration_id, devices in profiles.items()
        ],
        "quality_profiles": [
            {
                "configuration_id": configuration_id,
                "normalized_quality_lcb": value,
                "representative_output_bytes": output_bytes,
            }
            for configuration_id, value in quality.items()
        ],
        "execution_profiles": [
            {
                "configuration_id": configuration_id,
                "device_id": device_id,
                "warm_latency_p95_ms": latency,
                "gpu_memory_mib": memory,
            }
            for configuration_id, devices in profiles.items()
            for device_id, (latency, memory) in devices.items()
        ],
    }


def _view(
    dags: tuple[ToolCallPlan, ...],
    tools: tuple[dict[str, object], ...],
    *,
    scoring_context: ScoringContext | None = None,
    system_state: dict[str, object] | None = None,
) -> SimpleNamespace:
    devices = ("device", "edge", "cloud")
    transfers = tuple(
        {
            "source_device_id": source,
            "destination_device_id": destination,
            "propagation_delay_ms": 0.0,
            "bandwidth_bytes_per_second": 1_000_000,
        }
        for source in devices
        for destination in devices
        if source != destination
    )
    return SimpleNamespace(
        candidate_dags=dags,
        snapshot_evidence={
            "devices": tuple({"device_id": device_id} for device_id in devices),
            "tools": tools,
            "transfer_profiles": transfers,
        },
        system_state=system_state or {},
        scoring_context=scoring_context or ScoringContext(),
    )


def test_selects_longest_chain_and_highest_quality_compatible_configuration() -> None:
    short = _dag(_node("short", "tool"))
    long = _dag(_node("first", "tool"), _node("last", "tool", "first"))
    tool = _tool(
        "tool",
        {
            "small": {"device": (1.0, 0.0)},
            "large": {"edge": (2.0, 1.0)},
        },
        {"small": 0.8, "large": 0.95},
    )

    result = murakkab.propose(_view((short, long, short), (tool,)))

    assert result.path_index == 1
    assert result.dag == long
    assert set(result.assignments) == {"first", "last"}
    assert all(choice == {"configuration_id": "large", "device_id": "edge"}
               for choice in result.assignments.values())


def test_device_backlog_can_move_work_to_another_compatible_device() -> None:
    dag = _dag(_node("task", "tool"))
    tool = _tool("tool", {"cfg": {"edge": (1.0, 1.0), "cloud": (8.0, 1.0)}})
    view = _view(
        (dag, dag, dag),
        (tool,),
        scoring_context=ScoringContext(latency_weight=1.0, resource_weight=0.0, accuracy_weight=0.0),
        system_state={"time_ms": 20.0, "devices": {"edge": {"committed_work_ms": 100.0}}},
    )

    assert murakkab.propose(view).assignments["task"]["device_id"] == "cloud"


def test_resource_cost_can_outweigh_small_latency_advantage() -> None:
    dag = _dag(_node("task", "tool"))
    tool = _tool("tool", {"cfg": {"edge": (1.0, 100.0), "cloud": (2.0, 1.0)}})
    view = _view(
        (dag, dag, dag),
        (tool,),
        scoring_context=ScoringContext(
            accuracy_weight=0.0,
            latency_weight=0.0,
            resource_weight=1.0,
            resource_scale_mib=1.0,
        ),
    )

    assert murakkab.propose(view).assignments["task"]["device_id"] == "cloud"


def test_transfer_cost_affects_joint_placement() -> None:
    dag = _dag(_node("source", "source"), _node("consumer", "consumer", "source"))
    source = _tool("source", {"cfg": {"edge": (1.0, 1.0)}}, output_bytes=1_000)
    consumer = _tool("consumer", {"cfg": {"edge": (30.0, 1.0), "cloud": (1.0, 1.0)}})
    view = _view(
        (dag, dag, dag),
        (source, consumer),
        scoring_context=ScoringContext(latency_weight=1.0, resource_weight=0.0, accuracy_weight=0.0),
    )
    for transfer in view.snapshot_evidence["transfer_profiles"]:
        if transfer["source_device_id"] == "edge" and transfer["destination_device_id"] == "cloud":
            transfer["propagation_delay_ms"] = 100.0

    result = murakkab.propose(view)

    assert result.assignments["source"]["device_id"] == "edge"
    assert result.assignments["consumer"]["device_id"] == "edge"


def test_two_inputs_from_one_parent_require_two_transfers() -> None:
    consumer_node = ToolNode(
        "consumer",
        "consumer",
        {
            "left": InputSource.node("source", "text"),
            "right": InputSource.node("source", "text"),
        },
    )
    dag = _dag(_node("source", "source"), consumer_node)
    source = _tool("source", {"cfg": {"edge": (1.0, 1.0)}})
    consumer = _tool("consumer", {"cfg": {"edge": (25.0, 1.0), "cloud": (1.0, 1.0)}})
    view = _view(
        (dag, dag, dag),
        (source, consumer),
        scoring_context=ScoringContext(latency_weight=1.0, resource_weight=0.0, accuracy_weight=0.0),
    )
    for transfer in view.snapshot_evidence["transfer_profiles"]:
        if transfer["source_device_id"] == "edge" and transfer["destination_device_id"] == "cloud":
            transfer["propagation_delay_ms"] = 15.0

    assert murakkab.propose(view).assignments["consumer"]["device_id"] == "edge"
