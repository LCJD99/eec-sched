from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

from eec_sched import FinalOutput, InputSource, ToolCallPlan, ToolNode


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("sdts_baseline", ROOT / "schedulers/baselines/sdts.py")
assert SPEC and SPEC.loader
sdts = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sdts
SPEC.loader.exec_module(sdts)


def _node(node_id: str, tool_id: str, parent: str | None = None) -> ToolNode:
    source = InputSource.request("text") if parent is None else InputSource.node(parent, "text")
    return ToolNode(node_id, tool_id, {"text": source})


def _dag(*nodes: ToolNode) -> ToolCallPlan:
    return ToolCallPlan(tuple(nodes), (FinalOutput(nodes[-1].node_id, "text"),))


def _tool(
    tool_id: str,
    profiles_by_configuration: dict[str, dict[str, float]],
    *,
    quality: dict[str, float] | None = None,
    output_bytes: int = 100,
) -> dict[str, object]:
    quality = quality or {configuration_id: 1.0 for configuration_id in profiles_by_configuration}
    configurations = [
        {
            "configuration_id": configuration_id,
            "device_compatibility": [
                {"device_id": device_id, "status": "compatible"}
                for device_id in profiles
            ],
        }
        for configuration_id, profiles in profiles_by_configuration.items()
    ]
    quality_profiles = [
        {
            "configuration_id": configuration_id,
            "normalized_quality_lcb": quality_value,
            "representative_output_bytes": output_bytes,
        }
        for configuration_id, quality_value in quality.items()
    ]
    execution_profiles = [
        {
            "configuration_id": configuration_id,
            "device_id": device_id,
            "warm_latency_p95_ms": latency,
            "gpu_memory_mib": 1.0,
        }
        for configuration_id, profiles in profiles_by_configuration.items()
        for device_id, latency in profiles.items()
    ]
    return {
        "tool_id": tool_id,
        "configurations": configurations,
        "quality_profiles": quality_profiles,
        "execution_profiles": execution_profiles,
    }


def _view(
    dags: tuple[ToolCallPlan, ...],
    tools: tuple[dict[str, object], ...],
    *,
    system_state: dict[str, object] | None = None,
) -> SimpleNamespace:
    devices = tuple({"device_id": device_id} for device_id in ("device", "edge", "cloud"))
    transfers = tuple(
        {
            "source_device_id": source,
            "destination_device_id": destination,
            "propagation_delay_ms": 1.0,
            "bandwidth_bytes_per_second": 1_000_000,
        }
        for source in ("device", "edge", "cloud")
        for destination in ("device", "edge", "cloud")
        if source != destination
    )
    return SimpleNamespace(
        candidate_dags=dags,
        snapshot_evidence={"tools": tools, "devices": devices, "transfer_profiles": transfers},
        system_state=system_state or {},
    )


def test_selects_longest_path_and_highest_quality_usable_configuration() -> None:
    short = _dag(_node("short", "tool"))
    longest = _dag(_node("root", "tool"), _node("middle", "tool", "root"), _node("leaf", "tool", "middle"))
    tied = _dag(_node("other", "tool"), _node("other-leaf", "tool", "other"))
    tool = _tool(
        "tool",
        {"low": {"device": 10.0}, "best": {"edge": 20.0}},
        quality={"low": 0.7, "best": 0.95},
    )

    result = sdts.propose(_view((short, longest, tied), (tool,)))

    assert result.path_index == 1
    assert result.dag == longest
    assert {assignment["configuration_id"] for assignment in result.assignments.values()} == {"best"}
    assert {assignment["device_id"] for assignment in result.assignments.values()} == {"edge"}


def test_device_queue_changes_earliest_finish_choice() -> None:
    dag = _dag(_node("task", "tool"))
    tool = _tool("tool", {"cfg": {"edge": 1.0, "cloud": 8.0}})
    view = _view(
        (dag, dag, dag),
        (tool,),
        system_state={"time_ms": 20.0, "devices": {"edge": {"committed_work_ms": 100.0}}},
    )

    result = sdts.propose(view)

    assert result.assignments["task"]["device_id"] == "cloud"


def test_predecessor_transfer_is_included_in_successor_finish_estimate() -> None:
    dag = _dag(_node("source", "source"), _node("consumer", "consumer", "source"))
    source = _tool("source", {"cfg": {"edge": 1.0, "cloud": 20.0}}, output_bytes=1_000)
    consumer = _tool("consumer", {"cfg": {"edge": 30.0, "cloud": 1.0}})
    view = _view((dag, dag, dag), (source, consumer))
    for transfer in view.snapshot_evidence["transfer_profiles"]:
        if transfer["source_device_id"] == "edge" and transfer["destination_device_id"] == "cloud":
            transfer["propagation_delay_ms"] = 100.0
            transfer["bandwidth_bytes_per_second"] = 1_000.0

    result = sdts.propose(view)

    assert result.assignments["source"]["device_id"] == "edge"
    assert result.assignments["consumer"]["device_id"] == "edge"


def test_upward_rank_uses_mean_execution_and_transfer_costs() -> None:
    dag = _dag(_node("root", "root"), _node("leaf", "leaf", "root"))
    root = _tool("root", {"cfg": {"edge": 2.0, "cloud": 8.0}}, output_bytes=1_000)
    leaf = _tool("leaf", {"cfg": {"edge": 10.0, "cloud": 10.0}})
    view = _view((dag, dag, dag), (root, leaf))
    tools, devices, transfers = sdts._evidence(view)
    _, profiles, output_bytes = sdts._select_configurations(dag, tools, devices)

    ranks = sdts._upward_ranks(dag, profiles, output_bytes, transfers)

    assert ranks["root"] > ranks["leaf"]
    assert ranks["root"] == 5.0 + ranks["leaf"] + 1.0
