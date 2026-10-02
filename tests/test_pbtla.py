from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path

from eec_sched import FinalOutput, InputSource, ToolCallPlan, ToolNode


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("pbtla_baseline", ROOT / "schedulers/baselines/pbtla.py")
assert SPEC and SPEC.loader
pbtla = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pbtla
SPEC.loader.exec_module(pbtla)


def _node(node_id: str, tool_id: str, parent: str | None = None) -> ToolNode:
    source = InputSource.request("text") if parent is None else InputSource.node(parent, "text")
    return ToolNode(node_id, tool_id, {"text": source})


def _dag(*nodes: ToolNode) -> ToolCallPlan:
    return ToolCallPlan(tuple(nodes), (FinalOutput(nodes[-1].node_id, "text"),))


def _tool(tool_id: str, profiles: dict[str, float], *, quality: dict[str, float] | None = None, output_bytes: int = 100) -> dict[str, object]:
    quality = quality or {"best": 1.0}
    configurations = [
        {
            "configuration_id": configuration_id,
            "device_compatibility": [
                {"device_id": device_id, "status": "compatible"}
                for device_id in profiles
            ],
        }
        for configuration_id in quality
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
        for configuration_id in quality
        for device_id, latency in profiles.items()
    ]
    return {
        "tool_id": tool_id,
        "configurations": configurations,
        "quality_profiles": quality_profiles,
        "execution_profiles": execution_profiles,
    }


def _view(dags: tuple[ToolCallPlan, ...], tools: tuple[dict[str, object], ...], *, system_state: dict[str, object] | None = None) -> SimpleNamespace:
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


def test_selects_longest_dependency_chain_and_highest_quality_compatible_configuration() -> None:
    short = _dag(_node("short", "tool"))
    longest = _dag(_node("root", "tool"), _node("middle", "tool", "root"), _node("leaf", "tool", "middle"))
    tied = _dag(_node("other", "tool"), _node("other-leaf", "tool", "other"))
    tool = _tool("tool", {"device": 10.0, "edge": 10.0, "cloud": 10.0}, quality={"slow": 0.9, "best": 0.95})

    result = pbtla.propose(_view((short, longest, tied), (tool,)))

    assert result.path_index == 1
    assert result.dag == longest
    assert {assignment["configuration_id"] for assignment in result.assignments.values()} == {"best"}


def test_predecessor_transfer_can_keep_a_fast_cloud_successor_on_the_edge() -> None:
    root = _dag(_node("root", "source"), _node("child", "consumer", "root"))
    source = _tool("source", {"device": 100.0, "edge": 10.0, "cloud": 100.0}, output_bytes=1_000)
    consumer = _tool("consumer", {"device": 100.0, "edge": 30.0, "cloud": 1.0})
    view = _view((root, root, root), (source, consumer))
    for transfer in view.snapshot_evidence["transfer_profiles"]:
        if transfer["source_device_id"] == "edge" and transfer["destination_device_id"] == "cloud":
            transfer["propagation_delay_ms"] = 100.0
            transfer["bandwidth_bytes_per_second"] = 1_000.0

    result = pbtla.propose(view)

    assert result.assignments["root"]["device_id"] == "edge"
    assert result.assignments["child"]["device_id"] == "edge"
