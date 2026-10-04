from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

from eec_sched import FinalOutput, InputSource, ToolCallPlan, ToolNode
from eec_sched.evaluation.scoring import ScoringContext


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("qphh_baseline", ROOT / "schedulers/baselines/qphh.py")
assert SPEC and SPEC.loader
qphh = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = qphh
SPEC.loader.exec_module(qphh)


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


def test_longest_path_and_highest_quality_compatible_configuration() -> None:
    short = _dag(_node("short", "tool"))
    longest = _dag(_node("root", "tool"), _node("leaf", "tool", "root"))
    tool = _tool(
        "tool",
        {"small": {"device": (1.0, 1.0)}, "large": {"edge": (2.0, 2.0)}},
        {"small": 0.7, "large": 0.95},
    )

    result = qphh.propose(_view((short, longest, short), (tool,)))

    assert result.path_index == 1
    assert result.dag == longest
    assert {choice["configuration_id"] for choice in result.assignments.values()} == {"large"}
    assert {choice["device_id"] for choice in result.assignments.values()} == {"edge"}


def test_parent_transfer_cost_is_removed_by_parent_colocation() -> None:
    dag = _dag(_node("source", "source"), _node("consumer", "consumer", "source"))
    source = _tool("source", {"cfg": {"edge": (1.0, 1.0)}}, output_bytes=1_000)
    consumer = _tool("consumer", {"cfg": {"edge": (30.0, 1.0), "cloud": (1.0, 1.0)}})
    view = _view((dag, dag, dag), (source, consumer))
    for transfer in view.snapshot_evidence["transfer_profiles"]:
        if transfer["source_device_id"] == "edge" and transfer["destination_device_id"] == "cloud":
            transfer["propagation_delay_ms"] = 100.0
            transfer["bandwidth_bytes_per_second"] = 1_000.0

    result = qphh.propose(view)

    assert result.assignments["source"]["device_id"] == "edge"
    assert result.assignments["consumer"]["device_id"] == "edge"


def test_q_learning_bins_and_updates_are_nontrivial() -> None:
    assert qphh.state_from_delta(-0.1) == 0
    assert qphh.state_from_delta(0.0) == 1
    assert qphh.state_from_delta(1e-3) == 4
    assert qphh.state_from_delta(1.0) == 8
    assert qphh.reward_from_delta(-0.1) == -10.0
    assert qphh.reward_from_delta(0.05) == 9.0

    table = [list(row) for row in qphh.Q_TABLE]
    original = table[1][2]
    qphh.update_q_value(table, 1, 2, 10.0, 8, 0.5, 0.5)
    assert table[1][2] != original
    assert qphh.epsilon_greedy_action(table, 1, epsilon=0.0) in range(qphh.ACTION_COUNT)


def test_each_action_is_a_distinct_placement_operator() -> None:
    assert len(qphh.ACTION_NAMES) == 6
    assert len(set(qphh.ACTION_NAMES)) == 6
    assert len(qphh._OPERATORS) == qphh.ACTION_COUNT

    dag = _dag(_node("source", "source"), _node("consumer", "consumer", "source"))
    source = _tool("source", {"cfg": {"edge": (1.0, 1.0), "cloud": (5.0, 1.0)}})
    consumer = _tool("consumer", {"cfg": {"edge": (5.0, 1.0), "cloud": (1.0, 1.0)}})
    view = _view((dag, dag, dag), (source, consumer))
    prepared = qphh._prepare(view, dag)
    assignments = qphh._initial_assignments(dag, prepared)
    for action in range(qphh.ACTION_COUNT):
        result = qphh._apply_action(action, view, dag, assignments, prepared)
        assert set(result) == {"source", "consumer"}
        assert result is not assignments


def test_etrm_reuses_an_experienced_idle_device_after_probability_decays() -> None:
    nodes = tuple(
        _node(f"n{index}", "first" if index < 6 else "last", None if index == 0 else f"n{index - 1}")
        for index in range(7)
    )
    dag = _dag(*nodes)
    first = _tool("first", {"cfg": {"edge": (1.0, 1.0), "cloud": (10.0, 1.0)}})
    last = _tool("last", {"cfg": {"edge": (9.0, 1.0), "cloud": (1.0, 1.0)}})
    view = _view((dag, dag, dag), (first, last))
    prepared = qphh._prepare(view, dag)

    assignments = qphh._initial_assignments(dag, prepared, view)

    assert all(assignments[f"n{index}"]["device_id"] == "edge" for index in range(6))
    # The seventh deterministic draw chooses ETRM.  It considers the idle,
    # previously used edge device even though cloud has a faster profile.
    assert assignments["n6"]["device_id"] == "edge"
