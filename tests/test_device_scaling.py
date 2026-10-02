from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from eec_sched.profiling.device_scaling import build_device_scaling_scenario, calibration_from_snapshot


ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def calibration() -> dict:
    snapshot = json.loads((ROOT / "docs/examples/profiling-database.fake.json").read_text(encoding="utf-8"))
    return calibration_from_snapshot(snapshot)


@pytest.mark.parametrize("count", [10, 20, 50])
def test_scenario_has_exact_configurations_connected_sparse_network_and_capacity_tradeoff(calibration: dict, count: int) -> None:
    scenario = build_device_scaling_scenario(calibration, device_count=count, configurations_per_tool=5)
    assert scenario["data_kind"] == "synthetic"
    assert scenario["source_data_kind"] == "synthetic"
    assert len(scenario["devices"]) == count
    assert {item["gpu_model"] for item in scenario["devices"][:3]} == {"rtx5060", "rtx3090", "rtx4090"}
    assert len(scenario["tools"]) == len(calibration["tools"])
    assert len(scenario["links"]) < count * (count - 1)
    assert scenario == build_device_scaling_scenario(calibration, device_count=count, configurations_per_tool=5)

    adjacency = {device["device_id"]: set() for device in scenario["devices"]}
    for link in scenario["links"]:
        adjacency[link["source_device_id"]].add(link["destination_device_id"])
        assert link["bandwidth_bytes_per_second"] > 0
    visited = {scenario["devices"][0]["device_id"]}
    frontier = list(visited)
    while frontier:
        for neighbor in adjacency[frontier.pop()]:
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append(neighbor)
    assert visited == set(adjacency)

    for tool in scenario["tools"]:
        assert tool["fitted_reference_compute_demand"] > 0
        assert len(tool["configurations"]) == 5
        assert [item["compute_demand"] for item in tool["configurations"]] == sorted(
            item["compute_demand"] for item in tool["configurations"]
        )
        for configuration in tool["configurations"]:
            profiles = configuration["execution_profiles"]
            assert len(profiles) == count
            compatible = sorted(
                ((device["compute_capacity"], profile) for device, profile in zip(scenario["devices"], profiles) if profile["compatible"]),
                key=lambda item: item[0],
            )
            assert [profile["warm_latency_p95_ms"] for _, profile in compatible] == sorted(
                (profile["warm_latency_p95_ms"] for _, profile in compatible), reverse=True
            )
            assert [profile["execution_cost"] for _, profile in compatible] == sorted(
                profile["execution_cost"] for _, profile in compatible
            )
            for capacity, profile in compatible:
                assert profile["warm_latency_p95_ms"] == pytest.approx(configuration["compute_demand"] / capacity)


def test_rejects_anchor_costs_that_conflict_with_cost_constraint(calibration: dict) -> None:
    changed = dict(calibration)
    devices = [dict(device) for device in calibration["devices"]]
    devices[-1]["cost_per_hour"] = 3
    changed["devices"] = devices
    with pytest.raises(ValueError, match="cost per compute unit"):
        build_device_scaling_scenario(changed, device_count=10, configurations_per_tool=5)


def test_network_distance_controls_bandwidth_and_fit_diagnostics_are_recorded(calibration: dict) -> None:
    scenario = build_device_scaling_scenario(calibration, device_count=50, configurations_per_tool=1)
    by_distance: dict[int, list[float]] = {}
    for link in scenario["links"]:
        by_distance.setdefault(link["ordinal_distance"], []).append(link["bandwidth_bytes_per_second"])
    assert max(by_distance) > 1
    assert max(by_distance[max(by_distance)]) < min(by_distance[1])
    assert all(math.isfinite(tool["fit_rmse_log_latency"]) for tool in scenario["tools"])


def test_device_memory_marks_unrunnable_configuration_incompatible(calibration: dict) -> None:
    changed = dict(calibration)
    tools = [dict(tool) for tool in calibration["tools"]]
    tools[0]["reference_gpu_memory_mib"] = 9000
    changed["tools"] = tools
    scenario = build_device_scaling_scenario(changed, device_count=3, configurations_per_tool=1)
    profiles = scenario["tools"][0]["configurations"][0]["execution_profiles"]
    assert profiles[0]["compatible"] is False
    assert profiles[0]["warm_latency_p95_ms"] is None
    assert profiles[1]["compatible"] is True
