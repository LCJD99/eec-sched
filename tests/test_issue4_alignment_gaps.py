"""Independent Issue #4 alignment tests and blocked evaluator test plan.

This file intentionally does not import a not-yet-existing evaluator API.  The
first group exercises the profiling-database seam and the normative arithmetic
directly.  The skipped tests are executable placeholders for the evaluator
contract once that seam exists.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from eec_sched import FinalOutput, InputSource, ToolCallPlan, ToolNode, UTILITY_EPSILON, evaluate_scheduler_instance
from eec_sched.profiling_database import load_profiling_database


ROOT = Path(__file__).parents[1]
SCHEMA_PATH = ROOT / "docs/schemas/profiling-database.schema.json"
FAKE_DATABASE_PATH = ROOT / "docs/examples/profiling-database.fake.json"
def _payload() -> dict[str, Any]:
    return json.loads(FAKE_DATABASE_PATH.read_text(encoding="utf-8"))


def _normalized_quality_lcb(
    conservative_raw: float,
    semantic_floor: float,
    reference_raw: float,
    direction: str,
) -> float:
    if direction == "higher_is_better":
        value = (conservative_raw - semantic_floor) / (reference_raw - semantic_floor)
    else:
        value = (semantic_floor - conservative_raw) / (semantic_floor - reference_raw)
    return max(0.0, min(1.0, value))


def _plan_utility(
    accuracy_lcb: float,
    minimum_accuracy: float,
    latency_ms: float,
    maximum_latency_ms: float,
    gamma: float,
    incremental_execution_energy_j: float,
) -> float:
    accuracy_surplus = 0.0 if minimum_accuracy == 1 else (accuracy_lcb - minimum_accuracy) / (1 - minimum_accuracy)
    latency_surplus = (maximum_latency_ms - latency_ms) / maximum_latency_ms
    performance = gamma * accuracy_surplus + (1 - gamma) * latency_surplus
    return performance / (incremental_execution_energy_j + UTILITY_EPSILON)


def test_fake_snapshot_contains_the_complete_issue4_input_surface() -> None:
    snapshot = load_profiling_database(FAKE_DATABASE_PATH, SCHEMA_PATH)
    payload = snapshot.data

    assert snapshot.snapshot_digest == payload["snapshot_digest"]
    assert {device["device_id"] for device in payload["devices"]} == {"device", "edge", "cloud"}
    assert len(payload["transfer_profiles"]) == 6
    assert sum(len(tool["configurations"]) for tool in payload["tools"]) == 50
    assert sum(len(tool["quality_profiles"]) for tool in payload["tools"]) == 50
    assert sum(len(tool["execution_profiles"]) for tool in payload["tools"]) == 150

    for tool in payload["tools"]:
        if tool["eligibility"]["status"] != "eligible":
            continue
        assert all("normalized_quality_lcb" in profile for profile in tool["quality_profiles"])
        assert all("representative_output_bytes" in profile for profile in tool["quality_profiles"])
        assert all("mean_incremental_execution_energy_j" in profile for profile in tool["execution_profiles"])


def test_lower_confidence_bound_uses_conservative_endpoint_and_direction() -> None:
    assert _normalized_quality_lcb(0.88, 0.5, 0.95, "higher_is_better") == pytest.approx(0.8444444444)
    assert _normalized_quality_lcb(0.04, 0.0, 0.05, "lower_is_better") == pytest.approx(0.8)


def test_fake_quality_profiles_match_recomputed_lower_confidence_bounds() -> None:
    for tool in _payload()["tools"]:
        if tool["eligibility"]["status"] != "eligible":
            continue
        contract = tool["quality_contract"]
        reference_raw = contract["reference"]["raw_metric"]
        floor = contract["semantic_floor"]
        direction = contract["direction"]
        for profile in tool["quality_profiles"]:
            interval = profile["raw_metric"]["confidence_interval"]
            conservative = interval["lower"] if direction == "higher_is_better" else interval["upper"]
            assert profile["normalized_quality_lcb"] == pytest.approx(
                _normalized_quality_lcb(conservative, floor, reference_raw, direction)
            )


def test_cross_device_transfer_formula_uses_output_bytes_and_direction() -> None:
    profile = next(
        profile
        for profile in _payload()["transfer_profiles"]
        if profile["source_device_id"] == "device" and profile["destination_device_id"] == "edge"
    )
    output_bytes = 256
    expected_latency = profile["propagation_delay_ms"] + 1000 * output_bytes / profile["bandwidth_bytes_per_second"]
    expected_energy = profile["setup_energy_j"] + output_bytes * profile["energy_per_byte_j"]

    assert expected_latency == pytest.approx(4.00512)
    assert expected_energy == pytest.approx(0.02000256)


def test_utility_handles_A_m_equal_one_and_divides_by_incremental_energy() -> None:
    utility = _plan_utility(1.0, 1.0, 10.0, 20.0, 0.5, 2.0)
    assert utility == pytest.approx(0.25 / (2.0 + UTILITY_EPSILON))

    assert _plan_utility(0.999999, 1.0, 10.0, 20.0, 0.5, 2.0) == pytest.approx(0.25 / (2.0 + UTILITY_EPSILON))


def test_illegal_scheduler_return_is_rejected_before_metric_evaluation() -> None:
    plan = ToolCallPlan(
        nodes=(ToolNode("generate", "text_generation", {"prompt": InputSource.request("prompt")}),),
        final_outputs=(FinalOutput("generate", "text"),),
    )
    report = evaluate_scheduler_instance(
        load_profiling_database(FAKE_DATABASE_PATH, SCHEMA_PATH),
        plan,
        lambda dag: {},
        minimum_accuracy=0,
        maximum_latency_ms=100,
        gamma=0.5,
    )
    assert report.scheduler_status == "rejected"
    assert report.utility is None
    assert any("missing node assignments" in error for error in report.validation_errors)


def test_deterministic_replay_includes_makespan_transfer_and_energy(monkeypatch) -> None:
    """Replay the same snapshot, DAG, and Scheduler output twice and compare reports."""
    import eec_sched.trusted_evaluation as trusted_evaluation

    ticks = iter((1.0, 1.001, 2.0, 2.001))
    monkeypatch.setattr(trusted_evaluation, "perf_counter", lambda: next(ticks))
    plan = ToolCallPlan(
        nodes=(ToolNode("generate", "text_generation", {"prompt": InputSource.request("prompt")}),),
        final_outputs=(FinalOutput("generate", "text"),),
    )
    scheduler = lambda dag: {"generate": {"configuration_id": "synthetic-reference", "device_id": "cloud"}}
    snapshot = load_profiling_database(FAKE_DATABASE_PATH, SCHEMA_PATH)
    first = evaluate_scheduler_instance(snapshot, plan, scheduler, minimum_accuracy=0, maximum_latency_ms=100, gamma=0.5)
    second = evaluate_scheduler_instance(snapshot, plan, scheduler, minimum_accuracy=0, maximum_latency_ms=100, gamma=0.5)
    assert first == second
    assert first.simulated_makespan_ms == pytest.approx(10)
    assert first.incremental_execution_energy_j == pytest.approx(0.14)


def test_evaluator_report_separates_scheduler_time_from_plan_utility(monkeypatch) -> None:
    """Scheduler solving time must be reported and included in the latency proxy."""
    import eec_sched.trusted_evaluation as trusted_evaluation

    ticks = iter((1.0, 1.002))
    monkeypatch.setattr(trusted_evaluation, "perf_counter", lambda: next(ticks))
    plan = ToolCallPlan(
        nodes=(ToolNode("generate", "text_generation", {"prompt": InputSource.request("prompt")}),),
        final_outputs=(FinalOutput("generate", "text"),),
    )
    report = evaluate_scheduler_instance(
        load_profiling_database(FAKE_DATABASE_PATH, SCHEMA_PATH),
        plan,
        lambda dag: {"generate": {"configuration_id": "synthetic-reference", "device_id": "cloud"}},
        minimum_accuracy=0,
        maximum_latency_ms=100,
        gamma=0.5,
    )
    assert report.scheduler_solving_time_ms == pytest.approx(2)
    assert report.latency_proxy_ms == pytest.approx(12)
    assert report.utility is not None
