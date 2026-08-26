from __future__ import annotations

from pathlib import Path

import pytest

from eec_sched import (
    FinalOutput,
    InputSource,
    ToolCallPlan,
    ToolNode,
    UTILITY_EPSILON,
    evaluate_scheduler_instance,
    load_profiling_database,
)


ROOT = Path(__file__).parents[1]
SNAPSHOT = load_profiling_database(
    ROOT / "docs/examples/profiling-database.fake.json",
    ROOT / "docs/schemas/profiling-database.schema.json",
)


def one_node_plan() -> ToolCallPlan:
    return ToolCallPlan(
        nodes=(ToolNode("generate", "text_generation", {"prompt": InputSource.request("prompt")}),),
        final_outputs=(FinalOutput("generate", "text"),),
    )


def test_evaluator_returns_immutable_replayable_report_and_includes_trusted_scheduler_time(monkeypatch) -> None:
    import eec_sched.trusted_evaluation as trusted_evaluation

    ticks = iter((1.0, 1.001, 2.0, 2.001))
    monkeypatch.setattr(trusted_evaluation, "perf_counter", lambda: next(ticks))
    plan = one_node_plan()
    scheduler = lambda dag: {"generate": {"configuration_id": "synthetic-reference", "device_id": "cloud"}}

    first = evaluate_scheduler_instance(
        SNAPSHOT, plan, scheduler, minimum_accuracy=0.9, maximum_latency_ms=11, gamma=0.5,
    )
    second = evaluate_scheduler_instance(
        SNAPSHOT, plan, scheduler, minimum_accuracy=0.9, maximum_latency_ms=11, gamma=0.5,
    )

    assert first == second
    assert first.scheduler_status == "scheduled"
    assert first.simulated_makespan_ms == pytest.approx(68.0384)
    assert first.latency_proxy_ms == pytest.approx(69.0384)
    assert [(transfer.source_device_id, transfer.destination_device_id) for transfer in first.transfers] == [("device", "cloud"), ("cloud", "device")]
    assert first.incremental_execution_energy_j == pytest.approx(0.31001664)
    assert first.utility is not None
    with pytest.raises(TypeError):
        first.assignments["other"] = first.assignments["generate"]  # type: ignore[index]


def test_evaluator_does_not_accept_caller_supplied_scheduler_time() -> None:
    with pytest.raises(TypeError, match="scheduler_solving_time_ms"):
        evaluate_scheduler_instance(
            SNAPSHOT,
            one_node_plan(),
            lambda dag: {},
            minimum_accuracy=0,
            maximum_latency_ms=100,
            gamma=0.5,
            scheduler_solving_time_ms=0,  # type: ignore[call-arg]
        )


def test_evaluator_accounts_for_directional_transfer_latency_and_energy() -> None:
    plan = ToolCallPlan(
        nodes=(
            ToolNode("generate", "text_generation", {"prompt": InputSource.request("prompt")}),
            ToolNode("summarize", "text_summarization", {"text": InputSource.node("generate", "text")}),
        ),
        final_outputs=(FinalOutput("summarize", "text"),),
    )
    choices = {
        "generate": {"configuration_id": "synthetic-reference", "device_id": "device"},
        "summarize": {"configuration_id": "fast", "device_id": "edge"},
    }
    report = evaluate_scheduler_instance(
        SNAPSHOT, plan, lambda dag: choices, minimum_accuracy=0, maximum_latency_ms=100, gamma=0.5,
    )

    assert report.scheduler_status == "scheduled"
    assert len(report.transfers) == 2
    assert report.transfers[0].latency_ms == pytest.approx(4.00512)
    assert report.transfers[0].energy_j == pytest.approx(0.02000256)
    assert report.transfers[1].source_device_id == "edge"
    assert report.transfers[1].destination_device_id == "device"
    assert report.transfers[1].latency_ms == pytest.approx(4.5056888889)
    assert report.communication_energy_j == pytest.approx(0.045005632)


def test_evaluator_scores_over_budget_latency_instead_of_rejecting_it() -> None:
    report = evaluate_scheduler_instance(
        SNAPSHOT,
        one_node_plan(),
        lambda dag: {"generate": {"configuration_id": "synthetic-reference", "device_id": "cloud"}},
        minimum_accuracy=0.9,
        maximum_latency_ms=5,
        gamma=0.5,
    )

    assert report.scheduler_status == "scheduled"
    assert report.accuracy_feasible is True
    assert report.latency_feasible is False
    assert report.feasible is False
    assert report.utility is not None
    assert report.utility < 0


def test_invalid_and_incompatible_scheduler_outputs_are_rejected() -> None:
    plan = one_node_plan()

    missing = evaluate_scheduler_instance(
        SNAPSHOT, plan, lambda dag: {}, minimum_accuracy=0, maximum_latency_ms=100, gamma=0.5,
    )
    incompatible = evaluate_scheduler_instance(
        SNAPSHOT,
        plan,
        lambda dag: {"generate": {"configuration_id": "synthetic-reference", "device_id": "unknown"}},
        minimum_accuracy=0,
        maximum_latency_ms=100,
        gamma=0.5,
    )

    assert missing.scheduler_status == "rejected"
    assert any("missing node assignments" in error for error in missing.validation_errors)
    assert incompatible.scheduler_status == "rejected"
    assert incompatible.utility is None


def test_utility_epsilon_is_a_named_fixed_constant() -> None:
    assert UTILITY_EPSILON > 0
