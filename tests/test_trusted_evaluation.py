from __future__ import annotations

from pathlib import Path

import pytest

from eec_sched import (
    FinalOutput,
    InputSource,
    ScoringContext,
    ToolCallPlan,
    ToolNode,
    composite_score,
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
    import eec_sched.evaluation.evaluator as trusted_evaluation

    ticks = iter((1.0, 1.001, 2.0, 2.001))
    monkeypatch.setattr(trusted_evaluation, "perf_counter", lambda: next(ticks))
    plan = one_node_plan()
    scheduler = lambda dag: {"generate": {"configuration_id": "synthetic-reference", "device_id": "cloud"}}

    first = evaluate_scheduler_instance(SNAPSHOT, plan, scheduler)
    second = evaluate_scheduler_instance(SNAPSHOT, plan, scheduler)

    assert first == second
    assert first.scheduler_status == "scheduled"
    assert first.simulated_makespan_ms == pytest.approx(68.0384)
    assert first.latency == pytest.approx(69.0384)
    assert [(transfer.source_device_id, transfer.destination_device_id) for transfer in first.transfers] == [("device", "cloud"), ("cloud", "device")]
    assert first.resource == pytest.approx(sum(node.gpu_memory_mib for node in first.nodes))
    assert first.gpu_memory == first.resource
    assert first.raw_metrics["accuracy"] == first.accuracy
    assert first.raw_metrics["latency"] == first.latency
    assert first.raw_metrics["resource"] == first.resource
    assert first.composite_score is not None
    with pytest.raises(TypeError):
        first.assignments["other"] = first.assignments["generate"]  # type: ignore[index]


def test_evaluator_does_not_accept_caller_supplied_scheduler_time() -> None:
    with pytest.raises(TypeError, match="scheduler_solving_time_ms"):
        evaluate_scheduler_instance(
            SNAPSHOT,
            one_node_plan(),
            lambda dag: {},
            scheduler_solving_time_ms=0,  # type: ignore[call-arg]
        )


def test_evaluator_accounts_for_directional_transfer_latency_and_gpu_memory() -> None:
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
    report = evaluate_scheduler_instance(SNAPSHOT, plan, lambda dag: choices)

    assert report.scheduler_status == "scheduled"
    assert len(report.transfers) == 2
    assert report.transfers[0].latency_ms == pytest.approx(4.00512)
    assert report.transfers[1].source_device_id == "edge"
    assert report.transfers[1].destination_device_id == "device"
    assert report.transfers[1].latency_ms == pytest.approx(4.5056888889)
    assert report.resource == pytest.approx(sum(node.gpu_memory_mib for node in report.nodes))
    assert report.raw_accuracy_metrics.keys() == {"generate", "summarize"}


def test_evaluator_scores_any_latency_without_a_feasibility_gate() -> None:
    report = evaluate_scheduler_instance(
        SNAPSHOT,
        one_node_plan(),
        lambda dag: {"generate": {"configuration_id": "synthetic-reference", "device_id": "cloud"}},
    )

    assert report.scheduler_status == "scheduled"
    assert report.latency is not None and report.latency > 5
    assert report.composite_score is not None and report.composite_score > 0


def test_invalid_and_incompatible_scheduler_outputs_are_rejected() -> None:
    plan = one_node_plan()

    missing = evaluate_scheduler_instance(SNAPSHOT, plan, lambda dag: {})
    incompatible = evaluate_scheduler_instance(
        SNAPSHOT,
        plan,
        lambda dag: {"generate": {"configuration_id": "synthetic-reference", "device_id": "unknown"}},
    )

    assert missing.scheduler_status == "rejected"
    assert any("missing node assignments" in error for error in missing.validation_errors)
    assert incompatible.scheduler_status == "rejected"
    assert incompatible.composite_score is None


def test_composite_score_compares_all_three_metrics_without_constraints() -> None:
    context = ScoringContext(
        accuracy_weight=1.0,
        latency_weight=1.0,
        resource_weight=1.0,
        latency_scale_ms=100.0,
        resource_scale_mib=1000.0,
    )
    best = composite_score(0.9, 10.0, 100.0, context)
    worse_accuracy = composite_score(0.8, 10.0, 100.0, context)
    worse_latency = composite_score(0.9, 20.0, 100.0, context)
    worse_resource = composite_score(0.9, 10.0, 200.0, context)

    assert best > worse_accuracy
    assert best > worse_latency
    assert best > worse_resource
