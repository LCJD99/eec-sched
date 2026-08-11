from __future__ import annotations

from pathlib import Path

import pytest

from eec_sched import (
    EvaluationTrace,
    EvolutionModule,
    FinalOutput,
    InputSource,
    SchedulerCandidate,
    ToolCallPlan,
    ToolNode,
    evaluate_scheduler_candidate,
    load_profiling_database,
)


ROOT = Path(__file__).parents[1]
SNAPSHOT = load_profiling_database(
    ROOT / "docs/examples/profiling-database.fake.json",
    ROOT / "docs/schemas/profiling-database.schema.json",
)


def _trace(trace_id: str) -> EvaluationTrace:
    dag = ToolCallPlan(
        nodes=(ToolNode("generate", "text_generation", {"prompt": InputSource.request("prompt")}),),
        final_outputs=(FinalOutput("generate", "text"),),
    )
    return EvaluationTrace(
        trace_id=trace_id,
        task_input={"prompt": f"prompt for {trace_id}"},
        dag=dag,
        minimum_accuracy=0.0,
        maximum_latency_ms=100.0,
        gamma=0.5,
    )


def _reference_candidate(version: int = 1) -> SchedulerCandidate:
    return SchedulerCandidate(
        version=version,
        propose=lambda view: {
            "generate": {"configuration_id": "synthetic-reference", "device_id": "cloud"}
        },
    )


def test_candidate_evaluation_preserves_three_stages_and_averages_scored_traces() -> None:
    traces = (_trace("planning-1"), _trace("planning-2"))

    result = evaluate_scheduler_candidate(SNAPSHOT, traces, _reference_candidate())

    assert result.schema_version == "v1"
    assert result.candidate_version == 1
    assert [record.trace.trace_id for record in result.traces] == ["planning-1", "planning-2"]
    assert result.traces[0].trace.task_input == {"prompt": "prompt for planning-1"}
    assert result.traces[0].trace.dag == traces[0].dag
    assert result.traces[0].assignments["generate"].configuration_id == "synthetic-reference"
    assert all(record.status == "scored" and record.score is not None for record in result.traces)
    assert result.candidate_score == pytest.approx(sum(record.score for record in result.traces) / 2)


def test_candidate_evaluation_distinguishes_rejected_assignments_from_failed_candidates() -> None:
    rejected = SchedulerCandidate(version=2, propose=lambda view: {})
    failed = SchedulerCandidate(version=3, propose=lambda view: (_ for _ in ()).throw(RuntimeError("boom")))
    unknown_configuration = SchedulerCandidate(
        version=4,
        propose=lambda view: {"generate": {"configuration_id": "not-a-configuration", "device_id": "cloud"}},
    )
    incompatible_device = SchedulerCandidate(
        version=5,
        propose=lambda view: {"generate": {"configuration_id": "synthetic-reference", "device_id": "not-a-device"}},
    )

    rejected_result = evaluate_scheduler_candidate(SNAPSHOT, (_trace("rejected"),), rejected)
    failed_result = evaluate_scheduler_candidate(SNAPSHOT, (_trace("failed"),), failed)
    unknown_configuration_result = evaluate_scheduler_candidate(SNAPSHOT, (_trace("unknown"),), unknown_configuration)
    incompatible_device_result = evaluate_scheduler_candidate(SNAPSHOT, (_trace("incompatible"),), incompatible_device)

    assert rejected_result.traces[0].status == "rejected"
    assert rejected_result.traces[0].score is None
    assert rejected_result.traces[0].reason
    assert failed_result.traces[0].status == "failed"
    assert failed_result.traces[0].score is None
    assert "RuntimeError" in failed_result.traces[0].reason
    assert unknown_configuration_result.traces[0].status == "rejected"
    assert incompatible_device_result.traces[0].status == "rejected"


def test_scheduler_view_cannot_mutate_the_planner_dag() -> None:
    def mutate_view(view):
        view.dag.nodes[0].inputs["prompt"] = InputSource.request("other")
        return {}

    result = evaluate_scheduler_candidate(SNAPSHOT, (_trace("immutable-view"),), SchedulerCandidate(6, mutate_view))

    assert result.traces[0].status == "failed"
    assert result.traces[0].trace.dag.nodes[0].inputs["prompt"] == InputSource.request("prompt")


def test_concise_projections_are_versioned_and_final_projection_adds_oracle_comparison() -> None:
    traces = (_trace("trace"),)
    candidate = _reference_candidate()

    evolution = evaluate_scheduler_candidate(SNAPSHOT, traces, candidate)
    final = evaluate_scheduler_candidate(SNAPSHOT, traces, candidate, mode="final", oracle_reference=_reference_candidate(99))

    concise_evolution = evolution.concise_projection()
    concise_final = final.concise_projection()
    assert concise_evolution == {
        "schema_version": "v1",
        "scheduler_candidate_version": 1,
        "traces": [{"trace_id": "trace", "status": "scored", "score": pytest.approx(evolution.candidate_score)}],
        "candidate_score": pytest.approx(evolution.candidate_score),
    }
    assert "oracle_reference_score" not in concise_evolution["traces"][0]
    assert concise_final["traces"][0]["oracle_reference_score"] == pytest.approx(evolution.candidate_score)
    assert concise_final["traces"][0]["difference"] == pytest.approx(0)
    assert concise_final["average_difference"] == pytest.approx(0)


def test_evolution_module_runs_fixed_rounds_and_selects_highest_observed_candidate() -> None:
    selected_contexts: list[tuple[str, ...]] = []

    def select(records):
        selected_contexts.append(tuple(record.trace.trace_id for record in records))
        return records

    module = EvolutionModule(
        snapshot=SNAPSHOT,
        evolution_traces=(_trace("a"), _trace("b")),
        trace_selection_strategy=select,
        candidate_proposer=lambda context, current: _reference_candidate(current.version + 1),
    )

    result = module.run(_reference_candidate(), rounds=2)

    assert len(result.observed_evaluations) == 3
    assert selected_contexts == [("a", "b"), ("a", "b")]
    assert result.selected_candidate.version == 1
    assert result.selected_evaluation.candidate_score == pytest.approx(result.observed_evaluations[0].candidate_score)


def test_evolution_module_requires_a_separate_final_evaluation_trace_set() -> None:
    module = EvolutionModule(
        snapshot=SNAPSHOT,
        evolution_traces=(_trace("evolution"),),
        trace_selection_strategy=lambda records: records,
        candidate_proposer=lambda context, current: _reference_candidate(current.version + 1),
    )
    evolution = module.run(_reference_candidate(), rounds=0)

    with pytest.raises(ValueError, match="independent"):
        module.final_evaluate(evolution, (_trace("evolution"),), _reference_candidate(99))

    final = module.final_evaluate(evolution, (_trace("final"),), _reference_candidate(99))
    assert final.average_difference == pytest.approx(0)
