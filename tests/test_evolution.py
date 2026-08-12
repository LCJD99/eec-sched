from __future__ import annotations

from pathlib import Path

import pytest

from eec_sched import (
    EvolutionLoop,
    EvaluationTrace,
    EvaluationReport,
    EvolutionModule,
    FinalOutput,
    InputSource,
    SchedulerCandidate,
    TraceEvaluation,
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


def test_canonical_loop_injects_evaluator_and_oracle_at_the_application_seam() -> None:
    """The loop evaluates every candidate once, then pays the Oracle cost once."""
    calls: list[tuple[str, int]] = []
    selected: list[tuple[str, ...]] = []
    oracle_calls: list[tuple[int, tuple[str, ...]]] = []

    def evaluator(snapshot, trace, candidate):
        calls.append((trace.trace_id, candidate.version))
        score = 0.8 if candidate.version == 1 else 0.2
        report = EvaluationReport(snapshot.snapshot_digest, "scheduled", (), {})
        return TraceEvaluation(trace, {}, "scored", score, report)

    def select(records):
        selected.append(tuple(record.trace.trace_id for record in records))
        return records[:1]

    def oracle(snapshot, records, candidate):
        oracle_calls.append((candidate.version, tuple(record.trace.trace_id for record in records)))
        return (1.0,)

    loop = EvolutionLoop(
        snapshot=SNAPSHOT,
        evolution_traces=(_trace("evolution-a"), _trace("evolution-b")),
        final_evaluation_traces=(_trace("final"),),
        initial_candidate=_reference_candidate(),
        rounds=1,
        candidate_proposer=lambda context, current: _reference_candidate(2),
        trace_selection_strategy=select,
        trusted_evaluator=evaluator,
        oracle=oracle,
    )

    result = loop.run()

    assert calls == [("evolution-a", 1), ("evolution-b", 1), ("evolution-a", 2), ("evolution-b", 2), ("final", 1)]
    assert selected == [("evolution-a", "evolution-b")]
    assert result.selected_candidate.version == 1
    assert result.selected_evaluation.candidate_score == pytest.approx(0.8)
    assert oracle_calls == [(1, ("final",))]
    assert result.final_evaluation.average_candidate_score == pytest.approx(0.8)
    assert result.final_evaluation.average_oracle_reference_score == pytest.approx(1.0)
    assert result.final_evaluation.average_difference == pytest.approx(-0.2)


def test_canonical_loop_keeps_rejections_and_failures_visible_but_scores_them_as_zero() -> None:
    trace_a, trace_b, final_trace = _trace("a"), _trace("b"), _trace("final")

    def evaluator(snapshot, trace, candidate):
        report = EvaluationReport(snapshot.snapshot_digest, "rejected", ("short reason",), {})
        if trace.trace_id == "a":
            return TraceEvaluation(trace, {}, "scored", 0.6, report)
        return TraceEvaluation(trace, {}, "rejected", None, report, "short reason")

    result = EvolutionLoop(
        SNAPSHOT,
        (trace_a, trace_b),
        (final_trace,),
        _reference_candidate(),
        0,
        lambda context, candidate: candidate,
        lambda records: records,
        evaluator,
        lambda snapshot, records, candidate: (0.7,),
    ).run()

    assert result.selected_evaluation.candidate_score == pytest.approx(0.3)
    assert [(record.status, record.score, record.reason) for record in result.selected_evaluation.traces] == [
        ("scored", 0.6, None),
        ("rejected", None, "short reason"),
    ]
    assert result.selected_evaluation.concise_projection()["traces"] == [
        {"trace_id": "a", "status": "scored", "score": 0.6},
        {"trace_id": "b", "status": "rejected", "reason": "short reason"},
    ]
    assert result.selected_evaluation.concise_projection()["scheduler_version"] == 1


def test_canonical_loop_rejects_duplicate_versions_and_invalid_selected_contexts() -> None:
    trace, final_trace = _trace("evolution"), _trace("final")
    report = EvaluationReport(SNAPSHOT.snapshot_digest, "scheduled", (), {})
    record = TraceEvaluation(trace, {}, "scored", 0.4, report)

    duplicate = EvolutionLoop(
        SNAPSHOT, (trace,), (final_trace,), _reference_candidate(), 1,
        lambda context, candidate: _reference_candidate(), lambda records: records,
        lambda snapshot, trace, candidate: record,
        lambda snapshot, records, candidate: (0.4,),
    )
    with pytest.raises(ValueError, match="duplicate"):
        duplicate.run()

    foreign = EvolutionLoop(
        SNAPSHOT, (trace,), (final_trace,), _reference_candidate(), 1,
        lambda context, candidate: _reference_candidate(2),
        lambda records: (TraceEvaluation(trace, {}, "scored", 0.4, report),),
        lambda snapshot, trace, candidate: record,
        lambda snapshot, records, candidate: (0.4,),
    )
    with pytest.raises(ValueError, match="select"):
        foreign.run()


def test_canonical_loop_enforces_the_evaluation_status_vocabulary() -> None:
    trace, final_trace = _trace("evolution"), _trace("final")
    report = EvaluationReport(SNAPSHOT.snapshot_digest, "scheduled", (), {})
    loop = EvolutionLoop(
        SNAPSHOT, (trace,), (final_trace,), _reference_candidate(), 0,
        lambda context, candidate: candidate, lambda records: records,
        lambda snapshot, trace, candidate: TraceEvaluation(trace, {}, "other", None, report),  # type: ignore[arg-type]
        lambda snapshot, records, candidate: (0.4,),
    )
    with pytest.raises(ValueError, match="unknown Evaluation Status"):
        loop.run()


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


def test_model_context_cannot_mutate_the_fixed_trace_set() -> None:
    trace, final_trace = _trace("evolution"), _trace("final")
    report = EvaluationReport(SNAPSHOT.snapshot_digest, "scheduled", (), {})

    def evaluator(snapshot, evaluated_trace, candidate):
        return TraceEvaluation(evaluated_trace, {}, "scored", 0.5, report)

    def proposer(context, candidate):
        context.traces[0].trace.dag.nodes[0].inputs["prompt"] = InputSource.request("changed")
        return _reference_candidate(2)

    loop = EvolutionLoop(
        SNAPSHOT, (trace,), (final_trace,), _reference_candidate(), 1, proposer,
        lambda records: records, evaluator, lambda snapshot, records, candidate: (0.5,),
    )

    with pytest.raises(TypeError):
        loop.run()
    assert trace.dag.nodes[0].inputs["prompt"] == InputSource.request("prompt")


def test_concise_projections_are_versioned_and_final_projection_adds_oracle_comparison() -> None:
    traces = (_trace("trace"),)
    candidate = _reference_candidate()

    evolution = evaluate_scheduler_candidate(SNAPSHOT, traces, candidate)
    final = evaluate_scheduler_candidate(SNAPSHOT, traces, candidate, mode="final", oracle_reference=_reference_candidate(99))

    concise_evolution = evolution.concise_projection()
    concise_final = final.concise_projection()
    assert concise_evolution == {
        "schema_version": "v1",
        "scheduler_version": 1,
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
