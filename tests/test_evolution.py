from __future__ import annotations

from pathlib import Path

import pytest

from eec_sched import (
    EvolutionLoop,
    EvaluationTrace,
    EvaluationReport,
    FinalOutput,
    InputSource,
    SchedulerCandidate,
    SchedulerCandidateRegistry,
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
        calls.append((trace.trace_id, candidate.scheduler_version))
        score = 0.8 if candidate.scheduler_version == 1 else 0.2
        report = EvaluationReport(snapshot.snapshot_digest, "scheduled", (), {})
        return TraceEvaluation(trace, {}, "scored", score, report)

    def select(records):
        selected.append(tuple(record.trace.trace_id for record in records))
        return records[:1]

    def oracle(snapshot, records, candidate):
        oracle_calls.append((candidate.scheduler_version, tuple(record.trace.trace_id for record in records)))
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
    assert result.selected_candidate.scheduler_version == 1
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
        {"trace_id": "b", "status": "rejected", "score": 0.0, "reason": "short reason"},
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
    )


def _reference_candidate(scheduler_version: int = 1) -> SchedulerCandidate:
    return SchedulerCandidate(
        scheduler_version=scheduler_version,
        propose=lambda view: {
            "generate": {"configuration_id": "synthetic-reference", "device_id": "cloud"}
        },
    )


def _evaluate(
    traces: tuple[EvaluationTrace, ...],
    candidate: SchedulerCandidate,
    **kwargs: object,
):
    return evaluate_scheduler_candidate(
        SNAPSHOT,
        traces,
        candidate.scheduler_version,
        SchedulerCandidateRegistry({candidate.scheduler_version: candidate}),
        **kwargs,
    )


def test_candidate_evaluation_preserves_three_stages_and_averages_scored_traces() -> None:
    traces = (_trace("planning-1"), _trace("planning-2"))

    result = _evaluate(traces, _reference_candidate())

    assert result.schema_version == "v1"
    assert result.scheduler_version == 1
    assert [record.trace.trace_id for record in result.traces] == ["planning-1", "planning-2"]
    assert result.traces[0].trace.task_input == {"prompt": "prompt for planning-1"}
    assert result.traces[0].trace.dag == traces[0].dag
    assert result.traces[0].assignments["generate"].configuration_id == "synthetic-reference"
    assert all(record.status == "scored" and record.score is not None for record in result.traces)
    assert result.candidate_score == pytest.approx(sum(record.score for record in result.traces) / 2)


def test_candidate_evaluation_distinguishes_rejected_assignments_from_failed_candidates() -> None:
    rejected = SchedulerCandidate(scheduler_version=2, propose=lambda view: {})
    failed = SchedulerCandidate(scheduler_version=3, propose=lambda view: (_ for _ in ()).throw(RuntimeError("boom")))
    unknown_configuration = SchedulerCandidate(
        scheduler_version=4,
        propose=lambda view: {"generate": {"configuration_id": "not-a-configuration", "device_id": "cloud"}},
    )
    incompatible_device = SchedulerCandidate(
        scheduler_version=5,
        propose=lambda view: {"generate": {"configuration_id": "synthetic-reference", "device_id": "not-a-device"}},
    )
    duplicate_assignment = SchedulerCandidate(
        scheduler_version=6,
        propose=lambda view: (
            {"node_id": "generate", "configuration_id": "synthetic-reference", "device_id": "cloud"},
            {"node_id": "generate", "configuration_id": "synthetic-reference", "device_id": "cloud"},
        ),
    )

    rejected_result = _evaluate((_trace("rejected"),), rejected)
    failed_result = _evaluate((_trace("failed"),), failed)
    unknown_configuration_result = _evaluate((_trace("unknown"),), unknown_configuration)
    incompatible_device_result = _evaluate((_trace("incompatible"),), incompatible_device)
    duplicate_assignment_result = _evaluate((_trace("duplicate"),), duplicate_assignment)

    rejected_record = rejected_result.traces[0]
    failed_record = failed_result.traces[0]
    assert rejected_record.status == "rejected"
    assert rejected_record.score is None
    assert rejected_record.score_contribution == 0.0
    assert rejected_record.reason
    assert rejected_record.scheduler_view is not None
    assert rejected_record.scheduler_proposal == rejected_record.assignments
    assert rejected_record.scheduler_computation_time_ms is not None
    assert rejected_record.scheduler_version == 2
    assert rejected_result.concise_projection()["traces"][0]["score"] == 0.0
    assert failed_record.status == "failed"
    assert failed_record.score is None
    assert failed_record.score_contribution == 0.0
    assert "RuntimeError" in failed_record.reason
    assert unknown_configuration_result.traces[0].status == "rejected"
    assert incompatible_device_result.traces[0].status == "rejected"
    assert duplicate_assignment_result.traces[0].status == "rejected"
    assert "duplicate assignment" in duplicate_assignment_result.traces[0].reason


def test_scheduler_view_cannot_mutate_the_planner_dag() -> None:
    def mutate_view(view):
        view.dag.nodes[0].inputs["prompt"] = InputSource.request("other")
        return {}

    result = _evaluate((_trace("immutable-view"),), SchedulerCandidate(6, mutate_view))

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

    evolution = _evaluate(traces, candidate)
    final = _evaluate(traces, candidate, mode="final", oracle_reference=_reference_candidate(99))

    concise_evolution = evolution.concise_projection()
    concise_final = final.concise_projection()
    assert concise_evolution == {
        "schema_version": "v1",
        "scheduler_version": 1,
        "traces": [{"trace_id": "trace", "status": "scored", "score": pytest.approx(evolution.candidate_score)}],
        "candidate_score": pytest.approx(evolution.candidate_score),
    }
    assert "oracle_reference_score" not in concise_evolution["traces"][0]
    final_trace = concise_final["traces"][0]
    assert final_trace["oracle_reference_score"] is not None
    assert final_trace["difference"] == pytest.approx(final_trace["score"] - final_trace["oracle_reference_score"])
    assert concise_final["average_difference"] == pytest.approx(final_trace["difference"])


def test_candidate_evaluation_records_the_trusted_scheduler_boundary(monkeypatch) -> None:
    import eec_sched.trusted_evaluation as trusted_evaluation

    ticks = iter((10.0, 10.0025))
    monkeypatch.setattr(trusted_evaluation, "perf_counter", lambda: next(ticks))
    received = []

    def propose(view):
        received.append(view)
        return {"generate": {"configuration_id": "synthetic-reference", "device_id": "cloud"}}

    result = _evaluate(
        (_trace("canonical-boundary"),),
        SchedulerCandidate(scheduler_version=7, propose=propose),
    )

    record = result.traces[0]
    assert result.scheduler_version == 7
    assert received == [record.scheduler_view]
    assert record.scheduler_view.scoring_context.minimum_accuracy == 0.0
    assert record.scheduler_proposal == record.assignments
    assert record.scheduler_version == 7
    assert record.scheduler_computation_time_ms == pytest.approx(2.5)
    assert record.report.scheduler_solving_time_ms == pytest.approx(2.5)
    assert record.report.latency_proxy_ms == pytest.approx(12.5)
    assert set(record.scheduler_view.snapshot_evidence) == {"devices", "tools", "transfer_profiles"}
    with pytest.raises(TypeError):
        record.scheduler_view.snapshot_evidence["tools"] = ()  # type: ignore[index]


def test_trusted_registry_resolves_the_scheduler_version_without_source_in_the_request() -> None:
    candidate = _reference_candidate(7)
    registry = SchedulerCandidateRegistry({7: candidate})

    result = evaluate_scheduler_candidate(SNAPSHOT, (_trace("resolved-version"),), 7, registry)

    assert result.scheduler_version == 7
    with pytest.raises(ValueError, match="unknown Scheduler Version"):
        evaluate_scheduler_candidate(SNAPSHOT, (_trace("unknown-version"),), 8, registry)


def test_trusted_evaluator_fault_is_not_converted_to_a_candidate_status(monkeypatch) -> None:
    import eec_sched.trusted_evaluation as trusted_evaluation
    from eec_sched import TrustedEvaluationError

    def broken_simulation(snapshot, dag, assignments):
        raise TrustedEvaluationError("broken trusted simulator")

    monkeypatch.setattr(trusted_evaluation, "_simulate", broken_simulation)

    with pytest.raises(TrustedEvaluationError, match="trusted simulation failed"):
        _evaluate((_trace("system-error"),), _reference_candidate())


def test_noncompleting_candidate_is_failed_by_the_trusted_time_limit(monkeypatch) -> None:
    import eec_sched.evolution as evolution

    monkeypatch.setattr(evolution, "_CANDIDATE_TIMEOUT_SECONDS", 0.01)

    def never_returns(view):
        while True:
            pass

    result = _evaluate((_trace("timed-out"),), SchedulerCandidate(9, never_returns))

    record = result.traces[0]
    assert record.status == "failed"
    assert record.score_contribution == 0.0
    assert record.reason == "Scheduler Candidate exceeded the trusted computation time limit"


def test_candidate_cannot_bypass_the_time_limit_by_catching_the_timeout(monkeypatch) -> None:
    import eec_sched.evolution as evolution

    monkeypatch.setattr(evolution, "_CANDIDATE_TIMEOUT_SECONDS", 0.01)

    def catches_timeout(view):
        try:
            while True:
                pass
        except TimeoutError:
            return {"generate": {"configuration_id": "synthetic-reference", "device_id": "cloud"}}

    result = _evaluate((_trace("caught-timeout"),), SchedulerCandidate(11, catches_timeout))

    record = result.traces[0]
    assert record.status == "failed"
    assert record.score_contribution == 0.0
    assert record.reason == "Scheduler Candidate exceeded the trusted computation time limit"


def test_final_evaluation_averages_rejected_traces_as_zero() -> None:
    rejected_trace = EvaluationTrace(
        "rejected",
        {"prompt": "rejected"},
        ToolCallPlan(
            nodes=(ToolNode("other", "text_generation", {"prompt": InputSource.request("prompt")}),),
            final_outputs=(FinalOutput("other", "text"),),
        ),
    )
    candidate = SchedulerCandidate(
        10,
        lambda view: (
            {"generate": {"configuration_id": "synthetic-reference", "device_id": "cloud"}}
            if view.dag.nodes[0].node_id == "generate"
            else {}
        ),
    )
    traces = (_trace("scored"), rejected_trace)

    result = _evaluate(traces, candidate, mode="final", oracle_reference=_reference_candidate(99))

    assert result.candidate_evaluation.candidate_score == pytest.approx(result.traces[0].trace_evaluation.score / 2)
    assert result.average_candidate_score == pytest.approx(result.candidate_evaluation.candidate_score)


def test_candidate_cannot_control_canonical_order_for_ready_nodes() -> None:
    dag = ToolCallPlan(
        nodes=(
            ToolNode("beta", "text_generation", {"prompt": InputSource.request("beta")}),
            ToolNode("alpha", "text_generation", {"prompt": InputSource.request("alpha")}),
        ),
        final_outputs=(FinalOutput("alpha", "text"), FinalOutput("beta", "text")),
    )
    trace = EvaluationTrace("ready-node-order", {"alpha": "a", "beta": "b"}, dag)
    candidate = SchedulerCandidate(
        scheduler_version=8,
        propose=lambda view: {
            "beta": {"configuration_id": "synthetic-reference", "device_id": "cloud"},
            "alpha": {"configuration_id": "synthetic-reference", "device_id": "cloud"},
        },
    )

    first = _evaluate((trace,), candidate)
    second = _evaluate((trace,), candidate)

    assert [(node.node_id, node.start_ms) for node in first.traces[0].report.nodes] == [
        ("alpha", 0.0),
        ("beta", 10.0),
    ]
    assert first.traces[0].report.nodes == second.traces[0].report.nodes
