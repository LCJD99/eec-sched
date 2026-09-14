from __future__ import annotations

from pathlib import Path

import pytest

from eec_sched import (
    CandidateGenerationRequest,
    EvolutionGraph,
    EvolutionLoop,
    EvaluationTrace,
    EvaluationReport,
    FinalOutput,
    InputSource,
    NodeAssignment,
    SchedulerCandidate,
    SchedulerCandidateDraft,
    SchedulerCandidateRegistry,
    SimulatedNode,
    SimulatedTransfer,
    TraceEvaluation,
    ToolCallPlan,
    ToolNode,
    evaluate_scheduler_candidate,
    load_profiling_database,
)
from eec_sched.diagnosis import DiagnosisResult
from eec_sched.evolution import PostEvaluationDiagnosisEvolutionLoop, UnboundedRepertoire
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
    )


def _reference_candidate(
    scheduler_version: int = 1, *, parent_scheduler_versions: tuple[int, ...] = ()
) -> SchedulerCandidate:
    return SchedulerCandidate(
        scheduler_version=scheduler_version,
        propose=lambda view: {
            "generate": {"configuration_id": "synthetic-reference", "device_id": "cloud"}
        },
        source_code=f"def schedule_{scheduler_version}(view):\n    return {{}}\n",
        parent_scheduler_versions=parent_scheduler_versions,
        strategy_description=f"strategy for scheduler {scheduler_version}",
    )


def test_evolution_graph_starts_with_three_roots_and_retains_mutation_lineage() -> None:
    """The application seam retains each graph node and its direct parents."""
    report = EvaluationReport(SNAPSHOT.snapshot_digest, "scheduled", (), {})
    roots = tuple(_reference_candidate(version) for version in (1, 2, 3))
    generated_requests: list[CandidateGenerationRequest] = []
    reflection_calls = []

    def evaluator(snapshot, trace, candidate):
        score = {1: 0.1, 2: 0.4, 3: 0.3, 4: 0.9}[candidate.scheduler_version]
        return TraceEvaluation(trace, {}, "scored", score, report)

    def coding_agent(request):
        generated_requests.append(request)
        candidate = _reference_candidate(4)
        return SchedulerCandidateDraft(candidate.propose, candidate.source_code, candidate.strategy_description)

    result = EvolutionLoop(
        snapshot=SNAPSHOT,
        evolution_traces=(_trace("evolution"),),
        final_evaluation_traces=(_trace("final"),),
        initial_candidates=roots,
        rounds=1,
        mutation_probability=1.0,
        reflection_agent=lambda reflection: reflection_calls.append(reflection) or "strengthen the chosen strategy",
        coding_agent=coding_agent,
        trusted_evaluator=evaluator,
        oracle=lambda snapshot, records, candidate: (0.5,),
        random_float=lambda: 0.0,
        random_choice=lambda candidates: candidates[0],
    ).run()

    assert isinstance(result.evolution_graph, EvolutionGraph)
    assert tuple(result.evolution_graph.candidates) == (1, 2, 3, 4)
    assert result.evolution_graph.candidates[1].parent_scheduler_versions == ()
    assert result.evolution_graph.candidates[4].parent_scheduler_versions == (2,)
    assert result.selected_candidate.scheduler_version == 4
    assert generated_requests[0].parent_source_codes == ("def schedule_2(view):\n    return {}\n",)
    assert len(reflection_calls) == 1


def test_crossover_uses_least_similar_top_candidates_and_hides_source_from_reflection() -> None:
    report = EvaluationReport(SNAPSHOT.snapshot_digest, "scheduled", (), {})
    reflections = []
    requests = []
    scores = {
        1: (1.0, 0.0),
        2: (0.0, 1.0),
        3: (1.0, 1.0),
        4: (0.5, 0.5),
    }

    def evaluator(snapshot, trace, candidate):
        if trace.trace_id == "final":
            return TraceEvaluation(trace, {}, "scored", 0.5, report)
        index = ("left", "right").index(trace.trace_id)
        return TraceEvaluation(trace, {}, "scored", scores[candidate.scheduler_version][index], report)

    def coding_agent(request):
        requests.append(request)
        candidate = _reference_candidate(4)
        return SchedulerCandidateDraft(candidate.propose, candidate.source_code, candidate.strategy_description)

    result = EvolutionLoop(
        SNAPSHOT,
        (_trace("left"), _trace("right")),
        (_trace("final"),),
        tuple(_reference_candidate(version) for version in (1, 2, 3)),
        1,
        0.0,
        lambda reflection: reflections.append(reflection) or "combine complementary strengths",
        coding_agent,
        evaluator,
        lambda snapshot, records, candidate: (0.25,),
        random_float=lambda: 1.0,
    ).run()

    assert requests[0].operator == "crossover"
    assert requests[0].parent_scheduler_versions == (1, 2)
    assert requests[0].parent_source_codes == (
        "def schedule_1(view):\n    return {}\n",
        "def schedule_2(view):\n    return {}\n",
    )
    assert reflections[0].operator == "crossover"
    assert reflections[0].strategy_descriptions == {1: "strategy for scheduler 1", 2: "strategy for scheduler 2"}
    assert reflections[0].parent_source_codes == (
        "def schedule_1(view):\n    return {}\n",
        "def schedule_2(view):\n    return {}\n",
    )
    assert result.selected_candidate.scheduler_version == 3


def test_evolution_loop_requires_three_distinct_roots_without_parents() -> None:
    arguments = dict(
        snapshot=SNAPSHOT,
        evolution_traces=(_trace("evolution"),),
        final_evaluation_traces=(_trace("final"),),
        rounds=0,
        mutation_probability=1.0,
        reflection_agent=lambda reflection: "advice",
        coding_agent=lambda request: _reference_candidate(4, parent_scheduler_versions=request.parent_scheduler_versions),
        trusted_evaluator=lambda snapshot, trace, candidate: TraceEvaluation(
            trace, {}, "scored", 0.5, EvaluationReport(snapshot.snapshot_digest, "scheduled", (), {})
        ),
        oracle=lambda snapshot, records, candidate: (0.5,),
    )

    with pytest.raises(ValueError, match="exactly three"):
        EvolutionLoop(initial_candidates=(_reference_candidate(1), _reference_candidate(2)), **arguments)
    with pytest.raises(ValueError, match="distinct"):
        EvolutionLoop(initial_candidates=(_reference_candidate(1), _reference_candidate(1), _reference_candidate(2)), **arguments)
    with pytest.raises(ValueError, match="must not have"):
        EvolutionLoop(
            initial_candidates=(
                _reference_candidate(1),
                _reference_candidate(2, parent_scheduler_versions=(1,)),
                _reference_candidate(3),
            ),
            **arguments,
        )


def test_descendant_mutation_reflection_includes_direct_parent_evidence_and_strategy() -> None:
    report = EvaluationReport(SNAPSHOT.snapshot_digest, "scheduled", (), {})
    reflections = []
    scores = {1: (0.2, 0.2), 2: (0.1, 0.1), 3: (0.0, 0.0), 4: (0.9, 0.4), 5: (0.8, 0.8)}

    def evaluator(snapshot, trace, candidate):
        if trace.trace_id == "final":
            return TraceEvaluation(trace, {}, "scored", 0.5, report)
        return TraceEvaluation(
            trace, {}, "scored", scores[candidate.scheduler_version][("left", "right").index(trace.trace_id)], report
        )

    def coding_agent(request):
        version = 3 + len(reflections)
        candidate = _reference_candidate(version)
        return SchedulerCandidateDraft(candidate.propose, candidate.source_code, candidate.strategy_description)

    EvolutionLoop(
        SNAPSHOT,
        (_trace("left"), _trace("right")),
        (_trace("final"),),
        tuple(_reference_candidate(version) for version in (1, 2, 3)),
        2,
        1.0,
        lambda reflection: reflections.append(reflection) or "advice",
        coding_agent,
        evaluator,
        lambda snapshot, records, candidate: (0.5,),
        random_float=lambda: 0.0,
        random_choice=lambda candidates: candidates[-1] if len(candidates) > 1 else candidates[0],
    ).run()

    descendant_reflection = reflections[1]
    assert descendant_reflection.strategy_descriptions == {
        1: "strategy for scheduler 1",
        4: "strategy for scheduler 4",
    }
    assert [record.candidate_score for record in descendant_reflection.trace_evidence] == [0.9, 0.4]
    assert [record.compared_score for record in descendant_reflection.trace_evidence] == [0.2, 0.2]
    assert [record.score_delta for record in descendant_reflection.trace_evidence] == [0.7, 0.2]


def test_scheduler_candidate_requires_executable_source_code() -> None:
    with pytest.raises(ValueError, match="executable Python"):
        SchedulerCandidate(1, lambda view: {}, source_code="not valid python source !")


class _RecordingRepertoire:
    def __init__(self, log):
        self.log = log
        self.items_seen = []

    def add(self, candidate, evaluation):
        self.log.append(("repertoire", candidate.scheduler_version))
        self.items_seen.append((candidate, evaluation))

    def items(self):
        return tuple(self.items_seen)


class _RecordingMemory:
    def __init__(self, log):
        self.log = log
        self.experiences = []

    def retrieve(self, query="", *, limit=10):
        self.log.append(("memory_retrieve", query))
        return ()

    def record(self, experience):
        self.log.append(("memory", experience.strategy_version, experience.summary))
        self.experiences.append(experience)


def _post_evaluation_loop(*, operator="mutation", rounds=1, log=None, diagnoses=None, final_id="final"):
    log = log if log is not None else []
    diagnoses = diagnoses if diagnoses is not None else {}
    full_report = EvaluationReport(
        SNAPSHOT.snapshot_digest,
        "scheduled",
        (),
        {},
        nodes=(SimulatedNode("generate", "synthetic-reference", "cloud", 0.0, 2.0, 128.0),),
        transfers=(SimulatedTransfer("input", "generate", "cloud", "cloud", 0.0, 0.0, 0.0),),
        accuracy=0.9,
        raw_accuracy_metrics={"quality": 0.9},
        simulated_makespan_ms=2.0,
        scheduler_solving_time_ms=1.0,
        latency=3.0,
        resource=128.0,
        composite_score=0.8,
    )
    roots = tuple(_reference_candidate(version) for version in (1, 2, 3))
    traces = (_trace("evolution"),)
    final_traces = (_trace(final_id),)

    def evaluator(snapshot, trace, candidate):
        log.append(("evaluate", candidate.scheduler_version, trace.trace_id))
        return TraceEvaluation(
            trace,
            {"generate": NodeAssignment("synthetic-reference", "cloud")},
            "scored",
            0.5,
            full_report,
            scheduler_version=candidate.scheduler_version,
        )

    def diagnosis_agent(evidence):
        log.append(("diagnose", evidence["scheduler_version"], evidence))
        diagnoses[evidence["scheduler_version"]] = DiagnosisResult(
            f"diagnosis-{evidence['scheduler_version']}",
            (f"bottleneck-{evidence['scheduler_version']}",),
            (f"evidence-{evidence['scheduler_version']}",),
        )
        return diagnoses[evidence["scheduler_version"]]

    def recorder(event):
        log.append(("record", event["type"], event.get("scheduler_version"), event.get("trace_id")))

    generation_requests = []

    def coding(request):
        generation_requests.append(request)
        version = 4 + len(generation_requests) - 1
        candidate = _reference_candidate(version, parent_scheduler_versions=request.parent_scheduler_versions)
        return SchedulerCandidateDraft(candidate.propose, candidate.source_code, "generated")

    loop = PostEvaluationDiagnosisEvolutionLoop(
        SNAPSHOT,
        traces,
        final_traces,
        roots,
        rounds,
        1.0 if operator == "mutation" else 0.0,
        diagnosis_agent,
        coding,
        evaluator,
        lambda snapshot, records, candidate: (0.5,),
        random_float=lambda: 0.0 if operator == "mutation" else 1.0,
        random_choice=lambda candidates: candidates[0],
        repertoire=_RecordingRepertoire(log),
        memory=_RecordingMemory(log),
        trace_recorder=recorder,
    )
    return loop, log, generation_requests, diagnoses


def test_post_evaluation_loop_orders_evaluation_trace_diagnosis_archive_and_memory() -> None:
    loop, log, requests, diagnoses = _post_evaluation_loop()
    result = loop.run()

    # Every root and child has the complete sequence before the next phase.
    root_one = [index for index, item in enumerate(log) if item[0] == "evaluate" and item[1] == 1 and item[2] == "evolution"][0]
    assert log[root_one : root_one + 5] == [
        ("evaluate", 1, "evolution"),
        ("record", "evaluation_trace", 1, "evolution"),
        ("diagnose", 1, log[root_one + 2][2]),
        ("record", "diagnosis", 1, None),
        ("repertoire", 1),
    ]
    assert log[root_one + 5][0:2] == ("memory", 1)
    child = [index for index, item in enumerate(log) if item[0] == "evaluate" and item[1] == 4][0]
    assert log[child : child + 6] == [
        ("evaluate", 4, "evolution"),
        ("record", "evaluation_trace", 4, "evolution"),
        ("diagnose", 4, log[child + 2][2]),
        ("record", "diagnosis", 4, None),
        ("repertoire", 4),
        ("memory", 4, log[child + 5][2]),
    ]
    assert result.diagnoses[1] == diagnoses[1]
    assert result.diagnoses[4] == diagnoses[4]


def test_post_evaluation_diagnosis_receives_complete_source_free_trace() -> None:
    loop, log, _, _ = _post_evaluation_loop(rounds=0)
    loop.run()
    evidence = next(item[2] for item in log if item[0] == "diagnose" and item[1] == 1)
    trace = evidence["traces"][0]
    assert {"nodes", "transfers", "assignments", "dag", "snapshot_digest", "scheduler_version"} <= set(trace)
    assert trace["nodes"][0]["start_ms"] == 0.0

    def contains_source_code(value):
        if isinstance(value, dict):
            return any(key in {"source_code", "parent_source_codes"} or contains_source_code(item) for key, item in value.items())
        if isinstance(value, (list, tuple)):
            return any(contains_source_code(item) for item in value)
        return False

    assert not contains_source_code(evidence)


def test_post_evaluation_briefs_use_saved_mutation_and_crossover_diagnoses() -> None:
    mutation_loop, _, mutation_requests, mutation_diagnoses = _post_evaluation_loop()
    mutation_loop.run()
    mutation_request = mutation_requests[0]
    parent_version = mutation_request.parent_scheduler_versions[0]
    assert f"scheduler_version={parent_version}" in mutation_request.reflection_advice
    assert mutation_diagnoses[parent_version].advice in mutation_request.reflection_advice

    crossover_loop, _, crossover_requests, crossover_diagnoses = _post_evaluation_loop(operator="crossover")
    crossover_loop.run()
    crossover_request = crossover_requests[0]
    left, right = crossover_request.parent_scheduler_versions
    assert f"scheduler_versions={left},{right}" in crossover_request.reflection_advice
    assert crossover_diagnoses[left].advice in crossover_request.reflection_advice
    assert crossover_diagnoses[right].advice in crossover_request.reflection_advice


def test_post_evaluation_loop_keeps_final_traces_out_of_diagnosis_and_artifacts() -> None:
    loop, log, _, _ = _post_evaluation_loop()
    result = loop.run()
    assert result.final_evaluation.traces[0].trace_evaluation.trace.trace_id == "final"
    assert all(item[2] != "final" for item in log if item[0] == "record")
    assert all(
        trace["trace_id"] != "final"
        for item in log
        if item[0] == "diagnose"
        for trace in item[2]["traces"]
    )
    assert all(item[1] != "final" for item in log if item[0] in {"repertoire", "memory"})


def test_post_evaluation_results_are_immutable_and_legacy_result_has_no_diagnoses() -> None:
    loop, _, _, _ = _post_evaluation_loop(rounds=0)
    result = loop.run()
    with pytest.raises(TypeError):
        result.diagnoses[1] = DiagnosisResult("changed")  # type: ignore[index]

    legacy = EvolutionLoop(
        SNAPSHOT,
        (_trace("legacy-evolution"),),
        (_trace("legacy-final"),),
        tuple(_reference_candidate(version) for version in (1, 2, 3)),
        0,
        1.0,
        lambda reflection: "legacy advice",
        lambda request: SchedulerCandidateDraft(lambda view: {}, "def propose(view): return {}", "child"),
        lambda snapshot, trace, candidate: TraceEvaluation(trace, {}, "scored", 0.5, EvaluationReport(SNAPSHOT.snapshot_digest, "scheduled", (), {})),
        lambda snapshot, records, candidate: (0.5,),
    ).run()
    assert legacy.diagnoses == {}


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


def test_concise_projections_are_versioned_and_final_projection_adds_oracle_comparison() -> None:
    traces = (_trace("trace"),)
    candidate = _reference_candidate()

    evolution = _evaluate(traces, candidate)
    final = _evaluate(traces, candidate, mode="final", oracle_reference=_reference_candidate(99))

    concise_evolution = evolution.concise_projection()
    concise_final = final.concise_projection()
    concise_metrics = dict(evolution.traces[0].raw_metrics)
    assert concise_evolution == {
        "schema_version": "v1",
        "scheduler_version": 1,
        "traces": [{
            "trace_id": "trace",
            "status": "scored",
            "score": pytest.approx(evolution.candidate_score),
            "metrics": concise_metrics,
        }],
        "candidate_score": pytest.approx(evolution.candidate_score),
    }
    assert "oracle_reference_score" not in concise_evolution["traces"][0]
    final_trace = concise_final["traces"][0]
    assert final_trace["oracle_reference_score"] is not None
    assert final_trace["difference"] == pytest.approx(final_trace["score"] - final_trace["oracle_reference_score"])
    assert concise_final["average_difference"] == pytest.approx(final_trace["difference"])


def test_candidate_evaluation_records_the_trusted_scheduler_boundary(monkeypatch) -> None:
    import eec_sched.evaluation.evaluator as trusted_evaluation

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
    assert record.scheduler_view.scoring_context.accuracy_weight == pytest.approx(1 / 3)
    assert record.scheduler_view.scoring_context.latency_weight == pytest.approx(1 / 3)
    assert record.scheduler_view.scoring_context.resource_weight == pytest.approx(1 / 3)
    assert record.scheduler_proposal == record.assignments
    assert record.scheduler_version == 7
    assert record.scheduler_computation_time_ms == pytest.approx(2.5)
    assert record.report.scheduler_solving_time_ms == pytest.approx(2.5)
    assert record.report.latency == pytest.approx(70.5384)
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
    import eec_sched.evaluation.evaluator as trusted_evaluation
    from eec_sched import TrustedEvaluationError

    def broken_simulation(snapshot, dag, assignments):
        raise TrustedEvaluationError("broken trusted simulator")

    monkeypatch.setattr(trusted_evaluation, "simulate", broken_simulation)

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
        ("alpha", pytest.approx(28.0213333333)),
        ("beta", pytest.approx(56.0426666667)),
    ]
    assert first.traces[0].report.nodes == second.traces[0].report.nodes
