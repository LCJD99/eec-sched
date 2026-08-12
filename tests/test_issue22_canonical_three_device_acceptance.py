"""Issue #22 acceptance coverage for the canonical three-device flow."""

from __future__ import annotations

from pathlib import Path

import pytest

from eec_sched import (
    CandidateEvaluation,
    EvaluationTrace,
    FinalEvaluation,
    FinalOutput,
    InputSource,
    SchedulerCandidate,
    SchedulerCandidateRegistry,
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


def test_candidate_evaluation_retains_three_device_proposal_and_uses_trusted_simulation() -> None:
    observed_devices: list[tuple[str, ...]] = []

    def propose(view):
        observed_devices.append(tuple(device["device_id"] for device in view.snapshot_evidence["devices"]))
        return _three_device_assignments(view)

    trace = _three_device_trace("candidate-three-devices")
    candidate = SchedulerCandidate(101, propose)
    result = _candidate_evaluation((trace,), candidate)
    single_device_result = _candidate_evaluation(
        (trace,),
        SchedulerCandidate(101, _single_device_assignments),
    )

    assert observed_devices == [("device", "edge", "cloud")]
    assert isinstance(result, CandidateEvaluation)
    record = result.traces[0]
    assert record.scheduler_version == 101
    assert record.status == "scored"
    assert record.scheduler_computation_time_ms is not None
    assert record.scheduler_computation_time_ms >= 0
    assert {
        node_id: (assignment.configuration_id, assignment.device_id)
        for node_id, assignment in record.assignments.items()
    } == {
        "generate": ("synthetic-reference", "device"),
        "summarize": ("fast", "edge"),
        "classify": ("quality", "cloud"),
    }
    assert {node.node_id: node.device_id for node in record.report.nodes} == {
        "generate": "device",
        "summarize": "edge",
        "classify": "cloud",
    }
    assert len(record.report.transfers) == 2
    assert record.score is not None
    single_device_record = single_device_result.traces[0]
    assert single_device_record.status == "scored"
    assert single_device_record.score is not None
    assert {
        node_id: assignment.device_id for node_id, assignment in single_device_record.assignments.items()
    } == {"generate": "device", "summarize": "device", "classify": "device"}
    assert record.score > single_device_record.score


def test_evolution_candidates_share_one_trace_set_and_expose_only_candidate_scores() -> None:
    evolution_traces = (
        _three_device_trace("evolution-valid"),
        _rejected_trace("evolution-rejected"),
    )
    selected = SchedulerCandidate(
        102,
        lambda view: {} if view.dag.nodes[0].node_id == "reject" else _three_device_assignments(view),
    )

    def failing_proposal(view):
        if view.dag.nodes[0].node_id == "reject":
            raise RuntimeError("candidate could not schedule this Trace")
        return _single_device_assignments(view)

    competing = SchedulerCandidate(103, failing_proposal)
    selected_result = _candidate_evaluation(evolution_traces, selected)
    competing_result = _candidate_evaluation(evolution_traces, competing)

    assert [record.trace.trace_id for record in selected_result.traces] == [
        "evolution-valid",
        "evolution-rejected",
    ]
    assert [record.trace.trace_id for record in competing_result.traces] == [
        "evolution-valid",
        "evolution-rejected",
    ]
    assert selected_result.traces[1].status == "rejected"
    assert selected_result.traces[1].reason
    assert selected_result.traces[1].score_contribution == 0.0
    assert competing_result.traces[1].status == "failed"
    assert "RuntimeError" in competing_result.traces[1].reason
    assert competing_result.traces[1].score_contribution == 0.0
    assert selected_result.candidate_score == pytest.approx(selected_result.traces[0].score_contribution / 2)
    assert competing_result.candidate_score == pytest.approx(competing_result.traces[0].score_contribution / 2)
    assert selected_result.candidate_score > competing_result.candidate_score

    concise = selected_result.concise_projection()
    assert set(concise) == {"schema_version", "scheduler_version", "traces", "candidate_score"}
    assert concise["schema_version"] == "v1"
    assert concise["scheduler_version"] == 102
    concise_scored, concise_rejected = concise["traces"]
    assert set(concise_scored) == {"trace_id", "status", "score"}
    assert concise_scored["status"] == "scored"
    assert concise_scored["score"] == pytest.approx(selected_result.traces[0].score)
    assert set(concise_rejected) == {"trace_id", "status", "score", "reason"}
    assert concise_rejected["status"] == "rejected"
    assert concise_rejected["score"] == 0.0
    assert concise_rejected["reason"] == selected_result.traces[1].reason
    assert all("oracle_reference_score" not in trace and "difference" not in trace for trace in concise["traces"])


def test_final_evaluation_compares_only_selected_candidate_on_an_independent_trace_set() -> None:
    evolution_traces = (_three_device_trace("evolution-only"), _rejected_trace("evolution-rejected"))
    final_traces = (_three_device_trace("final-only"),)
    selected = SchedulerCandidate(104, _three_device_assignments)
    oracle = SchedulerCandidate(999, _cloud_assignments)
    oracle_candidate_evaluation = _candidate_evaluation(final_traces, oracle)

    result = evaluate_scheduler_candidate(
        SNAPSHOT,
        final_traces,
        selected.scheduler_version,
        SchedulerCandidateRegistry({selected.scheduler_version: selected}),
        mode="final",
        oracle_reference=oracle,
    )

    assert {trace.trace_id for trace in evolution_traces}.isdisjoint(trace.trace_id for trace in final_traces)
    assert isinstance(result, FinalEvaluation)
    assert result.candidate_evaluation.scheduler_version == selected.scheduler_version
    assert len(result.traces) == 1
    comparison = result.traces[0]
    assert comparison.trace_evaluation.trace.trace_id == "final-only"
    assert comparison.trace_evaluation.score is not None
    assert comparison.oracle_reference_score is not None
    assert comparison.oracle_reference_score == pytest.approx(
        oracle_candidate_evaluation.traces[0].score,
        rel=1e-3,
    )
    assert comparison.difference == pytest.approx(
        comparison.trace_evaluation.score - comparison.oracle_reference_score
    )
    assert result.average_candidate_score == pytest.approx(comparison.trace_evaluation.score)
    assert result.average_oracle_reference_score == pytest.approx(comparison.oracle_reference_score)
    assert result.average_difference == pytest.approx(comparison.difference)

    concise = result.concise_projection()
    assert set(concise) == {
        "schema_version",
        "scheduler_version",
        "traces",
        "candidate_score",
        "average_candidate_score",
        "average_oracle_reference_score",
        "average_difference",
    }
    assert concise["schema_version"] == "v1"
    assert concise["scheduler_version"] == selected.scheduler_version
    assert concise["candidate_score"] == pytest.approx(comparison.trace_evaluation.score)
    assert concise["average_candidate_score"] == pytest.approx(comparison.trace_evaluation.score)
    assert concise["average_oracle_reference_score"] == pytest.approx(comparison.oracle_reference_score)
    assert concise["average_difference"] == pytest.approx(comparison.difference)
    assert concise["traces"] == [
        {
            "trace_id": "final-only",
            "status": "scored",
            "score": pytest.approx(comparison.trace_evaluation.score),
            "oracle_reference_score": pytest.approx(comparison.oracle_reference_score),
            "difference": pytest.approx(comparison.difference),
        }
    ]


def _candidate_evaluation(
    traces: tuple[EvaluationTrace, ...], candidate: SchedulerCandidate
) -> CandidateEvaluation:
    result = evaluate_scheduler_candidate(
        SNAPSHOT,
        traces,
        candidate.scheduler_version,
        SchedulerCandidateRegistry({candidate.scheduler_version: candidate}),
    )
    assert isinstance(result, CandidateEvaluation)
    return result


def _three_device_trace(trace_id: str) -> EvaluationTrace:
    return EvaluationTrace(
        trace_id,
        {"prompt": "summarize and classify this input"},
        ToolCallPlan(
            nodes=(
                ToolNode("generate", "text_generation", {"prompt": InputSource.request("prompt")}),
                ToolNode("summarize", "text_summarization", {"text": InputSource.node("generate", "text")}),
                ToolNode("classify", "text_classification", {"text": InputSource.node("summarize", "text")}),
            ),
            final_outputs=(FinalOutput("classify", "label"),),
        ),
    )


def _rejected_trace(trace_id: str) -> EvaluationTrace:
    return EvaluationTrace(
        trace_id,
        {"prompt": "this Trace intentionally receives no assignment"},
        ToolCallPlan(
            nodes=(ToolNode("reject", "text_generation", {"prompt": InputSource.request("prompt")}),),
            final_outputs=(FinalOutput("reject", "text"),),
        ),
    )


def _three_device_assignments(view: object) -> dict[str, dict[str, str]]:
    return {
        "generate": {"configuration_id": "synthetic-reference", "device_id": "device"},
        "summarize": {"configuration_id": "fast", "device_id": "edge"},
        "classify": {"configuration_id": "quality", "device_id": "cloud"},
    }


def _single_device_assignments(view: object) -> dict[str, dict[str, str]]:
    return {
        "generate": {"configuration_id": "synthetic-reference", "device_id": "device"},
        "summarize": {"configuration_id": "fast", "device_id": "device"},
        "classify": {"configuration_id": "quality", "device_id": "device"},
    }


def _cloud_assignments(view: object) -> dict[str, dict[str, str]]:
    return {
        "generate": {"configuration_id": "synthetic-reference", "device_id": "cloud"},
        "summarize": {"configuration_id": "fast", "device_id": "cloud"},
        "classify": {"configuration_id": "quality", "device_id": "cloud"},
    }
