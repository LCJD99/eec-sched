from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

from eec_sched import (
    FinalOutput,
    InputSource,
    SchedulerOutput,
    ToolCallPlan,
    ToolCallPlanCandidates,
    ToolCallPlanDataset,
    ToolNode,
    evaluate_scheduler_instance,
    load_profiling_database,
)


ROOT = Path(__file__).parents[1]
SNAPSHOT = load_profiling_database(
    ROOT / "docs/examples/profiling-database.fake.json",
    ROOT / "docs/schemas/profiling-database.schema.json",
)


def _generate() -> ToolCallPlan:
    return ToolCallPlan(
        (ToolNode("generate", "text_generation", {"prompt": InputSource.request("prompt")}),),
        (FinalOutput("generate", "text"),),
    )


def _pipeline() -> ToolCallPlan:
    return ToolCallPlan(
        (
            ToolNode("generate", "text_generation", {"prompt": InputSource.request("prompt")}),
            ToolNode("summarize", "text_summarization", {"text": InputSource.node("generate", "text")}),
        ),
        (FinalOutput("summarize", "text"),),
    )


def test_scheduler_selects_one_of_three_candidates_and_evaluator_simulates_only_it() -> None:
    candidates = ToolCallPlanCandidates((_generate(), _pipeline(), _generate()))
    assignments = {
        "generate": {"configuration_id": "synthetic-reference", "device_id": "cloud"},
        "summarize": {"configuration_id": "fast", "device_id": "edge"},
    }

    report = evaluate_scheduler_instance(
        SNAPSHOT,
        candidates,
        lambda dags: SchedulerOutput(dags[1], assignments, path_index=1),
    )

    assert report.scheduler_status == "scheduled"
    assert report.selected_path_index == 1
    assert report.selected_dag == _pipeline()
    assert tuple(node.node_id for node in report.nodes) == ("generate", "summarize")
    assert len(report.candidate_dags) == 3


def test_evaluator_rejects_a_scheduler_dag_that_was_not_planned() -> None:
    candidates = ToolCallPlanCandidates((_generate(), _generate(), _generate()))
    foreign = _pipeline()
    report = evaluate_scheduler_instance(
        SNAPSHOT,
        candidates,
        lambda dags: SchedulerOutput(
            foreign,
            {"generate": {"configuration_id": "synthetic-reference", "device_id": "cloud"}},
        ),
    )

    assert report.scheduler_status == "rejected"
    assert any("not one of the Planner candidates" in error for error in report.validation_errors)
    assert report.composite_score is None


def test_duplicate_candidates_keep_explicit_path_index() -> None:
    plan = _generate()
    report = evaluate_scheduler_instance(
        SNAPSHOT,
        (plan, plan, plan),
        lambda dags: SchedulerOutput(
            dags[2],
            {"generate": {"configuration_id": "synthetic-reference", "device_id": "cloud"}},
            path_index=2,
        ),
    )

    assert report.scheduler_status == "scheduled"
    assert report.selected_path_index == 2


def test_dataset_imports_three_dag_rows_and_expands_legacy_rows(tmp_path: Path) -> None:
    plan = {
        "nodes": [{"node_id": "node-0", "tool_id": "text_classification", "inputs": {"text": {"kind": "request", "name": "text"}}}],
        "final_outputs": [{"node_id": "node-0", "port": "text"}],
    }
    path = tmp_path / "plans.jsonl"
    path.write_text(
        json.dumps({"trace_id": "new", "task_input": {"text": "x"}, "system_state": {"load": 1}, "dags": [plan, plan, plan]})
        + "\n"
        + json.dumps(plan)
        + "\n",
        encoding="utf-8",
    )

    dataset = ToolCallPlanDataset.load(path)

    assert len(dataset[0].candidate_dags) == 3
    assert dataset[0].trace_id == "new"
    assert dataset[0].system_state["load"] == 1
    assert len(dataset[1].candidate_dags) == 3


def test_dataset_rejects_non_three_candidate_rows(tmp_path: Path) -> None:
    plan = {
        "nodes": [{"node_id": "node-0", "tool_id": "text_classification", "inputs": {"text": {"kind": "request", "name": "text"}}}],
        "final_outputs": [{"node_id": "node-0", "port": "text"}],
    }
    path = tmp_path / "plans.jsonl"
    path.write_text(json.dumps({"dags": [plan, plan]}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly three"):
        ToolCallPlanDataset.load(path)


def test_candidates_schema_is_self_contained_and_rejects_wrong_cardinality_or_dag_shape() -> None:
    schema = json.loads((ROOT / "docs/schemas/tool-call-plan-candidates.schema.json").read_text(encoding="utf-8"))
    example = json.loads((ROOT / "data/planner-candidates.example.jsonl").read_text(encoding="utf-8").splitlines()[0])
    validator = Draft202012Validator(schema)

    validator.validate(example)
    with pytest.raises(ValidationError):
        validator.validate({**example, "dags": example["dags"][:2]})
    with pytest.raises(ValidationError):
        validator.validate({**example, "dags": [*example["dags"], example["dags"][0]]})
    malformed = {**example, "dags": [dict(example["dags"][0]), example["dags"][1], example["dags"][2]]}
    malformed["dags"][0]["nodes"][0]["inputs"]["text"] = {"kind": "request"}
    with pytest.raises(ValidationError):
        validator.validate(malformed)


def test_scheduler_view_freezes_nested_system_state() -> None:
    from eec_sched import SchedulerView, ScoringContext

    view = SchedulerView(
        _generate(),
        ScoringContext(),
        "snapshot",
        {},
        (_generate(), _generate(), _generate()),
        {"devices": {"cloud": {"queue_depth": 2}}},
    )

    with pytest.raises(TypeError):
        view.system_state["devices"]["cloud"]["queue_depth"] = 3  # type: ignore[index]
