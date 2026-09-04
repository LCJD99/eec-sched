from pathlib import Path

from eec_sched import (
    EvaluationTrace,
    FinalOutput,
    InputSource,
    SchedulerCandidate,
    SchedulerCandidateRegistry,
    ToolCallPlan,
    ToolNode,
    evaluate_scheduler_candidate,
    load_profiling_database,
)
from eec_sched.evolution import Experience, JsonlMemory, PerformanceBehaviorDescriptor


ROOT = Path(__file__).parents[1]


def test_performance_behavior_descriptor_records_metrics_and_placement() -> None:
    snapshot = load_profiling_database(
        ROOT / "docs/examples/profiling-database.fake.json",
        ROOT / "docs/schemas/profiling-database.schema.json",
    )
    trace = EvaluationTrace(
        "behavior",
        {},
        ToolCallPlan(
            (ToolNode("generate", "text_generation", {"prompt": InputSource.request("prompt")}),),
            (FinalOutput("generate", "text"),),
        ),
    )
    candidate = SchedulerCandidate(
        1,
        lambda view: {"generate": {"configuration_id": "synthetic-reference", "device_id": "cloud"}},
    )
    evaluation = evaluate_scheduler_candidate(
        snapshot, (trace,), 1, SchedulerCandidateRegistry({1: candidate})
    )

    point = PerformanceBehaviorDescriptor().describe(evaluation)

    assert len(point) == 6
    assert point[-3:] == (0.0, 0.0, 1.0)


def test_jsonl_memory_persists_cross_run_experience(tmp_path) -> None:
    path = tmp_path / "memory.jsonl"
    JsonlMemory(path).record(Experience("mutation improved edge placement", 4))

    recovered = JsonlMemory(path).retrieve("edge")

    assert recovered == (Experience("mutation improved edge placement", 4),)
