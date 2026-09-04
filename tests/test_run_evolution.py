from __future__ import annotations

from types import SimpleNamespace

from eec_sched.app.run_evolution import _result_json, build_agents
from eec_sched.artifacts import InMemoryArtifactStore
from eec_sched.evolution import CandidateGenerationRequest, ReflectionInput, SchedulerCandidate


class FakeModel:
    def __init__(self, *outputs: str) -> None:
        self.outputs = iter(outputs)

    def complete(self, **kwargs) -> str:
        return next(self.outputs)


def test_hydra_agents_record_diagnosis_and_candidate_generation() -> None:
    artifacts = InMemoryArtifactStore()
    reflection, coding = build_agents(
        "one_shot",
        FakeModel("diagnose the slow cloud placement"),
        FakeModel('{"source_code":"def propose(view): return {}","strategy_description":"new"}'),
        artifacts,
    )

    advice = reflection(ReflectionInput("mutation", "improve", {}, (), ()))
    draft = coding(CandidateGenerationRequest("mutation", (1,), ("source",), advice))

    assert advice == "diagnose the slow cloud placement"
    assert draft.strategy_description == "new"
    assert [event["type"] for event in artifacts.events] == ["diagnosis", "coding"]


def test_empty_diagnosis_uses_configured_ablation_advice() -> None:
    artifacts = InMemoryArtifactStore()
    reflection, _ = build_agents(
        "empty",
        FakeModel(),
        FakeModel(),
        artifacts,
        empty_diagnosis_advice="no diagnosis",
    )

    assert reflection(ReflectionInput("mutation", "improve", {}, (), ())) == "no diagnosis"


def test_result_json_retains_candidate_lineage_and_source() -> None:
    root = SchedulerCandidate(1, lambda view: {}, strategy_description="root")
    child = SchedulerCandidate(4, lambda view: {}, parent_scheduler_versions=(1,), strategy_description="child")
    evaluation = SimpleNamespace(
        concise_projection=lambda: {"scheduler_version": 4, "candidate_score": 0.5},
        traces=(),
    )
    result = SimpleNamespace(
        evolution_graph=SimpleNamespace(candidates={1: root, 4: child}, evaluations={1: evaluation, 4: evaluation}),
        selected_candidate=child,
        final_evaluation=SimpleNamespace(concise_projection=lambda: {"average_candidate_score": 0.5}),
    )

    payload = _result_json(result)

    assert payload["selected_scheduler_version"] == 4
    assert payload["evolution_evaluations"]["4"]["parent_scheduler_versions"] == [1]
    assert payload["evolution_evaluations"]["4"]["strategy_description"] == "child"
