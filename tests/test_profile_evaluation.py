from __future__ import annotations

from pathlib import Path

from eec_sched import Configuration
from eec_sched.profile_evaluation import (
    EvaluationSample,
    EvaluationSuite,
    load_persisted_samples,
    load_profile_artifact,
    profile_suite,
    save_profile_artifact,
    select_fixed_samples,
)


class Runner:
    def __init__(self) -> None:
        self.prepared: list[str] = []
        self.calls = 0

    def prepare(self, configuration: Configuration) -> None:
        self.prepared.append(configuration.configuration_id)

    def run(self, inputs: dict[str, object], configuration: Configuration) -> dict[str, object]:
        self.calls += 1
        return {"text": f"{configuration.parameters['prefix']}{inputs['text']}"}


def test_five_samples_are_selected_and_replayed_in_persisted_order() -> None:
    all_samples = [EvaluationSample(str(index), {"text": str(index)}, str(index)) for index in range(6)]
    selected = select_fixed_samples(all_samples)

    assert [sample.sample_id for sample in selected] == ["0", "1", "2", "3", "4"]
    assert [sample.sample_id for sample in load_persisted_samples(reversed(all_samples), ["3", "1", "4", "0", "2"])] == ["3", "1", "4", "0", "2"]


def test_profile_artifact_persists_raw_scores_and_fifty_latency_observations(tmp_path: Path) -> None:
    samples = tuple(EvaluationSample(str(index), {"text": str(index)}, f"x{index}") for index in range(5))
    suite = EvaluationSuite("example", "org/data", "abc123", "test", "accuracy", True, "official-1")
    runners: list[Runner] = []

    def factory() -> Runner:
        runner = Runner()
        runners.append(runner)
        return runner

    artifact = profile_suite(
        suite=suite,
        model_name="model@revision",
        configurations=(Configuration("fast", {"prefix": "x"}),),
        runner_factory=factory,
        samples=samples,
        scorer=lambda predictions, targets: sum(prediction["output"]["text"] == target for prediction, target in zip(predictions, targets)) / len(targets),
        device="cpu",
    )
    path = tmp_path / "profiles" / "example.json"
    save_profile_artifact(path, artifact)
    restored = load_profile_artifact(path)

    assert artifact.accuracy[0].raw_metric == 1.0
    assert artifact.latency[0].sample_count == 50
    assert runners[0].calls == 55
    assert restored.sample_ids == ("0", "1", "2", "3", "4")
    assert restored.accuracy == artifact.accuracy
