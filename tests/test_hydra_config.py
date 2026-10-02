from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from eec_sched.diagnosis import EmptyDiagnosis

ROOT = Path(__file__).parents[1]


def test_default_hydra_config_composes_replaceable_experiment_modules() -> None:
    with initialize_config_dir(
        config_dir=str(ROOT / "experiments/04_baseline"), version_base=None
    ):
        config = compose(config_name="config")

    assert config.search.behavior_descriptor == "performance_behavior"
    assert config.diagnosis.kind == "one_shot"
    assert config.loop.kind == "legacy_reflection"
    assert config.run.evolution_split == "train"
    assert instantiate(config.memory).__class__.__name__ == "InMemoryMemory"


def test_hydra_ablation_replaces_diagnosis_without_changing_source_layout() -> None:
    with initialize_config_dir(
        config_dir=str(ROOT / "experiments/04_baseline"), version_base=None
    ):
        config = compose(config_name="config", overrides=["diagnosis=empty"])

    diagnosis = EmptyDiagnosis(config.diagnosis.advice)
    assert diagnosis.diagnose({}).bottlenecks == ()


def test_hydra_composes_post_evaluation_diagnosis_experiment() -> None:
    with initialize_config_dir(
        config_dir=str(ROOT / "experiments/07_post_evaluation_diagnosis"),
        version_base=None,
    ):
        config = compose(config_name="config")

    assert config.loop.kind == "post_evaluation_diagnosis"
    assert config.run.experiment_name == "07_post_evaluation_diagnosis"


def test_hydra_composes_self_evolving_diagnosis_experiment() -> None:
    with initialize_config_dir(
        config_dir=str(ROOT / "experiments/08_self_evolving_diagnosis"),
        version_base=None,
    ):
        config = compose(config_name="config")

    assert config.diagnosis.kind == "self_evolving"
    assert config.diagnosis.history_limit == 5
    assert config.diagnosis.generated_tool_limit == 5
    assert config.diagnosis.trace_limit == 3
    assert config.loop.kind == "post_evaluation_diagnosis"
    assert config.run.experiment_name == "08_self_evolving_diagnosis"
