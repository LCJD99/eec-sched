import json
import re

from eec_sched.artifacts import InMemoryArtifactStore, JsonlArtifactStore


def test_in_memory_artifact_store_records_events_and_result() -> None:
    store = InMemoryArtifactStore()

    store.append_event("diagnosis", {"advice": "move work"})
    store.write_result({"winner": 4})

    assert store.events == [{"type": "diagnosis", "advice": "move work"}]
    assert store.result == {"winner": 4}


def test_in_memory_artifact_store_can_capture_composed_config() -> None:
    store = InMemoryArtifactStore()

    store.write_config({"run": {"experiment_name": "001_baseline"}})

    assert store.config == {"run": {"experiment_name": "001_baseline"}}


def test_jsonl_artifact_store_uses_one_run_identifier(tmp_path) -> None:
    store = JsonlArtifactStore(tmp_path, run_id="test-run")

    store.append_event("candidate", {"version": 4})
    result_path = store.write_result({"winner": 4})

    assert json.loads(store.events_path.read_text()) == {"type": "candidate", "version": 4}
    assert json.loads(result_path.read_text()) == {"winner": 4}


def test_jsonl_artifact_store_creates_a_complete_fresh_run_record(tmp_path) -> None:
    store = JsonlArtifactStore(tmp_path, experiment_name="001_baseline")
    config_path = store.write_config({"run": {"experiment_name": "001_baseline"}})
    store.append_event("candidate", {"version": 1})
    result_path = store.write_result({"winner": 1})

    assert re.fullmatch(r"001_baseline_\d{8}T\d{6}", store.run_dir.name)
    assert list(tmp_path.iterdir()) == [store.run_dir]
    assert config_path == store.run_dir / "config.yaml"
    assert result_path.parent == store.run_dir
    assert config_path.read_text(encoding="utf-8").startswith("run:\n")


def test_jsonl_artifact_store_never_reuses_an_explicit_run_directory(tmp_path) -> None:
    JsonlArtifactStore(tmp_path, run_id="001_baseline_20260904T000000")

    try:
        JsonlArtifactStore(tmp_path, run_id="001_baseline_20260904T000000")
    except FileExistsError:
        pass
    else:
        raise AssertionError("an existing run directory must not be reused")
