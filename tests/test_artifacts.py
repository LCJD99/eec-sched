import json

from eec_sched.artifacts import InMemoryArtifactStore, JsonlArtifactStore


def test_in_memory_artifact_store_records_events_and_result() -> None:
    store = InMemoryArtifactStore()

    store.append_event("diagnosis", {"advice": "move work"})
    store.write_result({"winner": 4})

    assert store.events == [{"type": "diagnosis", "advice": "move work"}]
    assert store.result == {"winner": 4}


def test_jsonl_artifact_store_uses_one_run_identifier(tmp_path) -> None:
    store = JsonlArtifactStore(tmp_path, run_id="test-run")

    store.append_event("candidate", {"version": 4})
    result_path = store.write_result({"winner": 4})

    assert json.loads(store.events_path.read_text()) == {"type": "candidate", "version": 4}
    assert json.loads(result_path.read_text()) == {"winner": 4}
