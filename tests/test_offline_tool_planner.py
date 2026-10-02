from __future__ import annotations

import json
from pathlib import Path

import pytest

from eec_sched.offline_tool_planner import (
    PlannerBatchError,
    WorkItem,
    load_tool_catalog,
    parse_three_dag_response,
    run_batch,
)
from eec_sched.workflow import ToolCallPlanDataset


ROOT = Path(__file__).parents[1]
SCHEMA = ROOT / "docs/schemas/tool-call-dag.schema.json"


def _dag(tool_id: str = "classify") -> dict[str, object]:
    return {
        "nodes": [
            {
                "node_id": "n0",
                "tool_id": tool_id,
                "inputs": {"text": {"kind": "request", "name": "request"}},
            }
        ],
        "final_outputs": [{"node_id": "n0", "port": "text"}],
    }


def _specs(tmp_path: Path):
    path = tmp_path / "catalog.json"
    path.write_text(
        json.dumps(
            {
                "tools": [
                    {
                        "tool_id": "classify",
                        "description": "Classify text.",
                        "inputs": {"text": {"modality": "text", "required": True}},
                        "outputs": {"text": {"modality": "text"}},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return load_tool_catalog(path), path


class _FakeModel:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.calls = 0

    def complete(self, **kwargs: object) -> str:
        self.calls += 1
        return next(self.responses)


def test_response_requires_three_real_dags_and_does_not_use_legacy_fallback() -> None:
    with pytest.raises(ValueError, match="dags array"):
        parse_three_dag_response(json.dumps(_dag()), schema_path=SCHEMA)
    with pytest.raises(ValueError, match="exactly three"):
        parse_three_dag_response(json.dumps({"dags": [_dag(), _dag()]}), schema_path=SCHEMA)
    with pytest.raises(ValueError, match="only a dags array"):
        parse_three_dag_response(
            json.dumps({"dags": [_dag(), _dag(), _dag()], "explanation": "extra"}),
            schema_path=SCHEMA,
        )


def test_catalog_schema_rejects_unknown_port_fields(tmp_path: Path) -> None:
    path = tmp_path / "catalog.json"
    path.write_text(
        json.dumps({
            "tools": [{
                "tool_id": "classify",
                "description": "Classify text.",
                "inputs": {"text": {"modality": "text", "unexpected": True}},
                "outputs": {"text": {"modality": "text"}},
            }]
        }),
        encoding="utf-8",
    )
    with pytest.raises(PlannerBatchError, match="tool catalog does not match schema"):
        load_tool_catalog(path)


def test_batch_calls_model_once_per_item_and_writes_canonical_sidecars(tmp_path: Path) -> None:
    specs, catalog_path = _specs(tmp_path)
    input_path = tmp_path / "requests.jsonl"
    input_path.write_text(
        '{"id":"a","request":"classify hello"}\n{"id":"b","request":"classify bye"}\n',
        encoding="utf-8",
    )
    content = json.dumps({"dags": [_dag(), _dag(), _dag()]})
    model = _FakeModel([content, content])
    output_path, failures_path, manifest_path = (tmp_path / name for name in ("dags.jsonl", "failures.jsonl", "manifest.json"))
    summary = run_batch(
        items=(WorkItem("a", "classify hello"), WorkItem("b", "classify bye")),
        specs=specs,
        system_prompt="Do not leak references.",
        model=model,
        output_path=output_path,
        failures_path=failures_path,
        manifest_path=manifest_path,
        input_path=input_path,
        catalog_path=catalog_path,
        model_name="test-model",
        endpoint="https://example.test/v1?secret=remove-me",
        prompt_version="test-v1",
        schema_path=SCHEMA,
    )

    assert model.calls == 2
    assert summary.succeeded == 2
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert [row["trace_id"] for row in rows] == ["a", "b"]
    assert set(rows[0]) == {"trace_id", "task_input", "system_state", "dags"}
    assert failures_path.read_text(encoding="utf-8") == ""
    replayed = ToolCallPlanDataset.load(output_path)
    assert [trace.trace_id for trace in replayed] == ["a", "b"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "remove-me" not in json.dumps(manifest)
    assert "token" not in manifest


def test_invalid_item_is_recorded_and_later_item_continues(tmp_path: Path) -> None:
    specs, catalog_path = _specs(tmp_path)
    input_path = tmp_path / "requests.jsonl"
    input_path.write_text('{"id":"a","request":"bad"}\n{"id":"b","request":"good"}\n', encoding="utf-8")
    model = _FakeModel([
        json.dumps({"dags": [_dag("unknown"), _dag("unknown"), _dag("unknown")]}),
        json.dumps({"dags": [_dag(), _dag(), _dag()]}),
    ])
    failures_path = tmp_path / "failures.jsonl"
    summary = run_batch(
        items=(WorkItem("a", "bad"), WorkItem("b", "good")),
        specs=specs,
        system_prompt="prompt",
        model=model,
        output_path=tmp_path / "dags.jsonl",
        failures_path=failures_path,
        manifest_path=tmp_path / "manifest.json",
        input_path=input_path,
        catalog_path=catalog_path,
        model_name="test-model",
        endpoint="https://example.test/v1",
        prompt_version="v1",
        schema_path=SCHEMA,
    )
    assert summary.succeeded == 1
    assert summary.failed == 1
    failure = json.loads(failures_path.read_text(encoding="utf-8"))
    assert failure["id"] == "a"
    assert "unknown_tool" in failure["reason"]
    assert model.calls == 2


def test_model_failure_is_item_scoped_and_does_not_persist_exception_body(tmp_path: Path) -> None:
    specs, catalog_path = _specs(tmp_path)
    input_path = tmp_path / "requests.jsonl"
    input_path.write_text('{"id":1,"request":"first"}\n{"id":2,"request":"second"}\n', encoding="utf-8")

    class _TransportFailureModel(_FakeModel):
        def complete(self, **kwargs: object) -> str:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("HTTP 500 response contained secret-body")
            return json.dumps({"dags": [_dag(), _dag(), _dag()]})

    model = _TransportFailureModel([])
    failures_path = tmp_path / "failures.jsonl"
    summary = run_batch(
        items=(WorkItem("1", "first"), WorkItem("2", "second")),
        specs=specs,
        system_prompt="prompt",
        model=model,
        output_path=tmp_path / "dags.jsonl",
        failures_path=failures_path,
        manifest_path=tmp_path / "manifest.json",
        input_path=input_path,
        catalog_path=catalog_path,
        model_name="test-model",
        endpoint="https://example.test/v1",
        prompt_version="v1",
        schema_path=SCHEMA,
    )
    assert summary.succeeded == 1
    assert summary.failed == 1
    failure = json.loads(failures_path.read_text(encoding="utf-8"))
    assert failure == {"id": "1", "stage": "model", "reason": "model request failed: RuntimeError"}
    assert "secret-body" not in failures_path.read_text(encoding="utf-8")


def test_explicit_media_references_become_typed_inputs_without_media_access(tmp_path: Path) -> None:
    catalog_path = tmp_path / "media-catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "tools": [
                    {
                        "tool_id": "caption",
                        "description": "Describe an image.",
                        "inputs": {"image": {"modality": "image"}},
                        "outputs": {"text": {"modality": "text"}},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    specs = load_tool_catalog(catalog_path)
    request = "Describe https://cdn.example/photo.png and /tmp/voice.wav."
    dag = {
        "nodes": [
            {
                "node_id": "caption-0",
                "tool_id": "caption",
                "inputs": {"image": {"kind": "request", "name": "image_0"}},
            }
        ],
        "final_outputs": [{"node_id": "caption-0", "port": "text"}],
    }
    input_path = tmp_path / "requests.jsonl"
    input_path.write_text(json.dumps({"id": "media", "request": request}) + "\n", encoding="utf-8")
    output_path = tmp_path / "dags.jsonl"
    run_batch(
        items=(WorkItem("media", request),),
        specs=specs,
        system_prompt="prompt",
        model=_FakeModel([json.dumps({"dags": [dag, dag, dag]})]),
        output_path=output_path,
        failures_path=tmp_path / "failures.jsonl",
        manifest_path=tmp_path / "manifest.json",
        input_path=input_path,
        catalog_path=catalog_path,
        model_name="test-model",
        endpoint="https://example.test/v1",
        prompt_version="v1",
        schema_path=SCHEMA,
    )
    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert row["task_input"]["image_0"] == "https://cdn.example/photo.png"
    assert row["task_input"]["audio_0"] == "/tmp/voice.wav"
    assert row["dags"][0]["nodes"][0]["inputs"]["image"]["data_type"] == "image"


def test_numeric_ids_are_normalized_and_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "requests.jsonl"
    path.write_text('{"id":7,"request":"one"}\n{"id":8.5,"request":"two"}\n', encoding="utf-8")
    from eec_sched.offline_tool_planner import load_work_items

    assert [item.item_id for item in load_work_items(path)] == ["7", "8.5"]
    path.write_text('{"id":7,"request":"one"}\n{"id":"7","request":"two"}\n', encoding="utf-8")
    with pytest.raises(PlannerBatchError, match="unique"):
        load_work_items(path)


def test_batch_refuses_to_overwrite_existing_artifacts(tmp_path: Path) -> None:
    specs, catalog_path = _specs(tmp_path)
    input_path = tmp_path / "requests.jsonl"
    input_path.write_text('{"id":"a","request":"hello"}\n', encoding="utf-8")
    output_path = tmp_path / "dags.jsonl"
    output_path.write_text("old\n", encoding="utf-8")
    with pytest.raises(PlannerBatchError, match="overwrite"):
        run_batch(
            items=(WorkItem("a", "hello"),),
            specs=specs,
            system_prompt="prompt",
            model=_FakeModel([json.dumps({"dags": [_dag(), _dag(), _dag()]})]),
            output_path=output_path,
            failures_path=tmp_path / "failures.jsonl",
            manifest_path=tmp_path / "manifest.json",
            input_path=input_path,
            catalog_path=catalog_path,
            model_name="test-model",
            endpoint="https://example.test/v1",
            prompt_version="v1",
            schema_path=SCHEMA,
        )

