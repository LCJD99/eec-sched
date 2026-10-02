import pytest

from eec_sched.diagnosis.tool_evolution import (
    CompositeToolSpec,
    DiagnosisToolStore,
    execute_tool_spec,
    render_tool_source,
    substitute_placeholders,
)


def _spec(name: str = "check_assignment") -> CompositeToolSpec:
    return CompositeToolSpec.from_dict(
        {
            "name": name,
            "description": "profile then validate a candidate assignment",
            "parameters": ["trace_id", "node_id", "device"],
            "steps": [
                {
                    "tool": "profile_trace",
                    "arguments": {"trace_id": "$trace_id"},
                },
                {
                    "tool": "intervene_assignment",
                    "arguments": {
                        "trace_id": "$trace_id",
                        "node_id": "$node_id",
                        "device": "$device",
                    },
                },
                {
                    "tool": "validate_intervention",
                    "arguments": {"trace_ids": ["$trace_id", "other"]},
                },
            ],
        }
    )


def test_substitute_placeholders_preserves_exact_values_and_interpolates_strings() -> (
    None
):
    value = {
        "trace": "$trace_ids",
        "label": "node=$node_id",
        "nested": ["$count", {"device": "$device"}],
    }

    assert substitute_placeholders(
        value,
        {"trace_ids": ["one", "two"], "node_id": "n1", "count": 3, "device": "edge"},
    ) == {
        "trace": ["one", "two"],
        "label": "node=n1",
        "nested": [3, {"device": "edge"}],
    }


def test_execute_continues_after_meta_tool_error() -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def dispatch(tool: str, arguments: dict[str, object]) -> object:
        calls.append((tool, arguments))
        if tool == "intervene_assignment":
            raise RuntimeError("intervention failed")
        return {"ok": tool}

    result = execute_tool_spec(
        _spec(),
        {"trace_id": "t1", "node_id": "n1", "device": "edge"},
        dispatch,
    )

    assert [tool for tool, _ in calls] == [
        "profile_trace",
        "intervene_assignment",
        "validate_intervention",
    ]
    assert result[1]["result"] == "RuntimeError: intervention failed"
    assert result[2]["result"] == {"ok": "validate_intervention"}


def test_store_persists_history_and_reads_recent_five(tmp_path) -> None:
    store = DiagnosisToolStore(tmp_path)
    for index in range(7):
        store.append_history(
            tool_calls=[
                {"tool": "profile_trace", "arguments": {"i": index}, "result": "ok"}
            ],
            diagnosis={"bottlenecks": [f"node-{index}"], "confidence": 0.5},
            snapshot_digest="snapshot-a",
        )

    recent = store.read_recent_history()

    assert len(recent) == 5
    assert recent[0]["diagnosis"]["bottlenecks"] == ["node-2"]
    assert recent[-1]["tool_calls"][0]["result_summary"] == "ok"
    assert len((tmp_path / "history.jsonl").read_text().splitlines()) == 7


def test_store_persists_specs_source_and_loads_recent_five(tmp_path) -> None:
    store = DiagnosisToolStore(tmp_path)
    for index in range(7):
        store.save_tool_spec(_spec(f"tool_{index}"))

    loaded = store.load_recent_tool_specs()

    assert [spec.name for spec in loaded] == [f"tool_{index}" for index in range(2, 7)]
    generated = list((tmp_path / "generated_tools").glob("*.py"))
    assert len(generated) == 7
    source = generated[-1].read_text()
    compile(source, str(generated[-1]), "exec")
    assert "execute_tool_spec" in source


def test_rendered_wrapper_executes_with_injected_dispatcher() -> None:
    spec = CompositeToolSpec.from_dict(
        {
            "name": "check_flags",
            "description": "values that exercise Python literal rendering",
            "parameters": [],
            "steps": [
                {
                    "tool": "profile_trace",
                    "arguments": {"enabled": True, "optional": None},
                }
            ],
        }
    )
    namespace: dict[str, object] = {}
    exec(compile(render_tool_source(spec), "generated.py", "exec"), namespace)
    assert namespace["TOOL_SPEC"]["steps"][0]["arguments"] == {  # type: ignore[index]
        "enabled": True,
        "optional": None,
    }

    namespace = {}
    exec(compile(render_tool_source(_spec()), "generated.py", "exec"), namespace)
    function = namespace["check_assignment"]
    assert callable(function)
    result = function(  # type: ignore[operator]
        lambda tool, arguments: {"tool": tool, "arguments": arguments},
        trace_id="t1",
        node_id="n1",
        device="edge",
    )
    assert result[0]["arguments"] == {"trace_id": "t1"}


def test_specs_reject_unknown_meta_tools_and_placeholders() -> None:
    with pytest.raises(ValueError, match="unknown meta-tool"):
        CompositeToolSpec.from_dict(
            {
                "name": "bad",
                "parameters": [],
                "steps": [{"tool": "shell", "arguments": {}}],
            }
        )
    with pytest.raises(ValueError, match="unknown parameter"):
        CompositeToolSpec.from_dict(
            {
                "name": "bad",
                "parameters": [],
                "steps": [{"tool": "profile_trace", "arguments": {"x": "$missing"}}],
            }
        )
    with pytest.raises(ValueError, match="must not shadow"):
        CompositeToolSpec.from_dict(
            {
                "name": "profile_trace",
                "parameters": [],
                "steps": [{"tool": "profile_trace", "arguments": {}}],
            }
        )
