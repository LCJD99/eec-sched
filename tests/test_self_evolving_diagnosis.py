from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from types import SimpleNamespace

import pytest

from eec_sched.diagnosis.agents import SelfEvolvingDiagnosis
from eec_sched.diagnosis.agents import _parse_diagnosis
from eec_sched.diagnosis.tool_evolution import CompositeToolSpec, DiagnosisToolStore


class FakeModel:
    def __init__(self, *outputs: str) -> None:
        self.outputs = iter(outputs)
        self.calls: list[dict[str, object]] = []

    def complete(self, **kwargs: object) -> str:
        self.calls.append(kwargs)
        return next(self.outputs)


def _event_recorder(
    events: list[dict[str, object]],
) -> Callable[[Mapping[str, object]], None]:
    def record(event: Mapping[str, object]) -> None:
        events.append(dict(event))

    return record


def test_self_evolving_diagnosis_records_result_and_persists_composite(
    tmp_path,
) -> None:
    model = FakeModel(
        json.dumps(
            {
                "bottlenecks": ["node-a"],
                "evidence": ["intervention reduced latency"],
                "impact_path": ["node-a queue", "makespan rises"],
                "advice": "move node-a to the faster device",
                "confidence": 0.7,
            }
        ),
        json.dumps(
            {
                "tool": {
                    "name": "inspect_path",
                    "description": "profile a trace",
                    "parameters": ["trace_id"],
                    "steps": [
                        {
                            "tool": "profile_trace",
                            "arguments": {"trace_id": "$trace_id"},
                        }
                    ],
                }
            }
        ),
    )
    store = DiagnosisToolStore(tmp_path)
    calls: list[tuple[str, dict[str, object]]] = []

    def provider(_evidence):
        return {
            "profile_trace": lambda **kwargs: (
                calls.append(("profile_trace", kwargs)) or "profiled"
            ),
            "intervene_assignment": lambda **kwargs: "intervened",
            "validate_intervention": lambda **kwargs: "validated",
        }

    agent = SelfEvolvingDiagnosis(
        model,
        tool_store=store,
        tool_provider=provider,
    )
    result = agent.diagnose({"snapshot_digest": "snapshot-a", "traces": []})

    assert result.bottlenecks == ("node-a",)
    assert result.impact_path == ("node-a queue", "makespan rises")
    assert store.read_recent_history()[0]["snapshot_digest"] == "snapshot-a"
    assert store.load_recent_tool_specs()[0].name == "inspect_path"


def test_fake_model_calls_emit_session_and_round_lifecycle(tmp_path) -> None:
    events: list[dict[str, object]] = []
    agent = SelfEvolvingDiagnosis(
        FakeModel(
            json.dumps(
                {
                    "bottlenecks": ["node-a"],
                    "evidence": ["e"],
                    "impact_path": ["node-a", "makespan"],
                    "advice": "move node-a",
                    "confidence": 0.4,
                }
            ),
            '{"tool": null}',
        ),
        tool_store=DiagnosisToolStore(tmp_path),
        event_recorder=_event_recorder(events),
    )

    agent.diagnose({"snapshot_digest": "snapshot-a", "traces": []})
    diagnosis_rounds = [
        event
        for event in events
        if event["type"] == "round_start" and event["phase"] == "diagnosis"
    ]
    assert len(diagnosis_rounds) == 1
    assert [event["round_number"] for event in diagnosis_rounds] == [1]
    assert events[0]["type"] == "session_start"
    assert events[-1]["type"] == "session_end"
    assert events[-1]["status"] == "completed"
    assert [event["type"] for event in events].count("round_end") == 2
    requests = [event for event in events if event["type"] == "llm_request"]
    responses = [event for event in events if event["type"] == "llm_response"]
    assert len(requests) == len(responses) == 2
    assert [event["round_number"] for event in requests] == [1, 2]
    assert requests[0]["phase"] == "diagnosis"
    request_input = requests[0]["input"]
    assert isinstance(request_input, dict)
    assert "evidence" in request_input
    assert responses[0]["status"] == "completed"
    assert "move node-a" in str(responses[0]["output"])


def test_meta_and_evolved_tool_events_share_call_ids(tmp_path) -> None:
    events: list[dict[str, object]] = []
    agent = SelfEvolvingDiagnosis(
        object(),
        tool_store=DiagnosisToolStore(tmp_path),
        tool_provider={
            "profile_trace": lambda **_: "profiled",
            "intervene_assignment": lambda **_: "intervened",
            "validate_intervention": lambda **_: "validated",
        },
        event_recorder=_event_recorder(events),
    )
    agent._active_provider = agent.tool_provider
    agent._invoke_meta("profile_trace", trace_id="trace-1")
    spec = CompositeToolSpec.from_dict(
        {
            "name": "inspect_saved_path",
            "parameters": ["trace_id"],
            "steps": [
                {"tool": "profile_trace", "arguments": {"trace_id": "$trace_id"}}
            ],
        }
    )
    agent._make_composite_callable(spec)(trace_id="trace-1")

    calls = {
        event["call_id"]: event
        for event in events
        if event["type"] == "tool_call"
    }
    results = {
        event["call_id"]: event
        for event in events
        if event["type"] == "tool_result"
    }
    evaluations = {
        event["call_id"]: event
        for event in events
        if event["type"] == "tool_evaluation"
    }
    assert set(calls) == set(results) == set(evaluations)
    assert {event["tool_kind"] for event in calls.values()} == {"meta", "evolved"}
    assert all(event["status"] == "success" for event in results.values())

    failing_events: list[dict[str, object]] = []
    failing = SelfEvolvingDiagnosis(
        object(),
        tool_store=DiagnosisToolStore(tmp_path / "failed"),
        tool_provider={"profile_trace": lambda **_: (_ for _ in ()).throw(RuntimeError("boom"))},
        event_recorder=_event_recorder(failing_events),
    )
    failing._active_provider = failing.tool_provider
    failing._invoke_meta("profile_trace", trace_id="trace-1")
    failed_call = next(event for event in failing_events if event["type"] == "tool_call")
    failed_result = next(event for event in failing_events if event["type"] == "tool_result")
    failed_evaluation = next(
        event for event in failing_events if event["type"] == "tool_evaluation"
    )
    assert failed_call["call_id"] == failed_result["call_id"] == failed_evaluation["call_id"]
    assert failed_result["status"] == failed_evaluation["status"] == "failed"


def test_generated_specs_bind_individual_specs_and_report_tool_errors(tmp_path) -> None:
    store = DiagnosisToolStore(tmp_path)
    first = CompositeToolSpec.from_dict(
        {
            "name": "first_tool",
            "description": "first",
            "parameters": ["trace_id"],
            "steps": [
                {"tool": "profile_trace", "arguments": {"trace_id": "$trace_id"}}
            ],
        }
    )
    second = CompositeToolSpec.from_dict(
        {
            "name": "second_tool",
            "description": "second",
            "parameters": ["trace_id"],
            "steps": [
                {"tool": "intervene_assignment", "arguments": {"trace_id": "$trace_id"}}
            ],
        }
    )
    store.save_tool_spec(first)
    store.save_tool_spec(second)
    agent = SelfEvolvingDiagnosis(object(), tool_store=store)
    first_call = agent._make_composite_callable(first)
    second_call = agent._make_composite_callable(second)
    seen: list[str] = []
    agent._active_provider = {
        "profile_trace": lambda **_: seen.append("profile") or "ok",
        "intervene_assignment": lambda **_: seen.append("intervene") or "ok",
        "validate_intervention": lambda **_: "ok",
    }

    first_call(trace_id="one")
    second_call(trace_id="two")

    assert seen == ["profile", "intervene"]


def test_diagnosis_bounds_evidence_to_three_traces(tmp_path) -> None:
    model = FakeModel(
        json.dumps(
            {
                "bottlenecks": ["node-a"],
                "evidence": ["observed delay"],
                "impact_path": ["node-a", "makespan"],
                "advice": "move node-a",
                "confidence": 0.4,
            }
        ),
        '{"tool": null}',
    )
    provided: list[dict[str, object]] = []

    def provider(evidence):
        provided.append(evidence)
        return {}

    agent = SelfEvolvingDiagnosis(
        model,
        tool_store=DiagnosisToolStore(tmp_path),
        tool_provider=provider,
        trace_limit=3,
    )
    agent.diagnose(
        {
            "snapshot_digest": "snapshot-a",
            "traces": [{"trace_id": f"trace-{index}"} for index in range(5)],
        }
    )

    provided_traces = provided[0].get("traces")
    assert isinstance(provided_traces, list)
    assert len(provided_traces) == 5
    user = model.calls[0].get("user")
    assert isinstance(user, dict)
    evidence = user.get("evidence")
    assert isinstance(evidence, dict)
    model_traces = evidence.get("trace_index")
    assert isinstance(model_traces, list)
    assert len(model_traces) == 3
    assert "task_input" not in str(user)
    assert "assignments" not in str(user)
    assert "nodes" not in str(user)
    assert "transfers" not in str(user)
    assert "raw_metrics" not in str(user)


def test_diagnosis_initial_context_includes_scheduler_source(tmp_path) -> None:
    source_code = "def propose(view):\n    return {}\n"
    model = FakeModel(
        json.dumps(
            {
                "bottlenecks": ["node-a"],
                "evidence": ["observed delay"],
                "impact_path": ["node-a", "makespan"],
                "advice": "move node-a",
                "confidence": 0.4,
            }
        ),
        '{"tool": null}',
    )
    agent = SelfEvolvingDiagnosis(
        model,
        tool_store=DiagnosisToolStore(tmp_path),
    )

    agent.diagnose(
        {
            "scheduler_version": 7,
            "strategy_description": "always choose the fastest compatible device",
            "source_code": source_code,
            "snapshot_digest": "snapshot-a",
            "traces": [{"trace_id": "trace-a"}],
        }
    )

    user = model.calls[0]["user"]
    assert isinstance(user, dict)
    evidence = user["evidence"]
    assert isinstance(evidence, dict)
    assert evidence["source_code"] == source_code
    assert evidence["strategy_description"] == (
        "always choose the fastest compatible device"
    )
    assert evidence["scheduler_version"] == 7


def test_agents_sdk_builds_meta_and_generated_tools(tmp_path) -> None:
    pytest.importorskip("agents")
    store = DiagnosisToolStore(tmp_path)
    store.save_tool_spec(
        CompositeToolSpec.from_dict(
            {
                "name": "inspect_saved_path",
                "description": "inspect one saved path",
                "parameters": ["trace_id"],
                "steps": [
                    {"tool": "profile_trace", "arguments": {"trace_id": "$trace_id"}}
                ],
            }
        )
    )

    class FakeRunner:
        @staticmethod
        def run_sync(agent, prompt, *, hooks=None):
            assert len(agent.tools) == 4
            assert "inspect_saved_path" in {tool.name for tool in agent.tools}
            profile_tool = next(
                tool for tool in agent.tools if tool.name == "profile_trace"
            )
            focus_schema = profile_tool.params_json_schema["properties"]["focus"]
            assert {
                "latency",
                "resource",
                "quality",
                "communication",
                "placement",
            } == set(focus_schema["anyOf"][0]["enum"])
            assert "section" not in profile_tool.params_json_schema["properties"]
            assert hooks is not None
            for number in range(2):
                asyncio.run(
                    hooks.on_llm_start(
                        None,
                        agent,
                        "sdk system prompt",
                        [{"role": "user", "content": f"sdk input {number}"}],
                    )
                )
                asyncio.run(
                    hooks.on_llm_end(
                        None,
                        agent,
                        {"output": [{"role": "assistant", "content": number}]},
                    )
                )
            return SimpleNamespace(
                final_output=json.dumps(
                    {
                        "bottlenecks": ["node-a"],
                        "evidence": ["observed delay"],
                        "impact_path": ["node-a", "makespan"],
                        "advice": "move node-a",
                        "confidence": 0.4,
                    }
                )
            )

    events: list[dict[str, object]] = []
    agent = SelfEvolvingDiagnosis(
        FakeModel('{"tool": null}'),
        tool_provider={
            "profile_trace": lambda **_: "profiled",
            "intervene_assignment": lambda **_: "intervened",
            "validate_intervention": lambda **_: "validated",
        },
        tool_store=store,
        runner=FakeRunner,
        event_recorder=_event_recorder(events),
    )

    assert (
        agent.diagnose({"snapshot_digest": "snapshot-a", "traces": []}).confidence
        == 0.4
    )
    diagnosis_rounds = [
        event
        for event in events
        if event["type"] == "round_start" and event["phase"] == "diagnosis"
    ]
    assert [event["round_number"] for event in diagnosis_rounds] == [1, 2]
    sdk_requests = [
        event
        for event in events
        if event["type"] == "llm_request" and event["phase"] == "diagnosis"
    ]
    sdk_responses = [
        event
        for event in events
        if event["type"] == "llm_response" and event["phase"] == "diagnosis"
    ]
    assert len(sdk_requests) == len(sdk_responses) == 2
    assert sdk_requests[0]["system_prompt"] == "sdk system prompt"
    assert "sdk input 0" in str(sdk_requests[0]["input"])
    assert "assistant" in str(sdk_responses[0]["output"])


@pytest.mark.parametrize(
    "payload, message",
    [
        ("not-json", "invalid JSON"),
        (
            {"advice": "x", "evidence": ["e"], "impact_path": ["p"], "confidence": 0.1},
            "identify a bottleneck",
        ),
        (
            {
                "advice": "x",
                "bottlenecks": ["b"],
                "impact_path": ["p"],
                "confidence": 0.1,
            },
            "include evidence",
        ),
        (
            {"advice": "x", "bottlenecks": ["b"], "evidence": ["e"], "confidence": 0.1},
            "impact_path",
        ),
        (
            {
                "advice": "x",
                "bottlenecks": ["b"],
                "evidence": ["e"],
                "impact_path": ["p"],
                "confidence": 2,
            },
            "between 0 and 1",
        ),
    ],
)
def test_diagnosis_output_is_strictly_structured(payload, message) -> None:
    with pytest.raises(ValueError, match=message):
        _parse_diagnosis(payload)
