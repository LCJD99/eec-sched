import json
from pathlib import Path

from eec_sched.diagnosis.models import DiagnosisResult
from eec_sched.diagnosis.tools import BottleneckMetaTools
from eec_sched.profiling.snapshot import load_profiling_database


ROOT = Path(__file__).parents[1]
SNAPSHOT = load_profiling_database(
    ROOT / "docs/examples/profiling-database.fake.json",
    ROOT / "docs/schemas/profiling-database.schema.json",
)


def _trace(trace_id: str, node_id: str = "node") -> dict[str, object]:
    return {
        "trace_id": trace_id,
        "dag": {
            "nodes": [
                {
                    "node_id": node_id,
                    "tool_id": "text_generation",
                    "inputs": {"text": {"kind": "request", "name": "request_text"}},
                }
            ],
            "final_outputs": [{"node_id": node_id, "port": "text"}],
        },
        "assignments": {
            node_id: {"configuration_id": "synthetic-reference", "device_id": "edge"}
        },
        "nodes": [
            {
                "node_id": node_id,
                "configuration_id": "synthetic-reference",
                "device_id": "edge",
                "start_ms": 2.5,
                "finish_ms": 12.0,
            }
        ],
        "transfers": [],
        "metrics": {"latency": 12.0, "composite_score": 0.5},
    }


def _serialized_chain_trace() -> dict[str, object]:
    request_finish = 4.00512
    first_finish = 17.00512
    second_finish = 30.00512
    final_finish = 34.50512
    return {
        "trace_id": "chain",
        "dag": {
            "nodes": [
                {
                    "node_id": "node-a",
                    "tool_id": "text_generation",
                    "inputs": {"text": {"kind": "request", "name": "request_text"}},
                },
                {
                    "node_id": "node-b",
                    "tool_id": "text_generation",
                    "inputs": {"text": {"kind": "node", "name": "node-a", "port": "text"}},
                },
            ],
            "final_outputs": [{"node_id": "node-b", "port": "text"}],
        },
        "assignments": {
            "node-a": {"configuration_id": "synthetic-reference", "device_id": "edge"},
            "node-b": {"configuration_id": "synthetic-reference", "device_id": "edge"},
        },
        "nodes": [
            {"node_id": "node-a", "device_id": "edge", "start_ms": request_finish, "finish_ms": first_finish},
            {"node_id": "node-b", "device_id": "edge", "start_ms": first_finish, "finish_ms": second_finish},
        ],
        "transfers": [
            {
                "source_node_id": "device",
                "destination_node_id": "node-a",
                "source_device_id": "device",
                "destination_device_id": "edge",
                "start_ms": 0.0,
                "finish_ms": request_finish,
                "latency_ms": request_finish,
            },
            {
                "source_node_id": "node-b",
                "destination_node_id": "device",
                "source_device_id": "edge",
                "destination_device_id": "device",
                "start_ms": second_finish,
                "finish_ms": final_finish,
                "latency_ms": 4.5,
            },
        ],
        "metrics": {"latency": final_finish, "composite_score": 0.5},
    }


def test_profile_trace_joins_tool_and_computes_latency() -> None:
    tools = BottleneckMetaTools({"traces": [_trace("one")]}, SNAPSHOT)

    result = json.loads(tools.profile_trace("one"))

    assert result["trace_id"] == "one"
    assert result["focus"] is None
    assert result["critical_path"]["nodes"] == ["node"]
    assert result["ranked_hotspots"][0]["latency_ms"] == 9.5
    assert result["placement_findings"][0]["current"]["device_id"] == "edge"
    assert result["configuration_alternatives"]
    assert "nodes" not in result
    focused = json.loads(tools.profile_trace("one", focus="communication", top_k=1))
    assert focused["focus"] == "communication"
    assert len(focused["ranked_hotspots"]) <= 1
    # Deliberately bypass the static Literal contract to exercise the JSON-facing
    # runtime validation used by tool callers.
    invalid = json.loads(tools.profile_trace("one", focus="unknown"))  # type: ignore[arg-type]
    assert invalid["allowed_focus"] == [
        "latency",
        "resource",
        "quality",
        "communication",
        "placement",
    ]
    invalid_top_k = json.loads(tools.profile_trace("one", top_k=0))
    assert "top_k" in invalid_top_k["error"]
    assert "error" in json.loads(tools.profile_trace("missing"))


def test_intervention_uses_complete_baseline_and_rejects_bad_nodes() -> None:
    tools = BottleneckMetaTools({"traces": [_trace("one")]}, SNAPSHOT)
    output = json.loads(
        tools.intervene_assignment(
            "one",
            {"node": {"configuration_id": "synthetic-reference", "device_id": "cloud"}},
        )
    )

    assert output["baseline"]["status"] == "scheduled"
    assert output["new"]["status"] == "scheduled"
    assert "latency" in output["delta"]
    assert "error" in json.loads(tools.intervene_assignment("one", {"unknown": {}}))


def test_critical_timeline_includes_request_final_and_device_queue() -> None:
    tools = BottleneckMetaTools({"traces": [_serialized_chain_trace()]}, SNAPSHOT)

    profile = json.loads(tools.profile_trace("chain", top_k=10))
    critical = profile["critical_path"]
    assert critical["approximation"] is False
    assert critical["nodes"] == ["node-a", "node-b"]
    activity_ids = [item["activity_id"] for item in critical["activities"]]
    assert any(item.startswith("transfer:0:device->node-a") for item in activity_ids)
    assert "node:node-a" in activity_ids
    assert "node:node-b" in activity_ids
    assert any(item.startswith("transfer:1:node-b->device") for item in activity_ids)
    assert critical["total_makespan_ms"] == 34.50512


def test_trace_profile_schema_survives_output_budget() -> None:
    tools = BottleneckMetaTools(
        {"traces": [_serialized_chain_trace()]}, SNAPSHOT, max_output_chars=80
    )

    profile = json.loads(tools.profile_trace("chain", top_k=100))
    assert {
        "trace_id",
        "focus",
        "metrics",
        "critical_path",
        "ranked_hotspots",
        "placement_findings",
        "configuration_alternatives",
        "bottleneck_hypotheses",
        "evidence_refs",
        "omitted",
    } <= set(profile)
    assert profile["omitted"].get("render_budget")


def test_configuration_alternatives_have_deterministic_node_coverage() -> None:
    tools = BottleneckMetaTools({"traces": [_serialized_chain_trace()]}, SNAPSHOT)

    profile = json.loads(tools.profile_trace("chain", top_k=2))
    assert {row["node_id"] for row in profile["configuration_alternatives"]} == {
        "node-a",
        "node-b",
    }
    assert profile["omitted"]["configuration_alternatives"] > 0


def test_validation_runs_explicit_patches_on_three_dags_and_summarizes() -> None:
    evidence = {"traces": [_trace("one"), _trace("two"), _trace("three")]}
    tools = BottleneckMetaTools(evidence, SNAPSHOT)
    patches = {
        trace_id: {
            "node": {"configuration_id": "synthetic-reference", "device_id": "cloud"}
        }
        for trace_id in ("one", "two", "three")
    }

    output = json.loads(tools.validate_intervention(patches))

    assert len(output["results"]) == 3
    assert output["summary"]["evaluated_count"] == 3
    assert "latency" in output["summary"]["delta"]


def test_impact_path_is_backward_compatible() -> None:
    assert DiagnosisResult("advice").impact_path == ()
