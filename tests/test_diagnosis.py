from eec_sched.diagnosis import EmptyDiagnosis, OneShotDiagnosis, TraceAnalysisTools


EVIDENCE = {
    "traces": [
        {
            "trace_id": "trace-1",
            "nodes": [
                {
                    "node_id": "caption",
                    "tool_id": "image_captioning",
                    "device_id": "edge",
                    "configuration_id": "small",
                    "latency_ms": 8.0,
                },
                {
                    "node_id": "answer",
                    "tool_id": "text_generation",
                    "device_id": "cloud",
                    "configuration_id": "large",
                    "latency_ms": 22.0,
                },
            ],
        }
    ]
}


def test_empty_diagnosis_is_a_real_ablation_adapter() -> None:
    result = EmptyDiagnosis().diagnose(EVIDENCE)

    assert result.advice
    assert result.bottlenecks == ()
    assert result.evidence == ()


def test_one_shot_diagnosis_rejects_empty_model_output() -> None:
    result = OneShotDiagnosis(lambda evidence: "move the bottleneck").diagnose(EVIDENCE)

    assert result.advice == "move the bottleneck"


def test_trace_tools_inspect_and_summarize_node_dimensions() -> None:
    tools = TraceAnalysisTools(EVIDENCE)

    assert '"node_id": "answer"' in tools.inspect_node_dimension("latency_ms")
    summary = tools.summarize_node_dimension("latency_ms", "device_id")
    assert '"cloud"' in summary
    assert '"mean": 22.0' in summary
