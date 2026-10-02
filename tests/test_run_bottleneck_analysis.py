from __future__ import annotations

import json
from pathlib import Path

import pytest

from eec_sched.app.run_bottleneck_analysis import (
    BottleneckAnalysisOptions,
    _options_from_args,
    build_parser,
    run_analysis,
)


ROOT = Path(__file__).parents[1]


class FakeModel:
    def __init__(self) -> None:
        self.calls = 0
        self.requests: list[dict[str, object]] = []

    def complete(self, **kwargs: object) -> str:
        self.calls += 1
        self.requests.append(kwargs)
        if self.calls == 1:
            return json.dumps(
                {
                    "bottlenecks": ["node-a"],
                    "evidence": ["observed delay"],
                    "impact_path": ["node-a", "makespan"],
                    "advice": "move node-a",
                    "confidence": 0.5,
                }
            )
        return '{"tool": null}'


def _options(tmp_path: Path, dataset: Path) -> BottleneckAnalysisOptions:
    return BottleneckAnalysisOptions(
        scheduler_source=ROOT / "schedulers/naive_fastest.py",
        dataset=dataset,
        split="all",
        output_dir=tmp_path / "events",
        state_dir=tmp_path / "state",
        model_token="do-not-persist-this-token",
    )


def test_run_analysis_writes_ordered_source_free_events(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        (ROOT / "data/mnms-ground-truth-dags.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
        + "\n",
        encoding="utf-8",
    )
    model = FakeModel()
    result = run_analysis(_options(tmp_path, dataset), model=model)

    assert result.output_path.parent == tmp_path / "events"
    assert result.output_path.name.endswith("Z.jsonl")
    rows = [json.loads(line) for line in result.output_path.read_text().splitlines()]
    assert [row["sequence"] for row in rows] == list(range(1, len(rows) + 1))
    assert all(row["timestamp"].endswith("Z") for row in rows)
    assert all(row["event_type"] == row["type"] for row in rows)
    assert rows[0]["type"] == "run_started"
    assert rows[-1]["type"] == "run_completed"
    types = [row["type"] for row in rows]
    expected_lifecycle = [
        "evaluation_completed",
        "diagnosis_started",
        "diagnosis_agent_started",
        "agent_context_prepared",
        "agent_started",
        "model_started",
        "model_completed",
        "agent_completed",
        "history_recorded",
        "tool_evolution_started",
        "model_started",
        "model_completed",
        "tool_evolution_completed",
        "diagnosis_agent_completed",
        "diagnosis_completed",
        "run_completed",
    ]
    cursor = 0
    for event_type in expected_lifecycle:
        cursor = types.index(event_type, cursor) + 1
    assert types.count("diagnosis_started") == 1
    assert types.count("diagnosis_completed") == 1
    assert "evaluation_trace" in types
    assert all("source_code" not in row for row in rows)
    assert "EEC_SCHED_REFLECTION_TOKEN" not in result.output_path.read_text()
    assert "do-not-persist-this-token" not in result.output_path.read_text()
    context = next(row for row in rows if row["type"] == "agent_context_prepared")
    assert len(context["context"]["evidence"]["trace_index"]) == 1
    assert "task_input" not in json.dumps(context["context"])
    assert "assignments" not in json.dumps(context["context"])
    assert "nodes" not in json.dumps(context["context"])
    assert context["context_chars"] < 4_000
    llm_requests = [row for row in rows if row["event_type"] == "llm_request"]
    llm_responses = [row for row in rows if row["event_type"] == "llm_response"]
    assert len(llm_requests) == len(llm_responses) == 2
    assert llm_requests[0]["phase"] == "diagnosis"
    assert "evidence" in llm_requests[0]["input"]
    assert llm_responses[0]["status"] == "completed"
    assert "move node-a" in llm_responses[0]["output"]
    diagnosis = next(row for row in rows if row["type"] == "run_completed")["result"][
        "diagnosis"
    ]
    assert diagnosis["advice"] == "move node-a"
    first_request = model.requests[0]["user"]
    assert isinstance(first_request, dict)
    first_evidence = first_request["evidence"]
    assert isinstance(first_evidence, dict)
    assert first_evidence["source_code"] == (
        ROOT / "schedulers/naive_fastest.py"
    ).read_text(encoding="utf-8")
    assert first_evidence["strategy_description"] == (
        f"loaded from {ROOT / 'schedulers/naive_fastest.py'}"
    )


def test_run_analysis_evaluates_three_dags_by_default(tmp_path: Path) -> None:
    source_lines = (ROOT / "data/mnms-ground-truth-dags.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("\n".join(source_lines[:4]) + "\n", encoding="utf-8")

    result = run_analysis(_options(tmp_path, dataset), model=FakeModel())
    rows = [json.loads(line) for line in result.output_path.read_text().splitlines()]
    loaded = next(row for row in rows if row["event_type"] == "dataset_loaded")
    started = next(row for row in rows if row["event_type"] == "evaluation_started")

    assert len(result.evaluation.traces) == 3
    assert loaded["selected_trace_count"] == 3
    assert loaded["split_trace_count"] == 4
    assert started["selected_trace_count"] == 3


def test_run_analysis_honors_explicit_dag_limit(tmp_path: Path) -> None:
    source_lines = (ROOT / "data/mnms-ground-truth-dags.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("\n".join(source_lines[:4]) + "\n", encoding="utf-8")

    result = run_analysis(
        BottleneckAnalysisOptions(
            **{
                **_options(tmp_path, dataset).__dict__,
                "dag_limit": 2,
            }
        ),
        model=FakeModel(),
    )
    assert len(result.evaluation.traces) == 2


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_options_reject_non_positive_integer_dag_limit(value) -> None:
    with pytest.raises(ValueError, match="dag_limit"):
        BottleneckAnalysisOptions(Path("scheduler.py"), dag_limit=value)


def test_run_analysis_records_failure_before_propagating(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_analysis(
            BottleneckAnalysisOptions(
                scheduler_source=tmp_path / "missing.py",
                output_dir=tmp_path / "events",
                state_dir=tmp_path / "state",
            ),
            model=FakeModel(),
        )
    path = next((tmp_path / "events").glob("*.jsonl"))
    assert json.loads(path.read_text().splitlines()[-1])["type"] == "run_failed"


def test_cli_rejects_more_than_one_scheduler() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--scheduler-source",
                "one.py",
                "--scheduler-source",
                "two.py",
            ]
        )


def test_cli_loads_model_config_and_allows_cli_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_MODEL_TOKEN", "env-token")
    config = tmp_path / "model.yaml"
    config.write_text(
        """\
_target_: eec_sched.llm.OpenAICompatibleChatModel
base_url: http://model-config.test/v1
token: ${oc.env:TEST_MODEL_TOKEN}
model: configured-model
timeout_seconds: 12
""",
        encoding="utf-8",
    )

    args = build_parser().parse_args(
        [
            "--scheduler-source",
            "scheduler.py",
            "--model-config",
            str(config),
            "--model",
            "cli-model",
        ]
    )
    options = _options_from_args(args)

    assert options.model_config == config
    assert options.model_base_url == "http://model-config.test/v1"
    assert options.model_token == "env-token"
    assert options.model_name == "cli-model"
    assert options.model_timeout_seconds == 12


def test_run_analysis_loads_model_config_for_programmatic_options(
    tmp_path: Path,
) -> None:
    config = tmp_path / "model.yaml"
    config.write_text(
        "base_url: http://model-config.test/v1\ntoken: config-token\nmodel: config-model\n",
        encoding="utf-8",
    )
    options = BottleneckAnalysisOptions(
        scheduler_source=tmp_path / "missing.py",
        model_config=config,
        output_dir=tmp_path / "events",
    )

    with pytest.raises(FileNotFoundError):
        run_analysis(options, model=FakeModel())

    rows = [
        json.loads(line)
        for line in next((tmp_path / "events").glob("*.jsonl")).read_text().splitlines()
    ]
    assert rows[0]["config"]["model"] == {
        "config": str(config),
        "base_url": "http://model-config.test/v1",
        "name": "config-model",
    }


@pytest.mark.parametrize(
    "overrides, message",
    [
        (
            {"accuracy_weight": 0.0, "latency_weight": 0.0, "resource_weight": 0.0},
            "weight",
        ),
        ({"latency_scale_ms": 0.0}, "latency_scale_ms"),
        ({"candidate_timeout_seconds": 0.0}, "candidate_timeout_seconds"),
    ],
)
def test_options_reject_invalid_scoring_and_timeout(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        BottleneckAnalysisOptions(Path("scheduler.py"), **overrides)
