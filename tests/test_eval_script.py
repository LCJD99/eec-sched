from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from eec_sched.workload import ArrivalManifest, ArrivalRecord, save_arrival_manifest


ROOT = Path(__file__).parents[1]


def _write_dataset(
    path: Path,
    *,
    trace_id: str = "template-0",
    tool_id: str = "text_classification",
) -> None:
    plan = {
        "nodes": [
            {
                "node_id": "work",
                "tool_id": tool_id,
                "inputs": {"text": {"kind": "request", "name": "text"}},
            }
        ],
        "final_outputs": [{"node_id": "work", "port": "text"}],
    }
    path.write_text(
        json.dumps(
            {
                "trace_id": trace_id,
                "task_input": {"text": "hello"},
                "system_state": {},
                "dags": [plan, plan, plan],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_scheduler(path: Path) -> None:
    path.write_text(
        "def propose(view):\n"
        "    return {node.node_id: {'configuration_id': 'quality', 'device_id': 'cloud'} for node in view.dag.nodes}\n",
        encoding="utf-8",
    )


def _write_path_selecting_scheduler(path: Path) -> None:
    path.write_text(
        "def propose(view):\n"
        "    dag = view.candidate_dags[1]\n"
        "    return {\n"
        "        'dag': dag,\n"
        "        'path_index': 1,\n"
        "        'assignments': {\n"
        "            node.node_id: {\n"
        "                'configuration_id': ('synthetic-reference' if node.tool_id == 'text_generation' else 'quality'),\n"
        "                'device_id': 'cloud',\n"
        "            }\n"
        "            for node in dag.nodes\n"
        "        },\n"
        "    }\n",
        encoding="utf-8",
    )


def _run_eval(
    dataset: Path,
    scheduler: Path,
    output_root: Path,
    *,
    arrivals: Path | None = None,
) -> tuple[Path, dict[str, object]]:
    overrides = [
        "mode=multi" if arrivals is not None else "mode=single",
        f"dataset={dataset}",
        f"scheduler_source={scheduler}",
        "scheduler_version=7",
        f"output_root={output_root}",
    ]
    if arrivals is not None:
        overrides.append(f"arrivals={arrivals}")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/eval.py"), *overrides],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    run_dir = Path(
        next(line for line in result.stdout.splitlines() if line.startswith("run_dir: "))
        .split(": ", 1)[1]
    )
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    return run_dir, summary


def _run_main_eval(
    video_dataset: Path,
    math_dataset: Path,
    scheduler: Path,
    arrivals: Path,
    output_root: Path,
) -> tuple[Path, dict[str, object]]:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/eval.py"),
            "--config-name",
            "main_exp",
            "method=Ours",
            "evolution_llm_family=F1",
            "planner_llm=planner-model-v1",
            "workload=Mixed",
            f"video_qa_dataset={video_dataset}",
            f"math_qa_dataset={math_dataset}",
            f"scheduler_source={scheduler}",
            "scheduler_version=7",
            f"arrivals={arrivals}",
            f"profiling_database={ROOT / 'docs/examples/profiling-database.fake.json'}",
            f"profiling_schema={ROOT / 'docs/schemas/profiling-database.schema.json'}",
            f"output_root={output_root}",
            "device_price_per_second={device:0.0,edge:0.0,cloud:2.0}",
            "currency=USD",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    run_dir = Path(
        next(line for line in result.stdout.splitlines() if line.startswith("run_dir: "))
        .split(": ", 1)[1]
    )
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    return run_dir, summary


def test_eval_help_loads_the_builtin_hydra_config() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/eval.py"), "--config-name", "config", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )

    assert result.returncode == 0, result.stderr
    assert "mode: single" in result.stdout


def test_single_eval_uses_three_distinct_planner_candidates(tmp_path: Path) -> None:
    scheduler = tmp_path / "scheduler.py"
    output_root = tmp_path / "runs"
    _write_path_selecting_scheduler(scheduler)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/eval.py"),
            "--config-name",
            "config",
            f"scheduler_source={scheduler}",
            "scheduler_version=8",
            f"output_root={output_root}",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    run_dir = Path(
        next(line for line in result.stdout.splitlines() if line.startswith("run_dir: "))
        .split(": ", 1)[1]
    )
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    row = json.loads((run_dir / "results.jsonl").read_text(encoding="utf-8"))
    candidates = row["candidate_dags"]

    assert summary["mode"] == "single"
    assert summary["trace_count"] == 1
    assert row["status"] == "scored"
    assert len(candidates) == 3
    assert len({json.dumps(dag, sort_keys=True) for dag in candidates}) == 3
    assert row["selected_path_index"] == 1
    assert row["dag"] == candidates[1]
    resolved = (run_dir / "config.yaml").read_text(encoding="utf-8")
    assert "experiments/09_offline_tool_planning/runs/tool-planner-20261002T085931Z/dags.jsonl" in resolved


def test_multi_eval_cli_writes_request_timelines_and_replays_manifest(tmp_path: Path) -> None:
    dataset = tmp_path / "dags.jsonl"
    scheduler = tmp_path / "scheduler.py"
    arrivals = tmp_path / "arrivals.json"
    _write_dataset(dataset)
    _write_scheduler(scheduler)
    save_arrival_manifest(
        arrivals,
        ArrivalManifest(
            arrivals=(
                ArrivalRecord(0.0, "template-0__request-000000", "template-0"),
                ArrivalRecord(1.0, "template-0__request-000001", "template-0"),
            ),
            request_rate_per_second=20.0,
            observation_window_ms=100.0,
            seed=19,
        ),
    )

    first_dir, first_summary = _run_eval(dataset, scheduler, tmp_path / "run-a", arrivals=arrivals)
    second_dir, second_summary = _run_eval(dataset, scheduler, tmp_path / "run-b", arrivals=arrivals)
    rows = [json.loads(line) for line in (first_dir / "results.jsonl").read_text().splitlines()]
    replay_rows = [json.loads(line) for line in (second_dir / "results.jsonl").read_text().splitlines()]

    assert first_summary["mode"] == "multi"
    assert first_summary["scheduler_version"] == 7
    assert first_summary["arrived_count"] == 2
    assert first_summary["completed_total"] == 2
    assert first_summary["rejected_count"] == 0
    assert first_summary["failed_count"] == 0
    assert first_summary["throughput_per_second"] == first_summary["completed_within_window"] / 0.1
    assert isinstance(first_summary["latency_p50_ms"], float)
    assert isinstance(first_summary["latency_p95_ms"], float)
    assert first_summary["selected_path_counts"] == {"0": 2, "1": 0, "2": 0}
    deterministic_summary_keys = set(first_summary) - {
        "latency_mean_ms",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_p99_ms",
    }
    assert {key: first_summary[key] for key in deterministic_summary_keys} == {
        key: second_summary[key] for key in deterministic_summary_keys
    }
    assert first_summary["latency_p50_ms"] <= first_summary["latency_p95_ms"]
    assert second_summary["latency_p50_ms"] <= second_summary["latency_p95_ms"]

    def stable_row(row: dict[str, object]) -> dict[str, object]:
        stable = {
            key: value
            for key, value in row.items()
            if key
            not in {
                "completion_time_ms",
                "latency_ms",
                "scheduler_computation_time_ms",
                "node_timeline",
                "transfer_timeline",
            }
        }
        stable["node_timeline"] = [
            {key: value for key, value in item.items() if key not in {"start_ms", "finish_ms"}}
            for item in row["node_timeline"]
        ]
        stable["transfer_timeline"] = [
            {key: value for key, value in item.items() if key not in {"start_ms", "finish_ms"}}
            for item in row["transfer_timeline"]
        ]
        return stable

    assert [stable_row(row) for row in rows] == [stable_row(row) for row in replay_rows]
    for row in rows + replay_rows:
        assert row["scheduler_computation_time_ms"] >= 0.0
        assert row["latency_ms"] == pytest.approx(
            row["completion_time_ms"] - row["arrival_time_ms"]
        )

    assert [row["request_id"] for row in rows] == [
        "template-0__request-000000",
        "template-0__request-000001",
    ]
    assert {
        "request_id",
        "trace_id",
        "template_id",
        "request_type",
        "arrival_time_ms",
        "status",
        "selected_path_index",
        "assignments",
        "node_timeline",
        "transfer_timeline",
        "completion_time_ms",
        "latency_ms",
        "quality",
        "scheduler_computation_time_ms",
        "execution_cost",
        "reason",
    } <= rows[0].keys()
    assert rows[0]["template_id"] == "template-0"
    assert rows[0]["trace_id"] == rows[0]["request_id"]
    assert rows[0]["node_timeline"]
    assert rows[0]["transfer_timeline"]
    assert isinstance(rows[0]["quality"], float)
    assert "dags" not in rows[0]


def test_multi_eval_cli_accepts_zero_arrivals(tmp_path: Path) -> None:
    dataset = tmp_path / "dags.jsonl"
    scheduler = tmp_path / "scheduler.py"
    arrivals = tmp_path / "arrivals.json"
    _write_dataset(dataset)
    _write_scheduler(scheduler)
    save_arrival_manifest(
        arrivals,
        ArrivalManifest(
            arrivals=(),
            request_rate_per_second=1.0,
            observation_window_ms=100.0,
            seed=0,
        ),
    )

    run_dir, summary = _run_eval(dataset, scheduler, tmp_path / "run", arrivals=arrivals)

    assert summary["arrived_count"] == 0
    assert summary["completed_total"] == 0
    assert summary["completed_within_window"] == 0
    assert summary["throughput_per_second"] == 0.0
    assert summary["latency_p50_ms"] is None
    assert (run_dir / "results.jsonl").read_text(encoding="utf-8") == ""


def test_multi_eval_cli_can_run_from_one_selected_yaml(tmp_path: Path) -> None:
    dataset = tmp_path / "dags.jsonl"
    scheduler = tmp_path / "scheduler.py"
    arrivals = tmp_path / "arrivals.json"
    config = tmp_path / "multi.yaml"
    output_root = tmp_path / "yaml-runs"
    _write_dataset(dataset)
    _write_scheduler(scheduler)
    save_arrival_manifest(
        arrivals,
        ArrivalManifest(
            arrivals=(ArrivalRecord(0.0, "template-0__request-000000", "template-0"),),
            request_rate_per_second=3.0,
            observation_window_ms=100.0,
            seed=23,
        ),
    )
    config.write_text(
        "\n".join(
            [
                "defaults:",
                "  - _self_",
                "  - override /hydra/job_logging: disabled",
                "  - override /hydra/hydra_logging: disabled",
                "mode: multi",
                f"dataset: {dataset}",
                f"arrivals: {arrivals}",
                f"scheduler_source: {scheduler}",
                "scheduler_version: 7",
                f"profiling_database: {ROOT / 'docs/examples/profiling-database.fake.json'}",
                f"profiling_schema: {ROOT / 'docs/schemas/profiling-database.schema.json'}",
                f"output_root: {output_root}",
                "scoring_context:",
                "  accuracy_weight: 0.5",
                "  latency_weight: 0.25",
                "  resource_weight: 0.25",
                "  latency_scale_ms: 100.0",
                "  resource_scale_mib: 8192.0",
                "hydra:",
                "  run:",
                "    dir: .",
                "  output_subdir: null",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/eval.py"),
            "--config-path",
            str(tmp_path),
            "--config-name",
            "multi",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    run_dir = Path(
        next(line for line in result.stdout.splitlines() if line.startswith("run_dir: "))
        .split(": ", 1)[1]
    )
    resolved = (run_dir / "config.yaml").read_text(encoding="utf-8")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert str(dataset.resolve()) in resolved
    assert str(arrivals.resolve()) in resolved
    assert "accuracy_weight: 0.5" in resolved
    assert summary["mode"] == "multi"
    assert summary["arrived_count"] == 1


def test_main_exp_mixed_records_metadata_types_and_node_execution_cost(tmp_path: Path) -> None:
    video_dataset = tmp_path / "video.jsonl"
    math_dataset = tmp_path / "math.jsonl"
    scheduler = tmp_path / "scheduler.py"
    arrivals = tmp_path / "mixed-arrivals.json"
    _write_dataset(video_dataset, trace_id="video-template")
    _write_dataset(math_dataset, trace_id="math-template")
    _write_scheduler(scheduler)
    save_arrival_manifest(
        arrivals,
        ArrivalManifest(
            arrivals=(
                ArrivalRecord(
                    0.0,
                    "video_qa__request-000000",
                    "video-template",
                    "video_qa",
                ),
                ArrivalRecord(
                    1.0,
                    "math_qa__request-000000",
                    "math-template",
                    "math_qa",
                ),
            ),
            request_rate_per_second=20.0,
            observation_window_ms=100.0,
            seed=19,
        ),
    )

    run_dir, summary = _run_main_eval(
        video_dataset, math_dataset, scheduler, arrivals, tmp_path / "main-runs"
    )
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]

    assert summary["method"] == "Ours"
    assert summary["evolution_llm_family"] == "F1"
    assert summary["planner_llm"] == "planner-model-v1"
    assert summary["workload"] == "Mixed"
    assert summary["request_rate_per_second"] == 20.0
    assert summary["actual_arrivals_per_second"] == 20.0
    assert summary["seed"] == 19
    assert summary["currency"] == "USD"
    assert summary["device_price_per_second"]["cloud"] == 2.0
    assert set(summary["per_request_type"]) == {"video_qa", "math_qa"}
    assert summary["latency_p99_ms"] >= summary["latency_p50_ms"]

    assert [row["request_type"] for row in rows] == ["video_qa", "math_qa"]
    assert all(row["quality"] is not None for row in rows)
    assert all(row["scheduler_computation_time_ms"] >= 0.0 for row in rows)
    expected_costs = []
    for row in rows:
        expected = sum(
            (node["finish_ms"] - node["start_ms"]) / 1000.0 * 2.0
            for node in row["node_timeline"]
        )
        expected_costs.append(expected)
        assert row["execution_cost"] == pytest.approx(expected)
        assert row["latency_ms"] == pytest.approx(
            row["completion_time_ms"] - row["arrival_time_ms"]
        )
    assert summary["total_cost"] == pytest.approx(sum(expected_costs))
    assert summary["mean_cost"] == pytest.approx(sum(expected_costs) / 2.0)
    resolved = (run_dir / "config.yaml").read_text(encoding="utf-8")
    assert "method: Ours" in resolved
    assert "planner_llm: planner-model-v1" in resolved
    assert "actual_arrivals_per_second: 20.0" in resolved
