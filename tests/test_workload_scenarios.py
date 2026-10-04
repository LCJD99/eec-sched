from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from eec_sched.workload import (
    generate_mixed_poisson_arrival_manifest,
    generate_poisson_arrival_manifest,
    load_mixed_workload_scenario,
    load_workload_scenario,
    save_arrival_manifest,
)


ROOT = Path(__file__).parents[1]
RUNNER = ROOT / "experiments/10_workload_scenarios/run.py"


def _planner_rows(prefix: str = "task") -> list[dict[str, object]]:
    plan = {
        "nodes": [
            {
                "node_id": "classify",
                "tool_id": "text_classification",
                "inputs": {"text": {"kind": "request", "name": "text", "data_type": "text"}},
            }
        ],
        "final_outputs": [{"node_id": "classify", "port": "text"}],
    }
    return [
        {"trace_id": f"{prefix}-{index}", "task_input": {"text": f"{prefix}-request-{index}"}, "system_state": {}, "dags": [plan, plan, plan]}
        for index in range(3)
    ]


def _write_planner_file(path: Path, prefix: str = "task") -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in _planner_rows(prefix)), encoding="utf-8")


def test_mixed_manifest_merges_two_independent_streams_and_replays_by_type(tmp_path: Path) -> None:
    video_path = tmp_path / "video.jsonl"
    math_path = tmp_path / "math.jsonl"
    manifest_path = tmp_path / "mixed-arrivals.json"
    _write_planner_file(video_path, "video")
    _write_planner_file(math_path, "math")

    from eec_sched.workflow import ToolCallPlanDataset

    datasets = {
        "video_qa": ToolCallPlanDataset.load(video_path),
        "math_qa": ToolCallPlanDataset.load(math_path),
    }
    manifest = generate_mixed_poisson_arrival_manifest(
        datasets,
        request_rate_per_second=100,
        observation_window_ms=1000,
        seed=3,
    )
    replayed_manifest = generate_mixed_poisson_arrival_manifest(
        datasets,
        request_rate_per_second=100,
        observation_window_ms=1000,
        seed=3,
    )
    save_arrival_manifest(manifest_path, manifest)

    assert manifest == replayed_manifest
    assert {arrival.request_type for arrival in manifest.arrivals} == {"video_qa", "math_qa"}
    assert len({arrival.request_id for arrival in manifest.arrivals}) == len(manifest.arrivals)
    assert all(
        arrival.request_id.startswith(f"{arrival.request_type}__request-")
        for arrival in manifest.arrivals
    )
    counts = {request_type: sum(item.request_type == request_type for item in manifest.arrivals) for request_type in ("video_qa", "math_qa")}
    assert counts["video_qa"] > 0
    assert counts["math_qa"] > 0

    replayed = load_mixed_workload_scenario(
        {"video_qa": video_path, "math_qa": math_path}, manifest_path
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert all("request_type" in item for item in payload["arrivals"])
    assert [time for time, _ in replayed] == [item.arrival_time_ms for item in manifest.arrivals]
    assert [trace.trace_id for _, trace in replayed] == [item.request_id for item in manifest.arrivals]
    assert all(
        trace.task_input["text"].startswith(
            "video-" if item.request_type == "video_qa" else "math-"
        )
        for item, (_, trace) in zip(manifest.arrivals, replayed)
    )


def test_scenario_manifest_replays_all_three_dag_templates_in_order(tmp_path: Path) -> None:
    planner_path = tmp_path / "dags.jsonl"
    manifest_path = tmp_path / "arrivals.json"
    _write_planner_file(planner_path)

    from eec_sched.workflow import ToolCallPlanDataset

    dataset = ToolCallPlanDataset.load(planner_path)
    manifest = generate_poisson_arrival_manifest(
        dataset,
        request_rate_per_second=100,
        observation_window_ms=1000,
        seed=3,
    )
    save_arrival_manifest(manifest_path, manifest)

    replayed = load_workload_scenario(planner_path, manifest_path)

    assert len(dataset) == 3
    assert len(replayed) == len(manifest.arrivals)
    assert replayed
    assert [trace.trace_id.split("__request-")[0] for _, trace in replayed] == [
        f"task-{index % 3}" for index in range(len(replayed))
    ]
    assert [time for time, _ in replayed] == [record.arrival_time_ms for record in manifest.arrivals]
    assert all(len(trace.candidate_dags) == 3 for _, trace in replayed)


def test_zero_arrivals_is_a_valid_manifest(tmp_path: Path) -> None:
    planner_path = tmp_path / "dags.jsonl"
    manifest_path = tmp_path / "arrivals.json"
    _write_planner_file(planner_path)

    from eec_sched.workflow import ToolCallPlanDataset

    manifest = generate_poisson_arrival_manifest(
        ToolCallPlanDataset.load(planner_path),
        request_rate_per_second=1,
        observation_window_ms=1e-9,
        seed=1,
    )
    save_arrival_manifest(manifest_path, manifest)

    assert manifest.arrivals == ()
    assert load_workload_scenario(planner_path, manifest_path) == ()


def test_hydra_entrypoint_writes_resolved_config_and_manifest(tmp_path: Path) -> None:
    planner_path = tmp_path / "dags.jsonl"
    output_dir = tmp_path / "scenario-run"
    _write_planner_file(planner_path)

    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            f"dag_path={planner_path}",
            "request_rate_per_second=1",
            "observation_window_ms=1e-9",
            "seed=1",
            f"output_dir={output_dir}",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "request_count: 0" in result.stdout
    assert (output_dir / "config.yaml").exists()
    arrivals_path = output_dir / "arrivals.json"
    assert arrivals_path.exists()
    assert json.loads(arrivals_path.read_text(encoding="utf-8"))["arrivals"] == []
    assert str(planner_path.resolve()) in (output_dir / "config.yaml").read_text(encoding="utf-8")


def test_hydra_entrypoint_nonzero_arrivals_replay_from_saved_manifest(tmp_path: Path) -> None:
    planner_path = tmp_path / "dags.jsonl"
    output_dir = tmp_path / "scenario-run"
    _write_planner_file(planner_path)

    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            f"dag_path={planner_path}",
            "request_rate_per_second=100",
            "observation_window_ms=1000",
            "seed=3",
            f"output_dir={output_dir}",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "request_count: 0" not in result.stdout
    manifest_path = output_dir / "arrivals.json"
    arrivals = load_workload_scenario(planner_path, manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert arrivals
    assert len(arrivals) == len(payload["arrivals"])
    assert [time for time, _ in arrivals] == [item["arrival_time_ms"] for item in payload["arrivals"]]
    assert [trace.trace_id for _, trace in arrivals] == [item["request_id"] for item in payload["arrivals"]]


def test_hydra_entrypoint_writes_one_mixed_manifest_from_two_datasets(tmp_path: Path) -> None:
    video_path = tmp_path / "video.jsonl"
    math_path = tmp_path / "math.jsonl"
    output_dir = tmp_path / "mixed-run"
    _write_planner_file(video_path, "video")
    _write_planner_file(math_path, "math")

    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            f"video_qa_dag_path={video_path}",
            f"math_qa_dag_path={math_path}",
            "request_rate_per_second=100",
            "observation_window_ms=1000",
            "seed=3",
            f"output_dir={output_dir}",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads((output_dir / "arrivals.json").read_text(encoding="utf-8"))
    assert {item["request_type"] for item in payload["arrivals"]} == {"video_qa", "math_qa"}
    resolved = (output_dir / "config.yaml").read_text(encoding="utf-8")
    assert str(video_path.resolve()) in resolved
    assert str(math_path.resolve()) in resolved
