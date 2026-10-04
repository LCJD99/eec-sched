import json
from pathlib import Path

import pytest

from server import _aggregate_group, config_identity, discover_runs, percentile, select_runs


def _run(identifier, timestamp, values, config_id="cfg-001"):
    return {
        "id": identifier,
        "scheduler": "demo",
        "rate": 1,
        "seed": 7,
        "timestamp": timestamp,
        "timestamp_value": timestamp,
        "config_id": config_id,
        "config_label": "demo",
        "config": {},
        "request_count": len(values),
        "completed_count": len(values),
        "failed_count": 0,
        "_values": {"quality": list(values), "resource": list(values), "latency": list(values)},
    }


def test_percentile_is_interpolated():
    assert percentile([1, 2, 3, 4]) == pytest.approx(3.85)


def test_group_p95_uses_merged_requests():
    first = _run("a", 1, [0, 1, 2, 3])
    second = _run("b", 2, [100])
    row = _aggregate_group([first, second])
    assert row["quality_p95"] == pytest.approx(80.6)
    assert row["quality_mean"] == pytest.approx(21.2)


def test_latest_selection_is_per_seed_and_config():
    old = _run("old", 1, [1])
    new = _run("new", 2, [2])
    other_seed = _run("seed-17", 3, [3])
    other_seed["seed"] = 17
    assert {run["id"] for run in select_runs([old, new, other_seed], "latest")} == {"new", "seed-17"}


def test_runtime_rate_does_not_split_config_but_scoring_does():
    base = {"dataset": "dags.jsonl", "request_rate_per_second": 1, "scoring_context": {"beta": 0.01}}
    same = {**base, "request_rate_per_second": 20, "resolved_seed": 27}
    changed = {**base, "scoring_context": {"beta": 0.02}}
    assert config_identity(base) == config_identity(same)
    assert config_identity(base) != config_identity(changed)


def test_discovery_excludes_incomplete_and_separates_summary_identity(tmp_path: Path):
    complete = tmp_path / "alpha" / "rate-1" / "seed-7" / "eval-20261003T000000Z"
    incomplete = tmp_path / "alpha" / "rate-1" / "seed-17" / "eval-20261003T000001Z"
    complete.mkdir(parents=True)
    incomplete.mkdir(parents=True)
    config = "dataset: dags.jsonl\nprofiling_database: db.json\n"
    (complete / "config.yaml").write_text(config, encoding="utf-8")
    (complete / "summary.json").write_text(json.dumps({"snapshot_digest": "snap", "observation_window_ms": 1000}), encoding="utf-8")
    (complete / "results.jsonl").write_text(json.dumps({"status": "completed", "quality": 1, "latency_ms": 1000, "node_timeline": [{"gpu_memory_mib": 2}]}) + "\n", encoding="utf-8")
    (incomplete / "config.yaml").write_text(config, encoding="utf-8")
    runs, stats = discover_runs(tmp_path)
    assert len(runs) == 1
    assert stats["incomplete_count"] == 1
    assert runs[0]["config"]["observation_window_ms"] == 1000
