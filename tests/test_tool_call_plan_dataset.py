from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from eec_sched import ToolCallPlanDataset


ROW = {
    "nodes": [{"node_id": "node-0", "tool_id": "text_classification", "inputs": {"text": {"kind": "request", "name": "text"}}}],
    "final_outputs": [{"node_id": "node-0", "port": "text"}],
}


def test_loads_tool_call_plan_jsonl_and_splits_in_4_3_3_ratio(tmp_path: Path) -> None:
    path = tmp_path / "dags.jsonl"
    path.write_text("".join(json.dumps(ROW) + "\n" for _ in range(10)), encoding="utf-8")

    dataset = ToolCallPlanDataset.load(path)
    splits = dataset.split()

    assert len(dataset) == 10
    assert [len(splits.train), len(splits.validation), len(splits.test)] == [4, 3, 3]
    assert splits.train[0].trace_id == "mnms-000000"
    assert splits.validation[0].trace_id == "mnms-000004"
    assert splits.test[0].trace_id == "mnms-000007"


def test_rejects_invalid_jsonl_with_its_line_number(tmp_path: Path) -> None:
    path = tmp_path / "dags.jsonl"
    path.write_text("not json\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"line 1"):
        ToolCallPlanDataset.load(path)


def test_eval_script_writes_one_dag_and_score_per_trace(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dags.jsonl"
    scheduler_path = tmp_path / "scheduler.py"
    output_path = tmp_path / "traces.jsonl"
    dataset_path.write_text(json.dumps(ROW) + "\n", encoding="utf-8")
    scheduler_path.write_text(
        "def propose(view):\n"
        "    return {node.node_id: {'configuration_id': 'quality', 'device_id': 'cloud'} for node in view.dag.nodes}\n",
        encoding="utf-8",
    )
    root = Path(__file__).parents[1]
    environment = {**os.environ, "PYTHONPATH": str(root / "src")}

    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/eval.py"),
            "--dataset",
            str(dataset_path),
            "--split",
            "all",
            "--scheduler-source",
            str(scheduler_path),
            "--scheduler-version",
            "12",
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    metrics = json.loads(completed.stdout)
    trace = json.loads(output_path.read_text(encoding="utf-8"))
    assert metrics["scheduler_version"] == 12
    assert metrics["trace_count"] == 1
    assert trace["trace_id"] == "mnms-000000"
    assert trace["dag"] == ROW
    assert isinstance(trace["score"], float)
