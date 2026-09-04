"""Contract tests for loading BO campaign observations into visualization points."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def load_visualizer():
    path = Path("experiments/02_bayesian_optimization/visualize_campaign.py")
    spec = importlib.util.spec_from_file_location("campaign_visualizer", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_points_keeps_successes_and_marks_selected_configurations(tmp_path: Path) -> None:
    campaign = {
        "observations": [
            {
                "configuration": {"imgsz": 640, "conf": 0.2},
                "status": "success",
                "box_ap": 0.5,
                "warm_latency_ms": 12.0,
                "peak_gpu_memory_mib": 100.0,
            },
            {
                "configuration": {"imgsz": 320, "conf": 0.1},
                "status": "oom",
                "box_ap": None,
                "warm_latency_ms": None,
                "peak_gpu_memory_mib": None,
            },
        ],
        "selected_configurations": [
            {"configuration": {"imgsz": 640, "conf": 0.2}}
        ],
    }
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(campaign), encoding="utf-8")

    visualizer = load_visualizer()
    points = visualizer.load_points(path)

    assert points == [
        visualizer.Point(640, 0.2, 0.5, 12.0, 100.0, selected=True)
    ]
