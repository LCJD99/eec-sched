"""Fast contract tests for the standalone Bayesian-optimization experiment."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_campaign():
    path = Path("experiments/02_bayesian_optimization/campaign.py")
    spec = importlib.util.spec_from_file_location("bayesian_campaign", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


campaign = load_campaign()


def observation(imgsz: int, conf: float, ap: float, latency: float, memory: float, status: str = "success"):
    return campaign.Observation(
        configuration=campaign.Configuration(imgsz, conf),
        status=status,
        box_ap=ap if status == "success" else None,
        warm_latency_ms=latency if status == "success" else None,
        peak_gpu_memory_mib=memory if status == "success" else None,
        error=None,
        timestamp_utc="2026-08-16T00:00:00+00:00",
        provenance={},
    )


def test_grid_quantizes_joint_parameters_and_removes_illegal_points() -> None:
    space = {
        "min_imgsz": 320,
        "max_imgsz": 384,
        "imgsz_step": 32,
        "min_conf": 0.10,
        "max_conf": 0.20,
        "conf_precision": 0.05,
        "illegal_combinations": [{"imgsz": 352, "conf": 0.15}],
    }

    assert campaign.canonicalize(349, 0.18, space) == campaign.Configuration(352, 0.20)
    assert campaign.canonicalize(352, 0.15, space) is None
    assert len(campaign.legal_grid(space)) == 8


def test_deprecated_half_is_migrated_to_equivalent_quantize_setting() -> None:
    assert campaign.normalize_yolo_parameters({"half": True, "batch": 1}) == {
        "quantize": "fp16",
        "batch": 1,
    }
    assert campaign.normalize_yolo_parameters({"half": False}) == {
        "quantize": "fp32"
    }


def test_sobol_initial_points_are_unique_and_reproducible() -> None:
    grid = [campaign.Configuration(size, conf) for size in (320, 352, 384) for conf in (0.1, 0.2, 0.3)]

    first = campaign.sobol_initial_candidates(grid, 6, seed=17)

    assert first == campaign.sobol_initial_candidates(grid, 6, seed=17)
    assert len(first) == len(set(first)) == 6


def test_pareto_frontier_uses_only_successful_measured_points() -> None:
    best = observation(320, 0.1, 0.60, 10, 4)
    dominated = observation(352, 0.1, 0.50, 12, 5)
    failed = observation(384, 0.1, 0.90, 1, 1, status="oom")

    assert campaign.pareto_frontier([best, dominated, failed]) == [best]


def test_selection_keeps_extrema_then_returns_at_most_ten() -> None:
    points = [
        observation(320 + index * 32, 0.1, 0.40 + index * 0.02, 10 + index, 4 + index * 0.4)
        for index in range(12)
    ]

    selected = campaign.select_configurations(points)

    assert len(selected) == 10
    assert max(points, key=lambda item: item.box_ap) in selected
    assert min(points, key=lambda item: item.warm_latency_ms) in selected
    assert min(points, key=lambda item: item.peak_gpu_memory_mib) in selected
