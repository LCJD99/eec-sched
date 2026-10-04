"""Prepare one deterministic, replayable workload scenario.

The command reads successful three-DAG Planner records, samples Poisson
arrival times in simulated milliseconds, and writes the realized arrival
manifest.  It does not invoke a Scheduler or evaluator.

Example::

    uv run python experiments/10_workload_scenarios/run.py \
        dag_path=experiments/09_offline_tool_planning/runs/tool-planner-20261002T085931Z/dags.jsonl \
        request_rate_per_second=2 observation_window_ms=60000 seed=7
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from eec_sched.workload import (
    generate_mixed_poisson_arrival_manifest,
    generate_poisson_arrival_manifest,
    save_arrival_manifest,
)
from eec_sched.workflow import ToolCallPlanDataset


@dataclass(frozen=True)
class WorkloadScenarioSummary:
    """Paths and count emitted by one scenario preparation run."""

    output_dir: Path
    config_path: Path
    arrival_manifest_path: Path
    request_count: int


def _root() -> Path:
    """Return the project root both under Hydra and in direct Python calls."""
    try:
        return Path(get_original_cwd())
    except (ValueError, RuntimeError):
        return Path.cwd()


def _path(root: Path, value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else root / candidate


def _new_output_dir(root: Path) -> Path:
    run_root = root / "experiments/10_workload_scenarios/runs"
    stamp = datetime.now(timezone.utc).strftime("workload-%Y%m%dT%H%M%SZ")
    candidate = run_root / stamp
    suffix = 1
    while candidate.exists():
        candidate = run_root / f"{stamp}-{suffix:02d}"
        suffix += 1
    return candidate


def _resolved_config(
    config: DictConfig | dict[str, Any],
    output_dir: Path,
    dag_paths: dict[str, Path],
) -> str:
    """Serialize the resolved run config and record paths for this run."""
    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("Hydra configuration must be a mapping")
    resolved["output_dir"] = str(output_dir)
    for key, path in dag_paths.items():
        resolved[key] = str(path.resolve())
    return OmegaConf.to_yaml(OmegaConf.create(resolved), resolve=True)


def run_from_config(config: DictConfig | dict[str, Any]) -> WorkloadScenarioSummary:
    """Prepare a scenario from a composed Hydra config.

    ``dag_path`` is the canonical key.  ``input_path`` is accepted as a small
    convenience for callers migrating from the offline Planner config.
    """
    root = _root()
    # Validate and materialize before creating a new run directory.  A bad
    # source file or workload parameter therefore leaves no partial run.
    dag_value = config.get("dag_path") or config.get("input_path")
    video_value = config.get("video_qa_dag_path")
    math_value = config.get("math_qa_dag_path")
    if video_value and math_value:
        dag_paths = {
            "video_qa_dag_path": _path(root, str(video_value)),
            "math_qa_dag_path": _path(root, str(math_value)),
        }
        datasets = {
            "video_qa": ToolCallPlanDataset.load(dag_paths["video_qa_dag_path"]),
            "math_qa": ToolCallPlanDataset.load(dag_paths["math_qa_dag_path"]),
        }
        manifest = generate_mixed_poisson_arrival_manifest(
            datasets,
            request_rate_per_second=config["request_rate_per_second"],
            observation_window_ms=config["observation_window_ms"],
            seed=config["seed"],
        )
    elif dag_value:
        dag_paths = {"dag_path": _path(root, str(dag_value))}
        dataset = ToolCallPlanDataset.load(dag_paths["dag_path"])
        manifest = generate_poisson_arrival_manifest(
            dataset,
            request_rate_per_second=config["request_rate_per_second"],
            observation_window_ms=config["observation_window_ms"],
            seed=config["seed"],
        )
    else:
        raise ValueError("configuration requires dag_path or both video_qa_dag_path and math_qa_dag_path")

    configured_output_dir = config.get("output_dir")
    output_dir = _path(root, str(configured_output_dir)) if configured_output_dir else _new_output_dir(root)
    if output_dir.exists():
        raise FileExistsError(f"refusing to use existing output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    config_path = output_dir / "config.yaml"
    config_path.write_text(_resolved_config(config, output_dir, dag_paths), encoding="utf-8")

    arrival_manifest_path = output_dir / "arrivals.json"
    save_arrival_manifest(arrival_manifest_path, manifest)
    return WorkloadScenarioSummary(
        output_dir=output_dir,
        config_path=config_path,
        arrival_manifest_path=arrival_manifest_path,
        request_count=len(manifest.arrivals),
    )


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config: DictConfig) -> None:
    try:
        summary = run_from_config(config)
    except (FileExistsError, OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"workload scenario configuration error: {exc}") from exc
    print(f"output_dir: {summary.output_dir}")
    print(f"arrivals: {summary.arrival_manifest_path}")
    print(f"request_count: {summary.request_count}")


if __name__ == "__main__":
    main()
