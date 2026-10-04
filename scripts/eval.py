"""Run one offline Scheduler evaluation from a Hydra configuration.

The composition root is ``experiments/11_scheduler_evaluation/config.yaml``.
All inputs belong in that YAML file; command-line arguments are Hydra
overrides such as ``mode=multi`` or ``dataset=/tmp/dags.jsonl``.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from eec_sched import (
    CandidateEvaluation,
    SchedulerCandidate,
    SchedulerCandidateRegistry,
    SchedulerView,
    ToolCallPlan,
    ToolCallPlanDataset,
    evaluate_scheduler_candidate,
    load_profiling_database,
)
from eec_sched.evaluation.scoring import ScoringContext
from eec_sched.evaluation.workload import WorkloadEvaluation, WorkloadRequest, evaluate_workload
from eec_sched.evolution import SchedulerProposal, TraceEvaluation
from eec_sched.workload import ArrivalManifest, load_arrival_manifest


ROOT = Path(__file__).parents[1]


def _root() -> Path:
    """Return the project root when called under Hydra or directly in tests."""
    try:
        return Path(get_original_cwd())
    except (RuntimeError, ValueError):
        return ROOT


def _path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_scheduler_candidate(source_path: Path, scheduler_version: int) -> SchedulerCandidate:
    """Load a versioned Candidate whose source defines ``propose(view)``."""
    source_code = source_path.read_text(encoding="utf-8")
    compiled = compile(source_code, str(source_path), "exec")

    def propose(view: SchedulerView) -> SchedulerProposal:
        namespace: dict[str, object] = {"__builtins__": __builtins__}
        exec(compiled, namespace)  # noqa: S102 - Candidate source is evaluated at a trusted boundary.
        loaded = namespace.get("propose")
        if not callable(loaded):
            raise ValueError(f"{source_path} must define callable propose(view)")
        return cast(Callable[[SchedulerView], SchedulerProposal], loaded)(view)

    return SchedulerCandidate(
        scheduler_version=scheduler_version,
        propose=propose,
        source_code=source_code,
        strategy_description=f"loaded from {source_path}",
    )


def dag_projection(dag: ToolCallPlan) -> dict[str, object]:
    """Project a DAG using the stable legacy single-request shape."""
    return {
        "nodes": [
            {
                "node_id": node.node_id,
                "tool_id": node.tool_id,
                "inputs": {
                    name: {
                        "kind": source.kind,
                        "name": source.name,
                        **({"port": source.port} if source.port is not None else {}),
                    }
                    for name, source in node.inputs.items()
                },
            }
            for node in dag.nodes
        ],
        "final_outputs": [
            {"node_id": output.node_id, "port": output.port}
            for output in dag.final_outputs
        ],
    }


def trace_projection(record: TraceEvaluation) -> dict[str, object]:
    """Keep the existing per-Trace output and Candidate Score semantics."""
    metrics = dict(record.raw_metrics)
    if record.report.raw_accuracy_metrics:
        metrics["raw_accuracy_metrics"] = dict(record.report.raw_accuracy_metrics)
    return {
        "trace_id": record.trace.trace_id,
        "system_state": dict(record.trace.system_state),
        "dag": dag_projection(record.selected_dag or record.trace.dag),
        "candidate_dags": [dag_projection(dag) for dag in record.trace.candidate_dags],
        "selected_path_index": record.selected_path_index,
        "status": record.status,
        "score": record.score_contribution,
        "metrics": metrics,
        "reason": record.reason,
        "assignments": {
            node_id: {
                "configuration_id": assignment.configuration_id,
                "device_id": assignment.device_id,
            }
            for node_id, assignment in record.assignments.items()
        },
    }


def write_trace_output(path: Path, evaluation: CandidateEvaluation) -> None:
    with path.open("w", encoding="utf-8") as output:
        for record in evaluation.traces:
            output.write(
                json.dumps(trace_projection(record), ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )


def _assignment_projection(record: WorkloadRequest) -> dict[str, object]:
    return {
        node_id: {
            "configuration_id": assignment.configuration_id,
            "device_id": assignment.device_id,
        }
        for node_id, assignment in record.assignments.items()
    }


def _default_request_type(workload: object) -> str | None:
    if not isinstance(workload, str):
        return None
    normalized = workload.lower().replace(" ", "_").replace("-", "_")
    if normalized in {"video", "video_qa", "videoqa"}:
        return "video_qa"
    if normalized in {"math", "math_qa", "mathqa"}:
        return "math_qa"
    return None


def workload_request_projection(
    record: WorkloadRequest,
    template_id: str,
    request_type: str | None = None,
    device_price_per_second: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """Project one workload request without copying its Planner template."""
    execution_cost = (
        None
        if device_price_per_second is None
        else sum(
            (node.finish_ms - node.start_ms) / 1000.0
            * device_price_per_second[node.device_id]
            for node in record.nodes
            if node.start_ms is not None and node.finish_ms is not None
        )
    )
    return {
        "request_id": record.request_id,
        "trace_id": record.trace_id,
        "template_id": template_id,
        "request_type": request_type,
        "arrival_time_ms": record.arrival_time_ms,
        "status": record.status,
        "selected_path_index": record.selected_path_index,
        "assignments": _assignment_projection(record),
        "node_timeline": [
            {
                "node_id": node.node_id,
                "configuration_id": node.configuration_id,
                "device_id": node.device_id,
                "start_ms": node.start_ms,
                "finish_ms": node.finish_ms,
                "gpu_memory_mib": node.gpu_memory_mib,
            }
            for node in record.nodes
        ],
        "transfer_timeline": [
            {
                "source_node_id": transfer.source_node_id,
                "destination_node_id": transfer.destination_node_id,
                "source_device_id": transfer.source_device_id,
                "destination_device_id": transfer.destination_device_id,
                "start_ms": transfer.start_ms,
                "finish_ms": transfer.finish_ms,
                "latency_ms": transfer.latency_ms,
            }
            for transfer in record.transfers
        ],
        "completion_time_ms": record.completion_time_ms,
        "latency_ms": record.latency_ms,
        "quality": record.quality,
        "scheduler_computation_time_ms": record.scheduler_computation_time_ms,
        "execution_cost": execution_cost,
        "reason": record.reason,
    }


def write_workload_output(
    path: Path,
    evaluation: WorkloadEvaluation,
    manifest: ArrivalManifest,
    device_price_per_second: Mapping[str, float] | None = None,
    default_request_type: str | None = None,
) -> None:
    arrival_by_request = {item.request_id: item for item in manifest.arrivals}
    with path.open("w", encoding="utf-8") as output:
        for record in evaluation.requests:
            arrival = arrival_by_request[record.request_id]
            projected = workload_request_projection(
                record,
                arrival.template_id,
                arrival.request_type or default_request_type,
                device_price_per_second,
            )
            output.write(json.dumps(projected, ensure_ascii=False, separators=(",", ":")) + "\n")


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _request_aggregates(
    records: list[WorkloadRequest],
    costs: Mapping[str, float] | None,
) -> dict[str, object]:
    quality = [
        float(record.quality)
        for record in records
        if record.status == "completed" and record.quality is not None
    ]
    latency = [
        float(record.latency_ms)
        for record in records
        if record.status == "completed" and record.latency_ms is not None
    ]
    total_cost = (
        None
        if costs is None
        else sum(costs.get(record.request_id, 0.0) for record in records)
    )
    return {
        "request_count": len(records),
        "quality_count": len(quality),
        "quality_mean": _mean(quality),
        "quality_p50": _percentile(quality, 0.50),
        "quality_p95": _percentile(quality, 0.95),
        "quality_p99": _percentile(quality, 0.99),
        "latency_count": len(latency),
        "latency_mean_ms": _mean(latency),
        "latency_p50_ms": _percentile(latency, 0.50),
        "latency_p95_ms": _percentile(latency, 0.95),
        "latency_p99_ms": _percentile(latency, 0.99),
        "total_execution_cost": total_cost,
        "mean_execution_cost": total_cost / len(records) if costs is not None and records else None,
    }


def _single_summary(
    evaluation: CandidateEvaluation, snapshot_digest: str, scheduler_version: int
) -> dict[str, object]:
    statuses = Counter(record.status for record in evaluation.traces)
    return {
        "mode": "single",
        "scheduler_version": scheduler_version,
        "trace_count": len(evaluation.traces),
        "candidate_score": evaluation.candidate_score,
        "status_counts": dict(sorted(statuses.items())),
        "snapshot_digest": snapshot_digest,
    }


def _multi_summary(
    evaluation: WorkloadEvaluation,
    manifest: ArrivalManifest,
    dataset_paths: Mapping[str, Path],
    arrivals_path: Path,
    metadata: Mapping[str, object],
    device_price_per_second: Mapping[str, float] | None,
    currency: str,
    default_request_type: str | None,
) -> dict[str, object]:
    records = list(evaluation.requests)
    completed_total = sum(record.status == "completed" for record in evaluation.requests)
    rejected_count = sum(record.status == "rejected" for record in evaluation.requests)
    failed_count = sum(record.status == "failed" for record in evaluation.requests)
    selected_path_counts = {
        str(index): sum(
            record.status == "completed" and record.selected_path_index == index
            for record in evaluation.requests
        )
        for index in range(3)
    }
    arrival_by_request = {item.request_id: item for item in manifest.arrivals}
    request_types = {
        request_id: arrival.request_type or default_request_type
        for request_id, arrival in arrival_by_request.items()
    }
    costs = (
        {
            record.request_id: sum(
                (node.finish_ms - node.start_ms) / 1000.0
                * device_price_per_second[node.device_id]
                for node in record.nodes
                if node.start_ms is not None and node.finish_ms is not None
            )
            for record in records
        }
        if device_price_per_second is not None
        else None
    )
    aggregates = _request_aggregates(records, costs)
    per_type: dict[str, object] = {}
    for request_type in sorted(
        {request_type for request_type in request_types.values() if request_type is not None}
    ):
        typed_records = [
            record
            for record in records
            if request_types.get(record.request_id) == request_type
        ]
        per_type[request_type] = _request_aggregates(typed_records, costs)
    dataset_value: object = (
        str(next(iter(dataset_paths.values())))
        if len(dataset_paths) == 1
        else {key: str(value) for key, value in dataset_paths.items()}
    )
    seconds = manifest.observation_window_ms / 1000.0
    summary = {
        "mode": "multi",
        "scheduler_version": evaluation.scheduler_version,
        "dataset": dataset_value,
        "arrivals": str(arrivals_path),
        "request_rate_per_second": manifest.request_rate_per_second,
        "actual_arrivals_per_second": len(records) / seconds,
        "seed": manifest.seed,
        "observation_window_ms": evaluation.observation_window_ms,
        "arrived_count": len(evaluation.requests),
        "completed_within_window": evaluation.completed_within_window,
        "completed_total": completed_total,
        "backlog_at_window_end": evaluation.backlog_at_window_end,
        "rejected_count": rejected_count,
        "failed_count": failed_count,
        "throughput_per_second": evaluation.throughput_per_second,
        "latency_p50_ms": evaluation.latency_p50_ms,
        "latency_p95_ms": evaluation.latency_p95_ms,
        "snapshot_digest": evaluation.snapshot_digest,
        "selected_path_counts": selected_path_counts,
        "device_price_per_second": (
            dict(device_price_per_second) if device_price_per_second is not None else None
        ),
        "currency": currency,
        "per_request_type": per_type,
    }
    summary.update(metadata)
    summary.update(aggregates)
    summary["total_cost"] = aggregates["total_execution_cost"]
    summary["mean_cost"] = aggregates["mean_execution_cost"]
    return summary


def _scoring_context(config: Mapping[str, Any]) -> ScoringContext:
    raw = config.get("scoring_context", {})
    if raw is None:
        return ScoringContext()
    if not isinstance(raw, Mapping):
        raise TypeError("scoring_context must be a mapping")
    return ScoringContext(**dict(raw))


def _resolved_config(
    config: DictConfig | Mapping[str, Any],
    paths: Mapping[str, Path],
    run_dir: Path,
    runtime: Mapping[str, object] | None = None,
) -> str:
    source = config if isinstance(config, DictConfig) else OmegaConf.create(dict(config))
    resolved = OmegaConf.to_container(source, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("Hydra configuration must be a mapping")
    for key, path in paths.items():
        resolved[key] = str(path.resolve())
    resolved["run_dir"] = str(run_dir.resolve())
    if runtime:
        resolved.update(runtime)
    return OmegaConf.to_yaml(OmegaConf.create(resolved), resolve=True)


def _new_run_dir(output_root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("eval-%Y%m%dT%H%M%SZ")
    candidate = output_root / stamp
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"{stamp}-{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def run_from_config(config: DictConfig | Mapping[str, Any]) -> tuple[Path, dict[str, object]]:
    """Evaluate one Scheduler and write its timestamped run directory."""
    root = _root()
    source = config if isinstance(config, DictConfig) else OmegaConf.create(dict(config))
    raw = OmegaConf.to_container(source, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("Hydra configuration must be a mapping")
    mode = raw.get("mode", "single")
    if mode not in {"single", "multi"}:
        raise ValueError("mode must be single or multi")
    if raw.get("final_evaluation"):
        required = (
            "method",
            "planner_llm",
            "scheduler_source",
            "scheduler_version",
            "arrivals",
            "profiling_database",
            "profiling_schema",
            "currency",
            "device_price_per_second",
        )
        missing = [key for key in required if raw.get(key) in (None, "")]
        if missing:
            raise ValueError(f"main final evaluation requires: {', '.join(missing)}")
        if mode != "multi":
            raise ValueError("main final evaluation requires mode=multi")

    dataset_value = raw.get("dataset")
    dataset_path = _path(root, str(dataset_value)) if dataset_value else None
    video_dataset_value = raw.get("video_qa_dataset") or raw.get("video_qa_dag_path")
    math_dataset_value = raw.get("math_qa_dataset") or raw.get("math_qa_dag_path")
    video_dataset_path = _path(root, str(video_dataset_value)) if video_dataset_value else None
    math_dataset_path = _path(root, str(math_dataset_value)) if math_dataset_value else None
    if (video_dataset_path is None) != (math_dataset_path is None):
        raise ValueError("mixed evaluation requires both video_qa and math_qa datasets")
    if raw.get("final_evaluation"):
        workload_name = str(raw.get("workload", "")).lower().replace(" ", "_").replace("-", "_")
        if workload_name in {"mixed"}:
            if dataset_path is not None or video_dataset_path is None or math_dataset_path is None:
                raise ValueError(
                    "main Mixed evaluation requires video_qa_dataset and math_qa_dataset"
                )
        elif workload_name in {"video", "video_qa", "videoqa", "math", "math_qa", "mathqa"}:
            if dataset_path is None or video_dataset_path is not None or math_dataset_path is not None:
                raise ValueError(
                    "main Video QA or Math QA evaluation requires dataset only"
                )
        else:
            raise ValueError("main final evaluation workload must be Video QA, Math QA, or Mixed")
    scheduler_source = _path(root, str(raw["scheduler_source"]))
    profiling_database = _path(root, str(raw["profiling_database"]))
    profiling_schema = _path(root, str(raw["profiling_schema"]))
    arrivals_value = raw.get("arrivals")
    arrivals_path = _path(root, str(arrivals_value)) if arrivals_value else None
    if mode == "multi" and arrivals_path is None:
        raise ValueError("multi mode requires arrivals")

    dataset: ToolCallPlanDataset | None = None
    datasets_by_type: dict[str, ToolCallPlanDataset] | None = None
    if video_dataset_path is not None and math_dataset_path is not None:
        datasets_by_type = {
            "video_qa": ToolCallPlanDataset.load(video_dataset_path),
            "math_qa": ToolCallPlanDataset.load(math_dataset_path),
        }
    elif dataset_path is not None:
        dataset = ToolCallPlanDataset.load(dataset_path)
    else:
        raise ValueError("evaluation requires dataset or both video_qa/math_qa datasets")
    candidate = load_scheduler_candidate(scheduler_source, int(raw["scheduler_version"]))
    snapshot = load_profiling_database(profiling_database, profiling_schema)
    scoring_context = _scoring_context(cast(Mapping[str, Any], raw))
    device_price_raw = raw.get("device_price_per_second")
    if device_price_raw is not None and not isinstance(device_price_raw, Mapping):
        raise TypeError("device_price_per_second must be a mapping")
    device_price_per_second = (
        {
            str(device): float(price) for device, price in device_price_raw.items()
        }
        if device_price_raw is not None
        else None
    )
    currency = str(raw.get("currency", ""))
    manifest: ArrivalManifest | None = None
    arrivals: tuple[tuple[float, Any], ...] = ()
    if mode == "multi":
        assert arrivals_path is not None
        manifest = load_arrival_manifest(arrivals_path)
        if datasets_by_type is not None:
            arrivals = manifest.materialize(datasets_by_type)
        else:
            assert dataset is not None
            arrivals = manifest.materialize(dataset)

    output_root = _path(root, str(raw["output_root"]))
    run_dir = _new_run_dir(output_root)
    paths: dict[str, Path] = {
        **({"dataset": dataset_path} if dataset_path is not None else {}),
        "scheduler_source": scheduler_source,
        "profiling_database": profiling_database,
        "profiling_schema": profiling_schema,
        "output_root": output_root,
    }
    if video_dataset_path is not None and math_dataset_path is not None:
        paths.update(
            {
                "video_qa_dataset": video_dataset_path,
                "math_qa_dataset": math_dataset_path,
            }
        )
    if arrivals_path is not None:
        paths["arrivals"] = arrivals_path
    runtime = {
        "manifest_request_rate_per_second": (
            manifest.request_rate_per_second if manifest is not None else None
        ),
        "actual_arrivals_per_second": (
            len(manifest.arrivals) / (manifest.observation_window_ms / 1000.0)
            if manifest is not None
            else None
        ),
        "actual_arrival_count": len(manifest.arrivals) if manifest is not None else None,
        "resolved_seed": manifest.seed if manifest is not None else None,
    }
    (run_dir / "config.yaml").write_text(
        _resolved_config(config, paths, run_dir, runtime), encoding="utf-8"
    )

    results_path = run_dir / "results.jsonl"
    if mode == "single":
        assert dataset is not None
        evaluation = evaluate_scheduler_candidate(
            snapshot,
            dataset.traces,
            candidate.scheduler_version,
            SchedulerCandidateRegistry({candidate.scheduler_version: candidate}),
            scoring_context=scoring_context,
        )
        assert isinstance(evaluation, CandidateEvaluation)
        write_trace_output(results_path, evaluation)
        summary = _single_summary(evaluation, snapshot.snapshot_digest, candidate.scheduler_version)
    else:
        assert manifest is not None and arrivals_path is not None
        workload = evaluate_workload(
            snapshot,
            arrivals,
            candidate,
            observation_window_ms=manifest.observation_window_ms,
            scoring_context=scoring_context,
        )
        if device_price_per_second is not None:
            used_devices = {
                node.device_id
                for record in workload.requests
                for node in record.nodes
            }
            missing_prices = sorted(used_devices - set(device_price_per_second))
            if missing_prices:
                raise ValueError(
                    "device_price_per_second missing used devices: "
                    + ", ".join(missing_prices)
                )
        default_request_type = _default_request_type(raw.get("workload"))
        write_workload_output(
            results_path,
            workload,
            manifest,
            device_price_per_second,
            default_request_type,
        )
        dataset_paths = (
            {
                "video_qa": video_dataset_path,
                "math_qa": math_dataset_path,
            }
            if video_dataset_path is not None and math_dataset_path is not None
            else {"dataset": dataset_path}
        )
        summary = _multi_summary(
            workload,
            manifest,
            cast(Mapping[str, Path], dataset_paths),
            arrivals_path,
            {
                "method": raw.get("method"),
                "evolution_llm_family": raw.get("evolution_llm_family"),
                "planner_llm": raw.get("planner_llm"),
                "workload": raw.get("workload"),
            },
            device_price_per_second,
            currency,
            default_request_type,
        )

    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return run_dir, summary


@hydra.main(
    version_base=None,
    config_path="../experiments/11_scheduler_evaluation",
    config_name="config",
)
def main(config: DictConfig) -> None:
    try:
        run_dir, summary = run_from_config(config)
    except (OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"evaluation configuration error: {exc}") from exc
    print(f"run_dir: {run_dir}")
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
