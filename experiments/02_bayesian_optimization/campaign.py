"""Offline YOLO11m configuration discovery with Sobol and constrained qNEHVI.

This module intentionally produces an experiment artifact, never a scheduler
input or Profiling Database snapshot.  Heavy dependencies are imported only
when an actual campaign is run so the deterministic selection rules remain
testable without a GPU or model download.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import math
import platform
import random
import sys
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, order=True)
class Configuration:
    imgsz: int
    conf: float

    @property
    def key(self) -> str:
        return f"imgsz={self.imgsz},conf={self.conf:.8f}"


@dataclass(frozen=True)
class Observation:
    configuration: Configuration
    status: str
    box_ap: float | None
    warm_latency_ms: float | None
    peak_gpu_memory_mib: float | None
    error: str | None
    timestamp_utc: str
    provenance: dict[str, Any]

    @property
    def successful(self) -> bool:
        return self.status == "success"


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "model_id",
        "dataset",
        "search_space",
        "fixed_yolo_parameters",
        "campaign",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Configuration is missing required keys: {sorted(missing)}")
    config["fixed_yolo_parameters"] = normalize_yolo_parameters(
        config["fixed_yolo_parameters"]
    )
    return config


def normalize_yolo_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """Migrate Ultralytics' deprecated ``half`` option without changing precision."""
    normalized = dict(parameters)
    if "half" not in normalized:
        return normalized
    if "quantize" in normalized:
        raise ValueError(
            "fixed_yolo_parameters cannot contain both deprecated 'half' and 'quantize'"
        )
    normalized["quantize"] = "fp16" if normalized.pop("half") else "fp32"
    return normalized


def canonicalize(
    raw_imgsz: float, raw_conf: float, search_space: dict[str, Any]
) -> Configuration | None:
    """Quantize one raw point onto the frozen joint grid, or reject it."""
    min_size, max_size, step = (
        int(search_space[key]) for key in ("min_imgsz", "max_imgsz", "imgsz_step")
    )
    precision = max(
        0, -Decimal(str(search_space["conf_precision"])).as_tuple().exponent
    )
    conf_step = float(search_space["conf_precision"])
    imgsz = min_size + round((raw_imgsz - min_size) / step) * step
    conf = round(
        float(search_space["min_conf"])
        + round((raw_conf - float(search_space["min_conf"])) / conf_step) * conf_step,
        precision,
    )
    if not (
        min_size <= imgsz <= max_size
        and float(search_space["min_conf"]) <= conf <= float(search_space["max_conf"])
    ):
        return None
    candidate = Configuration(imgsz=imgsz, conf=conf)
    illegal = {
        Configuration(int(item["imgsz"]), round(float(item["conf"]), precision))
        for item in search_space.get("illegal_combinations", [])
    }
    return None if candidate in illegal else candidate


def legal_grid(search_space: dict[str, Any]) -> list[Configuration]:
    step = int(search_space["imgsz_step"])
    conf_step = float(search_space["conf_precision"])
    precision = max(0, len(str(search_space["conf_precision"]).split(".")[-1]))
    sizes = range(
        int(search_space["min_imgsz"]), int(search_space["max_imgsz"]) + 1, step
    )
    count = round(
        (float(search_space["max_conf"]) - float(search_space["min_conf"])) / conf_step
    )
    candidates = [
        canonicalize(
            size,
            round(float(search_space["min_conf"]) + index * conf_step, precision),
            search_space,
        )
        for size in sizes
        for index in range(count + 1)
    ]
    return sorted(candidate for candidate in candidates if candidate is not None)


def sobol_initial_candidates(
    grid: list[Configuration], count: int, seed: int
) -> list[Configuration]:
    if count > len(grid):
        raise ValueError("initial_points exceeds the number of legal configurations")
    # torch SobolEngine keeps this deterministic and avoids an optional BO dependency.
    import torch

    engine = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=seed)
    selected: list[Configuration] = []
    remaining = set(grid)
    min_size, max_size = (
        min(item.imgsz for item in grid),
        max(item.imgsz for item in grid),
    )
    min_conf, max_conf = (
        min(item.conf for item in grid),
        max(item.conf for item in grid),
    )
    while len(selected) < count:
        point = engine.draw(1)[0].tolist()
        candidate = min(
            remaining,
            key=lambda item: (
                ((item.imgsz - min_size) / max(1, max_size - min_size) - point[0]) ** 2
                + ((item.conf - min_conf) / max(1e-12, max_conf - min_conf) - point[1])
                ** 2,
                item,
            ),
        )
        selected.append(candidate)
        remaining.remove(candidate)
    return selected


def dominates(left: Observation, right: Observation) -> bool:
    assert left.successful and right.successful
    assert (
        left.box_ap is not None
        and left.warm_latency_ms is not None
        and left.peak_gpu_memory_mib is not None
    )
    assert (
        right.box_ap is not None
        and right.warm_latency_ms is not None
        and right.peak_gpu_memory_mib is not None
    )
    no_worse = (
        left.box_ap >= right.box_ap
        and left.warm_latency_ms <= right.warm_latency_ms
        and left.peak_gpu_memory_mib <= right.peak_gpu_memory_mib
    )
    strict = (
        left.box_ap > right.box_ap
        or left.warm_latency_ms < right.warm_latency_ms
        or left.peak_gpu_memory_mib < right.peak_gpu_memory_mib
    )
    return no_worse and strict


def pareto_frontier(observations: Iterable[Observation]) -> list[Observation]:
    successful = [observation for observation in observations if observation.successful]
    return sorted(
        (
            item
            for item in successful
            if not any(
                other is not item and dominates(other, item) for other in successful
            )
        ),
        key=lambda item: item.configuration,
    )


def _normalized_point(
    item: Observation, frontier: list[Observation]
) -> tuple[float, float, float]:
    assert (
        item.box_ap is not None
        and item.warm_latency_ms is not None
        and item.peak_gpu_memory_mib is not None
    )
    values = [(x.box_ap, -x.warm_latency_ms, -x.peak_gpu_memory_mib) for x in frontier]
    columns = list(zip(*values))
    result = []
    for value, column in zip(values[frontier.index(item)], columns):
        low, high = min(column), max(column)
        result.append(1.0 if high == low else (value - low) / (high - low))
    return tuple(result)  # type: ignore[return-value]


def _hypervolume(points: Iterable[tuple[float, float, float]]) -> float:
    # Exact union volume for boxes [0, point] in three normalized dimensions.
    unique = sorted(set(points))
    volume = 0.0
    for count in range(1, len(unique) + 1):
        sign = 1 if count % 2 else -1
        for group in itertools.combinations(unique, count):
            volume += sign * math.prod(
                min(point[axis] for point in group) for axis in range(3)
            )
    return volume


def select_configurations(
    observations: Iterable[Observation], limit: int = 10
) -> list[Observation]:
    frontier = pareto_frontier(observations)
    if len(frontier) <= limit:
        return frontier
    extrema = [
        max(
            frontier,
            key=lambda x: (
                x.box_ap,
                -x.warm_latency_ms,
                -x.peak_gpu_memory_mib,
                x.configuration,
            ),
        ),
        min(
            frontier,
            key=lambda x: (
                x.warm_latency_ms,
                -x.box_ap,
                x.peak_gpu_memory_mib,
                x.configuration,
            ),
        ),
        min(
            frontier,
            key=lambda x: (
                x.peak_gpu_memory_mib,
                -x.box_ap,
                x.warm_latency_ms,
                x.configuration,
            ),
        ),
    ]
    selected: list[Observation] = []
    for item in extrema:
        if item not in selected and len(selected) < limit:
            selected.append(item)
    normalized = {
        item.configuration: _normalized_point(item, frontier) for item in frontier
    }
    while len(selected) < limit:
        baseline = _hypervolume(normalized[item.configuration] for item in selected)
        remaining = [item for item in frontier if item not in selected]
        chosen = max(
            remaining,
            key=lambda item: (
                _hypervolume(
                    [
                        *(normalized[x.configuration] for x in selected),
                        normalized[item.configuration],
                    ]
                )
                - baseline,
                item.box_ap,
                -item.warm_latency_ms,
                -item.peak_gpu_memory_mib,
                tuple(-ord(ch) for ch in item.configuration.key),
            ),
        )
        selected.append(chosen)
    return sorted(selected, key=lambda item: item.configuration)


def _run_model(
    config: Configuration,
    frozen: dict[str, Any],
    image_paths: list[Path],
    coco: Any,
    category_by_name: dict[str, int],
) -> tuple[float, float, float, dict[str, Any]]:
    """Perform warm, batch-one end-to-end inference and evaluate COCO box AP."""
    import torch
    from pycocotools.cocoeval import COCOeval
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA to measure peak GPU memory")
    model = YOLO(str(frozen["model_id"]))
    checkpoint_path = Path(getattr(model, "ckpt_path", frozen["model_id"]))
    if not checkpoint_path.is_file():
        raise RuntimeError(
            "Ultralytics did not expose a local checkpoint file; cannot record its actual revision"
        )
    warmup_path = str(image_paths[0])
    kwargs = {
        **frozen["fixed_yolo_parameters"],
        "imgsz": config.imgsz,
        "conf": config.conf,
        "verbose": False,
    }
    for _ in range(int(frozen["campaign"]["warmup_runs"])):
        model.predict(warmup_path, **kwargs)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    # The latency observation is deliberately one warm, batch-one call. The
    # subsequent 100-image loop is the separate fixed-quality evaluation.
    result = model.predict(warmup_path, **kwargs)[0]
    torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - started) * 1000
    peak_mib = torch.cuda.max_memory_allocated() / (1024 * 1024)
    detections: list[dict[str, Any]] = []
    for result in [
        result,
        *(model.predict(str(path), **kwargs)[0] for path in image_paths[1:]),
    ]:
        image_id = int(Path(result.path).stem)
        for box in result.boxes:
            cls_name = str(result.names[int(box.cls.item())])
            if cls_name not in category_by_name:
                continue
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
            detections.append(
                {
                    "image_id": image_id,
                    "category_id": category_by_name[cls_name],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(box.conf.item()),
                }
            )
    coco_dt = coco.loadRes(detections)
    evaluator = COCOeval(coco, coco_dt, "bbox")
    evaluator.params.imgIds = sorted(int(path.stem) for path in image_paths)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return (
        float(evaluator.stats[0]),
        latency_ms,
        peak_mib,
        {
            "detections": len(detections),
            "latency_image_id": int(image_paths[0].stem),
            "coco_evaluator": "pycocotools.COCOeval",
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "declared_model_revision": frozen.get("model_revision"),
        },
    )


def proposal_qnehvi(
    grid: list[Configuration], observations: list[Observation], frozen: dict[str, Any]
) -> tuple[Configuration, float]:
    """Choose one legal, unmeasured, conservatively non-OOM point with qNEHVI."""
    try:
        import torch
        from botorch.acquisition.multi_objective.monte_carlo import (
            qNoisyExpectedHypervolumeImprovement,
        )
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import ModelListGP, SingleTaskGP
        from botorch.models.transforms.outcome import Standardize
        from botorch.sampling.normal import SobolQMCNormalSampler
        from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
    except ImportError as error:
        raise RuntimeError(
            "Install the `bayesian-optimization` dependency group before adaptive sampling."
        ) from error
    successful = [item for item in observations if item.successful]
    if len(successful) < 2:
        raise RuntimeError(
            "At least two successful observations are required for qNEHVI"
        )
    measured = {item.configuration for item in observations}
    oom_sizes = [
        item.configuration.imgsz for item in observations if item.status == "oom"
    ]
    eligible = [
        item
        for item in grid
        if item not in measured and (not oom_sizes or item.imgsz < min(oom_sizes))
    ]
    if not eligible:
        raise RuntimeError(
            "No unmeasured legal candidates remain after the conservative OOM filter"
        )
    min_size, max_size = min(x.imgsz for x in grid), max(x.imgsz for x in grid)
    min_conf, max_conf = min(x.conf for x in grid), max(x.conf for x in grid)

    def vector(item: Configuration) -> list[float]:
        return [
            (item.imgsz - min_size) / max(1, max_size - min_size),
            (item.conf - min_conf) / max(1e-12, max_conf - min_conf),
        ]

    train_x = torch.tensor(
        [vector(item.configuration) for item in successful], dtype=torch.double
    )
    train_y = torch.tensor(
        [
            [item.box_ap, -item.warm_latency_ms, -item.peak_gpu_memory_mib]
            for item in successful
        ],
        dtype=torch.double,
    )
    noise = float(frozen["campaign"]["observation_noise_variance"])
    train_yvar = torch.full_like(train_y, noise)
    models = [
        SingleTaskGP(
            train_x,
            train_y[:, index : index + 1],
            train_Yvar=train_yvar[:, index : index + 1],
            outcome_transform=Standardize(m=1),
        )
        for index in range(3)
    ]
    model = ModelListGP(*models)
    fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))
    ref_point = train_y.min(dim=0).values - 0.01 * train_y.abs().max(
        dim=0
    ).values.clamp_min(1.0)
    acquisition = qNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        X_baseline=train_x,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([128])),
    )
    scores = [
        (
            float(acquisition(torch.tensor([vector(item)], dtype=torch.double)).item()),
            item,
        )
        for item in eligible
    ]
    score, candidate = max(
        scores, key=lambda pair: (pair[0], tuple(-ord(ch) for ch in pair[1].key))
    )
    return candidate, score


def selected_sample_ids(config: dict[str, Any]) -> list[int]:
    from pycocotools.coco import COCO

    dataset = config["dataset"]
    panoptic_path = Path(dataset["root"]) / dataset["panoptic_annotations"]
    instance_path = Path(dataset["root"]) / dataset["instance_annotations"]
    missing = [path for path in (panoptic_path, instance_path) if not path.is_file()]
    if missing:
        expected = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "COCO annotations required for this campaign are missing: " + expected
        )
    panoptic = json.loads(panoptic_path.read_text(encoding="utf-8"))
    ids = sorted(int(item["image_id"]) for item in panoptic["annotations"])
    if len(ids) < int(dataset["sample_count"]):
        raise ValueError("Panoptic annotations contain fewer images than sample_count")
    # Open the instances annotation now: it is the actual COCO box AP ground truth.
    COCO(str(instance_path))
    return sorted(
        random.Random(int(dataset["sample_seed"])).sample(
            ids, int(dataset["sample_count"])
        )
    )


def write_campaign(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "campaign.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(
    config: dict[str, Any], output_root: Path, campaign_id: str | None = None
) -> Path:
    campaign = config["campaign"]
    grid = legal_grid(config["search_space"])
    budget, initial_count = int(campaign["budget"]), int(campaign["initial_points"])
    if not 2 <= initial_count <= budget <= len(grid):
        raise ValueError(
            "Require 2 <= initial_points <= budget <= number of legal grid points"
        )
    if int(campaign["batch_size"]) != 1:
        raise ValueError(
            "This first implementation deliberately supports one GPU worker only; set batch_size to 1."
        )
    campaign_id = (
        campaign_id or f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    )
    output_dir = output_root / campaign_id
    if output_dir.exists():
        raise FileExistsError(f"Campaign directory already exists: {output_dir}")
    sample_ids = selected_sample_ids(config)
    dataset = config["dataset"]
    annotation_path = Path(dataset["root"]) / dataset["instance_annotations"]
    panoptic_path = Path(dataset["root"]) / dataset["panoptic_annotations"]
    image_paths = [
        Path(dataset["root"]) / dataset["images"] / f"{image_id:012d}.jpg"
        for image_id in sample_ids
    ]
    missing = [str(path) for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Selected COCO images are missing (first: {missing[0]})"
        )
    import torch
    import ultralytics
    from pycocotools.coco import COCO

    coco = COCO(str(annotation_path))
    category_by_name = {
        category["name"]: int(category["id"])
        for category in coco.loadCats(coco.getCatIds())
    }
    proposals = sobol_initial_candidates(grid, initial_count, int(campaign["seed"]))
    observations: list[Observation] = []
    rounds: list[dict[str, Any]] = []
    while len(observations) < budget:
        source = "sobol" if proposals else "qnehvi"
        acquisition = None
        if proposals:
            candidate = proposals.pop(0)
        else:
            candidate, acquisition = proposal_qnehvi(grid, observations, config)
        try:
            box_ap, latency, memory, provenance = _run_model(
                candidate, config, image_paths, coco, category_by_name
            )
            observation = Observation(
                candidate,
                "success",
                box_ap,
                latency,
                memory,
                None,
                utc_now(),
                provenance,
            )
        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            observation = Observation(
                candidate, "oom", None, None, None, str(error), utc_now(), {}
            )
        except (
            Exception
        ) as error:  # Persist every bad candidate instead of silently skipping it.
            observation = Observation(
                candidate,
                "error",
                None,
                None,
                None,
                "".join(traceback.format_exception_only(type(error), error)).strip(),
                utc_now(),
                {},
            )
        observations.append(observation)
        rounds.append(
            {
                "round": len(rounds) + 1,
                "source": source,
                "configuration": asdict(candidate),
                "acquisition_value": acquisition,
                "successful_training_points": sum(
                    item.successful for item in observations
                ),
            }
        )
        payload = {
            "experiment": "yolo11m-bayesian-optimization-demo",
            "campaign_id": campaign_id,
            "created_at_utc": utc_now(),
            "frozen_config": config,
            "sample_image_ids": sample_ids,
            "quality_protocol": {
                "raw_metric": "COCO detection box AP@[.50:.95]",
                "panoptic_source": str(panoptic_path),
                "panoptic_source_sha256": file_sha256(panoptic_path),
                "instance_annotations": str(annotation_path),
                "instance_annotations_sha256": file_sha256(annotation_path),
                "category_name_to_coco_thing_id": category_by_name,
                "evaluator": "pycocotools.COCOeval",
                "pycocotools_version": importlib.metadata.version("pycocotools"),
            },
            "observations": [asdict(item) for item in observations],
            "rounds": rounds,
            "pareto_frontier": [asdict(item) for item in pareto_frontier(observations)],
            "selected_configurations": [
                asdict(item) for item in select_configurations(observations)
            ],
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None,
                "ultralytics": ultralytics.__version__,
            },
            "model_assumptions": {
                "observation_noise_variance": campaign["observation_noise_variance"],
                "noise_note": "Assumed variance; no repeated measurement was collected.",
                "oom_filter": "After an OOM, candidates at or above the smallest OOM imgsz are conservatively excluded.",
            },
            "scope_note": "Demo artifact only: warm_latency_ms is not p95 and peak_gpu_memory_mib is not formal execution energy.",
        }
        if not output_dir.exists():
            write_campaign(output_dir, payload)
        else:
            (output_dir / "campaign.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/02_bayesian_optimization/config.example.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/02_bayesian_optimization/campaigns"),
    )
    parser.add_argument("--campaign-id")
    args = parser.parse_args()
    print(
        f"Wrote campaign to {run(load_config(args.config), args.output_root, args.campaign_id)}"
    )


if __name__ == "__main__":
    main()
