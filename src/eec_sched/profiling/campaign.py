"""Configuration-driven Bayesian optimisation for the MnMS profiling contract.

The module deliberately keeps the campaign independent from the scheduler.  A
campaign evaluates a finite, joint configuration space and writes an artifact;
it does not expose continuous parameters to the scheduler.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol


ACTIVE_MODELS: dict[str, dict[str, Any]] = {
    "text_summarization": {"model": "facebook/bart-large-cnn", "metric": "rougeL", "direction": "maximize", "dataset": "cnn_dailymail", "split": "test"},
    "text_classification": {"model": "distilbert-base-uncased-finetuned-sst-2-english", "metric": "accuracy", "direction": "maximize", "dataset": "glue/sst2", "split": "validation"},
    "question_answering": {"model": "deepset/roberta-base-squad2", "metric": "f1", "direction": "maximize", "dataset": "squad_v2", "split": "validation"},
    "automatic_speech_recognition": {"model": "openai/whisper-small", "metric": "wer", "direction": "minimize", "dataset": "librispeech_asr", "split": "test.clean"},
    "image_captioning": {"model": "Salesforce/blip-image-captioning-large", "metric": "CIDEr", "direction": "maximize", "dataset": "coco2014", "split": "validation"},
    "image_classification": {"model": "google/vit-base-patch16-224", "metric": "top1_accuracy", "direction": "maximize", "dataset": "imagenet-1k", "split": "validation"},
    "visual_question_answering": {"model": "Salesforce/blip-vqa-base", "metric": "VQA_accuracy", "direction": "maximize", "dataset": "vqav2", "split": "validation"},
    "object_detection": {"model": "facebook/detr-resnet-101", "metric": "box_AP", "direction": "maximize", "dataset": "coco2017", "split": "validation"},
    "image_segmentation": {"model": "facebook/maskformer-swin-base-coco", "metric": "mask_AP", "direction": "maximize", "dataset": "coco2017", "split": "validation"},
    "optical_character_recognition": {"model": "easyocr:multilingual", "metric": "MLT19_Hmean", "direction": "maximize", "dataset": "icdar2019_mlt", "split": "train"},
}

DISABLED_MODELS = {"text_generation", "image_generation", "image_editing"}


@dataclass(frozen=True, order=True)
class Candidate:
    values: tuple[tuple[str, Any], ...]

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "Candidate":
        return cls(tuple(sorted(values.items())))

    def as_dict(self) -> dict[str, Any]:
        return dict(self.values)

    @property
    def key(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))


@dataclass
class Measurement:
    candidate: Candidate
    status: str
    quality: float | None
    latency_ms: float | None
    resource_value: float | None
    error: str | None
    provenance: dict[str, Any]
    timestamp_utc: str


class Evaluator(Protocol):
    def __call__(self, candidate: Candidate, samples: list[dict[str, Any]], model: dict[str, Any]) -> tuple[float, float, float, dict[str, Any]]: ...


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def validate_config(config: dict[str, Any]) -> None:
    for key in ("dataset_dir", "models", "campaign"):
        if key not in config:
            raise ValueError(f"config is missing {key!r}")
    if not config["models"]:
        raise ValueError("models must select at least one model")
    unknown = set(config["models"]) - set(ACTIVE_MODELS)
    disabled = set(config["models"]) & DISABLED_MODELS
    if unknown or disabled:
        raise ValueError(f"models must be active MnMS models; unknown={sorted(unknown)}, disabled={sorted(disabled)}")
    campaign = config["campaign"]
    budget = int(campaign.get("budget", 0))
    initial_points = int(campaign.get("initial_points", 0))
    if budget < 100:
        raise ValueError("campaign.budget must be at least 100 for profiling data")
    if initial_points < 1 or initial_points > budget:
        raise ValueError("campaign.initial_points must be between 1 and campaign.budget")


def _value_grid(spec: dict[str, Any]) -> list[Any]:
    if "values" in spec:
        return list(spec["values"])
    low, high, step = float(spec["min"]), float(spec["max"]), float(spec["step"])
    if step <= 0 or high < low:
        raise ValueError(f"invalid search-space dimension: {spec}")
    values: list[Any] = []
    value = low
    while value <= high + step * 1e-8:
        values.append(int(round(value)) if spec.get("type") == "int" else round(value, int(spec.get("precision", 8))))
        value += step
    return values


def legal_candidates(space: dict[str, dict[str, Any]]) -> list[Candidate]:
    names = sorted(name for name in space if name != "illegal_combinations")
    candidates = [Candidate.from_dict(dict(zip(names, values))) for values in __import__("itertools").product(*[_value_grid(space[name]) for name in names])]
    illegal = {Candidate.from_dict(item) for item in space.get("illegal_combinations", [])} if isinstance(space.get("illegal_combinations"), list) else set()
    return [candidate for candidate in candidates if candidate not in illegal]


def sobol_candidates(grid: list[Candidate], count: int, seed: int) -> list[Candidate]:
    if count > len(grid):
        raise ValueError("initial_points exceeds legal search space")
    # A seeded shuffle is deterministic and dependency-free.  The optional
    # torch Sobol design is used when available for better space filling.
    try:
        import torch
        points = torch.quasirandom.SobolEngine(len(grid[0].values), scramble=True, seed=seed).draw(max(count * 4, count)).tolist()
        order = sorted(range(len(grid)), key=lambda i: (sum(points[i % len(points)]), grid[i].key))
        return [grid[i] for i in order[:count]]
    except ImportError:
        result = list(grid)
        random.Random(seed).shuffle(result)
        return result[:count]


def _continuous_dimensions(space: dict[str, dict[str, Any]]) -> list[str]:
    names = sorted(name for name in space if name != "illegal_combinations")
    if not names:
        raise ValueError("search_space must contain at least one dimension")
    for name in names:
        spec = space[name]
        if "min" not in spec or "max" not in spec:
            raise ValueError(f"continuous search dimensions require min/max: {name}")
        if float(spec["max"]) < float(spec["min"]):
            raise ValueError(f"search-space max is below min: {name}")
    return names


def canonicalize_candidate(raw: dict[str, float], space: dict[str, dict[str, Any]]) -> Candidate | None:
    """Canonicalize a continuous BO point without requiring a finite grid."""
    values: dict[str, Any] = {}
    for name in _continuous_dimensions(space):
        spec = space[name]
        value = min(float(spec["max"]), max(float(spec["min"]), float(raw[name])))
        if spec.get("type") == "int":
            values[name] = int(round(value))
        else:
            values[name] = round(value, int(spec.get("precision", 8)))
    candidate = Candidate.from_dict(values)
    illegal = {Candidate.from_dict(item) for item in space.get("illegal_combinations", [])}
    return None if candidate in illegal else candidate


def continuous_initial_candidates(space: dict[str, dict[str, Any]], count: int, seed: int) -> list[Candidate]:
    names = _continuous_dimensions(space)
    selected: list[Candidate] = []
    seen: set[Candidate] = set()
    try:
        import torch
        points = torch.quasirandom.SobolEngine(len(names), scramble=True, seed=seed).draw(max(count * 4, count)).tolist()
    except ImportError:
        points = [[random.Random(seed + index).random() for _ in names] for index in range(max(count * 4, count))]
    for point in points:
        raw = {name: float(space[name]["min"]) + value * (float(space[name]["max"]) - float(space[name]["min"])) for name, value in zip(names, point)}
        candidate = canonicalize_candidate(raw, space)
        if candidate is not None and candidate not in seen:
            selected.append(candidate)
            seen.add(candidate)
        if len(selected) == count:
            return selected
    raise ValueError(f"could not canonicalize {count} distinct initial candidates")


def propose_continuous(space: dict[str, dict[str, Any]], measurements: list[Measurement], config: dict[str, Any]) -> tuple[Candidate, float | None]:
    """Propose a canonical point from continuous bounds."""
    names = _continuous_dimensions(space)
    measured = {item.candidate for item in measurements}
    seed = int(config["campaign"].get("seed", 0)) + len(measurements)
    try:
        import torch
        from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import ModelListGP, SingleTaskGP
        from botorch.models.transforms.outcome import Standardize
        from botorch.optim import optimize_acqf
        from botorch.sampling.normal import SobolQMCNormalSampler
        from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
        successful = [item for item in measurements if item.status == "success"]
        if len(successful) >= 2:
            mins = {name: float(space[name]["min"]) for name in names}
            spans = {name: max(1e-12, float(space[name]["max"]) - mins[name]) for name in names}
            train_x = torch.tensor([[((float(item.candidate.as_dict()[name]) - mins[name]) / spans[name]) for name in names] for item in successful], dtype=torch.double)
            train_y = torch.tensor([[item.quality, -item.latency_ms, -item.resource_value] for item in successful], dtype=torch.double)
            noise = float(config["campaign"].get("observation_noise_variance", 1e-4))
            models = [SingleTaskGP(train_x, train_y[:, i:i + 1], train_Yvar=torch.full((len(successful), 1), noise, dtype=torch.double), outcome_transform=Standardize(m=1)) for i in range(3)]
            model = ModelListGP(*models)
            fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))
            acq = qNoisyExpectedHypervolumeImprovement(model=model, ref_point=(train_y.min(0).values - 0.01).tolist(), X_baseline=train_x, sampler=SobolQMCNormalSampler(sample_shape=torch.Size([128])))
            point, value = optimize_acqf(acq, bounds=torch.tensor([[0.0] * len(names), [1.0] * len(names)], dtype=torch.double), q=1, num_restarts=5, raw_samples=64)
            normalized = point[0].tolist()
            raw = {name: mins[name] + normalized[index] * spans[name] for index, name in enumerate(names)}
            candidate = canonicalize_candidate(raw, space)
            if candidate is not None and candidate not in measured:
                return candidate, float(value.item())
    except (ImportError, RuntimeError, ValueError, IndexError):
        pass
    for offset in range(1000):
        candidate = continuous_initial_candidates(space, len(measurements) + offset + 1, seed)[-1]
        if candidate not in measured:
            return candidate, None
    raise RuntimeError("could not find an unmeasured canonical continuous candidate")


def _fallback_proposal(grid: list[Candidate], measured: set[Candidate], seed: int) -> Candidate:
    remaining = [candidate for candidate in grid if candidate not in measured]
    if not remaining:
        raise RuntimeError("no legal unmeasured candidates remain")
    return random.Random(seed + len(measured)).choice(remaining)


def propose(grid: list[Candidate], measurements: list[Measurement], config: dict[str, Any]) -> tuple[Candidate, float | None]:
    measured = {item.candidate for item in measurements}
    successful = [item for item in measurements if item.status == "success"]
    if len(successful) < 2:
        return _fallback_proposal(grid, measured, int(config["campaign"].get("seed", 0))), None
    try:
        import torch
        from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import ModelListGP, SingleTaskGP
        from botorch.models.transforms.outcome import Standardize
        from botorch.sampling.normal import SobolQMCNormalSampler
        from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
        names = sorted(successful[0].candidate.as_dict())
        values = {name: sorted({float(c.as_dict()[name]) for c in grid}) for name in names}
        def vector(candidate: Candidate) -> list[float]:
            data = candidate.as_dict()
            return [(float(data[name]) - min(values[name])) / max(1e-12, max(values[name]) - min(values[name])) for name in names]
        train_x = torch.tensor([vector(item.candidate) for item in successful], dtype=torch.double)
        train_y = torch.tensor([[item.quality, -item.latency_ms, -item.resource_value] for item in successful], dtype=torch.double)
        noise = float(config["campaign"].get("observation_noise_variance", 1e-4))
        models = [SingleTaskGP(train_x, train_y[:, i:i + 1], train_Yvar=torch.full((len(successful), 1), noise, dtype=torch.double), outcome_transform=Standardize(m=1)) for i in range(3)]
        model = ModelListGP(*models)
        fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))
        acq = qNoisyExpectedHypervolumeImprovement(model=model, ref_point=(train_y.min(0).values - 0.01).tolist(), X_baseline=train_x, sampler=SobolQMCNormalSampler(sample_shape=torch.Size([128])))
        remaining = [candidate for candidate in grid if candidate not in measured]
        scored = [(float(acq(torch.tensor([vector(candidate)], dtype=torch.double)).item()), candidate) for candidate in remaining]
        score, candidate = max(scored, key=lambda item: (item[0], item[1].key))
        return candidate, score
    except (ImportError, RuntimeError, ValueError):
        return _fallback_proposal(grid, measured, int(config["campaign"].get("seed", 0))), None


def default_evaluator(candidate: Candidate, samples: list[dict[str, Any]], model: dict[str, Any]) -> tuple[float, float, float, dict[str, Any]]:
    """Run a model-specific adapter supplied by the selected model config.

    Keeping the adapter callable in config is intentionally not supported: a
    campaign must be serializable.  Implementations can register an evaluator
    in ``EVALUATORS`` or provide ``adapter`` as a Python import path.
    """
    adapter = model.get("adapter")
    if not adapter:
        raise RuntimeError(f"no evaluator adapter configured for {model['tool']}")
    module_name, function_name = adapter.rsplit(":", 1)
    module = __import__(module_name, fromlist=[function_name])
    return getattr(module, function_name)(candidate, samples, model)


def run_model(tool: str, model: dict[str, Any], samples: list[dict[str, Any]], config: dict[str, Any], output: Path, evaluator: Evaluator = default_evaluator) -> Path:
    if evaluator is default_evaluator and config["campaign"].get("resource_metric") != "peak_gpu_memory_mib":
        raise RuntimeError(
            "campaign.resource_metric must be 'peak_gpu_memory_mib' for this "
            "experiment"
        )
    campaign = config["campaign"]
    budget = int(campaign["budget"])
    initial = continuous_initial_candidates(model["search_space"], min(int(campaign["initial_points"]), budget), int(campaign.get("seed", 0)))
    measurements: list[Measurement] = []
    rounds: list[dict[str, Any]] = []
    for index in range(budget):
        source = "sobol" if initial else "qnehvi"
        acquisition = None
        if initial:
            candidate = initial.pop(0)
        else:
            candidate, acquisition = propose_continuous(model["search_space"], measurements, config)
        started = time.perf_counter()
        try:
            quality, latency, resource_value, provenance = evaluator(candidate, samples, model)
            if not math.isfinite(float(quality)) or not math.isfinite(float(latency)):
                raise RuntimeError("evaluator returned a non-finite quality or latency")
            if float(resource_value) <= 0:
                raise RuntimeError(
                    "evaluator did not return a positive resource value"
                )
            measurement = Measurement(candidate, "success", float(quality), float(latency), float(resource_value), None, provenance, utc_now())
        except Exception as error:
            measurement = Measurement(candidate, "error", None, None, None, "".join(traceback.format_exception_only(type(error), error)).strip(), {}, utc_now())
        measurements.append(measurement)
        rounds.append({"round": index + 1, "source": source, "configuration": candidate.as_dict(), "acquisition_value": acquisition, "elapsed_seconds": time.perf_counter() - started})
    serialized_observations = [asdict(item) for item in measurements]
    results = [
        {
            "configuration": dict(item["candidate"]["values"]),
            "quality": item["quality"],
            "quality_metric": model["metric"],
            "accuracy": item["quality"] if model["metric"] == "accuracy" else None,
            "latency_ms": item["latency_ms"],
            "gpu_memory_mib": item["resource_value"],
            "status": item["status"],
        }
        for item in serialized_observations
        if item["status"] == "success"
    ]
    retain_count = int(campaign.get("retain_count", 100))
    selected = results[:retain_count]
    payload = {"experiment": "mnms-bayesian-optimization", "tool": tool, "model": model["model"], "frozen_config": config, "evaluation_sample_ids": [item["id"] for item in samples], "observations": serialized_observations, "results": results, "selected_results": selected, "rounds": rounds, "campaign_status": "ready_for_candidate_discovery" if len(selected) >= retain_count else "incomplete", "publishable_for_profiling_database": False, "environment": {"python": platform.python_version(), "platform": platform.platform()}, "created_at_utc": utc_now()}
    target = output / tool / "campaign.json"
    target.parent.mkdir(parents=True, exist_ok=False)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def load_samples(dataset_dir: Path, tool: str, count: int = 100) -> list[dict[str, Any]]:
    if count < 1:
        raise ValueError("evaluation sample count must be positive")
    benchmark_name = {
        "text_summarization": "cnn_dailymail",
        "text_classification": "sst2",
        "question_answering": "squad_v2",
        "automatic_speech_recognition": "librispeech_test_clean",
        "image_classification": "imagenet_1k",
        "visual_question_answering": "vqav2",
        "optical_character_recognition": "mlt19",
        "image_captioning": "image_captioning",
        "object_detection": "object_detection",
        "image_segmentation": "image_segmentation",
    }.get(tool, tool)
    path = dataset_dir / f"{tool}.jsonl"
    if not path.is_file():
        path = dataset_dir / benchmark_name / "samples.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing fixed evaluation subset under {dataset_dir}: {path}; run the benchmark download/preparation commands first")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < count:
        raise ValueError(f"{path} must contain at least {count} samples, found {len(rows)}")
    rows = rows[:count]
    if any("id" not in row for row in rows):
        raise ValueError(f"{path} contains a sample without an id")
    portable_root = dataset_dir.resolve()
    for row in rows:
        for field in ("image", "audio", "annotations", "ground_truth"):
            value = row.get(field)
            if isinstance(value, str) and not Path(value).is_absolute():
                row[field] = str((portable_root / value).resolve())
    return rows


def write_unavailable(tool: str, model: dict[str, Any], config: dict[str, Any], output: Path, error: Exception) -> Path:
    """Persist a failed preflight as an artifact instead of aborting the campaign."""
    target = output / tool / "campaign.json"
    target.parent.mkdir(parents=True, exist_ok=False)
    payload = {
        "experiment": "mnms-bayesian-optimization",
        "tool": tool,
        "model": model["model"],
        "frozen_config": config,
        "evaluation_sample_ids": [],
        "observations": [],
        "results": [],
        "selected_results": [],
        "rounds": [],
        "campaign_status": "blocked",
        "error": f"{type(error).__name__}: {error}",
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "created_at_utc": utc_now(),
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def run(config: dict[str, Any], output: Path) -> list[Path]:
    validate_config(config)
    dataset_dir = Path(config["dataset_dir"])
    campaign_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    campaign_root = output / campaign_id
    paths: list[Path] = []
    for tool in config["models"]:
        model = {**ACTIVE_MODELS[tool], **config.get("model_overrides", {}).get(tool, {}), "tool": tool}
        try:
            samples = load_samples(dataset_dir, tool, int(config.get("evaluation_samples", 100)))
            paths.append(run_model(tool, model, samples, config, campaign_root))
        except Exception as error:
            paths.append(write_unavailable(tool, model, config, campaign_root, error))
    return paths
