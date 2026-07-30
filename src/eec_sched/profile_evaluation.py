"""Reproducible five-sample quality profiling and JSON profile artifacts.

Importing this module is intentionally cheap.  Dataset/evaluator packages are
loaded by callers only when a profiling command is actually executed.
"""

from __future__ import annotations

import json
import platform
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping, Sequence

from .domain import Configuration, ToolRunner
from .profiles import AccuracyProfile, LatencyProfile


@dataclass(frozen=True)
class EvaluationSample:
    """One public corpus item, with its stable dataset identifier and target."""

    sample_id: str
    inputs: Mapping[str, object]
    target: object


@dataclass(frozen=True)
class EvaluationSuite:
    tool_id: str
    dataset: str
    revision: str
    split: str
    metric_name: str
    higher_is_better: bool
    scorer_version: str


@dataclass(frozen=True)
class ProfileArtifact:
    """Portable profile data plus the provenance required to reproduce it."""

    suite: EvaluationSuite
    sample_ids: tuple[str, ...]
    model_name: str
    runtime: Mapping[str, str]
    accuracy: tuple[AccuracyProfile, ...]
    latency: tuple[LatencyProfile, ...]
    schema_version: int = 1


def select_fixed_samples(samples: Iterable[EvaluationSample], count: int = 5) -> tuple[EvaluationSample, ...]:
    """Select the first ``count`` corpus records and reject ambiguous IDs.

    Dataset iteration order is the pinned split order.  Persisting these IDs in
    the artifact means reruns use the exact same records even if a loader later
    changes its default ordering.
    """
    selected: list[EvaluationSample] = []
    seen: set[str] = set()
    for sample in samples:
        if sample.sample_id in seen:
            raise ValueError(f"duplicate evaluation sample ID: {sample.sample_id}")
        selected.append(sample)
        seen.add(sample.sample_id)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"evaluation suite requires exactly {count} samples; received {len(selected)}")
    return tuple(selected)


def load_persisted_samples(samples: Iterable[EvaluationSample], sample_ids: Sequence[str]) -> tuple[EvaluationSample, ...]:
    """Resolve a persisted selection without changing its order."""
    by_id = {sample.sample_id: sample for sample in samples}
    missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
    if missing:
        raise ValueError(f"persisted evaluation samples are absent from the pinned split: {missing}")
    return tuple(by_id[sample_id] for sample_id in sample_ids)


def load_huggingface_samples(
    *,
    dataset: str,
    revision: str,
    split: str,
    to_sample: Callable[[Mapping[str, object], int], EvaluationSample],
    persisted_ids: Sequence[str] = (),
) -> tuple[EvaluationSample, ...]:
    """Load a pinned public split lazily and select/resolve its five samples.

    ``to_sample`` owns each corpus's field names and must make the sample ID
    stable (for example, a COCO image ID rather than the local row offset).
    """
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("profiling data requires the optional 'datasets' package") from error
    rows = load_dataset(dataset, revision=revision, split=split)
    available = (to_sample(dict(row), index) for index, row in enumerate(rows))
    return load_persisted_samples(available, persisted_ids) if persisted_ids else select_fixed_samples(available)


def evaluate_configuration(
    runner: ToolRunner,
    configuration: Configuration,
    samples: Sequence[EvaluationSample],
    scorer: Callable[[Sequence[object], Sequence[object]], float],
) -> float:
    """Run one configuration and return only the suite's standard raw score."""
    if len(samples) != 5:
        raise ValueError("quality profiling requires the persisted five samples")
    runner.prepare(configuration)
    predictions: list[object] = []
    targets: list[object] = []
    for sample in samples:
        public_output = runner.run(sample.inputs, configuration)
        # Segmentation/OCR evaluators can consume private details when supplied;
        # no private value is returned to the planner or DAG executor.
        details = getattr(runner, "evaluation_output", lambda: {})()
        predictions.append({"output": public_output, "evaluation": details})
        targets.append(sample.target)
    return float(scorer(predictions, targets))


def profile_suite(
    *,
    suite: EvaluationSuite,
    model_name: str,
    configurations: Sequence[Configuration],
    runner_factory: Callable[[], ToolRunner],
    samples: Sequence[EvaluationSample],
    scorer: Callable[[Sequence[object], Sequence[object]], float],
    device: str,
    latency_input_bucket: str = "evaluation-five-samples",
    latency_observations_per_sample: int = 10,
) -> ProfileArtifact:
    """Evaluate and warm-profile every finite configuration on five inputs."""
    if len(samples) != 5:
        raise ValueError("quality profiling requires exactly five fixed samples")
    if not configurations:
        raise ValueError("a profile suite needs at least one configuration")
    accuracy: list[AccuracyProfile] = []
    latency: list[LatencyProfile] = []
    for configuration in configurations:
        runner = runner_factory()
        raw_metric = evaluate_configuration(runner, configuration, samples, scorer)
        accuracy.append(AccuracyProfile(suite.tool_id, configuration.configuration_id, raw_metric, model_name, dict(configuration.parameters)))
        latency.append(profile_warm_latency_samples(
            suite.tool_id, model_name, configuration, device, runner, samples,
            observations_per_sample=latency_observations_per_sample,
            input_bucket=latency_input_bucket,
        ))
    return ProfileArtifact(suite, tuple(sample.sample_id for sample in samples), model_name, runtime_metadata(device), tuple(accuracy), tuple(latency))


def runtime_metadata(device: str) -> dict[str, str]:
    """Discover runtime identity without requiring CUDA to be installed."""
    metadata = {"device": device, "platform": platform.platform()}
    try:
        import torch
        metadata.update({"torch_version": str(torch.__version__), "cuda_version": str(torch.version.cuda or "none")})
        if torch.cuda.is_available():
            metadata.update({"gpu_model": torch.cuda.get_device_name(0), "gpu_memory_bytes": str(torch.cuda.get_device_properties(0).total_memory)})
    except ImportError:
        metadata.update({"torch_version": "unavailable", "cuda_version": "unavailable"})
    return metadata


def profile_warm_latency_samples(
    tool_id: str,
    model_name: str,
    configuration: Configuration,
    device: str,
    runner: ToolRunner,
    samples: Sequence[EvaluationSample],
    *,
    observations_per_sample: int = 10,
    input_bucket: str = "evaluation-five-samples",
) -> LatencyProfile:
    """Measure the required 5 × 10 warm end-to-end observations as one record."""
    if len(samples) != 5 or observations_per_sample < 1:
        raise ValueError("latency profiling requires five samples and positive observations")
    runner.prepare(configuration)
    timings: list[float] = []
    for sample in samples:
        for _ in range(observations_per_sample):
            started = perf_counter()
            runner.run(sample.inputs, configuration)
            timings.append((perf_counter() - started) * 1000)
    ordered = sorted(timings)
    p95_index = min(len(ordered) - 1, max(0, round(0.95 * len(ordered)) - 1))
    return LatencyProfile(tool_id, configuration.configuration_id, device, input_bucket, median(timings), ordered[p95_index], len(timings), model_name, dict(configuration.parameters))


def save_profile_artifact(path: Path, artifact: ProfileArtifact) -> None:
    """Atomically persist profiles, five selected IDs, and replay metadata."""
    payload = asdict(artifact)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as file:
        json.dump(payload, file, sort_keys=True, indent=2)
        file.write("\n")
        temporary = Path(file.name)
    temporary.replace(path)


def load_profile_artifact(path: Path) -> ProfileArtifact:
    payload = json.loads(path.read_text(encoding="utf-8"))
    suite = EvaluationSuite(**payload["suite"])
    accuracy = tuple(AccuracyProfile(**profile) for profile in payload["accuracy"])
    latency = tuple(LatencyProfile(**profile) for profile in payload["latency"])
    return ProfileArtifact(suite, tuple(payload["sample_ids"]), payload["model_name"], payload["runtime"], accuracy, latency, payload.get("schema_version", 1))
