"""Deterministic arrival workloads for multi-request evaluation.

The evaluator consumes an ordered tuple of ``(arrival_time_ms,
EvaluationTrace)`` pairs.  This module owns only workload generation and its
portable arrival manifest; it deliberately does not know anything about
resource simulation or scoring.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .candidate import EvaluationTrace


def _validate_generation_inputs(
    traces: Sequence[EvaluationTrace],
    request_rate_per_second: float,
    observation_window_ms: float,
    seed: int,
) -> tuple[EvaluationTrace, ...]:
    """Validate and freeze the small set of inputs used by the generator."""
    templates = tuple(traces)
    if not templates:
        raise ValueError("at least one trace template is required")
    if any(not isinstance(trace, EvaluationTrace) for trace in templates):
        raise TypeError("trace templates must be EvaluationTrace instances")

    template_ids = [trace.trace_id for trace in templates]
    if len(set(template_ids)) != len(template_ids):
        raise ValueError("trace template IDs must be unique")

    if isinstance(request_rate_per_second, bool) or not isinstance(request_rate_per_second, (int, float)):
        raise TypeError("request_rate_per_second must be a number")
    if not math.isfinite(float(request_rate_per_second)) or request_rate_per_second <= 0:
        raise ValueError("request_rate_per_second must be positive and finite")

    if isinstance(observation_window_ms, bool) or not isinstance(observation_window_ms, (int, float)):
        raise TypeError("observation_window_ms must be a number")
    if not math.isfinite(float(observation_window_ms)) or observation_window_ms <= 0:
        raise ValueError("observation_window_ms must be positive and finite")

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    return templates


def _request_id(template_id: str, ordinal: int) -> str:
    """Return a stable ID that remains unique when templates are reused."""
    return f"{template_id}__request-{ordinal:06d}"


@dataclass(frozen=True)
class ArrivalRecord:
    """One persisted request admission in an arrival manifest."""

    arrival_time_ms: float
    request_id: str
    template_id: str
    request_type: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.arrival_time_ms, bool) or not isinstance(self.arrival_time_ms, (int, float)):
            raise TypeError("arrival_time_ms must be a number")
        if not math.isfinite(float(self.arrival_time_ms)) or self.arrival_time_ms < 0:
            raise ValueError("arrival_time_ms must be finite and non-negative")
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be a non-empty string")
        if not isinstance(self.template_id, str) or not self.template_id:
            raise ValueError("template_id must be a non-empty string")
        if self.request_type is not None and (
            not isinstance(self.request_type, str) or not self.request_type
        ):
            raise ValueError("request_type must be a non-empty string when provided")

    def as_dict(self) -> dict[str, object]:
        record = {
            "arrival_time_ms": self.arrival_time_ms,
            "request_id": self.request_id,
            "template_id": self.template_id,
        }
        if self.request_type is not None:
            record["request_type"] = self.request_type
        return record


@dataclass(frozen=True)
class ArrivalManifest:
    """Portable description of one generated workload scenario.

    The manifest stores request identities and times, while the templates
    themselves remain in the source dataset.  ``materialize`` joins those
    two pieces and creates fresh ``EvaluationTrace`` values with the
    persisted request IDs.
    """

    arrivals: tuple[ArrivalRecord, ...]
    request_rate_per_second: float
    observation_window_ms: float
    seed: int
    schema_version: int = 1

    def __post_init__(self) -> None:
        arrivals = tuple(self.arrivals)
        if any(not isinstance(arrival, ArrivalRecord) for arrival in arrivals):
            raise TypeError("arrivals must contain ArrivalRecord values")
        request_rate = self.request_rate_per_second
        window = self.observation_window_ms
        if isinstance(request_rate, bool) or not isinstance(request_rate, (int, float)):
            raise TypeError("request_rate_per_second must be a number")
        if not math.isfinite(float(request_rate)) or request_rate <= 0:
            raise ValueError("request_rate_per_second must be positive and finite")
        if isinstance(window, bool) or not isinstance(window, (int, float)):
            raise TypeError("observation_window_ms must be a number")
        if not math.isfinite(float(window)) or window <= 0:
            raise ValueError("observation_window_ms must be positive and finite")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if self.schema_version != 1:
            raise ValueError("unsupported arrival manifest schema version")

        request_ids = [arrival.request_id for arrival in arrivals]
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("arrival request IDs must be unique")
        if any(arrival.arrival_time_ms >= float(window) for arrival in arrivals):
            raise ValueError("arrival times must be inside the observation window")
        if any(
            later.arrival_time_ms < earlier.arrival_time_ms
            for earlier, later in zip(arrivals, arrivals[1:])
        ):
            raise ValueError("arrivals must be ordered by arrival_time_ms")
        object.__setattr__(self, "arrivals", arrivals)

    def as_dict(self) -> dict[str, object]:
        """Return the canonical JSON-compatible manifest representation."""
        return {
            "schema_version": self.schema_version,
            "request_rate_per_second": self.request_rate_per_second,
            "observation_window_ms": self.observation_window_ms,
            "seed": self.seed,
            "arrivals": [arrival.as_dict() for arrival in self.arrivals],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ArrivalManifest":
        """Load a manifest from its canonical dictionary representation."""
        if not isinstance(payload, dict):
            raise TypeError("arrival manifest must be a JSON object")
        raw_arrivals = payload.get("arrivals")
        if not isinstance(raw_arrivals, list):
            raise ValueError("arrival manifest arrivals must be an array")
        arrivals: list[ArrivalRecord] = []
        for index, raw_arrival in enumerate(raw_arrivals):
            if not isinstance(raw_arrival, dict):
                raise ValueError(f"arrival manifest record {index} must be an object")
            try:
                arrivals.append(
                    ArrivalRecord(
                        arrival_time_ms=raw_arrival["arrival_time_ms"],
                        request_id=raw_arrival["request_id"],
                        template_id=raw_arrival["template_id"],
                        request_type=raw_arrival.get("request_type"),
                    )
                )
            except KeyError as exc:
                raise ValueError(f"arrival manifest record {index} is missing {exc.args[0]}") from exc
        try:
            return cls(
                arrivals=tuple(arrivals),
                request_rate_per_second=payload["request_rate_per_second"],
                observation_window_ms=payload["observation_window_ms"],
                seed=payload["seed"],
                schema_version=payload.get("schema_version", 1),
            )
        except KeyError as exc:
            raise ValueError(f"arrival manifest is missing {exc.args[0]}") from exc

    def materialize(
        self,
        traces: Sequence[EvaluationTrace] | Mapping[str, Sequence[EvaluationTrace]],
    ) -> tuple[tuple[float, EvaluationTrace], ...]:
        """Resolve persisted template IDs into replayable request traces."""
        if isinstance(traces, Mapping):
            return self._materialize_mixed(traces)
        templates = _validate_generation_inputs(
            traces,
            self.request_rate_per_second,
            self.observation_window_ms,
            self.seed,
        )
        by_id = {trace.trace_id: trace for trace in templates}
        missing = sorted({arrival.template_id for arrival in self.arrivals} - by_id.keys())
        if missing:
            raise ValueError(f"arrival manifest references unknown template IDs: {missing}")
        return tuple(
            (
                arrival.arrival_time_ms,
                replace(by_id[arrival.template_id], trace_id=arrival.request_id),
            )
            for arrival in self.arrivals
        )

    def _materialize_mixed(
        self, traces_by_type: Mapping[str, Sequence[EvaluationTrace]]
    ) -> tuple[tuple[float, EvaluationTrace], ...]:
        datasets = {
            request_type: _validate_generation_inputs(
                traces,
                self.request_rate_per_second,
                self.observation_window_ms,
                self.seed,
            )
            for request_type, traces in traces_by_type.items()
        }
        by_type = {
            request_type: {trace.trace_id: trace for trace in traces}
            for request_type, traces in datasets.items()
        }
        result: list[tuple[float, EvaluationTrace]] = []
        for arrival in self.arrivals:
            if arrival.request_type is None:
                raise ValueError("mixed arrival records require request_type")
            try:
                templates = by_type[arrival.request_type]
            except KeyError as exc:
                raise ValueError(
                    f"arrival manifest references unknown request type: {arrival.request_type!r}"
                ) from exc
            try:
                template = templates[arrival.template_id]
            except KeyError as exc:
                raise ValueError(
                    f"arrival manifest references unknown template ID {arrival.template_id!r} "
                    f"for request type {arrival.request_type!r}"
                ) from exc
            result.append((arrival.arrival_time_ms, replace(template, trace_id=arrival.request_id)))
        return tuple(result)


def _generate(
    traces: Sequence[EvaluationTrace],
    *,
    request_rate_per_second: float,
    observation_window_ms: float,
    seed: int,
) -> tuple[tuple[tuple[float, EvaluationTrace], ...], ArrivalManifest]:
    templates = _validate_generation_inputs(traces, request_rate_per_second, observation_window_ms, seed)
    request_rate_per_ms = float(request_rate_per_second) / 1000.0
    observation_window = float(observation_window_ms)
    rng = random.Random(seed)

    records: list[ArrivalRecord] = []
    materialized: list[tuple[float, EvaluationTrace]] = []
    arrival_time = 0.0
    ordinal = 0
    while True:
        arrival_time += rng.expovariate(request_rate_per_ms)
        if arrival_time >= observation_window:
            break
        template = templates[ordinal % len(templates)]
        request_id = _request_id(template.trace_id, ordinal)
        records.append(ArrivalRecord(arrival_time, request_id, template.trace_id))
        materialized.append((arrival_time, replace(template, trace_id=request_id)))
        ordinal += 1

    manifest = ArrivalManifest(
        arrivals=tuple(records),
        request_rate_per_second=request_rate_per_second,
        observation_window_ms=observation_window_ms,
        seed=seed,
    )
    return tuple(materialized), manifest


def generate_poisson_arrivals(
    traces: Sequence[EvaluationTrace],
    *,
    request_rate_per_second: float,
    observation_window_ms: float,
    seed: int,
) -> tuple[tuple[float, EvaluationTrace], ...]:
    """Generate a deterministic Poisson arrival sequence.

    Inter-arrival gaps are exponential with rate ``request_rate_per_second``.
    The supplied templates are selected in round-robin order, and every
    returned trace receives a unique request ID.  Arrival times are in
    milliseconds and satisfy ``0 <= t < observation_window_ms``.
    """
    arrivals, _ = _generate(
        traces,
        request_rate_per_second=request_rate_per_second,
        observation_window_ms=observation_window_ms,
        seed=seed,
    )
    return arrivals


def generate_poisson_arrival_manifest(
    traces: Sequence[EvaluationTrace],
    *,
    request_rate_per_second: float,
    observation_window_ms: float,
    seed: int,
) -> ArrivalManifest:
    """Generate the persisted manifest corresponding to a workload."""
    _, manifest = _generate(
        traces,
        request_rate_per_second=request_rate_per_second,
        observation_window_ms=observation_window_ms,
        seed=seed,
    )
    return manifest


def _mixed_workloads(
    workloads: Mapping[str, Sequence[EvaluationTrace]] | Sequence[EvaluationTrace],
    second_workload: Sequence[EvaluationTrace] | None,
) -> dict[str, tuple[EvaluationTrace, ...]]:
    if isinstance(workloads, Mapping):
        if second_workload is not None:
            raise TypeError("second_workload is only valid with positional mixed datasets")
        result = {request_type: tuple(traces) for request_type, traces in workloads.items()}
    else:
        if second_workload is None:
            raise TypeError("mixed workloads require two datasets")
        result = {"video_qa": tuple(workloads), "math_qa": tuple(second_workload)}
    if len(result) != 2:
        raise ValueError("mixed workloads require exactly two request types")
    return result


def generate_mixed_poisson_arrival_manifest(
    workloads: Mapping[str, Sequence[EvaluationTrace]] | Sequence[EvaluationTrace],
    second_workload: Sequence[EvaluationTrace] | None = None,
    *,
    request_rate_per_second: float,
    observation_window_ms: float,
    seed: int,
) -> ArrivalManifest:
    """Generate one manifest by merging two independent half-rate streams."""
    streams = _mixed_workloads(workloads, second_workload)
    request_types = tuple(sorted(streams))
    for traces in streams.values():
        _validate_generation_inputs(
            traces,
            request_rate_per_second,
            observation_window_ms,
            seed,
        )
    stream_records: list[ArrivalRecord] = []
    stream_rate_per_ms = float(request_rate_per_second) / 2.0 / 1000.0
    for stream_index, request_type in enumerate(request_types):
        templates = tuple(streams[request_type])
        rng = random.Random(seed + stream_index)
        arrival_time = 0.0
        ordinal = 0
        while True:
            arrival_time += rng.expovariate(stream_rate_per_ms)
            if arrival_time >= float(observation_window_ms):
                break
            template = templates[ordinal % len(templates)]
            stream_records.append(
                ArrivalRecord(
                    arrival_time,
                    _request_id(request_type, ordinal),
                    template.trace_id,
                    request_type,
                )
            )
            ordinal += 1
    stream_records.sort(key=lambda item: (item.arrival_time_ms, item.request_type or "", item.request_id))
    return ArrivalManifest(
        arrivals=tuple(stream_records),
        request_rate_per_second=request_rate_per_second,
        observation_window_ms=observation_window_ms,
        seed=seed,
    )


def save_arrival_manifest(path: Path | str, manifest: ArrivalManifest) -> None:
    """Write an arrival manifest as UTF-8 JSON."""
    if not isinstance(manifest, ArrivalManifest):
        raise TypeError("manifest must be an ArrivalManifest")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest.as_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_arrival_manifest(path: Path | str) -> ArrivalManifest:
    """Read an arrival manifest previously written by ``save_arrival_manifest``."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source}: invalid arrival manifest JSON") from exc
    return ArrivalManifest.from_dict(payload)


def load_workload_scenario(
    planner_dags_path: Path | str | Mapping[str, Path | str],
    arrival_manifest_path: Path | str,
) -> tuple[tuple[float, EvaluationTrace], ...]:
    """Load an offline Planner JSONL file and replay one arrival manifest.

    The Planner output is kept as the source of the request templates.  The
    manifest supplies the simulated arrival times and the unique request IDs;
    no new random draws are made while replaying a scenario.  For a mixed
    manifest, pass a mapping from persisted request types to their separate
    Planner JSONL files.  Loading the dataset here keeps callers from
    accidentally applying its train, validation, or test split when they need
    the complete offline output.
    """
    # Import locally to keep ``workflow`` and its JSON/DAG adapters out of the
    # core workload module's import path for callers that only generate times.
    from .workflow import ToolCallPlanDataset

    manifest = load_arrival_manifest(arrival_manifest_path)
    if isinstance(planner_dags_path, Mapping):
        datasets = {
            request_type: ToolCallPlanDataset.load(path)
            for request_type, path in planner_dags_path.items()
        }
        return manifest.materialize(datasets)
    dataset = ToolCallPlanDataset.load(planner_dags_path)
    return manifest.materialize(dataset)


def load_mixed_workload_scenario(
    planner_dags_paths: Mapping[str, Path | str],
    arrival_manifest_path: Path | str,
) -> tuple[tuple[float, EvaluationTrace], ...]:
    """Load a mixed manifest from its request-type-specific Planner JSONLs."""
    return load_workload_scenario(planner_dags_paths, arrival_manifest_path)


__all__ = [
    "ArrivalManifest",
    "ArrivalRecord",
    "generate_poisson_arrival_manifest",
    "generate_poisson_arrivals",
    "generate_mixed_poisson_arrival_manifest",
    "load_arrival_manifest",
    "load_mixed_workload_scenario",
    "load_workload_scenario",
    "save_arrival_manifest",
]
