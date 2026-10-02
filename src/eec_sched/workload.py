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
from typing import Any, Sequence

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

    def __post_init__(self) -> None:
        if isinstance(self.arrival_time_ms, bool) or not isinstance(self.arrival_time_ms, (int, float)):
            raise TypeError("arrival_time_ms must be a number")
        if not math.isfinite(float(self.arrival_time_ms)) or self.arrival_time_ms < 0:
            raise ValueError("arrival_time_ms must be finite and non-negative")
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be a non-empty string")
        if not isinstance(self.template_id, str) or not self.template_id:
            raise ValueError("template_id must be a non-empty string")

    def as_dict(self) -> dict[str, object]:
        return {
            "arrival_time_ms": self.arrival_time_ms,
            "request_id": self.request_id,
            "template_id": self.template_id,
        }


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

    def materialize(self, traces: Sequence[EvaluationTrace]) -> tuple[tuple[float, EvaluationTrace], ...]:
        """Resolve persisted template IDs into replayable request traces."""
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


__all__ = [
    "ArrivalManifest",
    "ArrivalRecord",
    "generate_poisson_arrival_manifest",
    "generate_poisson_arrivals",
    "load_arrival_manifest",
    "save_arrival_manifest",
]
