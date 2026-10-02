from __future__ import annotations

import json
import random

import pytest

from eec_sched.candidate import EvaluationTrace
from eec_sched.domain import FinalOutput, InputSource, ToolCallPlan, ToolNode
from eec_sched.workload import (
    generate_poisson_arrival_manifest,
    generate_poisson_arrivals,
    load_arrival_manifest,
    save_arrival_manifest,
)


def _trace(trace_id: str) -> EvaluationTrace:
    plan = ToolCallPlan(
        nodes=(ToolNode("node", "tool", {"input": InputSource.request("prompt")}),),
        final_outputs=(FinalOutput("node", "output"),),
    )
    return EvaluationTrace(trace_id, {"prompt": trace_id}, plan)


def test_poisson_arrivals_are_deterministic_and_cycle_templates() -> None:
    templates = (_trace("alpha"), _trace("beta"))
    first = generate_poisson_arrivals(
        templates,
        request_rate_per_second=20,
        observation_window_ms=500,
        seed=7,
    )
    second = generate_poisson_arrivals(
        templates,
        request_rate_per_second=20,
        observation_window_ms=500,
        seed=7,
    )

    assert [(time, trace.trace_id) for time, trace in first] == [
        (time, trace.trace_id) for time, trace in second
    ]
    assert first
    assert all(0 <= time < 500 for time, _ in first)
    assert [trace.trace_id.split("__request-")[0] for _, trace in first] == [
        "alpha" if index % 2 == 0 else "beta" for index in range(len(first))
    ]
    assert len({trace.trace_id for _, trace in first}) == len(first)


def test_poisson_times_match_local_exponential_process() -> None:
    rate = 4
    window = 1000
    seed = 11
    arrivals = generate_poisson_arrivals(
        (_trace("template"),),
        request_rate_per_second=rate,
        observation_window_ms=window,
        seed=seed,
    )

    expected: list[float] = []
    clock = 0.0
    rng = random.Random(seed)
    while True:
        clock += rng.expovariate(rate / 1000)
        if clock >= window:
            break
        expected.append(clock)

    assert [time for time, _ in arrivals] == expected


def test_manifest_round_trip_materializes_same_requests(tmp_path) -> None:
    templates = (_trace("alpha"), _trace("beta"))
    manifest = generate_poisson_arrival_manifest(
        templates,
        request_rate_per_second=10,
        observation_window_ms=300,
        seed=3,
    )
    path = tmp_path / "arrivals.json"
    save_arrival_manifest(path, manifest)
    loaded = load_arrival_manifest(path)

    assert loaded == manifest
    replayed = loaded.materialize(templates)
    assert [(time, trace.trace_id) for time, trace in replayed] == [
        (arrival.arrival_time_ms, arrival.request_id) for arrival in manifest.arrivals
    ]
    assert [trace.task_input for _, trace in replayed] == [
        templates[index % 2].task_input for index in range(len(replayed))
    ]

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["request_rate_per_second"] == 10
    assert payload["observation_window_ms"] == 300
    assert payload["seed"] == 3
    assert set(payload["arrivals"][0]) == {"arrival_time_ms", "request_id", "template_id"}


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"request_rate_per_second": 0}, ValueError),
        ({"request_rate_per_second": float("inf")}, ValueError),
        ({"observation_window_ms": 0}, ValueError),
        ({"observation_window_ms": float("nan")}, ValueError),
        ({"seed": "7"}, TypeError),
    ],
)
def test_generation_rejects_invalid_inputs(kwargs, error) -> None:
    options = {
        "request_rate_per_second": 1,
        "observation_window_ms": 10,
        "seed": 1,
    }
    options.update(kwargs)
    with pytest.raises(error):
        generate_poisson_arrivals((_trace("only"),), **options)


def test_generation_requires_nonempty_unique_templates() -> None:
    with pytest.raises(ValueError, match="at least one"):
        generate_poisson_arrivals((), request_rate_per_second=1, observation_window_ms=10, seed=1)
    with pytest.raises(ValueError, match="unique"):
        generate_poisson_arrivals(
            (_trace("same"), _trace("same")),
            request_rate_per_second=1,
            observation_window_ms=10,
            seed=1,
        )

