"""Acceptance tests for the first multi-request evaluator contract.

These tests deliberately use the synthetic profiling snapshot.  The workload
evaluator is expected to derive its system state from the replay; the
``EvaluationTrace.system_state`` values in this file are sentinels to make
sure that the old per-trace state is not accidentally reused.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest

from eec_sched import (
    EvaluationTrace,
    FinalOutput,
    InputSource,
    SchedulerCandidate,
    SchedulerOutput,
    ToolCallPlan,
    ToolNode,
    TrustedEvaluationError,
    evaluate_scheduler_instance,
    load_profiling_database,
)
from eec_sched.evaluation.workload import evaluate_workload


ROOT = Path(__file__).parents[1]
SNAPSHOT = load_profiling_database(
    ROOT / "docs/examples/profiling-database.fake.json",
    ROOT / "docs/schemas/profiling-database.schema.json",
)


def _get(value: object, name: str) -> Any:
    """Read a field from either the public dataclass or mapping projection."""
    if isinstance(value, Mapping):
        return value[name]
    return getattr(value, name)


def _one_node_plan(node_id: str = "work") -> ToolCallPlan:
    return ToolCallPlan(
        nodes=(ToolNode(node_id, "text_generation", {"prompt": InputSource.request("prompt")}),),
        final_outputs=(FinalOutput(node_id, "text"),),
    )


def _chain_plan(root_id: str = "root", child_id: str = "child") -> ToolCallPlan:
    return ToolCallPlan(
        nodes=(
            ToolNode(root_id, "text_generation", {"prompt": InputSource.request("prompt")}),
            ToolNode(child_id, "text_summarization", {"text": InputSource.node(root_id, "text")}),
        ),
        final_outputs=(FinalOutput(child_id, "text"),),
    )


def _two_parent_plan() -> ToolCallPlan:
    return ToolCallPlan(
        nodes=(
            ToolNode("early", "text_generation", {"prompt": InputSource.request("prompt")}),
            ToolNode("late", "text_generation", {"prompt": InputSource.request("prompt")}),
            ToolNode(
                "join",
                "text_summarization",
                {
                    "left": InputSource.node("early", "text"),
                    "right": InputSource.node("late", "text"),
                },
            ),
        ),
        final_outputs=(FinalOutput("join", "text"),),
    )


def _trace(
    trace_id: str,
    plan: ToolCallPlan | None = None,
    *,
    system_state: Mapping[str, object] | None = None,
) -> EvaluationTrace:
    plan = plan or _one_node_plan()
    return EvaluationTrace(
        trace_id,
        {"prompt": trace_id},
        plan,
        system_state=system_state or {},
    )


def _assignment(device_id: str = "device") -> dict[str, dict[str, str]]:
    return {"work": {"configuration_id": "synthetic-reference", "device_id": device_id}}


def _node(record: object, node_id: str) -> object:
    return next(item for item in _get(record, "nodes") if _get(item, "node_id") == node_id)


def _request(result: object, trace_id: str) -> object:
    return next(item for item in _get(result, "requests") if _get(item, "trace_id") == trace_id)


def _transfer(record: object, source_device: str, destination_device: str) -> object:
    return next(
        item
        for item in _get(record, "transfers")
        if _get(item, "source_device_id") == source_device
        and _get(item, "destination_device_id") == destination_device
    )


def test_requests_on_one_device_are_serialized() -> None:
    trace_a = _trace("a")
    trace_b = _trace("b")
    candidate = SchedulerCandidate(1, lambda view: _assignment("device"))

    result = evaluate_workload(
        SNAPSHOT,
        ((0.0, trace_a), (1.0, trace_b)),
        candidate,
        observation_window_ms=100.0,
    )

    first, second = result.requests
    first_node = _node(first, "work")
    second_node = _node(second, "work")
    assert _get(second_node, "start_ms") >= _get(first_node, "finish_ms")
    assert _get(first, "completion_time_ms") is not None
    assert _get(second, "completion_time_ms") is not None
    assert _get(first, "trace_id") == "a"
    assert _get(first, "arrival_time_ms") == pytest.approx(0.0)
    assert _get(first, "selected_path_index") == 0
    assert _get(first, "assignments")
    assert _get(first, "scheduler_computation_time_ms") >= 0.0
    assert _get(first, "quality") == pytest.approx(
        SNAPSHOT.quality_profile("text_generation", "synthetic-reference").normalized_quality_lcb
    )
    assert _get(first, "reason") is None


def test_scheduler_computation_time_delays_request_activation(monkeypatch) -> None:
    import eec_sched.evaluation.workload as workload_module

    ticks = iter((1.0, 1.1))
    monkeypatch.setattr(workload_module, "perf_counter", lambda: next(ticks))
    result = evaluate_workload(
        SNAPSHOT,
        ((0.0, _trace("delayed")),),
        SchedulerCandidate(1, lambda view: _assignment("device")),
        observation_window_ms=200.0,
    )

    request = result.requests[0]
    node = _node(request, "work")
    assert _get(request, "scheduler_computation_time_ms") == pytest.approx(100.0)
    assert _get(node, "start_ms") == pytest.approx(100.0)
    assert _get(request, "latency_ms") == pytest.approx(
        _get(request, "completion_time_ms") - _get(request, "arrival_time_ms")
    )


def test_activation_events_preserve_resource_order_around_later_arrival(monkeypatch) -> None:
    import eec_sched.evaluation.workload as workload_module

    ticks = iter((1.0, 1.1, 2.0, 2.0))
    monkeypatch.setattr(workload_module, "perf_counter", lambda: next(ticks))
    first = _trace("delayed-first")
    second = _trace("immediate-second")
    observed_states: list[Mapping[str, object]] = []

    def propose(view: object) -> object:
        observed_states.append(_get(view, "system_state"))
        return _assignment("device")

    result = evaluate_workload(
        SNAPSHOT,
        ((0.0, first), (1.0, second)),
        SchedulerCandidate(1, propose),
        observation_window_ms=300.0,
    )

    first_request, second_request = result.requests
    first_node = _node(first_request, "work")
    second_node = _node(second_request, "work")
    assert _get(first_request, "scheduler_computation_time_ms") == pytest.approx(100.0)
    assert _get(second_request, "scheduler_computation_time_ms") == pytest.approx(0.0)
    assert _get(_get(observed_states[1], "devices"), "device")["committed_work_ms"] == 0.0
    assert _get(second_node, "start_ms") == pytest.approx(1.0)
    assert _get(first_node, "start_ms") >= _get(second_node, "finish_ms")


def test_uncontended_request_matches_single_request_profile_timing() -> None:
    trace = _trace("single")
    candidate = SchedulerCandidate(1, lambda view: _assignment("cloud"))
    workload = evaluate_workload(
        SNAPSHOT,
        ((7.0, trace),),
        candidate,
        observation_window_ms=100.0,
    )
    single = evaluate_scheduler_instance(SNAPSHOT, trace.dag, lambda dags: _assignment("cloud"))

    assert workload.requests[0].latency_ms == pytest.approx(
        single.simulated_makespan_ms
        + workload.requests[0].scheduler_computation_time_ms
    )
    assert workload.requests[0].completion_time_ms == pytest.approx(
        7.0
        + single.simulated_makespan_ms
        + workload.requests[0].scheduler_computation_time_ms
    )


def test_different_devices_can_overlap() -> None:
    # Give each request a distinct node ID so the proposal can select a device.
    traces = (_trace("device-request", _one_node_plan("device-work")), _trace("cloud-request", _one_node_plan("cloud-work")))
    candidate = SchedulerCandidate(
        1,
        lambda view: {
            _get(_get(view, "dag").nodes[0], "node_id"): {
                "configuration_id": "synthetic-reference",
                "device_id": "device"
                if _get(_get(view, "dag").nodes[0], "node_id") == "device-work"
                else "edge",
            }
        },
    )
    result = evaluate_workload(
        SNAPSHOT,
        ((0.0, traces[0]), (0.0, traces[1])),
        candidate,
        observation_window_ms=100.0,
    )

    device_node = _node(_request(result, "device-request"), "device-work")
    cloud_node = _node(_request(result, "cloud-request"), "cloud-work")
    assert max(_get(device_node, "start_ms"), _get(cloud_node, "start_ms")) < min(
        _get(device_node, "finish_ms"), _get(cloud_node, "finish_ms")
    )


def test_same_directed_ingress_link_is_serialized() -> None:
    traces = (_trace("a", _one_node_plan("a-work")), _trace("b", _one_node_plan("b-work")))
    candidate = SchedulerCandidate(
        1,
        lambda view: {
            _get(_get(view, "dag").nodes[0], "node_id"): {
                "configuration_id": "synthetic-reference",
                "device_id": "edge",
            }
        },
    )
    result = evaluate_workload(
        SNAPSHOT,
        ((0.0, traces[0]), (1.0, traces[1])),
        candidate,
        observation_window_ms=100.0,
    )

    first_transfer = _transfer(result.requests[0], "device", "edge")
    second_transfer = _transfer(result.requests[1], "device", "edge")
    assert _get(second_transfer, "start_ms") >= _get(first_transfer, "finish_ms")


def test_dependencies_enter_the_ready_queue_only_after_predecessor_transfer() -> None:
    trace = _trace("chain", _chain_plan())

    def propose(view: object) -> object:
        return {
            "root": {"configuration_id": "synthetic-reference", "device_id": "device"},
            "child": {"configuration_id": "fast", "device_id": "edge"},
        }

    result = evaluate_workload(
        SNAPSHOT,
        ((0.0, trace),),
        SchedulerCandidate(1, propose),
        observation_window_ms=100.0,
    )
    record = result.requests[0]
    root = _node(record, "root")
    child = _node(record, "child")
    dependency_transfer = next(
        transfer
        for transfer in _get(record, "transfers")
        if _get(transfer, "source_node_id") == "root"
        and _get(transfer, "destination_node_id") == "child"
    )
    assert _get(child, "start_ms") >= _get(dependency_transfer, "finish_ms")
    assert _get(root, "finish_ms") <= _get(dependency_transfer, "start_ms")
    root_quality = SNAPSHOT.quality_profile("text_generation", "synthetic-reference").normalized_quality_lcb
    child_quality = SNAPSHOT.quality_profile("text_summarization", "fast").normalized_quality_lcb
    assert _get(record, "quality") == pytest.approx((root_quality + child_quality) / 2 + 0.01)


def test_two_parent_dependencies_wait_for_each_parent_transfer() -> None:
    trace = _trace("two-parent", _two_parent_plan())

    def propose(view: object) -> object:
        return {
            "early": {"configuration_id": "synthetic-reference", "device_id": "device"},
            "late": {"configuration_id": "synthetic-reference", "device_id": "cloud"},
            "join": {"configuration_id": "fast", "device_id": "edge"},
        }

    result = evaluate_workload(
        SNAPSHOT,
        ((0.0, trace),),
        SchedulerCandidate(1, propose),
        observation_window_ms=200.0,
    )
    record = result.requests[0]
    early = _node(record, "early")
    late = _node(record, "late")
    join = _node(record, "join")
    early_transfer = next(
        transfer
        for transfer in _get(record, "transfers")
        if _get(transfer, "source_node_id") == "early"
        and _get(transfer, "destination_node_id") == "join"
    )
    late_transfer = next(
        transfer
        for transfer in _get(record, "transfers")
        if _get(transfer, "source_node_id") == "late"
        and _get(transfer, "destination_node_id") == "join"
    )

    assert _get(early, "finish_ms") < _get(late, "finish_ms")
    assert _get(early_transfer, "start_ms") >= _get(early, "finish_ms")
    assert _get(late_transfer, "start_ms") >= _get(late, "finish_ms")
    assert _get(join, "start_ms") >= max(
        _get(early_transfer, "finish_ms"), _get(late_transfer, "finish_ms")
    )


def test_later_scheduler_sees_busy_and_committed_state_derived_by_evaluator() -> None:
    first = _trace(
        "first",
        _chain_plan("first-root", "first-child"),
        system_state={"secret": "must-not-be-copied"},
    )
    second = _trace(
        "second",
        _one_node_plan("second-work"),
        system_state={"secret": "future-trace-state"},
    )
    observed: list[Mapping[str, object]] = []

    def propose(view: object) -> object:
        state = _get(view, "system_state")
        observed.append(state)
        node_ids = {_get(node, "node_id") for node in _get(_get(view, "dag"), "nodes")}
        if "first-root" in node_ids:
            return {
                "first-root": {"configuration_id": "synthetic-reference", "device_id": "device"},
                "first-child": {"configuration_id": "fast", "device_id": "cloud"},
            }
        return {"second-work": {"configuration_id": "synthetic-reference", "device_id": "edge"}}

    evaluate_workload(
        SNAPSHOT,
        ((0.0, first), (1.0, second)),
        SchedulerCandidate(1, propose),
        observation_window_ms=100.0,
    )

    assert len(observed) == 2
    state = observed[1]
    assert "future-trace-state" not in repr(state)
    assert "must-not-be-copied" not in repr(state)
    assert _get(state, "time_ms") == pytest.approx(1.0)
    device_state = _get(_get(state, "devices"), "device")
    cloud_state = _get(_get(state, "devices"), "cloud")
    assert _get(device_state, "busy") is True
    assert _get(cloud_state, "committed_work_ms") > 0.0


def test_request_and_node_ids_with_delimiters_keep_distinct_pending_work() -> None:
    def local_plan(node_id: str) -> ToolCallPlan:
        return ToolCallPlan(
            (ToolNode(node_id, "text_generation", {}),),
            (FinalOutput(node_id, "text"),),
        )

    arrivals = (
        (0.0, _trace("start", local_plan("work"))),
        (1.0, _trace("a|node|b", local_plan("c"))),
        (2.0, _trace("a", local_plan("b|node|c"))),
        (3.0, _trace("observer", local_plan("observe"))),
    )
    observed: list[Mapping[str, object]] = []

    def propose(view: object) -> object:
        state = _get(view, "system_state")
        if _get(state, "time_ms") == 3.0:
            observed.append(state)
        node_id = _get(_get(view, "dag").nodes[0], "node_id")
        return {node_id: {"configuration_id": "synthetic-reference", "device_id": "device"}}

    result = evaluate_workload(
        SNAPSHOT,
        arrivals,
        SchedulerCandidate(1, propose),
        observation_window_ms=120.0,
    )

    assert len(observed) == 1
    device = _get(_get(observed[0], "devices"), "device")
    assert _get(device, "queue_depth") == 2
    assert _get(device, "committed_work_ms") > 72.0
    assert all(record.status == "completed" for record in result.requests)


def test_same_timestamp_arrivals_are_admitted_in_trace_id_order() -> None:
    seen: list[str] = []
    trace_b = _trace("b", _one_node_plan("b-node"))
    trace_a = _trace("a", _one_node_plan("a-node"))

    def propose(view: object) -> object:
        node_id = _get(_get(view, "dag").nodes[0], "node_id")
        seen.append(node_id)
        return {node_id: {"configuration_id": "synthetic-reference", "device_id": "device"}}

    evaluate_workload(
        SNAPSHOT,
        ((0.0, trace_b), (0.0, trace_a)),
        SchedulerCandidate(1, propose),
        observation_window_ms=100.0,
    )

    assert seen == ["a-node", "b-node"]


def test_scheduler_can_select_a_nonzero_candidate_path() -> None:
    path_zero = _one_node_plan("zero")
    path_one = ToolCallPlan(
        nodes=(ToolNode("one", "text_summarization", {"text": InputSource.request("prompt")}),),
        final_outputs=(FinalOutput("one", "text"),),
    )
    path_two = _one_node_plan("two")
    trace = EvaluationTrace(
        "path-selection",
        {"prompt": "x"},
        path_zero,
        dags=(path_zero, path_one, path_two),
    )

    def propose(view: object) -> object:
        selected = _get(view, "candidate_dags")[1]
        return SchedulerOutput(
            selected,
            {"one": {"configuration_id": "fast", "device_id": "edge"}},
            path_index=1,
        )

    result = evaluate_workload(
        SNAPSHOT,
        ((0.0, trace),),
        SchedulerCandidate(1, propose),
        observation_window_ms=100.0,
    )
    request = result.requests[0]
    assert _get(request, "status") == "completed"
    assert _get(request, "selected_path_index") == 1
    assert _get(request, "assignments")["one"].device_id == "edge"


def test_invalid_path_and_incompatible_device_are_rejected_without_stopping_replay() -> None:
    invalid_path = _trace("invalid-path", _one_node_plan("invalid-path-node"))
    incompatible = _trace("incompatible", _one_node_plan("incompatible-node"))
    good = _trace("good-after-invalid", _one_node_plan("good-after-invalid-node"))

    def propose(view: object) -> object:
        node_id = _get(_get(view, "dag").nodes[0], "node_id")
        assignment = {
            node_id: {"configuration_id": "synthetic-reference", "device_id": "device"}
        }
        if node_id == "invalid-path-node":
            return {"dag": _get(view, "dag"), "path_index": 99, "assignments": assignment}
        if node_id == "incompatible-node":
            return {
                node_id: {"configuration_id": "synthetic-reference", "device_id": "unknown"}
            }
        return assignment

    result = evaluate_workload(
        SNAPSHOT,
        ((0.0, invalid_path), (1.0, incompatible), (2.0, good)),
        SchedulerCandidate(1, propose),
        observation_window_ms=100.0,
    )

    assert _get(result.requests[0], "status") == "rejected"
    assert "path_index" in _get(result.requests[0], "reason")
    assert _get(result.requests[1], "status") == "rejected"
    assert "incompatible device" in _get(result.requests[1], "reason")
    assert _get(result.requests[2], "completion_time_ms") is not None


def test_rejected_or_failed_request_does_not_stop_later_requests() -> None:
    bad = _trace("bad", _one_node_plan("bad-node"))
    broken = _trace("broken", _one_node_plan("broken-node"))
    good = _trace("good", _one_node_plan("good-node"))

    def propose(view: object) -> object:
        node_id = _get(_get(view, "dag").nodes[0], "node_id")
        if node_id == "bad-node":
            return {}
        if node_id == "broken-node":
            raise RuntimeError("candidate failure")
        return {node_id: {"configuration_id": "synthetic-reference", "device_id": "device"}}

    result = evaluate_workload(
        SNAPSHOT,
        ((0.0, bad), (1.0, broken), (2.0, good)),
        SchedulerCandidate(1, propose),
        observation_window_ms=100.0,
    )

    assert _get(result.requests[0], "status") in {"rejected", "failed"}
    assert _get(result.requests[0], "reason")
    assert _get(result.requests[0], "quality") is None
    assert _get(result.requests[1], "status") == "failed"
    assert _get(result.requests[1], "reason")
    assert _get(result.requests[1], "quality") is None
    assert _get(result.requests[2], "completion_time_ms") is not None
    assert _get(result.requests[2], "latency_ms") is not None


def test_window_metrics_count_only_completions_by_window_and_retain_backlog() -> None:
    trace_a = _trace("a")
    trace_b = _trace("b")
    result = evaluate_workload(
        SNAPSHOT,
        ((0.0, trace_a), (1.0, trace_b)),
        SchedulerCandidate(1, lambda view: _assignment("device")),
        observation_window_ms=30.0,
    )

    assert _get(result, "completed_within_window") == 1
    assert _get(result, "backlog_at_window_end") == 1
    assert _get(result, "throughput_per_second") == pytest.approx(1 / 0.03)
    assert _get(result, "latency_p50_ms") is not None
    assert _get(result, "latency_p95_ms") is not None
    assert _get(result.requests[1], "completion_time_ms") > 30.0


def test_scheduler_does_not_receive_future_arrivals_or_trace_state() -> None:
    seen: list[Mapping[str, object]] = []
    current = _trace("current", _one_node_plan("current-node"))
    future = _trace(
        "future",
        _one_node_plan("future-node"),
        system_state={"secret": "future-only"},
    )

    def propose(view: object) -> object:
        state = _get(view, "system_state")
        seen.append(state)
        node_id = _get(_get(view, "dag").nodes[0], "node_id")
        return {node_id: {"configuration_id": "synthetic-reference", "device_id": "device"}}

    evaluate_workload(
        SNAPSHOT,
        ((0.0, current), (50.0, future)),
        SchedulerCandidate(1, propose),
        observation_window_ms=100.0,
    )

    assert len(seen) == 2
    assert "future" not in repr(seen[0])
    assert "future-only" not in repr(seen[0])
    assert _get(seen[0], "time_ms") == pytest.approx(0.0)


@pytest.mark.parametrize(
    "arrivals",
    [
        ((-1.0, _trace("negative")),),
        ((10.0, _trace("at-end")),),
        ((float("nan"), _trace("nan")),),
        ((0.0, _trace("duplicate")), (1.0, _trace("duplicate"))),
    ],
)
def test_workload_input_validation_rejects_invalid_arrivals(arrivals: tuple[tuple[float, EvaluationTrace], ...]) -> None:
    with pytest.raises(ValueError):
        evaluate_workload(
            SNAPSHOT,
            arrivals,
            SchedulerCandidate(1, lambda view: _assignment()),
            observation_window_ms=10.0,
        )


def test_workload_input_validation_rejects_non_positive_window() -> None:
    with pytest.raises(ValueError):
        evaluate_workload(
            SNAPSHOT,
            (),
            SchedulerCandidate(1, lambda view: _assignment()),
            observation_window_ms=0.0,
        )


def test_trusted_workload_fault_is_not_reported_as_candidate_failure() -> None:
    cyclic = ToolCallPlan(
        nodes=(
            ToolNode("work", "text_generation", {"prompt": InputSource.node("work", "text")}),
        ),
        final_outputs=(FinalOutput("work", "text"),),
    )
    with pytest.raises(TrustedEvaluationError, match="simulation stalled"):
        evaluate_workload(
            SNAPSHOT,
            ((0.0, _trace("cycle", cyclic)),),
            SchedulerCandidate(1, lambda view: _assignment()),
            observation_window_ms=100.0,
        )
