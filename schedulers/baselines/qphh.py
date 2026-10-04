"""Placement-focused adaptation of the QPHH workflow scheduler.

The paper's QPHH uses Q-learning to select six low-level heuristics (LLHs)
which change both task order and task-resource mapping.  The repository's
trusted evaluator deliberately executes a selected DAG in one canonical,
deterministic order.  This baseline therefore keeps the part of QPHH that is
observable at the Scheduler boundary: Q-learning selects six deterministic
device-placement refinement operators.

The operators are:

``single_node_remap``
    remap a high-connectivity node to its best profiled device;
``parent_colocation``
    try placing a child on a compatible parent's device;
``chain_colocation``
    try colocating a parent-child pair on one compatible device;
``greedy_best_position``
    perform one bounded coordinate-descent pass over node placements;
``swap_placements``
    swap two compatible nodes' devices; and
``transfer_aware``
    optimize one child placement using all predecessor transfer costs.

The objective is the lower-is-better complement of the evaluator's reciprocal
latency and GPU-memory indicators.  Energy is intentionally not estimated:
the current ``SchedulerView`` exposes no energy evidence.  The selected DAG
and configuration policy are fixed for comparability (longest dependency
chain and highest compatible normalized quality).  The evaluator owns actual
execution order and scoring.

Training starts at the paper's all-one Q-table initialization.  The training
script exports a non-uniform table and embeds it here, so evaluation has no
dependency on a training run.  ``update_q_value``
and ``train_q_table`` are exposed for that offline entrypoint.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from eec_sched import SchedulerOutput


_END_DEVICE = "device"
_REQUEST_BYTES = {"text": 256, "image": 262_144, "audio": 1_048_576}

ACTION_NAMES = (
    "single_node_remap",
    "parent_colocation",
    "chain_colocation",
    "greedy_best_position",
    "swap_placements",
    "transfer_aware",
)
ACTION_COUNT = len(ACTION_NAMES)
STATE_COUNT = 10
DEFAULT_EPSILON = 0.3
DEFAULT_GAMMA = 0.5

# Learned offline with CUDA_VISIBLE_DEVICES=2, seed 17, 8 episodes over the
# 118-record training split.  train.py always initializes a fresh all-one
# table before updating it; this constant is used only for inference.
Q_TABLE: tuple[tuple[float, ...], ...] = (
    (10.35211597, 6.501112786, 15.47708846, 13.14959691, 13.32508467, 10.73269764,),
    (7.629683247, 6.364238308, 14.72215907, 11.60312006, 6.156059795, 9.248235542,),
    (2.021139274, 7.288733071, 12.54522656, 7.736353699, 5.449996058, 3.528117169,),
    (6.665908637, 8.959448241, 11.19690414, 11.54569708, 8.523093778, 9.828509511,),
    (1.203397924, 4.47201091, 12.10078907, 9.020785891, 9.3824473, -1.151259883,),
    (0.9780274622, 5.337965901, 8.674780914, 11.84239018, 7.917782044, 5.67521824,),
    (-1.482262209, 11.41515786, 3.179093233, 1, 1, 1,),
    (1, 1, 1, 1, 1, 1,),
    (1, 1, 1, 1, 1, 1,),
    (1, 1, 1, 1, 1, 1,),
)


@dataclass(frozen=True)
class _Prepared:
    tools: Mapping[str, Mapping[str, Any]]
    devices: tuple[str, ...]
    transfers: Mapping[tuple[str, str], Mapping[str, Any]]
    selected_configs: Mapping[str, str]
    profiles_by_tool: Mapping[str, Mapping[str, Mapping[str, Any]]]
    output_bytes_by_tool: Mapping[str, int]


def _parents(dag: Any) -> dict[str, tuple[str, ...]]:
    result: dict[str, list[str]] = {node.node_id: [] for node in dag.nodes}
    for node in dag.nodes:
        for source in node.inputs.values():
            if source.kind == "node":
                result[node.node_id].append(source.name)
    return {node_id: tuple(parents) for node_id, parents in result.items()}


def _children(parents: Mapping[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    result: dict[str, list[str]] = {node_id: [] for node_id in parents}
    for child, direct_parents in parents.items():
        for parent in direct_parents:
            if parent in result:
                result[parent].append(child)
    return {node_id: tuple(children) for node_id, children in result.items()}


def _longest_chain(dag: Any) -> int:
    """Return the number of nodes on the longest dependency path."""

    parents = _parents(dag)
    memo: dict[str, int] = {}
    visiting: set[str] = set()

    def depth(node_id: str) -> int:
        if node_id in memo:
            return memo[node_id]
        if node_id in visiting:
            raise ValueError("DAG contains a cycle")
        visiting.add(node_id)
        value = 1 + max((depth(parent) for parent in parents[node_id]), default=0)
        visiting.remove(node_id)
        memo[node_id] = value
        return value

    return max((depth(node_id) for node_id in parents), default=0)


def _evidence(
    view: Any,
) -> tuple[
    dict[str, Mapping[str, Any]],
    tuple[str, ...],
    dict[tuple[str, str], Mapping[str, Any]],
]:
    snapshot = view.snapshot_evidence
    tools = {tool["tool_id"]: tool for tool in snapshot["tools"]}
    devices = tuple(sorted(device["device_id"] for device in snapshot["devices"]))
    transfers = {
        (profile["source_device_id"], profile["destination_device_id"]): profile
        for profile in snapshot["transfer_profiles"]
    }
    return tools, devices, transfers


def _compatible_profiles(
    tool: Mapping[str, Any], configuration_id: str, devices: Iterable[str]
) -> dict[str, Mapping[str, Any]]:
    configuration = next(
        item for item in tool["configurations"]
        if item["configuration_id"] == configuration_id
    )
    allowed = {
        item["device_id"]
        for item in configuration["device_compatibility"]
        if item["status"] == "compatible"
    }
    profiles: dict[str, Mapping[str, Any]] = {}
    for profile in tool["execution_profiles"]:
        device_id = profile["device_id"]
        if device_id not in allowed or device_id not in devices:
            continue
        current = profiles.get(device_id)
        if current is None or profile["warm_latency_p95_ms"] < current["warm_latency_p95_ms"]:
            profiles[device_id] = profile
    return {device_id: profiles[device_id] for device_id in sorted(profiles)}


def _select_configurations(
    dag: Any,
    tools: Mapping[str, Mapping[str, Any]],
    devices: Iterable[str],
) -> tuple[
    dict[str, str],
    dict[str, dict[str, Mapping[str, Any]]],
    dict[str, int],
]:
    """Choose highest quality among configurations with usable profiles."""

    selected: dict[str, str] = {}
    profiles_by_tool: dict[str, dict[str, Mapping[str, Any]]] = {}
    output_bytes_by_tool: dict[str, int] = {}
    available_devices = tuple(devices)
    for tool_id in sorted({node.tool_id for node in dag.nodes}):
        tool = tools[tool_id]
        quality = {profile["configuration_id"]: profile for profile in tool["quality_profiles"]}
        candidates: list[tuple[float, str, dict[str, Mapping[str, Any]], Mapping[str, Any]]] = []
        for configuration in tool["configurations"]:
            configuration_id = configuration["configuration_id"]
            quality_profile = quality.get(configuration_id)
            if quality_profile is None:
                continue
            profiles = _compatible_profiles(tool, configuration_id, available_devices)
            if profiles:
                candidates.append((
                    float(quality_profile["normalized_quality_lcb"]),
                    configuration_id,
                    profiles,
                    quality_profile,
                ))
        if not candidates:
            raise ValueError(f"no compatible quality configuration for tool {tool_id}")
        best_quality = max(item[0] for item in candidates)
        tied = [item for item in candidates if item[0] == best_quality]
        _, configuration_id, profiles, quality_profile = min(tied, key=lambda item: item[1])
        selected[tool_id] = configuration_id
        profiles_by_tool[tool_id] = profiles
        output_bytes_by_tool[tool_id] = int(quality_profile["representative_output_bytes"])
    return selected, profiles_by_tool, output_bytes_by_tool


def _prepare(view: Any, dag: Any) -> _Prepared:
    tools, devices, transfers = _evidence(view)
    selected, profiles, output_bytes = _select_configurations(dag, tools, devices)
    return _Prepared(tools, devices, transfers, selected, profiles, output_bytes)


def _transfer_ms(
    transfers: Mapping[tuple[str, str], Mapping[str, Any]],
    source: str,
    destination: str,
    size_bytes: int,
) -> float:
    if source == destination:
        return 0.0
    profile = transfers[(source, destination)]
    return float(profile["propagation_delay_ms"]) + (
        1000.0 * size_bytes / float(profile["bandwidth_bytes_per_second"])
    )


def _resource_state(
    view: Any,
    devices: Iterable[str],
    transfers: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[float, dict[str, float], dict[tuple[str, str], float]]:
    state = view.system_state
    now = float(state.get("time_ms", 0.0))
    device_available = {device_id: now for device_id in devices}
    for device_id in device_available:
        details = state.get("devices", {}).get(device_id, {})
        backlog = details.get("committed_work_ms", details.get("remaining_ms", 0.0))
        device_available[device_id] = now + float(backlog)
    link_available: dict[tuple[str, str], float] = {}
    for source, destination in transfers:
        details = state.get("links", {}).get(f"{source}->{destination}", {})
        backlog = details.get("committed_work_ms", details.get("remaining_ms", 0.0))
        link_available[(source, destination)] = now + float(backlog)
    return now, device_available, link_available


def _schedule(
    view: Any,
    dag: Any,
    assignments: Mapping[str, Mapping[str, str]],
    prepared: _Prepared,
) -> tuple[float, float]:
    """Replay a placement with evaluator-visible queues and directed links."""

    parents = _parents(dag)
    by_id = {node.node_id: node for node in dag.nodes}
    used_devices = sorted({choice["device_id"] for choice in assignments.values()})
    now, device_available, link_available = _resource_state(
        view, used_devices, prepared.transfers
    )
    node_finish: dict[str, float] = {}
    transfer_ready = {node_id: now for node_id in by_id}
    remaining = set(by_id)

    def reserve(source: str, destination: str, ready: float, size_bytes: int) -> float:
        if source == destination:
            return ready
        key = (source, destination)
        start = max(ready, link_available[key])
        finish = start + _transfer_ms(prepared.transfers, source, destination, size_bytes)
        link_available[key] = finish
        return finish

    while remaining:
        ready_nodes = sorted(
            node_id for node_id in remaining
            if all(parent in node_finish for parent in parents[node_id])
        )
        if not ready_nodes:
            raise ValueError("DAG contains a cycle or unknown predecessor")
        node_id = ready_nodes[0]
        node = by_id[node_id]
        device_id = assignments[node_id]["device_id"]
        profile = prepared.profiles_by_tool[node.tool_id][device_id]
        input_ready = transfer_ready[node_id]
        for source in node.inputs.values():
            if source.kind == "request":
                input_ready = max(
                    input_ready,
                    reserve(
                        _END_DEVICE,
                        device_id,
                        now,
                        _REQUEST_BYTES[source.data_type or "text"],
                    ),
                )
        execution_start = max(input_ready, device_available[device_id])
        finish = execution_start + float(profile["warm_latency_p95_ms"])
        node_finish[node_id] = finish
        device_available[device_id] = finish

        for child_id, child in by_id.items():
            occurrences = sum(
                1 for source in child.inputs.values()
                if source.kind == "node" and source.name == node_id
            )
            if not occurrences:
                continue
            child_device = assignments[child_id]["device_id"]
            for _ in range(occurrences):
                if device_id == child_device:
                    transfer_ready[child_id] = max(transfer_ready[child_id], finish)
                else:
                    transfer_ready[child_id] = max(
                        transfer_ready[child_id],
                        reserve(
                            device_id,
                            child_device,
                            finish,
                            prepared.output_bytes_by_tool[node.tool_id],
                        ),
                    )
        remaining.remove(node_id)

    completion = max(node_finish.values(), default=now)
    for output in dag.final_outputs:
        node = by_id[output.node_id]
        device_id = assignments[node.node_id]["device_id"]
        completion = max(
            completion,
            reserve(
                device_id,
                _END_DEVICE,
                node_finish[node.node_id],
                prepared.output_bytes_by_tool[node.tool_id],
            ),
        )
    resource = sum(
        float(prepared.profiles_by_tool[by_id[node_id].tool_id][choice["device_id"]]["gpu_memory_mib"])
        for node_id, choice in assignments.items()
    )
    return completion - now, resource


def _placement_objective(view: Any, latency_ms: float, resource_mib: float) -> float:
    """Return the complement of ScoringContext's reciprocal placement score."""

    context = getattr(view, "scoring_context", None)
    latency_weight = float(getattr(context, "latency_weight", 1.0 / 3.0))
    resource_weight = float(getattr(context, "resource_weight", 1.0 / 3.0))
    latency_scale = float(getattr(context, "latency_scale_ms", 100.0))
    resource_scale = float(getattr(context, "resource_scale_mib", 8192.0))
    denominator = latency_weight + resource_weight
    if denominator <= 0.0:
        latency_weight, resource_weight, denominator = 1.0, 0.0, 1.0
    latency_weight /= denominator
    resource_weight /= denominator
    latency_indicator = 1.0 / (1.0 + max(0.0, latency_ms) / latency_scale)
    resource_indicator = 1.0 / (1.0 + max(0.0, resource_mib) / resource_scale)
    return latency_weight * (1.0 - latency_indicator) + resource_weight * (1.0 - resource_indicator)


def _evaluation_key(view: Any, dag: Any, assignments: Mapping[str, Mapping[str, str]], prepared: _Prepared) -> tuple[float, float, float]:
    latency, resource = _schedule(view, dag, assignments, prepared)
    return (_placement_objective(view, latency, resource), latency, resource)


def _topological_order(dag: Any) -> tuple[str, ...]:
    parents = _parents(dag)
    remaining = set(parents)
    order: list[str] = []
    while remaining:
        ready = sorted(node_id for node_id in remaining if all(parent in order for parent in parents[node_id]))
        if not ready:
            raise ValueError("DAG contains a cycle or unknown predecessor")
        order.extend(ready)
        remaining.difference_update(ready)
    return tuple(order)


def _idle_devices(view: Any, devices: Iterable[str]) -> set[str]:
    state = getattr(view, "system_state", {})
    idle: set[str] = set()
    for device_id in devices:
        details = state.get("devices", {}).get(device_id, {})
        backlog = details.get("committed_work_ms", details.get("remaining_ms", 0.0))
        if float(backlog) <= 0.0:
            idle.add(device_id)
    return idle


def _initial_assignments(
    dag: Any,
    prepared: _Prepared,
    view: Any | None = None,
) -> dict[str, dict[str, str]]:
    """Construct a placement with the paper's experience-driven mapping.

    The original ETRM tracks resource frequency, uses greedy mapping with
    ``p_greedy=1`` initially, then decays it by 0.95.  Energy is unavailable
    in this repository, so the measured warm latency plus observed queue is
    used for the greedy comparison.  ETRM candidates are restricted to
    previously used and currently idle compatible devices; when no such
    device exists, all compatible devices are considered.
    """

    assignments: dict[str, dict[str, str]] = {}
    frequency: dict[str, int] = {}
    idle_devices = _idle_devices(view, prepared.devices) if view is not None else set(prepared.devices)
    queue_ms = {
        device_id: float(
            getattr(view, "system_state", {}).get("devices", {}).get(device_id, {}).get(
                "committed_work_ms", 0.0
            )
        )
        for device_id in prepared.devices
    }
    rng = random.Random(0)
    p_greedy = 1.0
    by_id = {node.node_id: node for node in dag.nodes}
    for node_id in _topological_order(dag):
        node = by_id[node_id]
        profiles = prepared.profiles_by_tool[node.tool_id]
        compatible = tuple(sorted(profiles))
        experienced_idle = tuple(
            device_id for device_id in compatible
            if frequency.get(device_id, 0) > 0 and device_id in idle_devices
        )
        use_greedy = rng.random() < p_greedy
        choices = compatible if use_greedy or not experienced_idle else experienced_idle
        # ETRM's frequency guides the tie break after the observable
        # latency/queue estimate.  No energy value is invented.
        device_id = min(
            choices,
            key=lambda candidate: (
                float(profiles[candidate]["warm_latency_p95_ms"]) + queue_ms[candidate],
                -frequency.get(candidate, 0),
                float(profiles[candidate]["gpu_memory_mib"]),
                candidate,
            ),
        )
        assignments[node_id] = {
            "configuration_id": prepared.selected_configs[node.tool_id],
            "device_id": device_id,
        }
        frequency[device_id] = frequency.get(device_id, 0) + 1
        p_greedy *= 0.95
    return assignments


def _candidate_if_better(
    view: Any,
    dag: Any,
    current: Mapping[str, Mapping[str, str]],
    candidate: Mapping[str, Mapping[str, str]],
    prepared: _Prepared,
) -> dict[str, dict[str, str]]:
    current_key = _evaluation_key(view, dag, current, prepared)
    candidate_key = _evaluation_key(view, dag, candidate, prepared)
    if candidate_key < current_key:
        return {node_id: dict(choice) for node_id, choice in candidate.items()}
    return {node_id: dict(choice) for node_id, choice in current.items()}


def _op_single_node_remap(view: Any, dag: Any, current: Mapping[str, Mapping[str, str]], prepared: _Prepared) -> dict[str, dict[str, str]]:
    parents = _parents(dag)
    children = _children(parents)
    node_id = max(sorted(parents), key=lambda value: (len(parents[value]) + len(children[value]), value))
    best = {key: dict(value) for key, value in current.items()}
    for device_id in sorted(prepared.profiles_by_tool[next(node.tool_id for node in dag.nodes if node.node_id == node_id)]):
        candidate = {key: dict(value) for key, value in current.items()}
        candidate[node_id]["device_id"] = device_id
        best = _candidate_if_better(view, dag, best, candidate, prepared)
    return best


def _op_parent_colocation(view: Any, dag: Any, current: Mapping[str, Mapping[str, str]], prepared: _Prepared) -> dict[str, dict[str, str]]:
    by_id = {node.node_id: node for node in dag.nodes}
    best = {key: dict(value) for key, value in current.items()}
    for child_id in sorted(_parents(dag)):
        for parent_id in sorted(_parents(dag)[child_id]):
            target = current[parent_id]["device_id"]
            if target not in prepared.profiles_by_tool[by_id[child_id].tool_id]:
                continue
            candidate = {key: dict(value) for key, value in current.items()}
            candidate[child_id]["device_id"] = target
            best = _candidate_if_better(view, dag, best, candidate, prepared)
    return best


def _op_chain_colocation(view: Any, dag: Any, current: Mapping[str, Mapping[str, str]], prepared: _Prepared) -> dict[str, dict[str, str]]:
    by_id = {node.node_id: node for node in dag.nodes}
    parents = _parents(dag)
    best = {key: dict(value) for key, value in current.items()}
    for child_id in sorted(parents):
        for parent_id in sorted(parents[child_id]):
            shared = sorted(
                set(prepared.profiles_by_tool[by_id[parent_id].tool_id])
                & set(prepared.profiles_by_tool[by_id[child_id].tool_id])
            )
            for device_id in shared:
                candidate = {key: dict(value) for key, value in current.items()}
                candidate[parent_id]["device_id"] = device_id
                candidate[child_id]["device_id"] = device_id
                best = _candidate_if_better(view, dag, best, candidate, prepared)
    return best


def _op_greedy_best_position(view: Any, dag: Any, current: Mapping[str, Mapping[str, str]], prepared: _Prepared) -> dict[str, dict[str, str]]:
    best = {key: dict(value) for key, value in current.items()}
    by_id = {node.node_id: node for node in dag.nodes}
    for node_id in sorted(by_id):
        for device_id in sorted(prepared.profiles_by_tool[by_id[node_id].tool_id]):
            candidate = {key: dict(value) for key, value in best.items()}
            candidate[node_id]["device_id"] = device_id
            best = _candidate_if_better(view, dag, best, candidate, prepared)
    return best


def _op_swap_placements(view: Any, dag: Any, current: Mapping[str, Mapping[str, str]], prepared: _Prepared) -> dict[str, dict[str, str]]:
    by_id = {node.node_id: node for node in dag.nodes}
    node_ids = sorted(by_id)
    best = {key: dict(value) for key, value in current.items()}
    for left_index, left_id in enumerate(node_ids):
        for right_id in node_ids[left_index + 1:]:
            left_device = current[left_id]["device_id"]
            right_device = current[right_id]["device_id"]
            if right_device not in prepared.profiles_by_tool[by_id[left_id].tool_id] or left_device not in prepared.profiles_by_tool[by_id[right_id].tool_id]:
                continue
            candidate = {key: dict(value) for key, value in current.items()}
            candidate[left_id]["device_id"], candidate[right_id]["device_id"] = right_device, left_device
            best = _candidate_if_better(view, dag, best, candidate, prepared)
    return best


def _op_transfer_aware(view: Any, dag: Any, current: Mapping[str, Mapping[str, str]], prepared: _Prepared) -> dict[str, dict[str, str]]:
    by_id = {node.node_id: node for node in dag.nodes}
    parents = _parents(dag)
    best = {key: dict(value) for key, value in current.items()}
    candidates = sorted(parents, key=lambda node_id: (-len(parents[node_id]), node_id))
    for child_id in candidates:
        if not parents[child_id]:
            continue
        for device_id in sorted(prepared.profiles_by_tool[by_id[child_id].tool_id]):
            candidate = {key: dict(value) for key, value in current.items()}
            candidate[child_id]["device_id"] = device_id
            best = _candidate_if_better(view, dag, best, candidate, prepared)
    return best


_OPERATORS = (
    _op_single_node_remap,
    _op_parent_colocation,
    _op_chain_colocation,
    _op_greedy_best_position,
    _op_swap_placements,
    _op_transfer_aware,
)


def _apply_action(action: int, view: Any, dag: Any, current: Mapping[str, Mapping[str, str]], prepared: _Prepared) -> dict[str, dict[str, str]]:
    if not 0 <= action < ACTION_COUNT:
        raise ValueError(f"unknown QPHH action: {action}")
    return _OPERATORS[action](view, dag, current, prepared)


def state_from_delta(delta: float) -> int:
    """Map logarithmic objective improvement to the paper's ten states."""

    if delta < 0.0:
        return 0
    if delta == 0.0:
        return 1
    for z in range(1, 7):
        if 10.0 ** (-z) <= delta < 10.0 ** (-(z - 1)):
            return z + 1
    if delta >= 1.0:
        return 8
    return 9


def reward_from_delta(delta: float) -> float:
    """Return the paper's improvement reward bins."""

    if delta < 0.0:
        return -10.0
    for z in range(1, 6):
        if 10.0 ** (-(z + 1)) <= delta <= 10.0 ** (-z):
            return float(10 - z)
    if delta >= 10.0 ** -1:
        return 10.0
    return 0.0


def epsilon_greedy_action(
    table: Any,
    state: int,
    epsilon: float = 0.0,
    rng: random.Random | None = None,
    tie_break: int = 0,
) -> int:
    """Select an action with stable argmax ties and optional exploration."""

    if not 0 <= state < STATE_COUNT:
        raise ValueError(f"invalid QPHH state: {state}")
    generator = rng or random.Random(0)
    if epsilon > 0.0 and generator.random() < epsilon:
        return generator.randrange(ACTION_COUNT)
    row = table[state]
    values = [float(value) for value in row]
    best_value = max(values)
    tied = [action for action, value in enumerate(values) if value == best_value]
    return tied[tie_break % len(tied)]


def update_q_value(
    table: Any,
    state: int,
    action: int,
    reward: float,
    next_state: int,
    alpha: float,
    gamma: float = DEFAULT_GAMMA,
) -> float:
    """Apply one Bellman update to a list table or a torch tensor table."""

    if not (0 <= state < STATE_COUNT and 0 <= next_state < STATE_COUNT):
        raise ValueError("QPHH state outside table")
    if not 0 <= action < ACTION_COUNT:
        raise ValueError("QPHH action outside table")
    if not 0.0 <= alpha <= 1.0 or not 0.0 <= gamma <= 1.0:
        raise ValueError("QPHH alpha and gamma must be in [0, 1]")
    current = table[state, action] if _supports_two_index(table) else table[state][action]
    next_row = table[next_state]
    next_best = next_row.max() if hasattr(next_row, "max") else max(next_row)
    updated = current + alpha * (reward + gamma * next_best - current)
    if _supports_two_index(table):
        table[state, action] = updated
        try:
            return float(updated.detach().cpu().item())
        except AttributeError:
            return float(updated)
    table[state][action] = updated
    return float(updated)


def _supports_two_index(value: Any) -> bool:
    try:
        value[0, 0]
    except (TypeError, IndexError, KeyError):
        return False
    return True


def train_q_table(
    transitions: Iterable[tuple[int, int, float, int]],
    *,
    initial_table: Iterable[Iterable[float]] | None = None,
    alpha: float = 0.2,
    gamma: float = DEFAULT_GAMMA,
) -> tuple[tuple[float, ...], ...]:
    """Train a small Q table from ``(state, action, reward, next_state)`` rows.

    The function is intentionally framework-free so the experiment entrypoint
    can use the same update formula on a CUDA tensor and export its result.
    """

    table = [list(row) for row in (initial_table or Q_TABLE)]
    for state, action, reward, next_state in transitions:
        update_q_value(table, state, action, reward, next_state, alpha, gamma)
    return tuple(tuple(float(value) for value in row) for row in table)


def propose(view: Any) -> SchedulerOutput:
    """Select the longest candidate DAG and refine a quality-first placement."""

    candidates = tuple(view.candidate_dags)
    path_index = max(
        range(len(candidates)),
        key=lambda index: (_longest_chain(candidates[index]), -index),
    )
    dag = candidates[path_index]
    prepared = _prepare(view, dag)
    assignments = _initial_assignments(dag, prepared, view)
    current_key = _evaluation_key(view, dag, assignments, prepared)
    state = 1

    # Inference is deterministic (epsilon=0).  A finite budget keeps the
    # scheduler computation bounded even for large DAGs.
    budget = min(12, max(2, 2 * len(assignments)))
    for iteration in range(budget):
        # Tie rotation is only relevant for the paper's all-one
        # initialization.  A learned table normally has a unique argmax;
        # the rotation keeps a fresh table from exercising one LLH forever.
        action = epsilon_greedy_action(Q_TABLE, state, epsilon=0.0, tie_break=iteration)
        candidate = _apply_action(action, view, dag, assignments, prepared)
        candidate_key = _evaluation_key(view, dag, candidate, prepared)
        delta = 0.0
        if current_key[0] > 0.0 and candidate_key[0] > 0.0:
            delta = math.log(current_key[0]) - math.log(candidate_key[0])
        elif current_key[0] > candidate_key[0]:
            delta = 1.0
        if candidate_key < current_key:
            assignments, current_key = candidate, candidate_key
        state = state_from_delta(delta)
        # Once the chosen operator cannot improve the placement, the next
        # state remains deterministic and QPHH may select another operator.

    return SchedulerOutput(dag, assignments, path_index=path_index)


__all__ = [
    "ACTION_NAMES",
    "ACTION_COUNT",
    "DEFAULT_EPSILON",
    "DEFAULT_GAMMA",
    "Q_TABLE",
    "STATE_COUNT",
    "epsilon_greedy_action",
    "propose",
    "reward_from_delta",
    "state_from_delta",
    "train_q_table",
    "update_q_value",
]
