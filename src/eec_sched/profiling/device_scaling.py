"""Synthetic device-count scenarios with explicit capacity, work, and topology."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from typing import Any

# Relative experiment units. These are not vendor throughput or rental prices.
GPU_ARCHETYPES = {
    "rtx5060": {"compute_capacity": 1.0, "memory_capacity_mib": 8192, "cost_per_hour": 1.0},
    "rtx3090": {"compute_capacity": 1.8, "memory_capacity_mib": 24576, "cost_per_hour": 2.4},
    "rtx4090": {"compute_capacity": 3.2, "memory_capacity_mib": 24576, "cost_per_hour": 5.0},
}
DEFAULT_GPU_MODELS = {"device": "rtx5060", "edge": "rtx3090", "cloud": "rtx4090"}


def _number(value: Any, name: str, *, positive: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return float(value)


def calibration_from_snapshot(snapshot: Mapping[str, Any], gpu_models: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Extract three reference measurements per tool and a network scale."""

    models = DEFAULT_GPU_MODELS if gpu_models is None else gpu_models
    ids = [item["device_id"] for item in snapshot["devices"]]
    if len(ids) != 3 or len(set(ids)) != 3 or set(ids) != set(models):
        raise ValueError("provide one GPU model for each of three distinct anchor devices")
    if set(models.values()) != set(GPU_ARCHETYPES):
        raise ValueError("anchors must contain one each of rtx5060, rtx3090, and rtx4090")
    devices = [
        {"device_id": device_id, "device_ordinal": index, "gpu_model": models[device_id], **GPU_ARCHETYPES[models[device_id]]}
        for index, device_id in enumerate(ids)
    ]
    tools = []
    for tool in snapshot["tools"]:
        if tool["eligibility"]["status"] != "eligible":
            continue
        reference_id = tool["quality_contract"]["reference"]["configuration_id"]
        quality = next(item for item in tool["quality_profiles"] if item["configuration_id"] == reference_id)
        execution = {item["device_id"]: item for item in tool["execution_profiles"] if item["configuration_id"] == reference_id}
        if set(execution) != set(ids):
            raise ValueError(f"reference Configuration of {tool['tool_id']} needs all three Execution Profiles")
        tools.append({
            "tool_id": tool["tool_id"],
            "reference_configuration_id": reference_id,
            "reference_quality_lcb": quality["normalized_quality_lcb"],
            "reference_output_bytes": quality["representative_output_bytes"],
            "reference_gpu_memory_mib": execution[ids[0]]["gpu_memory_mib"],
            "reference_latency_ms": {device_id: execution[device_id]["warm_latency_p95_ms"] for device_id in ids},
        })
    transfers = snapshot["transfer_profiles"]
    if not transfers:
        raise ValueError("source snapshot needs transfer profiles for a network scale")
    bandwidths = [_number(item["bandwidth_bytes_per_second"], "bandwidth") for item in transfers]
    delays = [_number(item["propagation_delay_ms"], "delay", positive=False) for item in transfers]
    return {
        "source_snapshot_id": snapshot["snapshot_id"],
        "source_snapshot_digest": snapshot["snapshot_digest"],
        "source_data_kind": snapshot["data_kind"],
        "devices": devices,
        "tools": tools,
        "network_bandwidth_scale": math.exp(sum(map(math.log, bandwidths)) / len(bandwidths)),
        "network_delay_scale_ms": sum(delays) / len(delays),
    }


def _cost_for_capacity(capacity: float, anchors: list[dict[str, Any]]) -> float:
    ordered = sorted(anchors, key=lambda item: item["compute_capacity"])
    for left, right in zip(ordered, ordered[1:]):
        lo, hi = left["compute_capacity"], right["compute_capacity"]
        if lo <= capacity <= hi:
            fraction = math.log(capacity / lo) / math.log(hi / lo)
            return left["cost_per_hour"] * (right["cost_per_hour"] / left["cost_per_hour"]) ** fraction
    raise ValueError("generated capacity outside anchor range")


def _devices(anchors: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    capacities = [item["compute_capacity"] for item in anchors]
    low, high = min(capacities), max(capacities)
    devices = [{**item, "kind": "anchor"} for item in anchors]
    for index in range(count - 3):
        fraction = (index + 0.5) / (count - 3)
        capacity = low * (high / low) ** fraction
        if any(math.isclose(capacity, other, rel_tol=1e-12) for other in capacities):
            capacity = low * (high / low) ** ((index + 0.25) / (count - 3))
        nearest = min(anchors, key=lambda item: abs(math.log(capacity / item["compute_capacity"])))
        devices.append({
            "device_id": f"sim-{index + 3:03d}", "device_ordinal": index + 3,
            "gpu_model": f"synthetic-near-{nearest['gpu_model']}",
            "compute_capacity": capacity, "memory_capacity_mib": nearest["memory_capacity_mib"],
            "cost_per_hour": _cost_for_capacity(capacity, anchors), "kind": "synthetic",
        })
    if len({item["device_id"] for item in devices}) != count:
        raise ValueError("synthetic device IDs collide with anchor IDs")
    return devices


def _links(devices: list[dict[str, Any]], bandwidth: float, delay: float, decay: float, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    links = []
    for i, source in enumerate(devices):
        for destination in devices[i + 1:]:
            distance = abs(source["device_ordinal"] - destination["device_ordinal"])
            # Neighbor links guarantee connectivity; long links become rarer.
            if distance != 1 and rng.random() >= 0.45 * math.exp(-distance / 4):
                continue
            for start, end in ((source, destination), (destination, source)):
                links.append({
                    "source_device_id": start["device_id"], "destination_device_id": end["device_id"],
                    "ordinal_distance": distance,
                    "bandwidth_bytes_per_second": bandwidth * math.exp(-decay * distance) * rng.uniform(0.85, 1.15),
                    "propagation_delay_ms": delay * (1 + decay * distance),
                })
    return links


def build_device_scaling_scenario(
    calibration: Mapping[str, Any], *, device_count: int, configurations_per_tool: int,
    quality_gain: float = 0.10, work_gain: float = 0.50, work_exponent: float = 1.4,
    memory_gain: float = 0.25, network_distance_decay: float = 0.12, network_seed: int = 0,
) -> dict[str, Any]:
    """Generate a synthetic scenario; latency = Configuration work / device capacity."""

    if isinstance(device_count, bool) or not isinstance(device_count, int) or device_count < 3:
        raise ValueError("device_count must be an integer >= 3")
    if isinstance(configurations_per_tool, bool) or not isinstance(configurations_per_tool, int) or configurations_per_tool < 1:
        raise ValueError("configurations_per_tool must be a positive integer")
    for name, value in (("quality_gain", quality_gain), ("work_gain", work_gain),
                        ("memory_gain", memory_gain), ("network_distance_decay", network_distance_decay)):
        _number(value, name, positive=False)
    _number(work_exponent, "work_exponent")
    if isinstance(network_seed, bool) or not isinstance(network_seed, int):
        raise ValueError("network_seed must be an integer")
    anchors = list(calibration["devices"])
    if len(anchors) != 3 or len({item["device_id"] for item in anchors}) != 3:
        raise ValueError("exactly three distinct anchors are required")
    for item in anchors:
        for key in ("compute_capacity", "cost_per_hour", "memory_capacity_mib"):
            _number(item[key], key)
    ordered = sorted(anchors, key=lambda item: item["compute_capacity"])
    if len({item["compute_capacity"] for item in anchors}) != 3:
        raise ValueError("anchor compute capacities must differ")
    if any(right["cost_per_hour"] <= left["cost_per_hour"] or
           right["cost_per_hour"] / right["compute_capacity"] <= left["cost_per_hour"] / left["compute_capacity"]
           for left, right in zip(ordered, ordered[1:])):
        raise ValueError("cost and cost per compute unit must increase with compute capacity")
    devices = _devices(anchors, device_count)
    links = _links(devices, _number(calibration["network_bandwidth_scale"], "network bandwidth"),
                   _number(calibration["network_delay_scale_ms"], "network delay", positive=False),
                   network_distance_decay, network_seed)
    if not calibration["tools"]:
        raise ValueError("at least one eligible tool is required")
    tools = []
    for tool in calibration["tools"]:
        observed = tool["reference_latency_ms"]
        if set(observed) != {item["device_id"] for item in anchors}:
            raise ValueError(f"{tool['tool_id']}: missing reference latency on an anchor")
        log_work = [math.log(_number(observed[item["device_id"]], "reference latency") * item["compute_capacity"])
                    for item in anchors]
        mean_log_work = sum(log_work) / len(log_work)
        work = math.exp(mean_log_work)
        rmse = math.sqrt(sum((value - mean_log_work) ** 2 for value in log_work) / len(log_work))
        quality = _number(tool["reference_quality_lcb"], "quality", positive=False)
        if quality > 1:
            raise ValueError("quality must be in [0, 1]")
        memory = _number(tool["reference_gpu_memory_mib"], "GPU memory")
        output_bytes = tool["reference_output_bytes"]
        if isinstance(output_bytes, bool) or not isinstance(output_bytes, int) or output_bytes < 0:
            raise ValueError("output bytes must be a nonnegative integer")
        configurations = []
        for index in range(configurations_per_tool):
            z = index / (configurations_per_tool - 1) if configurations_per_tool > 1 else 0.0
            demand = work * (1 + work_gain * z ** work_exponent)
            required_memory = memory * (1 + memory_gain * z)
            profiles = []
            for device in devices:
                compatible = required_memory <= device["memory_capacity_mib"]
                latency = demand / device["compute_capacity"] if compatible else None
                profiles.append({
                    "device_id": device["device_id"], "compatible": compatible,
                    "warm_latency_p95_ms": latency, "gpu_memory_mib": required_memory,
                    "execution_cost": latency * device["cost_per_hour"] / 3_600_000 if compatible else None,
                })
            configurations.append({
                "configuration_id": f"synthetic-demand-{index + 1:03d}", "demand_level": z,
                "compute_demand": demand, "normalized_quality_lcb": min(1.0, quality + quality_gain * z),
                "representative_output_bytes": output_bytes, "execution_profiles": profiles,
            })
        tools.append({
            "tool_id": tool["tool_id"], "reference_configuration_id": tool["reference_configuration_id"],
            "fitted_reference_compute_demand": work, "fit_rmse_log_latency": rmse,
            "configurations": configurations,
        })
    scenario: dict[str, Any] = {
        "schema_version": "device-scaling-scenario/2", "data_kind": "synthetic",
        "source_snapshot_id": calibration["source_snapshot_id"],
        "source_snapshot_digest": calibration["source_snapshot_digest"],
        "source_data_kind": calibration["source_data_kind"],
        "assumptions": {
            "quality_gain": quality_gain, "work_gain": work_gain, "work_exponent": work_exponent,
            "memory_gain": memory_gain, "network_distance_decay": network_distance_decay,
            "network_seed": network_seed, "cost_unit": "relative_units_per_hour",
            "execution_cost_unit": "relative_units_per_execution",
            "latency_model": "compute_demand / device_compute_capacity",
            "network_model": "shared sparse ordinal-distance graph",
        },
        "devices": devices, "links": links, "tools": tools,
    }
    return scenario
