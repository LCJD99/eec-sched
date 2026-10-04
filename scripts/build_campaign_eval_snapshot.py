"""Adapt the three campaign device profiles to the evaluator's v1.1 snapshot."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from eec_sched.mnms_tools import mnms_tool_specs
from eec_sched.profiling.snapshot import validate_profiling_database


ROOT = Path(__file__).parents[1]
SOURCE_PATHS = {
    "device": ROOT / "profiles/5060.json",
    "edge": ROOT / "profiles/3090.json",
    "cloud": ROOT / "profiles/4090.json",
}
FALLBACK_PATH = ROOT / "docs/examples/profiling-database.fake.json"
SCHEMA_PATH = ROOT / "docs/schemas/profiling-database.schema.json"
DEFAULT_OUTPUT = ROOT / "profiles/eval-database.json"


def _provenance(
    provenance_id: str, subject_kind: str, source: dict[str, Any], source_path: Path
) -> dict[str, Any]:
    quality = subject_kind == "quality"
    protocol = (
        {
            "dataset": source["source"]["campaign_directory"],
            "dataset_revision": source["snapshot_id"],
            "split": "campaign-export",
            "sample_ids": ["campaign-export"],
            "preprocessing": "Values copied from the device-specific campaign export.",
            "scorer": "source-raw-metric",
            "scorer_version": "source-export",
            "confidence_method": "unavailable-point-estimate-only",
            "model_revision": "source-export",
            "random_seed": None,
        }
        if quality
        else {
            "input_sample_ids": ["campaign-export"],
            "warmup_count": 0,
            "observation_count": 1,
            "timing_method": "Exported latency_ms used as a p95 proxy; percentile observations unavailable.",
            "resource_measurement_method": "Exported adjusted gpu_memory_mib copied without change.",
            "model_revision": "source-export",
            "runtime_versions": {"adapter": "campaign-eval-v1", "source": "campaign-export"},
        }
    )
    return {
        "provenance_id": provenance_id,
        "subject_kind": subject_kind,
        "measurement_kind": "synthetic",
        "measured_at": source["created_at"],
        "operator": "campaign-profile-adapter",
        "source_revision": source["snapshot_id"],
        "command": "uv run python scripts/build_campaign_eval_snapshot.py",
        "environment": {"source_profile": str(source_path.relative_to(ROOT))},
        "protocol": protocol,
        "notes": (
            "Source quality is a normalized point estimate, not a confidence lower bound."
            if quality
            else "5060 Ti latency is a processed campaign value; 3090/4090 latencies are formula estimates."
        ),
    }


def _quality_contract(configurations: list[dict[str, Any]]) -> dict[str, Any]:
    first = configurations[0]
    direction = first["quality"]["normalization_direction"]
    raw_values = [item["raw_quality"]["value"] for item in configurations]
    low, high = min(raw_values), max(raw_values)
    if direction == "higher_is_better":
        reference = max(configurations, key=lambda item: item["raw_quality"]["value"])
        floor = low - 4 * (high - low) if low != high else high - 1
    else:
        reference = min(configurations, key=lambda item: item["raw_quality"]["value"])
        floor = high + 4 * (high - low) if low != high else low + 1
    return {
        "metric_name": first["raw_quality"]["metric"],
        "metric_unit": "source-metric-unit",
        "direction": direction,
        "semantic_floor": floor,
        "reference": {
            "configuration_id": reference["configuration_id"],
            "raw_metric": reference["raw_quality"]["value"],
        },
    }


def build_snapshot(
    sources: dict[str, dict[str, Any]], fallback: dict[str, Any]
) -> dict[str, Any]:
    base = sources["device"]
    source_tools = {
        device_id: {tool["tool_id"]: tool for tool in source["tools"]}
        for device_id, source in sources.items()
    }
    catalog = {spec.tool_id: spec for spec in mnms_tool_specs()}
    fallback_text_generation = deepcopy(
        next(tool for tool in fallback["tools"] if tool["tool_id"] == "text_generation")
    )
    tools = []
    for base_tool in base["tools"]:
        tool_id = base_tool["tool_id"]
        spec = catalog[tool_id]
        configurations = base_tool["configurations"]
        by_device = {
            device_id: {
                item["configuration_id"]: item
                for item in source_tools[device_id][tool_id]["configurations"]
            }
            for device_id in SOURCE_PATHS
        }
        output_bytes = sum(
            262144 if port.modality == "image" else 256
            for port in spec.outputs.values()
        )
        converted_configurations = []
        quality_profiles = []
        execution_profiles = []
        for item in configurations:
            configuration_id = item["configuration_id"]
            converted_configurations.append({
                "configuration_id": configuration_id,
                "parameters": item["parameters"],
                "taxonomy": {
                    "model_capacity": "not-classified",
                    "input_fidelity": "not-classified",
                    "inference_effort": "not-classified",
                },
                "runtime": {"backend": "campaign-export", "numeric_format": "source-unspecified"},
                "device_compatibility": [
                    {"device_id": device_id, "status": "compatible", "reason": None}
                    for device_id in SOURCE_PATHS
                ],
            })
            raw_quality = item["raw_quality"]["value"]
            quality_profiles.append({
                "configuration_id": configuration_id,
                "raw_metric": {
                    "point_estimate": raw_quality,
                    "confidence_interval": {
                        "level": 0.95,
                        "lower": raw_quality,
                        "upper": raw_quality,
                        "method": "unavailable-point-estimate-only",
                    },
                    "sample_count": 1,
                },
                "normalized_quality_lcb": item["quality"]["value"],
                "representative_output_bytes": output_bytes,
                "provenance_id": "campaign-quality",
            })
            for device_id in SOURCE_PATHS:
                execution = by_device[device_id][configuration_id]["execution"]
                execution_profiles.append({
                    "configuration_id": configuration_id,
                    "device_id": device_id,
                    "warm_latency_p95_ms": execution["latency_ms"],
                    "gpu_memory_mib": execution["gpu_memory_mib"],
                    "sample_count": 1,
                    "provenance_id": f"campaign-execution-{device_id}",
                })
        tools.append({
            "tool_id": tool_id,
            "eligibility": {"status": "eligible", "reason": None},
            "dimensions": {
                key: {"applicable": False, "values": ["not-classified"]}
                for key in ("model_capacity", "input_fidelity", "inference_effort")
            },
            "quality_contract": _quality_contract(configurations),
            "configurations": converted_configurations,
            "quality_profiles": quality_profiles,
            "execution_profiles": execution_profiles,
        })
    tools.append(fallback_text_generation)
    payload = {
        "schema_version": "1.1.0",
        "snapshot_id": "campaign-20260828-three-gpu-eval-v1",
        "snapshot_digest": "campaign-20260828-three-gpu-eval-v1",
        "data_kind": "synthetic",
        "created_at": max(source["created_at"] for source in sources.values()),
        "measurement_scope": {
            "input_bucket": "campaign-export",
            "batch_size": 1,
            "warm_execution": True,
            "execution_boundary": "Source campaign latency used as a warm p95 proxy; source does not provide percentile observations.",
        },
        "devices": [
            {
                "device_id": device_id,
                "hardware_class": source["device"]["hardware_class"],
                "description": source["device"]["description"],
            }
            for device_id, source in sources.items()
        ],
        "provenances": [
            _provenance("campaign-quality", "quality", base, SOURCE_PATHS["device"]),
            *(
                _provenance(
                    f"campaign-execution-{device_id}", "execution", source, SOURCE_PATHS[device_id]
                )
                for device_id, source in sources.items()
            ),
            *deepcopy(fallback["provenances"]),
        ],
        "tools": tools,
        "transfer_profiles": deepcopy(fallback["transfer_profiles"]),
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    sources = {
        device_id: json.loads(path.read_text(encoding="utf-8"))
        for device_id, path in SOURCE_PATHS.items()
    }
    fallback = json.loads(FALLBACK_PATH.read_text(encoding="utf-8"))
    payload = build_snapshot(sources, fallback)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validate_profiling_database(payload, schema)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
