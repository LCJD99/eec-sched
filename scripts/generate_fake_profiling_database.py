"""Generate a deterministic, catalog-complete synthetic profiling snapshot.

The generated values are deliberately invented.  This script imports the
side-effect-free MnMS catalog, but never constructs runners, loads models, or
calls external services.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from eec_sched.domain import Configuration, ToolSpec
from eec_sched.mnms_tools import mnms_tool_specs
from eec_sched.profiling.snapshot import snapshot_digest, validate_profiling_database


ROOT = Path(__file__).parents[1]
DEFAULT_OUTPUT = ROOT / "docs/examples/profiling-database.fake.json"
DEFAULT_SCHEMA = ROOT / "docs/schemas/profiling-database.schema.json"
CREATED_AT = "2026-08-07T00:00:00Z"
DEVICES = ("device", "edge", "cloud")


def _provenance() -> list[dict[str, Any]]:
    common = {
        "measurement_kind": "synthetic",
        "measured_at": CREATED_AT,
        "operator": "deterministic-fixture-generator",
        "source_revision": "synthetic-not-measured",
        "environment": {"synthetic": True, "external_services_used": False},
        "notes": "Invented deterministic values for development only; not a real measurement.",
    }
    return [
        {
            **common,
            "provenance_id": "synthetic-quality-v1",
            "subject_kind": "quality",
            "command": "uv run python scripts/generate_fake_profiling_database.py",
            "protocol": {
                "dataset": "synthetic/mnms-catalog-fixture",
                "dataset_revision": "synthetic-v1",
                "split": "synthetic",
                "sample_ids": [f"synthetic-{index:03d}" for index in range(1, 101)],
                "preprocessing": "No samples processed; deterministic formula only.",
                "scorer": "synthetic-unit-score",
                "scorer_version": "synthetic-v1",
                "confidence_method": "synthetic-fixed-width-interval",
                "model_revision": "catalog-declared-or-synthetic-reference",
                "random_seed": 0,
            },
        },
        {
            **common,
            "provenance_id": "synthetic-execution-v1",
            "subject_kind": "execution",
            "command": "uv run python scripts/generate_fake_profiling_database.py",
            "protocol": {
                "input_sample_ids": [f"synthetic-{index:03d}" for index in range(1, 51)],
                "warmup_count": 5,
                "observation_count": 50,
                "timing_method": "No timing performed; deterministic formula only.",
                "resource_measurement_method": "Synthetic peak GPU memory formula only; no device measurement performed.",
                "model_revision": "catalog-declared-or-synthetic-reference",
                "runtime_versions": {"generator": "synthetic-v1", "runtime": "not-invoked"},
            },
        },
        {
            **common,
            "provenance_id": "synthetic-transfer-v1",
            "subject_kind": "transfer",
            "command": "uv run python scripts/generate_fake_profiling_database.py",
            "protocol": {
                "payload_sizes_bytes": [0, 1024, 1048576],
                "repetition_count": 50,
                "timing_method": "No transfer performed; deterministic formula only.",
            },
        },
    ]


def _dimensions(configurations: Sequence[Configuration]) -> dict[str, Any]:
    effort_values = [configuration.configuration_id for configuration in configurations]
    effort_applicable = len(effort_values) > 1
    return {
        "model_capacity": {"applicable": False, "values": ["catalog-fixed"]},
        "input_fidelity": {"applicable": False, "values": ["catalog-fixed"]},
        "inference_effort": {
            "applicable": effort_applicable,
            "values": effort_values if effort_applicable else [effort_values[0]],
        },
    }


def _configuration(configuration: Configuration) -> dict[str, Any]:
    return {
        "configuration_id": configuration.configuration_id,
        "parameters": dict(configuration.parameters),
        "taxonomy": {
            "model_capacity": "catalog-fixed",
            "input_fidelity": "catalog-fixed",
            "inference_effort": configuration.configuration_id,
        },
        "runtime": {"backend": "synthetic-no-runtime", "numeric_format": "synthetic"},
        "device_compatibility": [
            {"device_id": device_id, "status": "compatible", "reason": None}
            for device_id in DEVICES
        ],
    }


def _fixture_configurations(spec: ToolSpec) -> tuple[Configuration, ...]:
    if spec.configurations:
        return spec.configurations
    return (
        Configuration(
            "synthetic-reference",
            {
                "synthetic_fixture_only": True,
                "execution_supported": False,
                "note": "Not declared by the MnMS catalog; never use for real execution.",
            },
        ),
    )


def _quality_values(
    configuration_index: int,
    configuration_count: int,
    higher_is_better: bool,
) -> tuple[float, float, float, float]:
    progress = (configuration_index + 1) / configuration_count
    point = (0.6 + 0.3 * progress) if higher_is_better else (0.4 - 0.3 * progress)
    point = round(point, 12)
    lower = round(point - 0.02, 12)
    upper = round(point + 0.02, 12)
    semantic_floor = 0.0 if higher_is_better else 1.0
    reference_metric = 0.9 if higher_is_better else 0.1
    conservative = lower if higher_is_better else upper
    normalized = (
        (conservative - semantic_floor) / (reference_metric - semantic_floor)
        if higher_is_better
        else (semantic_floor - conservative) / (semantic_floor - reference_metric)
    )
    return point, lower, upper, max(0.0, min(1.0, normalized))


def _representative_output_bytes(spec: ToolSpec) -> int:
    image_outputs = sum(port.modality == "image" for port in spec.outputs.values())
    text_outputs = sum(port.modality == "text" for port in spec.outputs.values())
    return image_outputs * 262144 + text_outputs * 256


def _eligible_tool(spec: ToolSpec, tool_index: int) -> dict[str, Any]:
    configurations = _fixture_configurations(spec)
    higher_is_better = spec.metric_higher_is_better is not False
    reference = configurations[-1]
    semantic_floor = 0.0 if higher_is_better else 1.0
    reference_metric = 0.9 if higher_is_better else 0.1
    quality_profiles: list[dict[str, Any]] = []
    execution_profiles: list[dict[str, Any]] = []
    device_latency_factors = {"device": 2.5, "edge": 1.3, "cloud": 1.0}
    device_memory_factors = {"device": 0.8, "edge": 1.0, "cloud": 1.2}

    for configuration_index, configuration in enumerate(configurations):
        point, lower, upper, normalized = _quality_values(
            configuration_index, len(configurations), higher_is_better
        )
        quality_profiles.append(
            {
                "configuration_id": configuration.configuration_id,
                "raw_metric": {
                    "point_estimate": point,
                    "confidence_interval": {
                        "level": 0.95,
                        "lower": lower,
                        "upper": upper,
                        "method": "synthetic-fixed-width-interval",
                    },
                    "sample_count": 100,
                },
                "normalized_quality_lcb": normalized,
                "representative_output_bytes": _representative_output_bytes(spec),
                "provenance_id": "synthetic-quality-v1",
            }
        )
        for device_id in DEVICES:
            base_latency = 10.0 + tool_index * 3.0 + configuration_index * 5.0
            base_memory = 1024.0 + tool_index * 128.0 + configuration_index * 64.0
            execution_profiles.append(
                {
                    "configuration_id": configuration.configuration_id,
                    "device_id": device_id,
                    "warm_latency_p95_ms": round(base_latency * device_latency_factors[device_id], 6),
                    "gpu_memory_mib": round(base_memory * device_memory_factors[device_id], 6),
                    "sample_count": 50,
                    "provenance_id": "synthetic-execution-v1",
                }
            )

    return {
        "tool_id": spec.tool_id,
        "eligibility": {"status": "eligible", "reason": None},
        "dimensions": _dimensions(configurations),
        "quality_contract": {
            "metric_name": "synthetic-unit-score",
            "metric_unit": "synthetic-score",
            "direction": "higher_is_better" if higher_is_better else "lower_is_better",
            "semantic_floor": semantic_floor,
            "reference": {
                "configuration_id": reference.configuration_id,
                "raw_metric": reference_metric,
            },
        },
        "configurations": [_configuration(configuration) for configuration in configurations],
        "quality_profiles": quality_profiles,
        "execution_profiles": execution_profiles,
    }


def _transfers() -> list[dict[str, Any]]:
    values = {
        ("device", "edge"): (4.0, 50_000_000),
        ("device", "cloud"): (28.0, 12_000_000),
        ("edge", "device"): (4.5, 45_000_000),
        ("edge", "cloud"): (18.0, 80_000_000),
        ("cloud", "device"): (30.0, 15_000_000),
        ("cloud", "edge"): (19.0, 75_000_000),
    }
    return [
        {
            "source_device_id": source,
            "destination_device_id": destination,
            "propagation_delay_ms": propagation,
            "bandwidth_bytes_per_second": bandwidth,
            "sample_count": 50,
            "provenance_id": "synthetic-transfer-v1",
        }
        for (source, destination), (propagation, bandwidth) in values.items()
    ]


def build_fake_profiling_database(specs: Sequence[ToolSpec] | None = None) -> dict[str, Any]:
    """Build a deterministic snapshot containing every supplied catalog tool."""

    catalog = tuple(mnms_tool_specs() if specs is None else specs)
    payload: dict[str, Any] = {
        "schema_version": "1.1.0",
        "snapshot_id": "synthetic-mnms-catalog-v1",
        "snapshot_digest": "sha256:" + "0" * 64,
        "data_kind": "synthetic",
        "created_at": CREATED_AT,
        "measurement_scope": {
            "input_bucket": "synthetic-default-v1",
            "batch_size": 1,
            "warm_execution": True,
            "execution_boundary": "Synthetic formula only; no preprocessing, inference, postprocessing, model loading, or download occurred.",
        },
        "devices": [
            {
                "device_id": "device",
                "hardware_class": "synthetic-device",
                "description": "Synthetic local device; not real hardware.",
            },
            {
                "device_id": "edge",
                "hardware_class": "synthetic-edge",
                "description": "Synthetic edge node; not real hardware.",
            },
            {
                "device_id": "cloud",
                "hardware_class": "synthetic-cloud",
                "description": "Synthetic cloud node; not real hardware.",
            },
        ],
        "provenances": _provenance(),
        "tools": [
            _eligible_tool(spec, tool_index)
            for tool_index, spec in enumerate(catalog)
        ],
        "transfer_profiles": _transfers(),
    }
    payload["snapshot_digest"] = snapshot_digest(payload)
    return payload


def write_fake_profiling_database(
    output: Path = DEFAULT_OUTPUT,
    schema_path: Path = DEFAULT_SCHEMA,
) -> Mapping[str, Any]:
    """Build, validate, then write one canonical development fixture."""

    payload = build_fake_profiling_database()
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validate_profiling_database(payload, schema)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    args = parser.parse_args()
    payload = write_fake_profiling_database(args.output, args.schema)
    print(
        f"wrote {args.output} with {len(payload['tools'])} catalog tools "
        f"({payload['snapshot_digest']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
