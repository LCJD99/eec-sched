"""Export Bayesian campaign results as a device-specific profiling database."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping


def _configuration_id(tool_id: str, parameters: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        {"tool_id": tool_id, "parameters": parameters},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "campaign-" + hashlib.sha256(encoded).hexdigest()[:24]


def _campaign_files(directory: Path) -> list[Path]:
    direct = directory / "campaign.json"
    if direct.is_file():
        return [direct]
    return sorted(directory.glob("*/campaign.json"))


def _device(device: Mapping[str, Any]) -> dict[str, str]:
    required = ("device_id", "hardware_class", "description")
    missing = [key for key in required if not str(device.get(key, "")).strip()]
    if missing:
        raise ValueError(f"device is missing required fields: {missing}")
    return {key: str(device[key]) for key in required}


def export_campaign_database(
    campaign_directory: Path,
    device: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    """Export successful campaign observations for one measured device.

    ``campaign_directory`` may be a model campaign directory containing one
    ``campaign.json`` or a campaign root containing one directory per tool.
    The exporter prefers ``selected_results`` and falls back to ``results``.
    Failed observations are omitted; no objective value is invented.
    """
    files = _campaign_files(campaign_directory)
    if not files:
        raise FileNotFoundError(f"no campaign.json found below {campaign_directory}")
    target_device = _device(device)
    tools: list[dict[str, Any]] = []
    source_files: list[str] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_files.append(str(path.resolve()))
        rows = payload.get("selected_results") or payload.get("results") or []
        configurations: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in rows:
            if row.get("status", "success") != "success":
                continue
            required = ("configuration", "quality", "latency_ms", "gpu_memory_mib")
            missing = [key for key in required if key not in row or row[key] is None]
            if missing:
                raise ValueError(f"{path}: successful result is missing {missing}")
            parameters = dict(row["configuration"])
            configuration_id = _configuration_id(str(payload["tool"]), parameters)
            if configuration_id in seen:
                continue
            seen.add(configuration_id)
            configurations.append(
                {
                    "configuration_id": configuration_id,
                    "parameters": parameters,
                    "quality": {
                        "metric": str(row.get("quality_metric", payload.get("tool", "quality"))),
                        "value": float(row["quality"]),
                    },
                    "execution": {
                        "device_id": target_device["device_id"],
                        "latency_ms": float(row["latency_ms"]),
                        "gpu_memory_mib": float(row["gpu_memory_mib"]),
                    },
                }
            )
        tools.append(
            {
                "tool_id": str(payload["tool"]),
                "model": str(payload.get("model", "")),
                "campaign_status": payload.get("campaign_status", "unknown"),
                "configuration_count": len(configurations),
                "configurations": configurations,
            }
        )
    exported = {
        "schema_version": "device-specific-campaign-profiling.v1",
        "database_kind": "device-specific",
        "data_kind": "measured",
        "snapshot_id": "campaign-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "created_at": datetime.now(UTC).isoformat(),
        "device": target_device,
        "source": {
            "campaign_directory": str(campaign_directory.resolve()),
            "campaign_files": source_files,
        },
        "resource_metric": "peak_gpu_memory_mib",
        "tools": tools,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(exported, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return exported
