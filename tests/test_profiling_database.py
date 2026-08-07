from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from eec_sched.mnms_tools import mnms_tool_specs
from eec_sched.profiling_database import ProfilingDatabaseValidationError, load_profiling_database, snapshot_digest, validate_profiling_database


ROOT = Path(__file__).parents[1]
SCHEMA_PATH = ROOT / "docs/schemas/profiling-database.schema.json"
FAKE_DATABASE_PATH = ROOT / "docs/examples/profiling-database.fake.json"


def _documents() -> tuple[dict[str, object], dict[str, object]]:
    return json.loads(FAKE_DATABASE_PATH.read_text(encoding="utf-8")), json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _redigest(payload: dict[str, object]) -> None:
    payload["snapshot_digest"] = snapshot_digest(payload)


def _first_eligible_tool(payload: dict[str, object]) -> dict[str, object]:
    return next(
        tool
        for tool in payload["tools"]  # type: ignore[union-attr]
        if tool["eligibility"]["status"] == "eligible"
    )


def test_fake_database_is_valid_replayable_and_deeply_immutable() -> None:
    first = load_profiling_database(FAKE_DATABASE_PATH, SCHEMA_PATH)
    second = load_profiling_database(FAKE_DATABASE_PATH, SCHEMA_PATH)

    assert first.snapshot_digest == second.snapshot_digest
    assert first.data == second.data
    with pytest.raises(TypeError):
        first.data["snapshot_id"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        first.data["measurement_scope"]["batch_size"] = 2  # type: ignore[index]


def test_generated_fake_database_covers_the_authoritative_mnms_catalog() -> None:
    payload, schema = _documents()
    catalog = {spec.tool_id: spec for spec in mnms_tool_specs()}
    generated = {tool["tool_id"]: tool for tool in payload["tools"]}  # type: ignore[index]

    assert set(generated) == set(catalog)
    for tool_id, spec in catalog.items():
        tool = generated[tool_id]
        assert tool["eligibility"]["status"] == "eligible"
        if spec.configurations:
            assert [record["configuration_id"] for record in tool["configurations"]] == [
                configuration.configuration_id for configuration in spec.configurations
            ]
        else:
            assert [record["configuration_id"] for record in tool["configurations"]] == ["synthetic-reference"]
            assert tool["configurations"][0]["parameters"]["synthetic_fixture_only"] is True
            assert tool["configurations"][0]["parameters"]["execution_supported"] is False
        assert len(tool["quality_profiles"]) == len(tool["configurations"])
        assert len(tool["execution_profiles"]) == 3 * len(tool["configurations"])

    assert sum(len(tool["configurations"]) for tool in generated.values()) == 50
    assert sum(len(tool["quality_profiles"]) for tool in generated.values()) == 50
    assert sum(len(tool["execution_profiles"]) for tool in generated.values()) == 150

    validate_profiling_database(payload, schema)


def test_fake_database_generation_is_reproducible_and_fixture_is_current(tmp_path: Path) -> None:
    output = tmp_path / "profiling-database.fake.json"
    command = [
        sys.executable,
        str(ROOT / "scripts/generate_fake_profiling_database.py"),
        "--output",
        str(output),
        "--schema",
        str(SCHEMA_PATH),
    ]

    subprocess.run(command, check=True, cwd=ROOT, capture_output=True, text=True)
    first_bytes = output.read_bytes()
    subprocess.run(command, check=True, cwd=ROOT, capture_output=True, text=True)
    first = json.loads(first_bytes)

    assert output.read_bytes() == first_bytes
    assert json.loads(FAKE_DATABASE_PATH.read_text(encoding="utf-8")) == first
    assert first["data_kind"] == "synthetic"
    assert all(provenance["measurement_kind"] == "synthetic" for provenance in first["provenances"])
    assert first["snapshot_digest"] == snapshot_digest(first)


def test_compatible_devices_and_execution_profiles_must_match_exactly() -> None:
    payload, schema = _documents()
    tool = _first_eligible_tool(payload)
    tool["execution_profiles"] = tool["execution_profiles"][:-1]  # type: ignore[index]
    _redigest(payload)

    with pytest.raises(ProfilingDatabaseValidationError, match="must exactly match compatible devices"):
        validate_profiling_database(payload, schema)


def test_all_six_directed_transfer_profiles_are_required() -> None:
    payload, schema = _documents()
    payload["transfer_profiles"] = payload["transfer_profiles"][:-1]  # type: ignore[index]
    _redigest(payload)

    with pytest.raises(ProfilingDatabaseValidationError, match="too short"):
        validate_profiling_database(payload, schema)


def test_unknown_fields_are_rejected_by_schema() -> None:
    payload, schema = _documents()
    payload["untrusted_evaluator_override"] = True
    _redigest(payload)

    with pytest.raises(ProfilingDatabaseValidationError, match="Additional properties are not allowed"):
        validate_profiling_database(payload, schema)


def test_digest_detects_changed_measurements() -> None:
    payload, schema = _documents()
    changed = copy.deepcopy(payload)
    tool = _first_eligible_tool(changed)
    tool["execution_profiles"][0]["warm_latency_p95_ms"] = 1.0  # type: ignore[index]

    with pytest.raises(ProfilingDatabaseValidationError, match="snapshot_digest mismatch"):
        validate_profiling_database(changed, schema)
