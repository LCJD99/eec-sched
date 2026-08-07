"""Validation and immutable loading for profiling database snapshots."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Hashable, Mapping, Sequence, TypeVar, cast

from jsonschema import Draft202012Validator


@dataclass(frozen=True)
class ProfilingDatabaseSnapshot:
    """A validated snapshot whose nested containers cannot be mutated."""

    snapshot_id: str
    schema_version: str
    snapshot_digest: str
    data: Mapping[str, Any]

    @property
    def metadata(self) -> "SnapshotMetadata":
        return SnapshotMetadata(
            schema_version=self.schema_version,
            snapshot_id=self.snapshot_id,
            snapshot_digest=self.snapshot_digest,
            data_kind=cast(str, self.data["data_kind"]),
            created_at=cast(str, self.data["created_at"]),
        )

    @property
    def measurement_scope(self) -> "MeasurementScope":
        return _record(MeasurementScope, self.data["measurement_scope"])

    def devices(self) -> tuple["Device", ...]:
        return tuple(_record(Device, item) for item in cast(Sequence[Mapping[str, Any]], self.data["devices"]))

    def device(self, device_id: str) -> "Device":
        return _lookup(self.devices(), "device_id", device_id, f"device {device_id!r}")

    def tools(self, *, eligible_only: bool = False) -> tuple["Tool", ...]:
        tools = tuple(_record(Tool, item) for item in cast(Sequence[Mapping[str, Any]], self.data["tools"]))
        return tuple(tool for tool in tools if not eligible_only or tool.eligibility["status"] == "eligible")

    def tool(self, tool_id: str) -> "Tool":
        return _lookup(self.tools(), "tool_id", tool_id, f"tool {tool_id!r}")

    def configurations(self, tool_id: str) -> tuple["Configuration", ...]:
        return tuple(self.tool(tool_id).configurations)

    def configuration(self, tool_id: str, configuration_id: str) -> "Configuration":
        return _lookup(self.configurations(tool_id), "configuration_id", configuration_id, f"configuration {tool_id!r}/{configuration_id!r}")

    def quality_profile(self, tool_id: str, configuration_id: str) -> "QualityProfile":
        profile = _lookup(self.tool(tool_id).quality_profiles, "configuration_id", configuration_id, f"quality profile {tool_id!r}/{configuration_id!r}")
        return cast(QualityProfile, profile)

    def execution_profile(self, tool_id: str, configuration_id: str, device_id: str) -> "ExecutionProfile":
        profiles = tuple(profile for profile in self.tool(tool_id).execution_profiles if profile.configuration_id == configuration_id and profile.device_id == device_id)
        if not profiles:
            raise KeyError(f"execution profile {tool_id!r}/{configuration_id!r}/{device_id!r}")
        return profiles[0]

    def compatible_devices(self, tool_id: str, configuration_id: str) -> tuple[str, ...]:
        configuration = self.configuration(tool_id, configuration_id)
        return tuple(device_id for device_id, entry in configuration.device_compatibility.items() if entry["status"] == "compatible")

    def representative_output_bytes(self, tool_id: str, configuration_id: str) -> int:
        return self.quality_profile(tool_id, configuration_id).representative_output_bytes

    def transfer_profile(self, source_device_id: str, destination_device_id: str) -> "TransferProfile":
        for profile in self.transfer_profiles():
            if profile.source_device_id == source_device_id and profile.destination_device_id == destination_device_id:
                return profile
        raise KeyError(f"transfer profile {source_device_id!r}->{destination_device_id!r}")

    def transfer_profiles(self) -> tuple["TransferProfile", ...]:
        return tuple(_record(TransferProfile, item) for item in cast(Sequence[Mapping[str, Any]], self.data["transfer_profiles"]))


@dataclass(frozen=True)
class SnapshotMetadata:
    schema_version: str
    snapshot_id: str
    snapshot_digest: str
    data_kind: str
    created_at: str


@dataclass(frozen=True)
class MeasurementScope:
    input_bucket: str
    batch_size: int
    warm_execution: bool
    execution_boundary: str


@dataclass(frozen=True)
class Device:
    device_id: str
    hardware_class: str
    description: str


@dataclass(frozen=True)
class Tool:
    tool_id: str
    eligibility: Mapping[str, Any]
    dimensions: Mapping[str, Any]
    quality_contract: Mapping[str, Any]
    configurations: tuple["Configuration", ...]
    quality_profiles: tuple["QualityProfile", ...]
    execution_profiles: tuple["ExecutionProfile", ...]


@dataclass(frozen=True)
class Configuration:
    configuration_id: str
    parameters: Mapping[str, Any]
    taxonomy: Mapping[str, str]
    runtime: Mapping[str, str]
    device_compatibility: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class QualityProfile:
    configuration_id: str
    raw_metric: Mapping[str, Any]
    normalized_quality_lcb: float
    representative_output_bytes: int
    provenance_id: str


@dataclass(frozen=True)
class ExecutionProfile:
    configuration_id: str
    device_id: str
    warm_latency_p95_ms: float
    mean_incremental_execution_energy_j: float
    sample_count: int
    provenance_id: str


@dataclass(frozen=True)
class TransferProfile:
    source_device_id: str
    destination_device_id: str
    propagation_delay_ms: float
    bandwidth_bytes_per_second: int
    setup_energy_j: float
    energy_per_byte_j: float
    sample_count: int
    provenance_id: str


def _record(record_type: type[Any], value: Mapping[str, Any]) -> Any:
    """Convert one already-frozen JSON record into a typed frozen view."""
    if record_type is Tool:
        return Tool(
            tool_id=value["tool_id"],
            eligibility=value["eligibility"],
            dimensions=value["dimensions"],
            quality_contract=value["quality_contract"],
            configurations=tuple(_record(Configuration, item) for item in cast(Sequence[Mapping[str, Any]], value["configurations"])),
            quality_profiles=tuple(_record(QualityProfile, item) for item in cast(Sequence[Mapping[str, Any]], value["quality_profiles"])),
            execution_profiles=tuple(_record(ExecutionProfile, item) for item in cast(Sequence[Mapping[str, Any]], value["execution_profiles"])),
        )
    if record_type is Configuration:
        compatibility = {
            entry["device_id"]: entry
            for entry in cast(Sequence[Mapping[str, Any]], value["device_compatibility"])
        }
        return Configuration(value["configuration_id"], value["parameters"], value["taxonomy"], value["runtime"], MappingProxyType(compatibility))
    return record_type(**value)


def _lookup(records: Sequence[Any], field: str, expected: str, label: str) -> Any:
    for record in records:
        if getattr(record, field) == expected:
            return record
    raise KeyError(label)


class ProfilingDatabaseValidationError(ValueError):
    """Raised with every detected structural or cross-record problem."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("invalid profiling database:\n- " + "\n- ".join(errors))


def snapshot_digest(payload: Mapping[str, Any]) -> str:
    """Return the digest of canonical JSON, excluding the digest field itself."""

    digest_payload = dict(payload)
    digest_payload.pop("snapshot_digest", None)
    encoded = json.dumps(digest_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def validate_profiling_database(payload: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    """Validate JSON Schema constraints and database-wide completeness rules."""

    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=Draft202012Validator.FORMAT_CHECKER)
    errors = [f"{_json_path(error.absolute_path)}: {error.message}" for error in sorted(validator.iter_errors(payload), key=lambda error: list(error.absolute_path))]
    if errors:
        raise ProfilingDatabaseValidationError(errors)

    semantic_errors: list[str] = []
    expected_devices = {"device", "edge", "cloud"}
    devices = payload["devices"]
    device_ids = [device["device_id"] for device in devices]  # type: ignore[index]
    _check_unique(device_ids, "device_id", semantic_errors)
    if set(device_ids) != expected_devices:
        semantic_errors.append("devices must contain exactly device, edge, and cloud")

    provenances = payload["provenances"]
    provenance_ids = [record["provenance_id"] for record in provenances]  # type: ignore[index]
    _check_unique(provenance_ids, "provenance_id", semantic_errors)
    provenance_by_id = {record["provenance_id"]: record for record in provenances}  # type: ignore[index]
    expected_measurement_kind = payload["data_kind"]
    for record in provenances:  # type: ignore[assignment]
        if record["measurement_kind"] != expected_measurement_kind:
            semantic_errors.append(f"provenance {record['provenance_id']} measurement_kind does not match snapshot data_kind")

    tools = payload["tools"]
    tool_ids = [tool["tool_id"] for tool in tools]  # type: ignore[index]
    _check_unique(tool_ids, "tool_id", semantic_errors)
    for tool in tools:  # type: ignore[assignment]
        if tool["eligibility"]["status"] != "eligible":
            continue
        tool_id = tool["tool_id"]
        dimensions = tool["dimensions"]
        configurations = tool["configurations"]
        configuration_ids = [configuration["configuration_id"] for configuration in configurations]
        _check_unique(configuration_ids, f"configuration_id in tool {tool_id}", semantic_errors)
        known_configuration_ids = set(configuration_ids)
        identity_keys: list[str] = []
        for configuration in configurations:
            identity_keys.append(json.dumps({"parameters": configuration["parameters"], "runtime": configuration["runtime"]}, sort_keys=True, separators=(",", ":")))
            for dimension_name, declaration in dimensions.items():
                choice = configuration["taxonomy"][dimension_name]
                if choice not in declaration["values"]:
                    semantic_errors.append(f"{tool_id}/{configuration['configuration_id']} uses undeclared {dimension_name} value {choice}")
                if not declaration["applicable"] and (len(declaration["values"]) != 1 or choice != declaration["values"][0]):
                    semantic_errors.append(f"{tool_id} inapplicable dimension {dimension_name} must remain one explicit singleton")
        if len(set(identity_keys)) != len(identity_keys):
            semantic_errors.append(f"tool {tool_id} has duplicate effective Configuration identities")

        reference_id = tool["quality_contract"]["reference"]["configuration_id"]
        if reference_id not in known_configuration_ids:
            semantic_errors.append(f"tool {tool_id} references unknown quality reference Configuration {reference_id}")
        quality_profiles = tool["quality_profiles"]
        quality_ids = [profile["configuration_id"] for profile in quality_profiles]
        _check_exact_ids(quality_ids, known_configuration_ids, f"quality profiles for {tool_id}", semantic_errors)
        point_estimates = {profile["configuration_id"]: profile["raw_metric"]["point_estimate"] for profile in quality_profiles}
        reference_metric = tool["quality_contract"]["reference"]["raw_metric"]
        metric_direction = tool["quality_contract"]["direction"]
        semantic_floor = tool["quality_contract"]["semantic_floor"]
        if reference_metric == semantic_floor:
            semantic_errors.append(f"tool {tool_id} quality reference and semantic floor must differ")
        if reference_id in point_estimates and point_estimates[reference_id] != reference_metric:
            semantic_errors.append(f"tool {tool_id} reference raw metric does not match its Quality Profile")

        for profile in quality_profiles:
            raw_metric = profile["raw_metric"]
            interval = raw_metric["confidence_interval"]
            if not interval["lower"] <= raw_metric["point_estimate"] <= interval["upper"]:
                semantic_errors.append(f"quality profile {tool_id}/{profile['configuration_id']} has a contradictory confidence interval")
            conservative_raw = interval["lower"] if metric_direction == "higher_is_better" else interval["upper"]
            if reference_metric != semantic_floor:
                normalized_lcb = ((conservative_raw - semantic_floor) / (reference_metric - semantic_floor)) if metric_direction == "higher_is_better" else ((semantic_floor - conservative_raw) / (semantic_floor - reference_metric))
                normalized_lcb = max(0.0, min(1.0, normalized_lcb))
                if abs(profile["normalized_quality_lcb"] - normalized_lcb) > 1e-9:
                    semantic_errors.append(f"quality profile {tool_id}/{profile['configuration_id']} normalized_quality_lcb does not match its raw confidence bound")
            _check_provenance(profile["provenance_id"], "quality", provenance_by_id, f"quality profile {tool_id}/{profile['configuration_id']}", semantic_errors)

        execution_profiles = tool["execution_profiles"]
        execution_keys = [(profile["configuration_id"], profile["device_id"]) for profile in execution_profiles]
        _check_unique(execution_keys, f"execution profile key in tool {tool_id}", semantic_errors)
        for configuration in configurations:
            configuration_id = configuration["configuration_id"]
            compatibility = configuration["device_compatibility"]
            compatibility_ids = [entry["device_id"] for entry in compatibility]
            _check_exact_ids(compatibility_ids, expected_devices, f"device compatibility for {tool_id}/{configuration_id}", semantic_errors)
            compatible = {entry["device_id"] for entry in compatibility if entry["status"] == "compatible"}
            profiled = {device_id for config_id, device_id in execution_keys if config_id == configuration_id}
            if compatible != profiled:
                semantic_errors.append(f"execution profiles for {tool_id}/{configuration_id} must exactly match compatible devices")
        for profile in execution_profiles:
            if profile["configuration_id"] not in known_configuration_ids:
                semantic_errors.append(f"tool {tool_id} has Execution Profile for unknown Configuration {profile['configuration_id']}")
            _check_provenance(profile["provenance_id"], "execution", provenance_by_id, f"execution profile {tool_id}/{profile['configuration_id']}/{profile['device_id']}", semantic_errors)

    transfers = payload["transfer_profiles"]
    transfer_keys = [(profile["source_device_id"], profile["destination_device_id"]) for profile in transfers]  # type: ignore[index]
    expected_transfers = {(source, destination) for source in expected_devices for destination in expected_devices if source != destination}
    _check_exact_ids(transfer_keys, expected_transfers, "directed transfer profiles", semantic_errors)
    for profile in transfers:  # type: ignore[assignment]
        _check_provenance(profile["provenance_id"], "transfer", provenance_by_id, f"transfer profile {profile['source_device_id']}->{profile['destination_device_id']}", semantic_errors)

    actual_digest = payload["snapshot_digest"]
    expected_digest = snapshot_digest(payload)
    if actual_digest != expected_digest:
        semantic_errors.append(f"snapshot_digest mismatch: expected {expected_digest}")
    if semantic_errors:
        raise ProfilingDatabaseValidationError(semantic_errors)


def load_profiling_database(database_path: Path, schema_path: Path) -> ProfilingDatabaseSnapshot:
    """Load, validate, and deeply freeze one database snapshot."""

    payload = json.loads(database_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validate_profiling_database(payload, schema)
    frozen = cast(Mapping[str, Any], _deep_freeze(payload))
    return ProfilingDatabaseSnapshot(payload["snapshot_id"], payload["schema_version"], payload["snapshot_digest"], frozen)


_Hashable = TypeVar("_Hashable", bound=Hashable)


def _check_unique(values: Sequence[Hashable], label: str, errors: list[str]) -> None:
    if len(set(values)) != len(values):
        errors.append(f"duplicate {label}")


def _check_exact_ids(actual: Sequence[_Hashable], expected: set[_Hashable], label: str, errors: list[str]) -> None:
    _check_unique(actual, label, errors)
    if set(actual) != expected:
        missing = sorted(expected - set(actual), key=repr)
        extra = sorted(set(actual) - expected, key=repr)
        errors.append(f"{label} are incomplete or contradictory (missing={missing}, extra={extra})")


def _check_provenance(provenance_id: str, subject_kind: str, known: Mapping[str, Any], label: str, errors: list[str]) -> None:
    if provenance_id not in known:
        errors.append(f"{label} references unknown provenance {provenance_id}")
    elif known[provenance_id]["subject_kind"] != subject_kind:
        errors.append(f"{label} references {known[provenance_id]['subject_kind']} provenance {provenance_id}")


def _json_path(parts: object) -> str:
    path = "$"
    for part in parts:  # type: ignore[union-attr]
        path += f"[{part}]" if isinstance(part, int) else f".{part}"
    return path


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value
