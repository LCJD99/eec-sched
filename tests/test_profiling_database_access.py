from pathlib import Path

import pytest

from eec_sched.profiling.snapshot import load_profiling_database


ROOT = Path(__file__).parents[1]
SNAPSHOT = load_profiling_database(
    ROOT / "docs/examples/profiling-database.fake.json",
    ROOT / "docs/schemas/profiling-database.schema.json",
)


def test_snapshot_exposes_typed_metadata_and_measurement_scope() -> None:
    assert SNAPSHOT.metadata.snapshot_id == "synthetic-mnms-catalog-v1"
    assert SNAPSHOT.metadata.data_kind == "synthetic"
    assert SNAPSHOT.measurement_scope.input_bucket == "synthetic-default-v1"
    assert SNAPSHOT.measurement_scope.batch_size == 1
    assert SNAPSHOT.measurement_scope.warm_execution is True


def test_tool_configuration_quality_and_execution_queries() -> None:
    tool = SNAPSHOT.tool("text_generation")
    configuration = SNAPSHOT.configuration("text_generation", "synthetic-reference")
    quality = SNAPSHOT.quality_profile("text_generation", "synthetic-reference")
    execution = SNAPSHOT.execution_profile("text_generation", "synthetic-reference", "device")

    assert tool.tool_id == "text_generation"
    assert configuration.runtime["backend"] == "synthetic-no-runtime"
    assert SNAPSHOT.compatible_devices("text_generation", "synthetic-reference") == ("device", "edge", "cloud")
    assert quality.normalized_quality_lcb == pytest.approx(0.9777777777777777)
    assert SNAPSHOT.representative_output_bytes("text_generation", "synthetic-reference") == 256
    assert execution.warm_latency_p95_ms == 25.0


def test_device_and_transfer_queries_are_directional() -> None:
    assert SNAPSHOT.device("edge").hardware_class == "synthetic-edge"
    transfer = SNAPSHOT.transfer_profile("device", "edge")
    assert transfer.propagation_delay_ms == 4.0
    assert transfer.bandwidth_bytes_per_second == 50_000_000
    assert SNAPSHOT.transfer_profile("edge", "device").propagation_delay_ms == 4.5
    with pytest.raises(KeyError):
        SNAPSHOT.transfer_profile("device", "device")


def test_typed_views_do_not_allow_mutation() -> None:
    configuration = SNAPSHOT.configuration("text_generation", "synthetic-reference")
    with pytest.raises(TypeError):
        configuration.parameters["new"] = True  # type: ignore[index]
    with pytest.raises(TypeError):
        configuration.device_compatibility["device"]["status"] = "incompatible"  # type: ignore[index]


def test_query_methods_report_unknown_records() -> None:
    with pytest.raises(KeyError):
        SNAPSHOT.tool("missing")
    with pytest.raises(KeyError):
        SNAPSHOT.configuration("text_generation", "missing")
    with pytest.raises(KeyError):
        SNAPSHOT.execution_profile("text_generation", "synthetic-reference", "missing")
