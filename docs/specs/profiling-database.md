# Profiling database schema v1

This document defines the schema-first deliverable for [issue #3](https://github.com/LCJD99/eec-sched/issues/3). It freezes the evaluator input shape and supplies synthetic development data. It does **not** claim that the required real profiling campaign is complete.

## Artifacts

- [`../schemas/profiling-database.schema.json`](../schemas/profiling-database.schema.json) is the normative JSON Schema Draft 2020-12 document.
- [`../examples/profiling-database.fake.json`](../examples/profiling-database.fake.json) is a complete synthetic snapshot for downstream development.
- `scripts/generate_fake_profiling_database.py` deterministically rebuilds that snapshot from the side-effect-free `mnms_tool_specs()` catalog without loading models or using external services.
- `eec_sched.profiling_database` validates cross-record invariants, verifies the snapshot digest, and returns deeply immutable mappings and tuples.
- `scripts/validate_profiling_database.py` is the command-line validation entry point.

The existing files under `profiles/*.json` are v1 profiling-run artifacts from the earlier single-device implementation. They are useful source evidence, but they are **not** profiling database v1 snapshots and must not be silently interpreted as such.

## Snapshot boundary

One snapshot is the complete fixed input used for replay. Its digest is SHA-256 over canonical JSON with `snapshot_digest` omitted. Arrays are significant: producers must emit their canonical order before calculating the digest. The loader validates before recursively replacing JSON objects and arrays with immutable mappings and tuples.

`data_kind` is mandatory:

- `measured` means every referenced provenance record represents an actual measurement.
- `synthetic` means the snapshot exists only for development or testing.

Mixing measured and synthetic provenance inside one snapshot is rejected. A real snapshot must therefore replace the fake values as a whole rather than gradually acquiring an ambiguous status.

## Record model

### Tool and eligibility

Every known tool has one machine-readable eligibility status:

- `eligible`: dimensions, configurations, quality profiles, and execution profiles are required.
- `excluded`: the tool is deliberately outside the benchmark and has a coded reason.
- `unprofileable`: the desired tool cannot currently satisfy the measurement contract and has a coded reason.

Excluded and unprofileable tools have no Configuration or Profile records. This prevents partial data from looking like a schedulable choice.

### Configuration taxonomy

Each eligible tool declares all three dimensions:

- Model Capacity
- Input Fidelity
- Inference Effort

An inapplicable dimension is represented by exactly one explicit value and `applicable: false`. A Configuration chooses one declared value from each dimension. Its `parameters`, `backend`, and `numeric_format` jointly define its effective identity. The semantic validator rejects duplicate identities, so a quality-changing backend or numeric format cannot hide behind device placement.

A Configuration declares exactly one compatibility record for each of `device`, `edge`, and `cloud`. `compatible` requires exactly one matching Execution Profile; `incompatible` requires a machine-readable reason and forbids an Execution Profile. There is no “unknown” state in a complete snapshot.

### Quality Profile

Quality is device-independent because every quality-changing runtime choice creates a distinct Configuration. The tool-level quality contract freezes:

- raw metric name, unit, and direction;
- semantic floor;
- reference Configuration and its fixed raw value.

Each Configuration has one Quality Profile containing the raw point estimate, confidence interval and method, sample count, normalized quality lower confidence bound, representative output bytes, and provenance.

Quality provenance pins the dataset and revision, split, ordered sample IDs, preprocessing, scorer and version, confidence method, model revision, and random seed.

For a higher-is-better metric, with lower confidence endpoint `c`, floor `f`, and reference `r`:

```text
quality_lcb = clip((c - f) / (r - f), 0, 1)
```

For a lower-is-better metric, the upper confidence endpoint is conservative:

```text
quality_lcb = clip((f - c) / (f - r), 0, 1)
```

The semantic validator recomputes this value. Raw metrics remain the source evidence; the normalized value is a deterministic evaluator input.

### Execution Profile

One Execution Profile identifies a `(tool, Configuration, compatible device)` tuple and stores:

- warm p95 end-to-end node latency in milliseconds;
- mean incremental execution energy in joules;
- observation count;
- measurement provenance.

The snapshot has one global fixed input bucket, batch size one, and a declared execution boundary. Model download and loading remain outside the warm execution boundary in v1.

Execution provenance pins input IDs, warmup and observation counts, timing and energy methods, model revision, and runtime versions. Transfer provenance separately pins payload sizes, repetition count, timing method, and energy method. A Profile cannot reference provenance of the wrong subject type.

### Transfer Profile

The database contains exactly six records: every ordered pair of distinct `device`, `edge`, and `cloud`. Direction is significant. Each record stores propagation delay, bandwidth, setup energy, per-byte energy, observation count, and provenance.

For representative output size `S` bytes, the evaluator derives:

```text
transfer_latency_ms = propagation_delay_ms + 1000 * S / bandwidth_bytes_per_second
transfer_energy_j = setup_energy_j + S * energy_per_byte_j
```

Same-device DAG edges have zero transfer cost and therefore have no Transfer Profile.

## Validation responsibilities

JSON Schema rejects unknown fields and invalid primitive or conditional shapes. `validate_profiling_database` additionally rejects:

- duplicate devices, tools, provenances, Configurations, execution keys, or transfer keys;
- any device set other than exactly `device`, `edge`, and `cloud`;
- undeclared taxonomy choices or non-singleton inapplicable dimensions;
- duplicate effective Configuration identities;
- missing or extra Quality Profiles;
- missing, extra, or incompatible Execution Profiles;
- unknown Configuration or provenance references;
- contradictory raw confidence intervals or normalized quality bounds;
- missing, duplicate, or same-device transfer records;
- provenance whose measurement kind disagrees with the snapshot;
- a stale or fabricated snapshot digest.

Run the development fixture validation with:

```bash
uv run python scripts/validate_profiling_database.py \
  docs/examples/profiling-database.fake.json
```

Rebuild the fixture after the MnMS catalog or its Configurations change, then validate it:

```bash
uv run python scripts/generate_fake_profiling_database.py
uv run python scripts/validate_profiling_database.py \
  docs/examples/profiling-database.fake.json
```

The generator emits one eligible record for every catalog tool and profiles every catalog-declared Configuration on all three synthetic devices. For a tool whose runtime catalog declares no finite Configuration, the fixture adds exactly one `synthetic-reference` Configuration so downstream database and DAG flows can exercise that tool. Its parameters mark it `synthetic_fixture_only` and `execution_supported: false`: it is a database-fixture seam, is not added to `mnms_tool_specs()`, and must never be passed to a real runner. All values and provenance are marked `synthetic` and must never be cited as benchmark measurements.

## Replacing fake data

The next profiling campaign should create a new `measured` snapshot and leave the fake fixture unchanged for tests. Before publishing that snapshot:

1. Freeze the eligible MnMS tool list and coded exclusions.
2. Freeze every valid joint Configuration, including backend and numeric format.
3. Adopt one fixed quality suite, floor, reference, confidence method, input bucket, and execution boundary.
4. Measure Quality Profiles and every compatible Configuration-device Execution Profile.
5. Record explicit incompatibility for every unsupported placement.
6. Measure all six directed Transfer Profiles.
7. Canonically order records, calculate the digest, validate, and preserve the resulting file unchanged for benchmark replay.

Schema changes require a new semantic `schema_version`; existing snapshots are never upgraded in place.
