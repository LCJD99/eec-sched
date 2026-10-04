"""Generic, one-shot offline Tool Planner batch generation.

This module deliberately knows nothing about MnMS.  A workload is a JSONL file
whose records contain only ``id`` and ``request``; a separate JSON catalog
describes the tools available to the Planner.  The experiment entry point uses
this module to call an OpenAI-compatible model once per record and to write the
Scheduler's canonical three-DAG JSONL format.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol
from urllib.parse import urlsplit

from jsonschema import Draft202012Validator, SchemaError
from PIL import Image

from .domain import Port, ToolCallPlan, ToolCallPlanCandidates, ToolRegistry, ToolSpec
from .openai_compatible import plans_from_dict
from .planning import validate_plan_candidates


class PlannerModel(Protocol):
    def complete(
        self,
        *,
        system: str,
        user: Mapping[str, object],
        json_output: bool = False,
    ) -> str: ...


class PlannerBatchError(RuntimeError):
    """A configuration or transport error that should stop a batch."""


class _ModelCallError(RuntimeError):
    """An item-scoped model failure; the batch can continue after recording it."""


@dataclass(frozen=True)
class WorkItem:
    item_id: str
    request: str


@dataclass(frozen=True)
class BatchSummary:
    total: int
    selected: int
    succeeded: int
    failed: int
    output_path: Path
    failures_path: Path
    manifest_path: Path
    trace_path: Path


_MEDIA_REFERENCE = re.compile(
    r"(?:https?://[^\s)\]>]+|(?<!\S)[^\s)\]>]+\.(?:png|jpe?g|webp|gif|bmp|tiff?|svg|wav|mp3|flac|m4a|aac|ogg|opus))",
    re.IGNORECASE,
)
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".svg"}
_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus"}
_DEFAULT_INPUT_SCHEMA = (
    Path(__file__).resolve().parents[2] / "docs/schemas/tool-planner-input.schema.json"
)
_DEFAULT_CATALOG_SCHEMA = (
    Path(__file__).resolve().parents[2] / "docs/schemas/tool-planner-catalog.schema.json"
)
_DEFAULT_DAG_SCHEMA = (
    Path(__file__).resolve().parents[2] / "docs/schemas/tool-call-dag.schema.json"
)
_DEFAULT_CANDIDATES_SCHEMA = (
    Path(__file__).resolve().parents[2] / "docs/schemas/tool-call-plan-candidates.schema.json"
)


def load_work_items(
    path: Path | str,
    *,
    schema_path: Path | str | None = None,
) -> tuple[WorkItem, ...]:
    """Load the deliberately small generic ``id``/``request`` JSONL schema.

    Extra fields are rejected so reference answers or dataset-specific
    metadata cannot accidentally enter the Planner context.
    """

    source = Path(path)
    input_schema_path = Path(schema_path) if schema_path is not None else _DEFAULT_INPUT_SCHEMA
    try:
        input_schema = json.loads(input_schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlannerBatchError(f"could not read workload schema {input_schema_path}") from exc
    input_validator = Draft202012Validator(input_schema)
    records: list[WorkItem] = []
    try:
        handle = source.open(encoding="utf-8")
    except OSError as exc:
        raise PlannerBatchError(f"could not read workload {source}") from exc
    with handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                raise PlannerBatchError(f"input line {line_number} is empty")
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise PlannerBatchError(f"input line {line_number} is not valid JSON") from exc
            schema_errors = sorted(
                input_validator.iter_errors(value), key=lambda error: list(error.path)
            )
            if schema_errors:
                raise PlannerBatchError(
                    f"input line {line_number} does not match workload schema: "
                    f"{schema_errors[0].message}"
                )
            if not isinstance(value, dict):
                raise PlannerBatchError(f"input line {line_number} must be a JSON object")
            item_id, request = value["id"], value["request"]
            if isinstance(item_id, bool) or not isinstance(item_id, (str, int, float)):
                raise PlannerBatchError(f"input line {line_number} id must be a non-empty string or number")
            item_id = str(item_id)
            if not item_id.strip():
                raise PlannerBatchError(f"input line {line_number} id must be a non-empty string or number")
            if not isinstance(request, str) or not request.strip():
                raise PlannerBatchError(f"input line {line_number} request must be a non-empty string")
            records.append(WorkItem(item_id, request))
    ids = [item.item_id for item in records]
    if len(ids) != len(set(ids)):
        raise PlannerBatchError("workload ids must be unique")
    return tuple(records)


def _port(value: object, *, default_required: bool) -> Port:
    if not isinstance(value, dict):
        raise PlannerBatchError("catalog ports must be objects")
    modality = value.get("modality")
    if modality not in {"audio", "image", "text"}:
        raise PlannerBatchError("catalog port modality must be audio, image, or text")
    required = value.get("required", default_required)
    if not isinstance(required, bool):
        raise PlannerBatchError("catalog port required must be boolean")
    return Port("", modality, required)


def load_tool_catalog(path: Path | str) -> tuple[ToolSpec, ...]:
    """Load a dataset-independent tool catalog JSON file.

    Canonical shape::

        {"tools": [{"tool_id": "...", "description": "...",
                    "inputs": {"name": {"modality": "text", "required": true}},
                    "outputs": {"name": {"modality": "text"}}}]}

    ``required`` is meaningful for inputs and defaults to true.  It is
    accepted on outputs for a uniform catalog shape and defaults to false.
    """

    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlannerBatchError(f"could not read tool catalog {source}") from exc
    try:
        schema = json.loads(_DEFAULT_CATALOG_SCHEMA.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlannerBatchError("could not read tool catalog schema") from exc
    schema_errors = sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda error: list(error.path),
    )
    if schema_errors:
        raise PlannerBatchError(
            f"tool catalog does not match schema: {schema_errors[0].message}"
        )
    if not isinstance(value, dict) or set(value) != {"tools"} or not isinstance(value["tools"], list):
        raise PlannerBatchError("tool catalog must be an object containing only a tools array")
    specs: list[ToolSpec] = []
    seen: set[str] = set()
    for index, raw in enumerate(value["tools"]):
        if not isinstance(raw, dict) or not {"tool_id", "description", "inputs", "outputs"} <= set(raw):
            raise PlannerBatchError(f"catalog tool {index} is missing required fields")
        if not set(raw) <= {"tool_id", "description", "inputs", "outputs"}:
            raise PlannerBatchError(f"catalog tool {index} has unknown fields")
        tool_id, description = raw["tool_id"], raw["description"]
        if not isinstance(tool_id, str) or not tool_id.strip() or tool_id in seen:
            raise PlannerBatchError(f"catalog tool {index} has a duplicate or invalid tool_id")
        if not isinstance(description, str) or not description.strip():
            raise PlannerBatchError(f"catalog tool {tool_id} description must be non-empty")
        inputs, outputs = raw["inputs"], raw["outputs"]
        if not isinstance(inputs, dict) or not isinstance(outputs, dict):
            raise PlannerBatchError(f"catalog tool {tool_id} inputs and outputs must be objects")
        input_ports: dict[str, Port] = {}
        for name, raw_port in inputs.items():
            if not isinstance(name, str) or not name:
                raise PlannerBatchError(f"catalog tool {tool_id} has an invalid input port name")
            parsed = _port(raw_port, default_required=True)
            input_ports[name] = Port(name, parsed.modality, parsed.required)
        output_ports: dict[str, Port] = {}
        for name, raw_port in outputs.items():
            if not isinstance(name, str) or not name:
                raise PlannerBatchError(f"catalog tool {tool_id} has an invalid output port name")
            parsed = _port(raw_port, default_required=False)
            output_ports[name] = Port(name, parsed.modality, parsed.required)
        specs.append(ToolSpec(tool_id, description, input_ports, output_ports, ()))
        seen.add(tool_id)
    if not specs:
        raise PlannerBatchError("tool catalog must contain at least one tool")
    return tuple(specs)


def build_registry(specs: Sequence[ToolSpec]) -> ToolRegistry:
    """Build the validator registry without constructing any tool runner."""

    registry = ToolRegistry()
    for spec in specs:
        registry.register(spec, lambda: None)  # type: ignore[arg-type]
    return registry


def catalog_for_prompt(specs: Sequence[ToolSpec]) -> list[dict[str, object]]:
    return [
        {
            "tool_id": spec.tool_id,
            "description": spec.description,
            "inputs": {
                name: {"modality": port.modality, "required": port.required}
                for name, port in spec.inputs.items()
            },
            "outputs": {
                name: {"modality": port.modality, "required": port.required}
                for name, port in spec.outputs.items()
            },
        }
        for spec in specs
    ]


def parse_three_dag_response(
    content: str,
    *,
    schema_path: Path | str | None = None,
) -> ToolCallPlanCandidates:
    """Parse one raw model response, requiring a real top-level three-DAG list."""

    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("model response is not JSON") from exc
    if not isinstance(value, dict) or set(value) != {"dags"} or not isinstance(value["dags"], list):
        raise ValueError("model response must contain only a dags array")
    raw_dags = value["dags"]
    if len(raw_dags) != 3 or any(not isinstance(item, dict) for item in raw_dags):
        raise ValueError("model response dags must contain exactly three objects")
    schema = json.loads(Path(schema_path or _DEFAULT_DAG_SCHEMA).read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    for index, dag in enumerate(raw_dags):
        errors = sorted(validator.iter_errors(dag), key=lambda error: list(error.path))
        if errors:
            raise ValueError(f"DAG {index} does not match Tool-Call DAG schema: {errors[0].message}")
    # plans_from_dict is retained for the common conversion, but the strict
    # shape checks above happen first, so its legacy single-DAG fallback can
    # never turn an invalid response into three candidates here.
    return plans_from_dict({"dags": raw_dags})


def plan_to_dict(
    plan: ToolCallPlan, request_modalities: Mapping[str, str]
) -> dict[str, object]:
    """Serialize validated bindings using the harness's request modalities."""
    return {
        "nodes": [
            {
                "node_id": node.node_id,
                "tool_id": node.tool_id,
                "inputs": {
                    name: {
                        "kind": source.kind,
                        "name": source.name,
                        **({"port": source.port} if source.kind == "node" else {}),
                        **(
                            {"data_type": request_modalities[source.name]}
                            if source.kind == "request"
                            else {}
                        ),
                    }
                    for name, source in node.inputs.items()
                },
            }
            for node in plan.nodes
        ],
        "final_outputs": [
            {"node_id": output.node_id, "port": output.port}
            for output in plan.final_outputs
        ],
    }


def safe_endpoint(value: str) -> str:
    """Return only the endpoint origin, excluding path/query credentials."""

    parts = urlsplit(value)
    if not parts.scheme or not parts.hostname:
        return "<configured-endpoint>"
    host = parts.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = host
    if parts.port is not None:
        netloc = f"{netloc}:{parts.port}"
    return f"{parts.scheme}://{netloc}"


def _safe_reason(exc: BaseException, token: str | None = None) -> str:
    reason = str(exc).replace("\n", " ").strip()[:500]
    if token:
        reason = reason.replace(token, "[REDACTED]")
    return reason or type(exc).__name__


def _request_inputs(request: str) -> dict[str, object]:
    """Return JSON-safe request inputs, including explicit media references."""

    result: dict[str, object] = {"request": request}
    for name, modality, reference in _detected_media_inputs(request):
        result[name] = reference
    return result


def _detected_media_inputs(request: str) -> tuple[tuple[str, str, str], ...]:
    """Detect explicit media paths/URLs without reading or fetching them."""

    counters = {"image": 0, "audio": 0}
    detected: list[tuple[str, str, str]] = []
    for raw_reference in _MEDIA_REFERENCE.findall(request):
        reference = raw_reference.rstrip(".,;:!?\"'")
        path = urlsplit(reference).path.lower()
        suffix = Path(path).suffix
        if suffix in _IMAGE_SUFFIXES:
            modality = "image"
        elif suffix in _AUDIO_SUFFIXES:
            modality = "audio"
        else:
            continue
        name = f"{modality}_{counters[modality]}"
        counters[modality] += 1
        detected.append((name, modality, reference))
    return tuple(detected)


def _validation_inputs(request: str) -> dict[str, object]:
    """Build typed placeholders for deterministic modality validation only."""

    inputs: dict[str, object] = {"request": request}
    for name, modality, _reference in _detected_media_inputs(request):
        if modality == "image":
            inputs[name] = Image.new("RGB", (1, 1))
        else:
            # A Path is enough for validate_plan to identify audio. It is never
            # opened and may contain a URL or a non-existent local path.
            inputs[name] = Path(_reference)
    return inputs


def _planner_user(work_item: WorkItem, specs: Sequence[ToolSpec]) -> dict[str, object]:
    detected_inputs = _detected_media_inputs(work_item.request)
    return {
        "request": work_item.request,
        "inputs": {
            "request": "text",
            **{name: modality for name, modality, _reference in detected_inputs},
        },
        "input_references": {
            name: reference for name, _modality, reference in detected_inputs
        },
        "tool_catalog": catalog_for_prompt(specs),
        "instruction": (
            "Return one JSON object with only a dags array of exactly three DAG objects. "
            "Each array item must directly contain nodes and final_outputs."
        ),
    }


def run_batch(
    *,
    items: Sequence[WorkItem],
    specs: Sequence[ToolSpec],
    system_prompt: str,
    model: PlannerModel,
    output_path: Path | str,
    failures_path: Path | str,
    manifest_path: Path | str,
    trace_path: Path | str | None = None,
    input_path: Path | str,
    catalog_path: Path | str,
    model_name: str,
    endpoint: str,
    prompt_version: str,
    batch_size: int = 8,
    limit: int | None = None,
    item_id: str | None = None,
    schema_path: Path | str | None = None,
    api_token: str | None = None,
) -> BatchSummary:
    """Run model requests in bounded concurrent batches and write ordered results."""

    if limit is not None and limit < 0:
        raise PlannerBatchError("limit must be non-negative")
    if batch_size < 1:
        raise PlannerBatchError("batch_size must be positive")
    normalized_items = tuple(
        WorkItem(str(item.item_id), item.request) for item in items
    )
    if len({item.item_id for item in normalized_items}) != len(normalized_items):
        raise PlannerBatchError("workload ids must be unique")
    selected = [item for item in normalized_items if item_id is None or item.item_id == item_id]
    if item_id is not None and not selected:
        raise PlannerBatchError(f"requested id was not found: {item_id}")
    if limit is not None:
        selected = selected[:limit]

    dag_schema_path = Path(schema_path or _DEFAULT_DAG_SCHEMA)
    try:
        dag_schema = json.loads(dag_schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlannerBatchError(f"could not read DAG schema {dag_schema_path}") from exc
    try:
        Draft202012Validator.check_schema(dag_schema)
    except SchemaError as exc:
        raise PlannerBatchError(f"invalid DAG schema {dag_schema_path}") from exc

    output = Path(output_path)
    failures = Path(failures_path)
    manifest = Path(manifest_path)
    trace = Path(trace_path) if trace_path is not None else output.with_name("trace.jsonl")
    existing = [path for path in (output, failures, manifest, trace) if path.exists()]
    if existing:
        names = ", ".join(str(path) for path in existing)
        raise PlannerBatchError(f"refusing to overwrite existing Planner artifacts: {names}")
    output.parent.mkdir(parents=True, exist_ok=True)
    failures.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    trace.parent.mkdir(parents=True, exist_ok=True)
    registry = build_registry(specs)
    candidates_schema_path = _DEFAULT_CANDIDATES_SCHEMA
    try:
        candidates_validator = Draft202012Validator(
            json.loads(candidates_schema_path.read_text(encoding="utf-8"))
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise PlannerBatchError("could not read canonical candidate schema") from exc
    success_count = 0
    failure_count = 0
    with (
        output.open("w", encoding="utf-8") as success_handle,
        failures.open("w", encoding="utf-8") as failure_handle,
        trace.open("w", encoding="utf-8") as trace_handle,
        ThreadPoolExecutor(max_workers=batch_size) as executor,
    ):
        for start in range(0, len(selected), batch_size):
            batch = selected[start : start + batch_size]
            futures = [
                executor.submit(
                    model.complete,
                    system=system_prompt,
                    user=_planner_user(work_item, specs),
                    json_output=True,
                )
                for work_item in batch
            ]
            for work_item, future in zip(batch, futures, strict=True):
                try:
                    detected_inputs = _detected_media_inputs(work_item.request)
                    try:
                        content = future.result()
                    except Exception as exc:  # model/network failures are item-scoped
                        raise _ModelCallError(type(exc).__name__) from exc
                    trace_handle.write(
                        json.dumps(
                            {"id": work_item.item_id, "llm_output": content},
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    trace_handle.flush()
                    candidates = parse_three_dag_response(content, schema_path=schema_path)
                    # The validator only needs the input mapping.  It must not see
                    # benchmark constraints that do not exist in this interface.
                    validation_request = SimpleNamespace(
                        prompt=work_item.request,
                        inputs=_validation_inputs(work_item.request),
                    )
                    errors_by_dag = validate_plan_candidates(candidates, validation_request, registry)  # type: ignore[arg-type]
                    errors = [error for dag_errors in errors_by_dag for error in dag_errors]
                    if errors:
                        reason = "; ".join(f"{error.code}: {error.message}" for error in errors[:4])
                        raise ValueError(f"DAG validation failed: {reason}")
                    record = {
                        "trace_id": work_item.item_id,
                        "task_input": _request_inputs(work_item.request),
                        "system_state": {},
                        "dags": [
                            plan_to_dict(
                                dag,
                                {
                                    "request": "text",
                                    **{
                                        name: modality
                                        for name, modality, _reference in detected_inputs
                                    },
                                },
                            )
                            for dag in candidates.dags
                        ],
                    }
                    errors = sorted(
                        candidates_validator.iter_errors(record),
                        key=lambda error: list(error.path),
                    )
                    if errors:
                        raise ValueError(
                            f"canonical candidate schema failed: {errors[0].message}"
                        )
                    success_handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    success_count += 1
                except _ModelCallError as exc:
                    failure_handle.write(
                        json.dumps(
                            {
                                "id": work_item.item_id,
                                "stage": "model",
                                "reason": f"model request failed: {exc}",
                            },
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    failure_count += 1
                except (ValueError, KeyError, TypeError) as exc:
                    failure_handle.write(
                        json.dumps(
                            {
                                "id": work_item.item_id,
                                "stage": "parse_or_validate",
                                "reason": _safe_reason(exc, api_token),
                            },
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    failure_count += 1

    manifest_value = {
        "schema_version": 1,
        "model": {"name": model_name, "endpoint": safe_endpoint(endpoint)},
        "prompt": {"version": prompt_version},
        "tool_catalog": {
            "path": Path(catalog_path).name,
            "tool_count": len(specs),
        },
        "input": {
            "path": Path(input_path).name,
            "records_total": len(normalized_items),
            "records_selected": len(selected),
        },
        "results": {"succeeded": success_count, "failed": failure_count},
    }
    manifest.write_text(json.dumps(manifest_value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return BatchSummary(
        len(normalized_items), len(selected), success_count, failure_count,
        output, failures, manifest, trace,
    )


__all__ = [
    "BatchSummary",
    "PlannerBatchError",
    "PlannerModel",
    "WorkItem",
    "build_registry",
    "catalog_for_prompt",
    "load_tool_catalog",
    "load_work_items",
    "parse_three_dag_response",
    "plan_to_dict",
    "run_batch",
    "safe_endpoint",
]
