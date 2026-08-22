"""Export validated scheduler DAGs from the real MnMS ground-truth dataset.

The MnMS records describe a plan as a Python-literal list of tool calls.  This
script downloads one of those JSONL files, converts every eligible plan into
the device-independent ``ToolCallPlan`` JSON projection, and writes one DAG
per line.  A query is omitted as a whole when any of its tools is absent from
``mnms_tool_specs()`` or when the converted DAG cannot be validated.

The output intentionally contains *only* values conforming to
``tool-call-dag.schema.json``.  Literal MnMS arguments become request-input
bindings: scheduling needs the input modality and dependency graph, not the
literal value.  A ``<node-N>.port`` reference becomes a node-input binding.
For MnMS text templates that embed such a reference in larger text, this
preserves the dependency but not the string-template operation, which is not
part of the current ToolCallPlan domain model.

Example:
    uv run python scripts/export_mnms_ground_truth_dags.py \
        --output data/mnms-ground-truth-dags.jsonl
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from urllib.request import urlopen

from jsonschema import Draft202012Validator

from eec_sched.domain import PlanningRequest, ToolRegistry, ToolSpec
from eec_sched.mnms_tools import mnms_tool_specs
from eec_sched.openai_compatible import plan_from_dict
from eec_sched.planning import validate_plan


DEFAULT_DATASET_URL = "https://huggingface.co/datasets/zixianma/mnms/resolve/e9cb0112a0a88788fc7559e27cdf40646f30ad3f/test_human_verified_filtered.json"
NODE_REFERENCE = re.compile(r"<node-(\d+)>\.([A-Za-z_][A-Za-z0-9_]*)")


class SkipQuery(ValueError):
    """The complete MnMS query cannot be projected into a scheduler DAG."""


def _tool_id(mnms_name: object) -> str:
    if not isinstance(mnms_name, str):
        raise SkipQuery("tool name is not a string")
    return mnms_name.replace(" ", "_")


def _node_id(mnms_id: object) -> str:
    if not isinstance(mnms_id, int):
        raise SkipQuery("node id is not an integer")
    return f"node-{mnms_id}"


def _reference(value: object) -> tuple[str, str] | None:
    if not isinstance(value, str):
        return None
    match = NODE_REFERENCE.search(value)
    return (match.group(1), match.group(2)) if match else None


def _parse_plan(raw_plan: object) -> list[dict[str, object]]:
    if not isinstance(raw_plan, str):
        raise SkipQuery("plan_str is not a string")
    try:
        plan = ast.literal_eval(raw_plan)
    except (SyntaxError, ValueError) as error:
        raise SkipQuery(f"plan_str is not a Python literal: {error}") from error
    if not isinstance(plan, list) or not plan:
        raise SkipQuery("plan is not a non-empty list")
    if not all(isinstance(node, dict) for node in plan):
        raise SkipQuery("plan has a non-object node")
    return plan


def project_plan(raw_plan: object, catalog: Mapping[str, ToolSpec]) -> tuple[dict[str, object], dict[str, object]]:
    """Return a schema DAG and request values used only for domain validation."""
    source_nodes = _parse_plan(raw_plan)
    source_ids = {_node_id(node.get("id")) for node in source_nodes}
    if len(source_ids) != len(source_nodes):
        raise SkipQuery("plan contains duplicate node ids")

    nodes: list[dict[str, object]] = []
    request_inputs: dict[str, object] = {}
    for raw_node in source_nodes:
        node_id = _node_id(raw_node.get("id"))
        tool_id = _tool_id(raw_node.get("name"))
        if tool_id not in catalog:
            raise SkipQuery(f"tool is outside mnms_tool_specs: {tool_id}")
        raw_args = raw_node.get("args")
        if not isinstance(raw_args, dict):
            raise SkipQuery(f"{node_id}.args is not an object")

        inputs: dict[str, dict[str, str]] = {}
        for input_name, value in raw_args.items():
            if not isinstance(input_name, str):
                raise SkipQuery(f"{node_id} has a non-string argument name")
            reference = _reference(value)
            if reference:
                upstream_id = f"node-{reference[0]}"
                if upstream_id not in source_ids:
                    raise SkipQuery(f"{node_id}.{input_name} refers to missing {upstream_id}")
                inputs[input_name] = {"kind": "node", "name": upstream_id, "port": reference[1]}
            else:
                request_name = f"request_{node_id}_{input_name}"
                inputs[input_name] = {"kind": "request", "name": request_name}
                spec = catalog[tool_id]
                modality = spec.inputs.get(input_name).modality if input_name in spec.inputs else None
                request_inputs[request_name] = _request_value(value, modality)
        nodes.append({"node_id": node_id, "tool_id": tool_id, "inputs": inputs})

    referenced_nodes = {source["name"] for node in nodes for source in node["inputs"].values() if source["kind"] == "node"}
    terminals = [node for node in nodes if node["node_id"] not in referenced_nodes]
    final_outputs = [
        {"node_id": node["node_id"], "port": port_name}
        for node in terminals
        for port_name in catalog[node["tool_id"]].outputs
    ]
    return {"nodes": nodes, "final_outputs": final_outputs}, request_inputs


def _request_value(value: object, modality: str | None) -> object:
    """Use Path for media so ``validate_plan`` checks the intended modality."""
    if modality in {"image", "audio"}:
        return Path(str(value))
    return str(value)


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for spec in mnms_tool_specs():
        registry.register(spec, lambda: None)  # Runners are never constructed by validation.
    return registry


def iter_dataset(url: str) -> Iterable[dict[str, object]]:
    with urlopen(url, timeout=60) as response:  # noqa: S310 - user controls the dataset URL.
        for line_number, raw_line in enumerate(response, start=1):
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at line {line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"dataset line {line_number} is not an object")
            yield row


def export(dataset_url: str, output_path: Path) -> tuple[int, int]:
    schema_path = Path(__file__).parents[1] / "docs/schemas/tool-call-dag.schema.json"
    validator = Draft202012Validator(json.loads(schema_path.read_text()))
    catalog = {spec.tool_id: spec for spec in mnms_tool_specs()}
    registry = build_registry()
    exported = skipped = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for row in iter_dataset(dataset_url):
            try:
                dag, request_inputs = project_plan(row.get("plan_str"), catalog)
                validator.validate(dag)
                plan = plan_from_dict(dag)
                errors = validate_plan(
                    plan,
                    PlanningRequest(prompt=str(row.get("user_request", "")), inputs=request_inputs, minimum_accuracy=0.0, maximum_latency_ms=float("inf"), gamma=0.0),
                    registry,
                )
                if errors:
                    raise SkipQuery("; ".join(error.message for error in errors))
            except SkipQuery:
                skipped += 1
                continue
            except Exception as error:
                raise RuntimeError(f"unexpected conversion error for MnMS query {row.get('id')!r}") from error
            output.write(json.dumps(dag, separators=(",", ":")) + "\n")
            exported += 1
    return exported, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-url", default=DEFAULT_DATASET_URL, help="Direct URL to a MnMS JSONL file.")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSONL containing schema DAG values.")
    args = parser.parse_args()
    exported, skipped = export(args.dataset_url, args.output)
    print(f"Exported {exported} validated DAGs; skipped {skipped} queries.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
