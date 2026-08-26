"""Minimal OpenAI-compatible structured-output planner adapter.

The adapter is intentionally independent of the OpenAI SDK: deployments which
provide the compatible HTTP contract can use it without becoming a framework
dependency.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Callable
from urllib.request import Request, urlopen

from .domain import FinalOutput, InputSource, PlannerClient, PlanningRequest, ToolCallPlan, ToolNode, ToolSpec, ValidationError


def plan_from_dict(value: dict[str, Any]) -> ToolCallPlan:
    nodes = tuple(
        ToolNode(
            node_id=node["node_id"], tool_id=node["tool_id"],
            inputs={name: InputSource(source["kind"], source["name"], source.get("port"), source.get("data_type", "text") if source["kind"] == "request" else None) for name, source in node["inputs"].items()},
        )
        for node in value["nodes"]
    )
    return ToolCallPlan(nodes, tuple(FinalOutput(item["node_id"], item["port"]) for item in value["final_outputs"]))


class OpenAICompatiblePlannerClient(PlannerClient):
    def __init__(self, base_url: str, api_key: str, model: str, transport: Callable[[str, str, dict[str, Any]], dict[str, Any]] | None = None) -> None:
        self.base_url, self.api_key, self.model = base_url.rstrip("/"), api_key, model
        self._transport = transport or self._post

    def plan(self, request: PlanningRequest, catalog: tuple[ToolSpec, ...], repair_errors: tuple[ValidationError, ...] = ()) -> ToolCallPlan:
        payload = {
            "model": self.model,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": "Plan only tool IDs, named ports, input bindings, and final outputs. Never select model configurations."}, {"role": "user", "content": json.dumps({
                "request": request.prompt,
                "inputs": sorted(request.inputs),
                "constraints": {"minimum_accuracy": request.minimum_accuracy, "maximum_latency_ms": request.maximum_latency_ms, "gamma": request.gamma},
                "catalog": [{"tool_id": spec.tool_id, "description": spec.description, "inputs": {name: port.modality for name, port in spec.inputs.items()}, "outputs": {name: port.modality for name, port in spec.outputs.items()}} for spec in catalog],
                "repair_errors": [asdict(error) for error in repair_errors],
            })}],
        }
        response = self._transport(self.base_url + "/chat/completions", self.api_key, payload)
        return plan_from_dict(json.loads(response["choices"][0]["message"]["content"]))

    @staticmethod
    def _post(url: str, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(url, data=json.dumps(payload).encode(), headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=60) as response:  # noqa: S310 - caller controls compatible endpoint
            return json.loads(response.read())
