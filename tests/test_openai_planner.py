from __future__ import annotations

import json

from eec_sched import Configuration, OpenAICompatiblePlannerClient, PlanningRequest, Port, ToolSpec


def test_openai_compatible_planner_sends_structured_constraints() -> None:
    captured: dict[str, object] = {}

    def transport(url: str, api_key: str, payload: dict[str, object]) -> dict[str, object]:
        captured.update(payload)
        return {"choices": [{"message": {"content": json.dumps({"nodes": [], "final_outputs": []})}}]}

    client = OpenAICompatiblePlannerClient("https://example.test/v1", "secret", "model", transport)
    client.plan(PlanningRequest("nothing", {}, 0.8, 50, 0.25), (ToolSpec("x", "x", {}, {"text": Port("text", "text")}, (Configuration("x"),)),))
    user_message = captured["messages"][1]["content"]  # type: ignore[index]
    assert json.loads(user_message)["constraints"] == {"minimum_accuracy": 0.8, "maximum_latency_ms": 50, "gamma": 0.25}
