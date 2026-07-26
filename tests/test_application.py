from __future__ import annotations

import unittest

from eec_sched import (
    AccuracyProfile,
    Configuration,
    FakePlannerClient,
    FinalOutput,
    InMemoryProfileRepository,
    InputSource,
    LatencyProfile,
    PlanningRequest,
    Port,
    ToolCallPlan,
    ToolNode,
    ToolRegistry,
    ToolSpec,
    execute_request,
)


class PrefixRunner:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix

    def prepare(self, configuration: Configuration) -> None:
        pass

    def run(self, inputs: dict[str, object], configuration: Configuration) -> dict[str, object]:
        return {"text": f"{self.prefix}{inputs['text']}"}


def tool(tool_id: str) -> ToolSpec:
    return ToolSpec(
        tool_id=tool_id,
        description=tool_id,
        inputs={"text": Port("text", "text")},
        outputs={"text": Port("text", "text")},
        configurations=(Configuration("accurate"), Configuration("fast")),
        metric_higher_is_better=True,
        metric_floor=0.0,
    )


class ApplicationContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ToolRegistry()
        self.registry.register(tool("caption"), lambda: PrefixRunner("caption: "))
        self.registry.register(tool("summarize"), lambda: PrefixRunner("summary: "))
        self.profiles = InMemoryProfileRepository(
            accuracy=(
                AccuracyProfile("caption", "accurate", 1.0),
                AccuracyProfile("caption", "fast", 0.8),
                AccuracyProfile("summarize", "accurate", 1.0),
                AccuracyProfile("summarize", "fast", 0.75),
            ),
            latency=(
                LatencyProfile("caption", "accurate", "cpu", "default", 10, 12, 3),
                LatencyProfile("caption", "fast", "cpu", "default", 4, 5, 3),
                LatencyProfile("summarize", "accurate", "cpu", "default", 10, 12, 3),
                LatencyProfile("summarize", "fast", "cpu", "default", 4, 5, 3),
            ),
        )

    def test_executes_valid_multi_node_plan_with_greedy_downgrade(self) -> None:
        plan = ToolCallPlan(
            nodes=(
                ToolNode("caption", "caption", {"text": InputSource.request("prompt")}),
                ToolNode("summary", "summarize", {"text": InputSource.node("caption", "text")}),
            ),
            final_outputs=(FinalOutput("summary", "text"),),
        )
        result = execute_request(
            PlanningRequest("describe and shorten", {"prompt": "a sunny beach"}, 0.75, 18, 0.5),
            registry=self.registry,
            planner=FakePlannerClient([plan]),
            profiles=self.profiles,
            device="cpu",
        )
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.outputs, {"text": "summary: caption: a sunny beach"})
        self.assertEqual(result.schedule.predicted_latency_ms, 17)
        self.assertEqual(
            {node.node_id: node.configuration_id for node in result.schedule.nodes},
            {"caption": "fast", "summary": "accurate"},
        )

    def test_repairs_invalid_plan_without_running_it(self) -> None:
        invalid = ToolCallPlan((ToolNode("bad", "missing", {}),), (FinalOutput("bad", "text"),))
        valid = ToolCallPlan(
            (ToolNode("caption", "caption", {"text": InputSource.request("prompt")}),),
            (FinalOutput("caption", "text"),),
        )
        result = execute_request(
            PlanningRequest("caption", {"prompt": "a fox"}, 0.5, 30, 0.5),
            registry=self.registry,
            planner=FakePlannerClient([invalid, valid]),
            profiles=self.profiles,
            device="cpu",
            max_repairs=1,
        )
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.repair_attempts, 1)


if __name__ == "__main__":
    unittest.main()
