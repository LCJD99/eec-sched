from __future__ import annotations

from eec_sched import (
    Configuration,
    FinalOutput,
    InputSource,
    PlanningRequest,
    Port,
    ToolCallPlan,
    ToolNode,
    ToolRegistry,
    ToolSpec,
)
from eec_sched.planning import validate_plan


def registry() -> ToolRegistry:
    result = ToolRegistry()
    result.register(
        ToolSpec("text", "text", {"text": Port("text", "text")}, {"text": Port("text", "text")}, (Configuration("base"),), True, 0),
        lambda: None,  # validation must never instantiate it
    )
    result.register(
        ToolSpec("image", "image", {"image": Port("image", "image")}, {"image": Port("image", "image")}, (Configuration("base"),), True, 0),
        lambda: None,
    )
    return result


def request() -> PlanningRequest:
    return PlanningRequest("work", {"prompt": "hello"}, 0.5, 20, 0.5)


def test_rejects_modality_mismatch_and_cycle() -> None:
    plan = ToolCallPlan(
        (
            ToolNode("a", "text", {"text": InputSource.node("b", "image")}),
            ToolNode("b", "image", {"image": InputSource.node("a", "text")}),
        ),
        (FinalOutput("a", "text"),),
    )
    codes = {error.code for error in validate_plan(plan, request(), registry())}
    assert {"modality_mismatch", "cycle"} <= codes


def test_rejects_text_request_bound_to_image_port() -> None:
    plan = ToolCallPlan((ToolNode("image", "image", {"image": InputSource.request("prompt")}),), (FinalOutput("image", "image"),))
    codes = {error.code for error in validate_plan(plan, request(), registry())}
    assert "modality_mismatch" in codes


def test_rejects_non_connectable_request_value() -> None:
    plan = ToolCallPlan((ToolNode("text", "text", {"text": InputSource.request("payload")}),), (FinalOutput("text", "text"),))
    invalid_request = PlanningRequest("work", {"payload": object()}, 0.5, 20, 0.5)
    codes = {error.code for error in validate_plan(plan, invalid_request, registry())}
    assert "modality_mismatch" in codes
