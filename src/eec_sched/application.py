"""The public request-execution application seam."""

from __future__ import annotations

from dataclasses import dataclass

from .domain import PlannerClient, PlanningRequest, ToolCallPlan, ToolRegistry, ValidationError
from .execution import ExecutionResult, execute
from .planning import validate_plan
from .profiles import InMemoryProfileRepository
from .scheduling import NaiveScheduler, SchedulingPlan


@dataclass(frozen=True)
class RequestResult:
    status: str
    plan: ToolCallPlan | None = None
    schedule: SchedulingPlan | None = None
    outputs: dict[str, object] | None = None
    validation_errors: tuple[ValidationError, ...] = ()
    repair_attempts: int = 0
    execution: ExecutionResult | None = None


def execute_request(request: PlanningRequest, *, registry: ToolRegistry, planner: PlannerClient, profiles: InMemoryProfileRepository, device: str, max_repairs: int = 2, scheduler: NaiveScheduler | None = None) -> RequestResult:
    errors: tuple[ValidationError, ...] = ()
    plan: ToolCallPlan | None = None
    for attempt in range(max_repairs + 1):
        plan = planner.plan(request, registry.catalog(), errors)
        errors = validate_plan(plan, request, registry)
        if not errors:
            break
    if errors or plan is None:
        return RequestResult("planning_failed", plan, validation_errors=errors, repair_attempts=max_repairs)
    schedule = (scheduler or NaiveScheduler()).schedule(plan, request, registry, profiles, device)
    if schedule.status != "scheduled":
        return RequestResult(schedule.status, plan, schedule, repair_attempts=attempt)
    execution = execute(plan, schedule, dict(request.inputs), registry)
    return RequestResult(execution.status, plan, schedule, execution.outputs, repair_attempts=attempt, execution=execution)
