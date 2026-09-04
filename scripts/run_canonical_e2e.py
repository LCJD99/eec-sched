"""Run the deterministic three-device Scheduler Candidate flow end to end."""

from __future__ import annotations

import json
from pathlib import Path

from eec_sched import (
    EvaluationTrace,
    FinalOutput,
    InputSource,
    SchedulerCandidate,
    SchedulerCandidateRegistry,
    ToolCallPlan,
    ToolNode,
    evaluate_scheduler_candidate,
    load_profiling_database,
)


ROOT = Path(__file__).parents[1]


def main() -> None:
    snapshot = load_profiling_database(
        ROOT / "docs/examples/profiling-database.fake.json",
        ROOT / "docs/schemas/profiling-database.schema.json",
    )
    candidate = SchedulerCandidate(1, three_device_proposal)
    oracle = SchedulerCandidate(2, cloud_proposal)
    registry = SchedulerCandidateRegistry({candidate.scheduler_version: candidate})

    candidate_evaluation = evaluate_scheduler_candidate(
        snapshot,
        (trace("evolution-trace"),),
        candidate.scheduler_version,
        registry,
    )
    final_evaluation = evaluate_scheduler_candidate(
        snapshot,
        (trace("final-trace"),),
        candidate.scheduler_version,
        registry,
        mode="final",
        oracle_reference=oracle,
    )

    print(
        json.dumps(
            {
                "candidate_evaluation": candidate_evaluation.concise_projection(),
                "final_evaluation": final_evaluation.concise_projection(),
            },
            indent=2,
        )
    )


def trace(trace_id: str) -> EvaluationTrace:
    return EvaluationTrace(
        trace_id,
        {"prompt": "summarize and classify this input"},
        ToolCallPlan(
            nodes=(
                ToolNode("generate", "text_generation", {"prompt": InputSource.request("prompt")}),
                ToolNode("summarize", "text_summarization", {"text": InputSource.node("generate", "text")}),
                ToolNode("classify", "text_classification", {"text": InputSource.node("summarize", "text")}),
            ),
            final_outputs=(FinalOutput("classify", "label"),),
        ),
    )


def three_device_proposal(view: object) -> dict[str, dict[str, str]]:
    return {
        "generate": {"configuration_id": "synthetic-reference", "device_id": "device"},
        "summarize": {"configuration_id": "fast", "device_id": "edge"},
        "classify": {"configuration_id": "quality", "device_id": "cloud"},
    }


def cloud_proposal(view: object) -> dict[str, dict[str, str]]:
    return {
        "generate": {"configuration_id": "synthetic-reference", "device_id": "cloud"},
        "summarize": {"configuration_id": "fast", "device_id": "cloud"},
        "classify": {"configuration_id": "quality", "device_id": "cloud"},
    }


if __name__ == "__main__":
    main()
