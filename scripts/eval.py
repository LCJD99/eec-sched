"""Evaluate one Scheduler Candidate version over a ToolCallPlan JSONL split.

The scheduler source file must define ``propose(view)``.  It is invoked only
through the existing trusted Scheduler Candidate boundary.

Example:
    uv run python scripts/eval.py \
        --scheduler-source schedulers/version-12.py --scheduler-version 12 \
        --output evaluation-traces.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

from eec_sched import (
    CandidateEvaluation,
    EvaluationTrace,
    SchedulerCandidate,
    SchedulerCandidateRegistry,
    SchedulerView,
    ToolCallPlan,
    ToolCallPlanDataset,
    evaluate_scheduler_candidate,
    load_profiling_database,
)
from eec_sched.evolution import SchedulerProposal, TraceEvaluation


ROOT = Path(__file__).parents[1]
DEFAULT_DATASET = ROOT / "data/mnms-ground-truth-dags.jsonl"
DEFAULT_PROFILING_DATABASE = ROOT / "docs/examples/profiling-database.fake.json"
DEFAULT_PROFILING_SCHEMA = ROOT / "docs/schemas/profiling-database.schema.json"


def load_scheduler_candidate(source_path: Path, scheduler_version: int) -> SchedulerCandidate:
    """Load a versioned Candidate whose source defines ``propose(view)``."""
    source_code = source_path.read_text(encoding="utf-8")
    compiled = compile(source_code, str(source_path), "exec")

    def propose(view: SchedulerView) -> SchedulerProposal:
        namespace: dict[str, object] = {"__builtins__": __builtins__}
        exec(compiled, namespace)  # noqa: S102 - Candidate source executes only in trusted evaluation.
        loaded = namespace.get("propose")
        if not callable(loaded):
            raise ValueError(f"{source_path} must define callable propose(view)")
        return cast(Callable[[SchedulerView], SchedulerProposal], loaded)(view)

    return SchedulerCandidate(
        scheduler_version=scheduler_version,
        propose=propose,
        source_code=source_code,
        strategy_description=f"loaded from {source_path}",
    )


def traces_for_split(dataset: ToolCallPlanDataset, split: str) -> Sequence[EvaluationTrace]:
    splits = dataset.split()
    return {"train": splits.train, "validation": splits.validation, "test": splits.test, "all": dataset.traces}[split]


def dag_projection(dag: ToolCallPlan) -> dict[str, object]:
    return {
        "nodes": [
            {
                "node_id": node.node_id,
                "tool_id": node.tool_id,
                "inputs": {
                    name: {"kind": source.kind, "name": source.name, **({"port": source.port} if source.port is not None else {})}
                    for name, source in node.inputs.items()
                },
            }
            for node in dag.nodes
        ],
        "final_outputs": [{"node_id": output.node_id, "port": output.port} for output in dag.final_outputs],
    }


def trace_projection(record: TraceEvaluation) -> dict[str, object]:
    return {
        "trace_id": record.trace.trace_id,
        "dag": dag_projection(record.trace.dag),
        "status": record.status,
        "score": record.score_contribution,
        "reason": record.reason,
        "assignments": {
            node_id: {"configuration_id": assignment.configuration_id, "device_id": assignment.device_id}
            for node_id, assignment in record.assignments.items()
        },
    }


def write_trace_output(path: Path, evaluation: CandidateEvaluation) -> None:
    with path.open("w", encoding="utf-8") as output:
        for record in evaluation.traces:
            output.write(json.dumps(trace_projection(record), ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=("train", "validation", "test", "all"), default="test")
    parser.add_argument("--scheduler-source", type=Path, required=True)
    parser.add_argument("--scheduler-version", type=int, required=True)
    parser.add_argument("--profiling-database", type=Path, default=DEFAULT_PROFILING_DATABASE)
    parser.add_argument("--profiling-schema", type=Path, default=DEFAULT_PROFILING_SCHEMA)
    parser.add_argument("--output", type=Path, required=True, help="Destination JSONL with one evaluated Trace per line.")
    args = parser.parse_args()

    dataset = ToolCallPlanDataset.load(args.dataset)
    candidate = load_scheduler_candidate(args.scheduler_source, args.scheduler_version)
    snapshot = load_profiling_database(args.profiling_database, args.profiling_schema)
    evaluation = evaluate_scheduler_candidate(
        snapshot,
        traces_for_split(dataset, args.split),
        candidate.scheduler_version,
        SchedulerCandidateRegistry({candidate.scheduler_version: candidate}),
    )
    assert isinstance(evaluation, CandidateEvaluation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_trace_output(args.output, evaluation)
    print(json.dumps({"split": args.split, "trace_count": len(evaluation.traces), **evaluation.concise_projection()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
