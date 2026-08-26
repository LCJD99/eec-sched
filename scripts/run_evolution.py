"""Run the LLM-backed Scheduler Candidate evolution loop.

The runner keeps all evaluation inside the trusted public evaluator boundary.
It uses the dataset's deterministic train/test split for evolution and final
evaluation, respectively, and loads three root Candidates plus one Oracle
Candidate from source files.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence, cast
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from eec_sched import (
    CandidateEvaluation,
    EvaluationTrace,
    EvolutionAgentLlmConfig,
    EvolutionLoop,
    EvolutionLoopResult,
    SchedulerCandidate,
    SchedulerCandidateDraft,
    SchedulerCandidateRegistry,
    ToolCallPlanDataset,
    evaluate_scheduler_candidate,
    load_evolution_runner_config,
    load_profiling_database,
)
from eec_sched.evolution import (
    CandidateGenerationRequest,
    CodingAgent,
    ReflectionAgent,
    ReflectionInput,
    ReflectionTraceEvidence,
    SchedulerProposal,
    SchedulerView,
    TraceEvaluation,
)


@dataclass(frozen=True)
class EvolutionAgents:
    """The two model-backed dependencies required by ``EvolutionLoop``."""

    reflection_agent: ReflectionAgent
    coding_agent: CodingAgent


class LlmTraceWriter:
    """Write one JSON object for every completed LLM call in this run."""

    def __init__(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = output_dir / f"evolution-trace_{self.timestamp}.jsonl"
        self.result_path = output_dir / f"evolution-result_{self.timestamp}.json"
        self.path.write_text("", encoding="utf-8")

    def write(self, call_type: str, output: str) -> None:
        with self.path.open("a", encoding="utf-8") as trace:
            trace.write(json.dumps({"type": call_type, "output": output}, ensure_ascii=False) + "\n")


def build_evolution_agents(
    reflection_config: EvolutionAgentLlmConfig,
    coding_config: EvolutionAgentLlmConfig | None = None,
    trace_writer: LlmTraceWriter | None = None,
) -> EvolutionAgents:
    """Construct Reflection and Coding Agents from independent LLM configs."""
    coding_config = reflection_config if coding_config is None else coding_config
    pending_reflection_context: str | None = None

    def reflection_agent(reflection: ReflectionInput) -> str:
        nonlocal pending_reflection_context
        pending_reflection_context = _reflection_prompt(reflection)
        output = _chat_completion(
            reflection_config,
            system=(
                "You are the Reflection Agent for Scheduler Candidate evolution. "
                "Diagnose the supplied strategy descriptions and trusted Trace evidence. "
                "Return concise textual Reflection Advice. You never receive or request "
                "Scheduler source code, and you must not propose evaluator changes."
            ),
            user={"reflection_prompt": pending_reflection_context},
        )
        if trace_writer is not None:
            trace_writer.write("reflection", output)
        return output

    def coding_agent(request: CandidateGenerationRequest) -> SchedulerCandidateDraft:
        content = _chat_completion(
            coding_config,
            system=(
                "You are the Coding Agent for Scheduler Candidate evolution. "
                "Return one JSON object with exactly source_code and "
                "strategy_description. source_code must define propose(view), which returns "
                "one configuration_id and device_id assignment for every DAG node. Keep the "
                "trusted evaluator, scoring, timing, and execution order unchanged."
            ),
            user={"candidate_generation_request": _jsonable(request)},
            json_output=True,
        )
        if trace_writer is not None:
            trace_writer.write("coding", content)
        try:
            candidate = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("Coding Agent must return a JSON object") from exc
        if not isinstance(candidate, dict) or set(candidate) != {"source_code", "strategy_description"}:
            raise ValueError(
                "Coding Agent JSON must contain only source_code and strategy_description"
            )
        source_code = candidate["source_code"]
        description = candidate["strategy_description"]
        if not isinstance(source_code, str) or not isinstance(description, str):
            raise ValueError("Coding Agent source_code and strategy_description must be strings")
        return SchedulerCandidateDraft(
            propose=_proposal_from_source(source_code),
            source_code=source_code,
            strategy_description=description,
            reflection_context=pending_reflection_context,
            reflection_feedback=request.reflection_advice,
            coding_context=cast(dict[str, object], _jsonable(request)),
        )

    return EvolutionAgents(reflection_agent, coding_agent)


def _proposal_from_source(source_code: str) -> Callable[[SchedulerView], SchedulerProposal]:
    compiled = compile(source_code, "generated-scheduler.py", "exec")

    def propose(view: SchedulerView) -> SchedulerProposal:
        namespace: dict[str, object] = {}
        exec(compiled, namespace)  # noqa: S102 - SchedulerCandidate code executes only within trusted evaluation.
        generated_propose = namespace.get("propose")
        if not callable(generated_propose):
            raise ValueError("Coding Agent source_code must define callable propose(view)")
        return cast(Callable[[SchedulerView], SchedulerProposal], generated_propose)(view)

    return propose


def _reflection_prompt(reflection: ReflectionInput) -> str:
    """Render bounded trusted evidence into the natural-language prompt seam."""
    strategies = "\n".join(
        f"- Candidate {version}: {description}"
        for version, description in sorted(reflection.strategy_descriptions.items())
    ) or "- No strategy description was supplied."
    parent_sources = "\n\n".join(
        f"Parent Candidate {index + 1} source code:\n```python\n{source}\n```"
        for index, source in enumerate(reflection.parent_source_codes)
    ) or "No parent source code is available."
    evidence = "\n\n".join(_format_trace_evidence(item) for item in reflection.trace_evidence)
    return (
        "You are the Reflection Agent for Scheduler Candidate evolution.\n"
        f"Current operation: {reflection.operator_description}\n"
        "Using the strategy descriptions and trusted evaluation evidence below, identify the most likely scheduling strategy issue "
        "and give the Coding Agent concise, actionable, verifiable improvement advice. "
        "Do not modify the evaluator. You may inspect the parent source code, but output advice only.\n\n"
        "Relevant strategies:\n"
        f"{strategies}\n\n"
        "Parent source code:\n"
        f"{parent_sources}\n\n"
        "Selected evaluation evidence:\n"
        f"{evidence or 'No Trace evidence is available.'}\n\n"
        "Output only Reflection Advice. State what scheduling tendency should change and why."
    )


def _format_trace_evidence(evidence: ReflectionTraceEvidence) -> str:
    lines = [
        f"Trace {evidence.trace_id} (Candidate {evidence.candidate_scheduler_version})",
        f"- Task input summary: {evidence.task_input}",
        f"- DAG: {'; '.join(evidence.dag) or 'empty'}",
        f"- Candidate result: status={evidence.candidate_status}, score={evidence.candidate_score:.3f}",
        f"- Candidate assignments: {'; '.join(evidence.assignments) or 'none'}",
    ]
    if evidence.candidate_reason:
        lines.append(f"- Failure or rejection reason: {evidence.candidate_reason}")
    metrics = ", ".join(
        f"{name}={value:.3f}" if isinstance(value, float) else f"{name}={value}"
        for name, value in evidence.metrics.items()
        if value is not None
    )
    if metrics:
        lines.append(f"- Trusted evaluation metrics: {metrics}")
    if evidence.compared_scheduler_version is not None:
        lines.extend(
            [
                f"- Compared with parent Candidate {evidence.compared_scheduler_version}: "
                f"status={evidence.compared_status}, score={evidence.compared_score:.3f}, "
                f"score_delta={evidence.score_delta:+.3f}",
                f"- Parent assignments: {'; '.join(evidence.compared_assignments) or 'none'}",
            ]
        )
    return "\n".join(lines)


def _chat_completion(
    config: EvolutionAgentLlmConfig,
    *,
    system: str,
    user: Mapping[str, object],
    json_output: bool = False,
) -> str:
    payload: dict[str, object] = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, sort_keys=True)},
        ],
    }
    if json_output:
        payload["response_format"] = {"type": "json_object"}
    request = Request(
        config.base_url + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {config.token}", "Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=360) as response:  # noqa: S310 - configured compatible endpoint
        body = json.loads(response.read())
    try:
        content = body["choices"][0]["message"]["content"]
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError("OpenAI-compatible endpoint returned no chat completion content") from exc
    if not isinstance(content, str):
        raise ValueError("OpenAI-compatible endpoint returned non-text chat completion content")
    return content


def _jsonable(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise TypeError(f"cannot serialize {type(value).__name__} for an Evolution Agent")


def _load_candidate(source_path: Path, scheduler_version: int, description: str) -> SchedulerCandidate:
    source_code = source_path.read_text(encoding="utf-8")
    compiled = compile(source_code, str(source_path), "exec")

    def propose(view: SchedulerView) -> SchedulerProposal:
        namespace: dict[str, object] = {"__builtins__": __builtins__}
        exec(compiled, namespace)  # noqa: S102 - source runs only in trusted evaluation.
        generated = namespace.get("propose")
        if not callable(generated):
            raise ValueError(f"{source_path} must define callable propose(view)")
        return cast(Callable[[SchedulerView], SchedulerProposal], generated)(view)

    return SchedulerCandidate(
        scheduler_version=scheduler_version,
        propose=propose,
        source_code=source_code,
        strategy_description=description,
    )


def _split_traces(dataset: ToolCallPlanDataset, split: str) -> Sequence[EvaluationTrace]:
    if split == "all":
        return dataset.traces
    splits = dataset.split()
    return {"train": splits.train, "validation": splits.validation, "test": splits.test}[split]


def _trusted_evaluator(snapshot: object, trace: EvaluationTrace, candidate: SchedulerCandidate) -> TraceEvaluation:
    result = evaluate_scheduler_candidate(
        snapshot,
        (trace,),
        candidate.scheduler_version,
        SchedulerCandidateRegistry({candidate.scheduler_version: candidate}),
    )
    assert isinstance(result, CandidateEvaluation)
    return result.traces[0]


def _oracle(snapshot: object, records: tuple[TraceEvaluation, ...], oracle_candidate: SchedulerCandidate) -> Sequence[float | None]:
    traces = tuple(record.trace for record in records)
    result = evaluate_scheduler_candidate(
        snapshot,
        traces,
        oracle_candidate.scheduler_version,
        SchedulerCandidateRegistry({oracle_candidate.scheduler_version: oracle_candidate}),
    )
    assert isinstance(result, CandidateEvaluation)
    return tuple(record.score_contribution for record in result.traces)


def _result_json(result: object) -> dict[str, object]:
    loop_result = cast(EvolutionLoopResult, result)
    evolution_evaluations = {}
    for version, evaluation in loop_result.evolution_graph.evaluations.items():
        candidate = loop_result.evolution_graph.candidates[version]
        projection = {
            **evaluation.concise_projection(),
            "parent_scheduler_versions": list(candidate.parent_scheduler_versions),
            "strategy_description": candidate.strategy_description,
            "source_code": candidate.source_code,
            "reflection_context": candidate.reflection_context,
            "reflection_feedback": candidate.reflection_feedback,
            "coding_context": candidate.coding_context,
        }
        # Keep lightweight test doubles and older callers compatible while
        # recording the complete payload for real CandidateEvaluation values.
        if hasattr(evaluation, "traces"):
            projection.update(
                traces=[_trace_json(record) for record in evaluation.traces],
            )
        evolution_evaluations[str(version)] = projection
    return {
        "candidate_versions": sorted(loop_result.evolution_graph.candidates),
        "selected_scheduler_version": loop_result.selected_candidate.scheduler_version,
        "evolution_evaluations": evolution_evaluations,
        "final_evaluation": loop_result.final_evaluation.concise_projection(),
    }


def _trace_json(record: TraceEvaluation) -> dict[str, object]:
    """Serialize the trace evidence needed by the evolution visualizer."""
    return {
        "trace_id": record.trace.trace_id,
        "task_input": dict(record.trace.task_input),
        "dag": {
            "nodes": [
                {
                    "node_id": node.node_id,
                    "tool_id": node.tool_id,
                    "inputs": {
                        name: {
                            "kind": source.kind,
                            "name": source.name,
                            **({"port": source.port} if source.port is not None else {}),
                        }
                        for name, source in node.inputs.items()
                    },
                }
                for node in record.trace.dag.nodes
            ],
            "final_outputs": [
                {"node_id": output.node_id, "port": output.port}
                for output in record.trace.dag.final_outputs
            ],
        },
        "status": record.status,
        "score": record.score_contribution,
        "reason": record.reason,
        "assignments": {
            node_id: {
                "configuration_id": assignment.configuration_id,
                "device_id": assignment.device_id,
            }
            for node_id, assignment in record.assignments.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to private Evolution Agent YAML configuration.")
    parser.add_argument("--check", action="store_true", help="Only call Reflection Agent once and skip the evolution run.")
    arguments = parser.parse_args()
    load_dotenv(Path.cwd() / ".env")
    runner_config = load_evolution_runner_config(arguments.config)
    trace_writer = LlmTraceWriter(runner_config.output_dir)
    agents = build_evolution_agents(runner_config.reflection_llm, runner_config.coding_llm, trace_writer)
    if arguments.check:
        agents.reflection_agent(ReflectionInput("mutation", "Modify an existing strategy based on evaluation evidence.", {}, (), ()))
        print("Reflection Agent connection succeeded.")
        return

    snapshot = load_profiling_database(runner_config.profiling_database, runner_config.profiling_schema)
    dataset = ToolCallPlanDataset.load(runner_config.dataset)
    evolution_traces = _split_traces(dataset, runner_config.evolution_split)
    final_traces = _split_traces(dataset, runner_config.final_split)
    if {trace.trace_id for trace in evolution_traces} & {trace.trace_id for trace in final_traces}:
        parser.error("evolution and final Trace splits must be disjoint")
    initial_candidates = tuple(
        _load_candidate(item.source, item.scheduler_version, f"root Candidate loaded from {item.source}")
        for item in runner_config.root_candidates
    )
    oracle_candidate = _load_candidate(runner_config.oracle_source, runner_config.oracle_version, "Oracle reference Candidate")
    result = EvolutionLoop(
        snapshot=snapshot,
        evolution_traces=evolution_traces,
        final_evaluation_traces=final_traces,
        initial_candidates=initial_candidates,
        rounds=runner_config.rounds,
        mutation_probability=runner_config.mutation_probability,
        reflection_agent=agents.reflection_agent,
        coding_agent=agents.coding_agent,
        trusted_evaluator=_trusted_evaluator,
        oracle=lambda current_snapshot, records, candidate: _oracle(current_snapshot, records, oracle_candidate),
        show_progress=True,
    ).run()
    payload = _result_json(result)
    payload["evolution_trace_count"] = len(evolution_traces)
    payload["final_trace_count"] = len(final_traces)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    trace_writer.result_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
