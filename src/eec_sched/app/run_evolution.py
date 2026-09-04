"""Hydra composition root for model-backed Scheduler evolution."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence, cast
from random import Random

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

from ..artifacts import ArtifactStore
from ..candidate import SchedulerProposal, SchedulerView
from ..diagnosis import EmptyDiagnosis, ReactDiagnosis
from ..evolution import (
    CandidateEvaluation,
    CandidateGenerationRequest,
    ComplementaryBehaviorCrossoverParentSelector,
    EvolutionLoop,
    EvolutionLoopResult,
    FeatureDiverseMutationParentSelector,
    InMemoryRepertoire,
    ParetoMutationParentSelector,
    PerformanceBehaviorDescriptor,
    ProbabilisticOperatorSelector,
    ReflectionInput,
    SchedulerCandidate,
    SchedulerCandidateDraft,
    SchedulerCandidateRegistry,
    ScoringContext,
    TopKCosineCrossoverParentSelector,
    TraceEvaluation,
    TraceScoreBehaviorDescriptor,
    UnboundedRepertoire,
    evaluate_scheduler_candidate,
)
from ..llm import OpenAICompatibleChatModel
from ..profiling.snapshot import load_profiling_database
from ..workflow import ToolCallPlanDataset


def _jsonable(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _jsonable(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise TypeError(f"cannot serialize {type(value).__name__} for an Evolution Agent")


def _compile_candidate(source_code: str, source_name: str) -> Callable[[SchedulerView], SchedulerProposal]:
    compiled = compile(source_code, source_name, "exec")

    def propose(view: SchedulerView) -> SchedulerProposal:
        namespace: dict[str, object] = {"__builtins__": __builtins__}
        exec(compiled, namespace)  # noqa: S102 - Candidate execution is timed and validated by the trusted evaluator.
        generated = namespace.get("propose")
        if not callable(generated):
            raise ValueError(f"{source_name} must define callable propose(view)")
        return cast(Callable[[SchedulerView], SchedulerProposal], generated)(view)

    return propose


def load_candidate(source: str | Path, scheduler_version: int, description: str) -> SchedulerCandidate:
    path = Path(source)
    source_code = path.read_text(encoding="utf-8")
    return SchedulerCandidate(
        scheduler_version,
        _compile_candidate(source_code, str(path)),
        source_code=source_code,
        strategy_description=description,
    )


def build_agents(
    diagnosis_kind: str,
    diagnosis_model: OpenAICompatibleChatModel,
    coding_model: OpenAICompatibleChatModel,
    artifacts: ArtifactStore,
    *,
    empty_diagnosis_advice: str = "",
):
    def reflection_agent(reflection: ReflectionInput) -> str:
        evidence = cast(dict[str, object], _jsonable(reflection))
        if diagnosis_kind == "empty":
            advice = EmptyDiagnosis(empty_diagnosis_advice).diagnose(evidence).advice
        elif diagnosis_kind == "one_shot":
            advice = diagnosis_model.complete(
                system=(
                    "Diagnose the supplied trusted End-Edge-Cloud scheduling evidence. "
                    "Return concise, testable advice for changing the Scheduler Candidate; "
                    "do not modify evaluation rules."
                ),
                user={"candidate_evaluation_evidence": evidence},
            )
        elif diagnosis_kind == "react":
            advice = ReactDiagnosis(
                diagnosis_model.base_url, diagnosis_model.token, diagnosis_model.model
            ).diagnose(evidence).advice
        else:
            raise ValueError(f"unknown diagnosis adapter: {diagnosis_kind}")
        artifacts.append_event("diagnosis", {"input": evidence, "advice": advice})
        return advice

    def coding_agent(request: CandidateGenerationRequest) -> SchedulerCandidateDraft:
        context = cast(dict[str, object], _jsonable(request))
        content = coding_model.complete(
            system=(
                "Generate one complete Scheduler Candidate. Return a JSON object containing "
                "only source_code and strategy_description. source_code must define propose(view) "
                "and assign a Configuration and Compatible Device to every DAG node."
            ),
            user={"candidate_generation_request": context},
            json_output=True,
        )
        try:
            value = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("Coding Agent must return a JSON object") from exc
        if not isinstance(value, dict) or set(value) != {"source_code", "strategy_description"}:
            raise ValueError("Coding Agent must return only source_code and strategy_description")
        source_code, description = value["source_code"], value["strategy_description"]
        if not isinstance(source_code, str) or not isinstance(description, str):
            raise ValueError("Coding Agent fields must be strings")
        artifacts.append_event("coding", {"input": context, "output": value})
        return SchedulerCandidateDraft(
            _compile_candidate(source_code, "generated-scheduler.py"),
            source_code,
            description,
            reflection_feedback=request.reflection_advice,
            coding_context=context,
        )

    return reflection_agent, coding_agent


def _split(dataset: ToolCallPlanDataset, name: str):
    if name == "all":
        return dataset.traces
    value = dataset.split()
    return {"train": value.train, "validation": value.validation, "test": value.test}[name]


def _result_json(result: EvolutionLoopResult) -> dict[str, object]:
    evaluations: dict[str, object] = {}
    for version, evaluation in result.evolution_graph.evaluations.items():
        candidate = result.evolution_graph.candidates[version]
        evaluations[str(version)] = {
            **evaluation.concise_projection(),
            "parent_scheduler_versions": list(candidate.parent_scheduler_versions),
            "strategy_description": candidate.strategy_description,
            "source_code": candidate.source_code,
            "traces": [_jsonable(record) for record in evaluation.traces],
        }
    return {
        "candidate_versions": sorted(result.evolution_graph.candidates),
        "selected_scheduler_version": result.selected_candidate.scheduler_version,
        "evolution_evaluations": evaluations,
        "final_evaluation": result.final_evaluation.concise_projection(),
    }


def run(config: DictConfig) -> Path | None:
    load_dotenv(Path.cwd() / ".env")
    artifacts = cast(ArtifactStore, instantiate(config.artifacts))
    diagnosis_model = cast(OpenAICompatibleChatModel, instantiate(config.models.diagnosis))
    coding_model = cast(OpenAICompatibleChatModel, instantiate(config.models.coding))
    reflection_agent, coding_agent = build_agents(
        config.diagnosis.kind,
        diagnosis_model,
        coding_model,
        artifacts,
        empty_diagnosis_advice=config.diagnosis.get("advice", ""),
    )
    snapshot = load_profiling_database(Path(config.evidence.database), Path(config.evidence.schema))
    dataset = ToolCallPlanDataset.load(config.workflow.dataset)
    evolution_traces = _split(dataset, config.run.evolution_split)
    final_traces = _split(dataset, config.run.final_split)
    roots = tuple(
        load_candidate(item.source, item.scheduler_version, f"root Candidate loaded from {item.source}")
        for item in config.run.root_candidates
    )
    oracle = load_candidate(config.run.oracle_source, config.run.oracle_version, "Oracle reference Candidate")
    scoring_context = ScoringContext(
        minimum_accuracy=float(config.evaluation.minimum_accuracy),
        maximum_latency_ms=float(config.evaluation.maximum_latency_ms),
        gamma=float(config.evaluation.gamma),
    )

    descriptor = (
        PerformanceBehaviorDescriptor()
        if config.search.behavior_descriptor == "performance_behavior"
        else TraceScoreBehaviorDescriptor()
    )
    mutation_selector = (
        FeatureDiverseMutationParentSelector(descriptor)
        if config.search.mutation_parent_selector == "feature_diverse"
        else ParetoMutationParentSelector()
    )
    crossover_selector = (
        ComplementaryBehaviorCrossoverParentSelector(descriptor)
        if config.search.crossover_parent_selector == "complementary_behavior"
        else TopKCosineCrossoverParentSelector()
    )
    memory = instantiate(config.memory)
    repertoire = (
        InMemoryRepertoire(descriptor)
        if config.search.repertoire == "feature_archive"
        else UnboundedRepertoire()
    )
    random_source = Random(config.run.seed)
    import eec_sched.evolution as evolution_package

    if config.candidate_runtime.kind != "in_process":
        raise ValueError(f"unknown Candidate runtime: {config.candidate_runtime.kind}")
    evolution_package._CANDIDATE_TIMEOUT_SECONDS = float(config.candidate_runtime.timeout_seconds)

    def trusted(current_snapshot, trace, candidate) -> TraceEvaluation:
        value = evaluate_scheduler_candidate(
            current_snapshot,
            (trace,),
            candidate.scheduler_version,
            SchedulerCandidateRegistry({candidate.scheduler_version: candidate}),
            scoring_context=scoring_context,
        )
        assert isinstance(value, CandidateEvaluation)
        return value.traces[0]

    def oracle_scores(current_snapshot, records, selected) -> Sequence[float | None]:
        traces = tuple(record.trace for record in records)
        value = evaluate_scheduler_candidate(
            current_snapshot,
            traces,
            oracle.scheduler_version,
            SchedulerCandidateRegistry({oracle.scheduler_version: oracle}),
            scoring_context=scoring_context,
        )
        assert isinstance(value, CandidateEvaluation)
        return tuple(record.score_contribution for record in value.traces)

    result = EvolutionLoop(
        snapshot=snapshot,
        evolution_traces=evolution_traces,
        final_evaluation_traces=final_traces,
        initial_candidates=roots,
        rounds=config.run.rounds,
        mutation_probability=config.search.mutation_probability,
        reflection_agent=reflection_agent,
        coding_agent=coding_agent,
        trusted_evaluator=trusted,
        oracle=oracle_scores,
        show_progress=True,
        random_float=random_source.random,
        random_choice=random_source.choice,
        mutation_parent_selector=mutation_selector,
        crossover_parent_selector=crossover_selector,
        behavior_descriptor=descriptor,
        operator_selector=ProbabilisticOperatorSelector(config.search.mutation_probability),
        repertoire=repertoire,
        memory=memory,
    ).run()
    payload = _result_json(result)
    payload.update(
        resolved_config=OmegaConf.to_container(config, resolve=True),
        evolution_trace_count=len(evolution_traces),
        final_trace_count=len(final_traces),
    )
    return artifacts.write_result(payload)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig) -> None:
    result_path = run(config)
    if result_path is not None:
        print(result_path)


if __name__ == "__main__":
    main()
