"""Construct OpenAI-compatible Evolution Agents from a private YAML file.

This runner deliberately does not create an Evolution Loop: traces, root
Scheduler Candidates, and the Oracle are trusted application inputs, not LLM
configuration.  Import ``build_evolution_agents`` when wiring those inputs
into an ``EvolutionLoop``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, is_dataclass
import json
from pathlib import Path
from typing import Callable, Mapping, cast
from urllib.request import Request, urlopen

from eec_sched import EvolutionAgentLlmConfig, load_evolution_agent_llm_config
from eec_sched.evolution import CandidateGenerationRequest, CodingAgent, ReflectionAgent, ReflectionInput, SchedulerCandidate, SchedulerProposal, SchedulerView


@dataclass(frozen=True)
class EvolutionAgents:
    """The two model-backed dependencies required by ``EvolutionLoop``."""

    reflection_agent: ReflectionAgent
    coding_agent: CodingAgent


def build_evolution_agents(config: EvolutionAgentLlmConfig) -> EvolutionAgents:
    """Construct Reflection and Coding Agents sharing one LLM configuration."""

    def reflection_agent(reflection: ReflectionInput) -> str:
        return _chat_completion(
            config,
            system=(
                "You are the Reflection Agent for Scheduler Candidate evolution. "
                "Diagnose the supplied strategy descriptions and trusted Trace evidence. "
                "Return concise textual Reflection Advice. You never receive or request "
                "Scheduler source code, and you must not propose evaluator changes."
            ),
            user={"reflection": _jsonable(reflection)},
        )

    def coding_agent(request: CandidateGenerationRequest) -> SchedulerCandidate:
        content = _chat_completion(
            config,
            system=(
                "You are the Coding Agent for Scheduler Candidate evolution. "
                "Return one JSON object with exactly scheduler_version, source_code, and "
                "strategy_description. source_code must define propose(view), which returns "
                "one configuration_id and device_id assignment for every DAG node. Keep the "
                "trusted evaluator, scoring, timing, and execution order unchanged."
            ),
            user={"candidate_generation_request": _jsonable(request)},
            json_output=True,
        )
        try:
            candidate = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("Coding Agent must return a JSON object") from exc
        if not isinstance(candidate, dict) or set(candidate) != {
            "scheduler_version",
            "source_code",
            "strategy_description",
        }:
            raise ValueError(
                "Coding Agent JSON must contain only scheduler_version, source_code, and strategy_description"
            )
        version = candidate["scheduler_version"]
        source_code = candidate["source_code"]
        description = candidate["strategy_description"]
        if not isinstance(version, int) or isinstance(version, bool):
            raise ValueError("Coding Agent scheduler_version must be an integer")
        if not isinstance(source_code, str) or not isinstance(description, str):
            raise ValueError("Coding Agent source_code and strategy_description must be strings")
        return SchedulerCandidate(
            scheduler_version=version,
            propose=_proposal_from_source(source_code),
            source_code=source_code,
            parent_scheduler_versions=request.parent_scheduler_versions,
            strategy_description=description,
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
    with urlopen(request, timeout=60) as response:  # noqa: S310 - configured compatible endpoint
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Construct OpenAI-compatible Evolution Agents from YAML.")
    parser.add_argument("--config", required=True, type=Path, help="Path to private Evolution Agent YAML configuration.")
    parser.add_argument("--check", action="store_true", help="Call the Reflection Agent once to verify the configured endpoint.")
    arguments = parser.parse_args()
    agents = build_evolution_agents(load_evolution_agent_llm_config(arguments.config))
    if arguments.check:
        agents.reflection_agent(ReflectionInput("mutation", {}, ()))
        print("Reflection Agent connection succeeded.")
    else:
        print("Reflection and Coding Agents constructed. Supply them to an EvolutionLoop with trusted inputs.")


if __name__ == "__main__":
    main()
