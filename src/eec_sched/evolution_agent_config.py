"""Configuration for the Evolution Agents and their complete runner."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any, Mapping

import yaml

_ENVIRONMENT_TOKEN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


@dataclass(frozen=True)
class EvolutionAgentLlmConfig:
    """One OpenAI-compatible endpoint used by an Evolution Agent."""

    base_url: str
    token: str
    model: str

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise ValueError("Evolution Agent LLM base_url must not be empty")
        if not self.token.strip():
            raise ValueError("Evolution Agent LLM token must not be empty")
        if not self.model.strip():
            raise ValueError("Evolution Agent LLM model must not be empty")


@dataclass(frozen=True)
class EvolutionRootCandidateConfig:
    scheduler_version: int
    source: Path


@dataclass(frozen=True)
class EvolutionRunnerConfig:
    reflection_llm: EvolutionAgentLlmConfig
    coding_llm: EvolutionAgentLlmConfig
    dataset: Path
    profiling_database: Path
    profiling_schema: Path
    root_candidates: tuple[EvolutionRootCandidateConfig, ...]
    oracle_source: Path
    oracle_version: int
    rounds: int
    mutation_probability: float
    evolution_split: str
    final_split: str
    output_dir: Path


def load_evolution_agent_llm_config(
    path: Path, environ: Mapping[str, str] | None = None
) -> EvolutionAgentLlmConfig:
    """Load ``llm.base_url``, ``llm.token``, and ``llm.model`` from YAML.

    A token written as ``${VARIABLE_NAME}`` is resolved from the supplied
    environment (or the process environment), so credentials need not be kept
    in the YAML file.
    """
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read Evolution Agent LLM config: {path}") from exc
    if not isinstance(document, dict) or set(document) != {"llm"}:
        raise ValueError("Evolution Agent config must contain only an 'llm' mapping")
    return _parse_llm(document["llm"], os.environ if environ is None else environ, "llm")


def load_evolution_runner_config(
    path: Path, environ: Mapping[str, str] | None = None
) -> EvolutionRunnerConfig:
    """Load the complete runner configuration from YAML.

    Paths are interpreted relative to the process working directory. Tokens
    may use ``${ENVIRONMENT_VARIABLE}`` references.
    """
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read Evolution Agent config: {path}") from exc
    if not isinstance(document, dict) or set(document) != {"llm", "run"}:
        raise ValueError("Evolution Agent config must contain only 'llm' and 'run' mappings")
    environment = os.environ if environ is None else environ
    llm = document["llm"]
    if not isinstance(llm, dict) or set(llm) != {"reflection", "coding"}:
        raise ValueError("Evolution Agent llm config requires reflection and coding mappings")
    reflection_llm = _parse_llm(llm["reflection"], environment, "llm.reflection")
    coding_llm = _parse_llm(llm["coding"], environment, "llm.coding")
    run = document["run"]
    if not isinstance(run, dict):
        raise ValueError("Evolution Agent run config must be a mapping")
    required = {
        "dataset",
        "profiling_database",
        "profiling_schema",
        "root_candidates",
        "oracle_source",
        "oracle_version",
        "rounds",
        "mutation_probability",
        "evolution_split",
        "final_split",
        "output_dir",
    }
    if set(run) != required:
        raise ValueError(f"Evolution Agent run config requires exactly: {', '.join(sorted(required))}")
    roots = run["root_candidates"]
    if not isinstance(roots, list) or len(roots) != 3:
        raise ValueError("run.root_candidates must contain exactly three entries")
    root_configs = tuple(_parse_root(item, index) for index, item in enumerate(roots))
    if len({item.scheduler_version for item in root_configs}) != 3:
        raise ValueError("run.root_candidates scheduler_version values must be distinct")
    for key in ("dataset", "profiling_database", "profiling_schema", "oracle_source"):
        if not isinstance(run[key], str) or not run[key].strip():
            raise ValueError(f"run.{key} must be a non-empty string")
    integer_values = ("oracle_version", "rounds")
    if any(not isinstance(run[key], int) or isinstance(run[key], bool) for key in integer_values):
        raise ValueError("run.oracle_version and run.rounds must be integers")
    if run["oracle_version"] <= 0:
        raise ValueError("run.oracle_version must be a positive integer")
    probability = run["mutation_probability"]
    if not isinstance(probability, (int, float)) or isinstance(probability, bool) or not 0 <= probability <= 1:
        raise ValueError("run.mutation_probability must be between zero and one")
    splits = (run["evolution_split"], run["final_split"])
    if any(split not in {"train", "validation", "test", "all"} for split in splits):
        raise ValueError("run.evolution_split and run.final_split must be train, validation, test, or all")
    output_dir = run["output_dir"]
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ValueError("run.output_dir must be a non-empty string")
    return EvolutionRunnerConfig(
        reflection_llm,
        coding_llm,
        Path(run["dataset"]),
        Path(run["profiling_database"]),
        Path(run["profiling_schema"]),
        root_configs,
        Path(run["oracle_source"]),
        run["oracle_version"],
        run["rounds"],
        float(probability),
        run["evolution_split"],
        run["final_split"],
        Path(output_dir),
    )


def _parse_llm(value: Any, environ: Mapping[str, str], name: str) -> EvolutionAgentLlmConfig:
    if not isinstance(value, dict) or set(value) != {"base_url", "token", "model"}:
        raise ValueError(f"{name} requires base_url, token, and model")
    if not all(isinstance(value[key], str) for key in value):
        raise ValueError(f"{name} values must be strings")
    token = _resolve_token(value["token"], environ)
    return EvolutionAgentLlmConfig(value["base_url"].rstrip("/"), token, value["model"])


def _parse_root(value: Any, index: int) -> EvolutionRootCandidateConfig:
    if not isinstance(value, dict) or set(value) != {"scheduler_version", "source"}:
        raise ValueError(f"run.root_candidates[{index}] requires scheduler_version and source")
    if not isinstance(value["scheduler_version"], int) or isinstance(value["scheduler_version"], bool):
        raise ValueError(f"run.root_candidates[{index}].scheduler_version must be an integer")
    if not isinstance(value["source"], str) or not value["source"].strip():
        raise ValueError(f"run.root_candidates[{index}].source must be a non-empty string")
    return EvolutionRootCandidateConfig(value["scheduler_version"], Path(value["source"]))


def _resolve_token(value: str, environ: Mapping[str, str]) -> str:
    match = _ENVIRONMENT_TOKEN.fullmatch(value)
    if match is None:
        return value
    try:
        return environ[match.group(1)]
    except KeyError as exc:
        raise ValueError(f"Evolution Agent LLM token environment variable is not set: {match.group(1)}") from exc
