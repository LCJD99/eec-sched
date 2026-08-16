"""Configuration for the LLMs used by the Evolution Agents."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Mapping

import yaml

_ENVIRONMENT_TOKEN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


@dataclass(frozen=True)
class EvolutionAgentLlmConfig:
    """One OpenAI-compatible endpoint shared by Reflection and Coding Agents."""

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
    llm = document["llm"]
    if not isinstance(llm, dict) or set(llm) != {"base_url", "token", "model"}:
        raise ValueError("Evolution Agent llm config requires base_url, token, and model")
    if not all(isinstance(llm[key], str) for key in llm):
        raise ValueError("Evolution Agent LLM values must be strings")
    token = _resolve_token(llm["token"], environ or os.environ)
    return EvolutionAgentLlmConfig(llm["base_url"].rstrip("/"), token, llm["model"])


def _resolve_token(value: str, environ: Mapping[str, str]) -> str:
    match = _ENVIRONMENT_TOKEN.fullmatch(value)
    if match is None:
        return value
    try:
        return environ[match.group(1)]
    except KeyError as exc:
        raise ValueError(f"Evolution Agent LLM token environment variable is not set: {match.group(1)}") from exc
