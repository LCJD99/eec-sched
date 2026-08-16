from __future__ import annotations

from pathlib import Path

import pytest

from eec_sched import load_evolution_agent_llm_config


def test_loads_evolution_agent_llm_config_and_resolves_token_from_environment(tmp_path: Path) -> None:
    config_path = tmp_path / "evolution-agent.yaml"
    config_path.write_text(
        "llm:\n  base_url: https://example.test/v1/\n  token: ${EVOLUTION_TOKEN}\n  model: test-model\n",
        encoding="utf-8",
    )

    config = load_evolution_agent_llm_config(config_path, {"EVOLUTION_TOKEN": "secret"})

    assert config.base_url == "https://example.test/v1"
    assert config.token == "secret"
    assert config.model == "test-model"


def test_rejects_missing_token_environment_variable(tmp_path: Path) -> None:
    config_path = tmp_path / "evolution-agent.yaml"
    config_path.write_text(
        "llm:\n  base_url: https://example.test/v1\n  token: ${EVOLUTION_TOKEN}\n  model: test-model\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not set"):
        load_evolution_agent_llm_config(config_path, {})
