from __future__ import annotations

from pathlib import Path

import pytest

from eec_sched import load_evolution_agent_llm_config, load_evolution_runner_config


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


def test_loads_complete_runner_config_with_independent_agent_llms(tmp_path: Path) -> None:
    config_path = tmp_path / "evolution-agent.yaml"
    config_path.write_text(
        """
llm:
  reflection:
    base_url: https://reflection.test/v1/
    token: ${REFLECTION_TOKEN}
    model: reflection-model
  coding:
    base_url: https://coding.test/v1
    token: ${CODING_TOKEN}
    model: coding-model
run:
  dataset: data/dataset.jsonl
  profiling_database: data/profile.json
  profiling_schema: data/profile.schema.json
  root_candidates:
    - scheduler_version: 1
      source: schedulers/one.py
    - scheduler_version: 2
      source: schedulers/two.py
    - scheduler_version: 3
      source: schedulers/three.py
  oracle_source: schedulers/oracle.py
  oracle_version: 99
  rounds: 4
  mutation_probability: 0.75
  evolution_split: train
  final_split: test
  output_dir: outputs
""",
        encoding="utf-8",
    )

    config = load_evolution_runner_config(
        config_path,
        {"REFLECTION_TOKEN": "reflection-secret", "CODING_TOKEN": "coding-secret"},
    )

    assert config.reflection_llm.base_url == "https://reflection.test/v1"
    assert config.reflection_llm.token == "reflection-secret"
    assert config.coding_llm.base_url == "https://coding.test/v1"
    assert config.coding_llm.token == "coding-secret"
    assert tuple(item.scheduler_version for item in config.root_candidates) == (1, 2, 3)
    assert config.rounds == 4
    assert config.mutation_probability == 0.75
    assert config.output_dir == Path("outputs")


def test_rejects_non_three_root_candidates(tmp_path: Path) -> None:
    config_path = tmp_path / "evolution-agent.yaml"
    config_path.write_text(
        """
llm:
  reflection: {base_url: https://example.test, token: token, model: reflection}
  coding: {base_url: https://example.test, token: token, model: coding}
run:
  dataset: data.jsonl
  profiling_database: profile.json
  profiling_schema: schema.json
  root_candidates: []
  oracle_source: oracle.py
  oracle_version: 0
  rounds: 1
  mutation_probability: 0.5
  evolution_split: train
  final_split: test
  output_dir: outputs
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly three"):
        load_evolution_runner_config(config_path, {})


def test_rejects_non_positive_oracle_version(tmp_path: Path) -> None:
    config_path = tmp_path / "evolution-agent.yaml"
    config_path.write_text(
        """
llm:
  reflection: {base_url: https://example.test, token: token, model: reflection}
  coding: {base_url: https://example.test, token: token, model: coding}
run:
  dataset: data.jsonl
  profiling_database: profile.json
  profiling_schema: schema.json
  root_candidates:
    - {scheduler_version: 1, source: one.py}
    - {scheduler_version: 2, source: two.py}
    - {scheduler_version: 3, source: three.py}
  oracle_source: oracle.py
  oracle_version: 0
  rounds: 1
  mutation_probability: 0.5
  evolution_split: train
  final_split: test
  output_dir: outputs
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="oracle_version must be a positive integer"):
        load_evolution_runner_config(config_path, {})
