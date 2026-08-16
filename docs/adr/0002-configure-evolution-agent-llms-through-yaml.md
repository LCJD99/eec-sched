# Configure Evolution Agent LLMs through YAML

Status: proposed

The Reflection Agent and Coding Agent will use one OpenAI-compatible LLM configuration loaded from a YAML file. The file contains `llm.base_url`, `llm.token`, and `llm.model`; `token` may be a `${VARIABLE_NAME}` reference resolved from the process environment, so committed example configuration contains no credential. A future `run_evolution.py --config <path>` entry point will load this configuration and construct the two agents, while `EvolutionLoop` continues to receive them as injected dependencies and trusted evaluation continues to own validation, timing, simulation, and scoring.

## Consequences

- One model and endpoint serve both agents initially; configuring them separately is deferred until there is an explicit need.
- The YAML file configures LLM connectivity only. It does not contain Profiling Database evidence, scheduler source code, Oracle data, or trusted-evaluation settings.
- The existing deterministic `run.sh` remains an evaluation demo and does not invoke an LLM or evolution graph.
