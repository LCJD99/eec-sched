# eec-sched

End-Edge-Cloud Tool Use Scheduler Heuristic Code Generation

To run the complete model-backed evolution loop, copy
`config/evolution-agent.example.yaml` to a private location, copy
`.env.example` to `.env` and set both tokens, then adjust the paths and
parameters in the YAML:

```bash
uv run python scripts/run_evolution.py --config /private/evolution-agent.yaml
```

The YAML contains independent Reflection/Coding LLM settings and all loop
parameters. The default evolution split is `train` and the default final split is `test`.
The runner automatically loads `.env` from the current directory; existing
environment variables take precedence.
Use `--check` with only `--config` to send a source-free Reflection request to
the configured OpenAI-compatible endpoint without running the loop.
