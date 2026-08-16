# eec-sched

End-Edge-Cloud Tool Use Scheduler Heuristic Code Generation

To construct the model-backed Evolution Agents, copy
`config/evolution-agent.example.yaml` to a private location, set its token
environment variable, then run:

```bash
uv run python scripts/run_evolution.py --config /private/evolution-agent.yaml
```

Add `--check` to send a source-free Reflection request to the configured
OpenAI-compatible endpoint. The script does not start an Evolution Loop;
trusted traces, root Scheduler Candidates, and the Oracle are separate inputs.
