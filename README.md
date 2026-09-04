# eec-sched

End–Edge–Cloud Scheduler Candidate evolution for planned Agentic Workflow DAGs.

The application is composed with Hydra. Tokens stay in the environment; the
default local model adapters read `EEC_SCHED_REFLECTION_TOKEN` and
`EEC_SCHED_CODING_TOKEN`.

```bash
uv run eec-sched
```

The composition root is the selected experiment under `experiments/`. Shared
Hydra options live in `configs/`; the default is
`experiments/001_baseline/config.yaml`. Every execution creates one fresh
record under `runs/<experiment>_<YYYYMMDDTHHmmss>/` containing the resolved
`config.yaml`, event trace, result, and any run-local persistent memory.
Credential values from environment-backed settings are redacted in the saved
record. A run directory is never reused.

Select an explicit experiment or use an ablation without changing Python code:

```bash
uv run eec-sched --config-name experiments/002_ablation_no_memory/config
uv run eec-sched --config-name experiments/003_ablation_no_diagnosis/config
```

Every replaceable experiment seam can be overridden without editing source:

```bash
uv run eec-sched diagnosis=empty
uv run eec-sched memory=jsonl
uv run eec-sched search=legacy
uv run eec-sched experiment=ablation_no_memory
```

Inspect the fully composed configuration before a run:

```bash
uv run eec-sched --cfg job --resolve
```

For a deterministic local smoke run that does not call a model endpoint:

```bash
uv run eec-sched diagnosis=empty run.rounds=0
```

The source packages are grouped by implementation depth: profiling, trusted
evaluation, diagnosis, and evolution are directories; small replaceable
adapters remain colocated in files such as `evolution/strategies.py` and
`evolution/memory.py`.
