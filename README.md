# eec-sched

End–Edge–Cloud Scheduler Candidate evolution for planned Agentic Workflow DAGs.

The application is composed with Hydra. Tokens stay in the environment; the
default local model adapters read `EEC_SCHED_REFLECTION_TOKEN` and
`EEC_SCHED_CODING_TOKEN`.

```bash
uv run eec-sched
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

The source packages are grouped by implementation depth: profiling, trusted
evaluation, diagnosis, and evolution are directories; small replaceable
adapters remain colocated in files such as `evolution/strategies.py` and
`evolution/memory.py`.
