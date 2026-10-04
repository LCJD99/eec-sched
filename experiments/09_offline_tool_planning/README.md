```python
uv run python experiments/09_offline_tool_planning/run.py
```

Requests run concurrently in batches of 8 by default. Set `batch_size=4` on
the command line to use a different batch size.

Each invocation creates a new directory under `runs/` with `dags.jsonl`,
`failures.jsonl`, `manifest.json`, and `trace.jsonl`. The trace contains one
JSONL record per model response, keyed by input `id`, with the raw text in
`llm_output`; inspect it alongside `failures.jsonl` when parsing or validation
fails.
