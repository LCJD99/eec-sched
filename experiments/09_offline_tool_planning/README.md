```python
uv run python experiments/09_offline_tool_planning/run.py
```

Each invocation creates a new directory under `runs/` with `dags.jsonl`,
`failures.jsonl`, `manifest.json`, and `trace.jsonl`. The trace contains one
JSONL record per model response, keyed by input `id`, with the raw text in
`llm_output`; inspect it alongside `failures.jsonl` when parsing or validation
fails.
