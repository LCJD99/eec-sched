# eec-sched

End–Edge–Cloud Scheduler Candidate evolution for planned Agentic Workflow DAGs.

当前 evaluation 实验的实际数据流、目标三模块接口、缺口与 baseline
位置见 [evaluation 实验主链路](docs/evaluation-pipeline.md)。六种对照方法的
开发过程放在 [`experiments/baseline_methods/`](experiments/baseline_methods/)，
最终可评估的 Scheduler 源码放在 [`schedulers/baselines/`](schedulers/baselines/)。

The application is composed with Hydra. Tokens stay in the environment; the
default local model adapters read `EEC_SCHED_REFLECTION_TOKEN` and
`EEC_SCHED_CODING_TOKEN`.

```bash
uv run eec-sched
```

The composition root is the selected experiment under `experiments/`. Shared
Hydra options live in `configs/`; the default is
`experiments/04_baseline/config.yaml`. Every execution creates one fresh
record under `runs/<experiment>_<YYYYMMDDTHHmmss>/` containing the resolved
`config.yaml`, event trace, result, and any run-local persistent memory.
Credential values from environment-backed settings are redacted in the saved
record. A run directory is never reused.

Select an explicit experiment or use an ablation without changing Python code:

```bash
uv run eec-sched --config-name experiments/05_ablation_no_memory/config
uv run eec-sched --config-name experiments/06_ablation_no_diagnosis/config
uv run --group agentic-reflection eec-sched --config-name experiments/08_self_evolving_diagnosis/config
```

The self-evolving diagnosis experiment analyzes three Evolution Traces by
default (`diagnosis.trace_limit=3`). Its last five diagnosis trajectories and
generated composite tools persist under
`outputs/diagnosis-tools/<experiment_name>/` for reuse by later runs.

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

To independently debug one self-evolving bottleneck analysis, provide exactly
one Scheduler Candidate source.  The command evaluates that Candidate first,
then runs only `SelfEvolvingDiagnosis` (it does not start the coding/evolution
loop):

```bash
uv run --group agentic-reflection eec-sched-analyze-bottleneck \
  --scheduler-source schedulers/naive_fastest.py \
  --model-config configs/model/local_qwen.yaml
```

By default, exactly the first three DAGs in the selected split are evaluated
(`--dag-limit` changes this positive limit), and the same three scored DAGs are
passed to the diagnosis agent. One fresh UTC-timestamped JSONL event log is
written under `outputs/bottleneck-analysis/` (override with `--output-dir`).
Every flushed line contains both canonical `event_type` and compatibility
`type` fields, and records trusted evaluation events plus the diagnosis agent
loop (`session_start`, `round_start`, `tool_call`, `tool_result`,
`tool_evaluation`, and matching end events). Every model round also records the
actual JSON-safe request and response as `llm_request` and `llm_response`.
Scheduler source code and model tokens are never recorded.
Persistent diagnosis history and generated tools default to
`outputs/diagnosis-tools/bottleneck-analysis/` and can be changed with
`--state-dir`.  `--model-config` loads the LLM fields `base_url`, `token`,
`model`, and `timeout_seconds` from YAML; explicit model CLI options override
values from the file.  Without a config file, the model endpoint defaults to
`http://localhost:9888/v1`, `qwen3.8-27b`, and reads its token from
`EEC_SCHED_REFLECTION_TOKEN`.

The source packages are grouped by implementation depth: profiling, trusted
evaluation, diagnosis, and evolution are directories; small replaceable
adapters remain colocated in files such as `evolution/strategies.py` and
`evolution/memory.py`.
