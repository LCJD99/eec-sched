# Workload scenarios

This Hydra entry point prepares one fixed request intensity for later
multi-request evaluation.  It reads every successful record from an offline
Planner `dags.jsonl`, generates exponential inter-arrival gaps with the
configured Poisson rate, and cycles through the Planner records in their input
order.  Each admitted request gets a unique ID.  Arrival times are simulated
milliseconds; the command never sleeps and does not call a Scheduler.

For a mixed Video QA and Math QA workload, provide two separate offline DAG
files. The generator draws two independent Poisson streams at half the
configured total rate, merges them by arrival time, and stores the request
type in every mixed arrival record:

```bash
uv run python experiments/10_workload_scenarios/run.py \
  video_qa_dag_path=data/video-qa-dags.jsonl \
  math_qa_dag_path=data/math-qa-dags.jsonl \
  request_rate_per_second=2 \
  observation_window_ms=60000 \
  seed=7
```

Run it with:

```bash
uv run python experiments/10_workload_scenarios/run.py \
  dag_path=experiments/09_offline_tool_planning/runs/tool-planner-20261002T085931Z/dags.jsonl \
  request_rate_per_second=2 \
  observation_window_ms=60000 \
  seed=7
```

Each invocation creates a new directory under
`experiments/10_workload_scenarios/runs/`, writes the resolved `config.yaml`,
and writes `arrivals.json`. The manifest contains the realized arrival times,
request IDs, template IDs, and, for mixed workloads, request types, together
with the rate, window, and seed. It does not copy or hash the Planner file. A
configured existing output directory is rejected.

The later evaluator can replay the same request sequence for every Scheduler:

```python
from eec_sched.workload import load_workload_scenario

arrivals = load_workload_scenario(
    "experiments/09_offline_tool_planning/runs/tool-planner-20261002T085931Z/dags.jsonl",
    "experiments/10_workload_scenarios/runs/workload-.../arrivals.json",
)
```

For a mixed manifest, pass both Planner files keyed by the persisted request
types:

```python
arrivals = load_workload_scenario(
    {
        "video_qa": "data/video-qa-dags.jsonl",
        "math_qa": "data/math-qa-dags.jsonl",
    },
    "experiments/10_workload_scenarios/runs/workload-.../arrivals.json",
)
```

`arrivals` is an ordered tuple of `(arrival_time_ms, EvaluationTrace)` values.
The tuple can be empty when a low-rate, short-window Poisson draw produces no
arrivals.  Different request rates are separate scenarios and should be
evaluated independently.
