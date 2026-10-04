#!/usr/bin/env bash
# Preliminary MNMS multi-request evaluation for the selected baselines.
# The rates and synthetic profiling snapshot below are for a dry run of the
# experiment pipeline; main-exp.md's five rates and real prices are not frozen.
# All baselines replay the same arrivals for each rate and seed.
# Set a different SCENARIO_ROOT when changing DATASET.

set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT_DIR"

DATASET=${DATASET:-experiments/09_offline_tool_planning/runs/tool-planner-20261002T092747Z/dags.jsonl}
PLANNER_LLM=${PLANNER_LLM:-qwen3.8-27b}
PROFILING_DATABASE=${PROFILING_DATABASE:-profiles/eval-database.json}
PROFILING_SCHEMA=${PROFILING_SCHEMA:-docs/schemas/profiling-database.schema.json}
SCENARIO_ROOT=${SCENARIO_ROOT:-experiments/10_workload_scenarios/runs/mnms-preliminary}
OUTPUT_ROOT=${OUTPUT_ROOT:-experiments/11_scheduler_evaluation/runs/mnms-preliminary}
OBSERVATION_WINDOW_MS=${OBSERVATION_WINDOW_MS:-60000}
SCHEDULER_VERSION=${SCHEDULER_VERSION:-1}

# Space-separated overrides, e.g. BASELINES="sdts pbtla" REQUEST_RATES="2 4" TEST_SEEDS="7 17" ./eval.sh
read -r -a baselines <<<"${BASELINES:-sdts pbtla qphh murakkab}"
read -r -a rates <<<"${REQUEST_RATES:-5 6 7}"
read -r -a seeds <<<"${TEST_SEEDS:-7 17 27}"

for rate in "${rates[@]}"; do
  for seed in "${seeds[@]}"; do
    scenario_dir="$SCENARIO_ROOT/window-$OBSERVATION_WINDOW_MS/rate-$rate/seed-$seed"
    arrivals="$scenario_dir/arrivals.json"

    if [[ ! -f "$arrivals" ]]; then
      uv run python experiments/10_workload_scenarios/run.py \
        "dag_path=$DATASET" \
        "request_rate_per_second=$rate" \
        "observation_window_ms=$OBSERVATION_WINDOW_MS" \
        "seed=$seed" \
        "output_dir=$scenario_dir"
    fi

    for baseline in "${baselines[@]}"; do
      case "$baseline" in
        sdts) method=SDTS ;;
        pbtla) method=PBTLA ;;
        qphh) method=QPHH ;;
        murakkab) method=Murakkab ;;
        *)
          printf 'Unknown baseline "%s" (choose sdts, pbtla, qphh, murakkab)\n' "$baseline" >&2
          exit 2
          ;;
      esac

      printf '%s MNMS: rate=%s requests/s, seed=%s, arrivals=%s\n' "$method" "$rate" "$seed" "$arrivals"
      uv run python scripts/eval.py --config-name multi \
        "dataset=$DATASET" \
        "arrivals=$arrivals" \
        "scheduler_source=schedulers/baselines/$baseline.py" \
        "scheduler_version=$SCHEDULER_VERSION" \
        "profiling_database=$PROFILING_DATABASE" \
        "profiling_schema=$PROFILING_SCHEMA" \
        "output_root=$OUTPUT_ROOT/$baseline/rate-$rate/seed-$seed" \
        "+method=$method" \
        "+workload=MNMS" \
        "+planner_llm=$PLANNER_LLM"
    done
  done
done
