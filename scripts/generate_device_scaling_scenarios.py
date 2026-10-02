"""Generate synthetic 10/20/50-device compute-capacity scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eec_sched.profiling.device_scaling import DEFAULT_GPU_MODELS, GPU_ARCHETYPES, build_device_scaling_scenario, calibration_from_snapshot
from eec_sched.profiling.snapshot import load_profiling_database


ROOT = Path(__file__).parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=ROOT / "docs/examples/profiling-database.fake.json")
    parser.add_argument("--schema", type=Path, default=ROOT / "docs/schemas/profiling-database.schema.json")
    parser.add_argument("--gpu", action="append", metavar="DEVICE_ID=RTX_MODEL",
                        help="override the default device=rtx5060 edge=rtx3090 cloud=rtx4090 mapping")
    parser.add_argument("--configurations-per-tool", type=int, required=True)
    parser.add_argument("--device-counts", nargs="+", type=int, default=[10, 20, 50])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/device-scaling-scenarios")
    parser.add_argument("--quality-gain", type=float, default=0.10)
    parser.add_argument("--work-gain", type=float, default=0.50)
    parser.add_argument("--work-exponent", type=float, default=1.4)
    parser.add_argument("--memory-gain", type=float, default=0.25)
    parser.add_argument("--network-distance-decay", type=float, default=0.1)
    parser.add_argument("--network-seed", type=int, default=0)
    args = parser.parse_args()
    gpu_models = dict(DEFAULT_GPU_MODELS)
    for entry in args.gpu or []:
        try:
            device_id, value = entry.split("=", 1)
            if device_id not in gpu_models or value not in GPU_ARCHETYPES:
                raise ValueError
            gpu_models[device_id] = value
        except ValueError:
            parser.error(f"invalid --gpu {entry!r}; use DEVICE_ID=rtx5060|rtx3090|rtx4090")
    snapshot = load_profiling_database(args.snapshot, args.schema)
    calibration = calibration_from_snapshot(snapshot.data, gpu_models)
    scenarios = [
        build_device_scaling_scenario(
            calibration,
            device_count=count,
            configurations_per_tool=args.configurations_per_tool,
            quality_gain=args.quality_gain,
            work_gain=args.work_gain,
            work_exponent=args.work_exponent,
            memory_gain=args.memory_gain,
            network_distance_decay=args.network_distance_decay,
            network_seed=args.network_seed,
        )
        for count in args.device_counts
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for scenario in scenarios:
        count = len(scenario["devices"])
        path = args.output_dir / f"devices-{count}-configurations-{args.configurations_per_tool}.json"
        path.write_text(json.dumps(scenario, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{path} {scenario['scenario_digest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
