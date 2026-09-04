"""Render the three measured BO objectives against the joint Configuration grid."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Point:
    imgsz: int
    conf: float
    box_ap: float
    warm_latency_ms: float
    peak_gpu_memory_mib: float
    selected: bool


def configuration_key(item: dict[str, Any]) -> tuple[int, float]:
    configuration = item["configuration"]
    return int(configuration["imgsz"]), float(configuration["conf"])


def load_points(path: Path) -> list[Point]:
    """Load successful observations and mark the final selected cover set."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = {
        configuration_key(item) for item in payload.get("selected_configurations", [])
    }
    points = []
    for item in payload.get("observations", []):
        if item.get("status") != "success":
            continue
        if any(item.get(field) is None for field, _, _ in METRICS):
            continue
        imgsz, conf = configuration_key(item)
        points.append(
            Point(
                imgsz=imgsz,
                conf=conf,
                box_ap=float(item["box_ap"]),
                warm_latency_ms=float(item["warm_latency_ms"]),
                peak_gpu_memory_mib=float(item["peak_gpu_memory_mib"]),
                selected=(imgsz, conf) in selected,
            )
        )
    if not points:
        raise ValueError("Campaign contains no successful observations to visualize")
    return points


METRICS = (
    ("box_ap", "box AP@[.50:.95]", "higher is better"),
    ("warm_latency_ms", "warm latency (ms)", "single warm observation"),
    ("peak_gpu_memory_mib", "peak GPU memory (MiB)", "lower is better"),
)


def render(points: list[Point], output: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_points = [point for point in points if not point.selected]
    selected_points = [point for point in points if point.selected]
    figure = plt.figure(figsize=(18, 6), constrained_layout=True)
    for index, (field, label, direction) in enumerate(METRICS, start=1):
        axis = figure.add_subplot(1, 3, index, projection="3d")
        axis.scatter(
            [point.imgsz for point in all_points],
            [point.conf for point in all_points],
            [getattr(point, field) for point in all_points],
            color="#457b9d",
            s=38,
            alpha=0.78,
            label="measured",
        )
        axis.scatter(
            [point.imgsz for point in selected_points],
            [point.conf for point in selected_points],
            [getattr(point, field) for point in selected_points],
            color="#e63946",
            marker="*",
            s=180,
            edgecolors="#111111",
            linewidths=0.6,
            label="selected Configuration",
        )
        axis.set_title(f"{label}\n({direction})")
        axis.set_xlabel("imgsz")
        axis.set_ylabel("conf")
        axis.set_zlabel(label)
        axis.view_init(elev=24, azim=-58)
        axis.legend(loc="upper left")
    figure.suptitle(title, fontsize=16)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_json", type=Path, help="Final campaign.json file")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PNG (default: <campaign directory>/configuration-effects.png)",
    )
    args = parser.parse_args()
    output = args.output or args.campaign_json.with_name("configuration-effects.png")
    points = load_points(args.campaign_json)
    render(points, output, f"Configuration effects — {args.campaign_json.parent.name}")
    print(f"Wrote {output} from {len(points)} successful observations")


if __name__ == "__main__":
    main()
