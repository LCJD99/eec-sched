"""Small, dependency-light server for scheduler evaluation results.

The evaluator writes one directory per run.  This module discovers completed
runs, computes request-level aggregates (including p95), and serves those
aggregates to the static browser client.  It intentionally reads the raw
``results.jsonl`` records instead of trusting summary percentiles: a p95 over
multiple seeds must be computed from the merged request values.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable
from urllib.parse import parse_qs, urlsplit

import yaml


PROTOTYPE_DIR = Path(__file__).resolve().parent
REPO_DIR = PROTOTYPE_DIR.parent.parent
DEFAULT_RUNS_ROOT = REPO_DIR / "experiments" / "11_scheduler_evaluation" / "runs" / "mnms-preliminary"
STATIC_FILES = {
    "/": PROTOTYPE_DIR / "index.html",
    "/index.html": PROTOTYPE_DIR / "index.html",
    "/app.js": PROTOTYPE_DIR / "app.js",
    "/styles.css": PROTOTYPE_DIR / "styles.css",
}
RUNTIME_CONFIG_KEYS = {
    "seed",
    "resolved_seed",
    "manifest_request_rate_per_second",
    "actual_arrivals_per_second",
    "actual_arrival_count",
    "arrivals",
    "output_root",
    "run_dir",
    "scheduler_source",
    "method",
    "request_rate_per_second",
}
RATE_RE = re.compile(r"^rate-(.+)$")
SEED_RE = re.compile(r"^seed-(.+)$")
EVAL_RE = re.compile(r"^eval-(.+)$")


def _json_number(value: Any) -> float | int | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def percentile(values: Iterable[float], fraction: float = 0.95) -> float | None:
    """Return an interpolated percentile over request-level values.

    The ``(n - 1) * p`` convention is deterministic and agrees with the
    common linear interpolation used by numpy/pandas.  Empty data is kept as
    ``None`` so the UI can distinguish missing data from a real zero.
    """

    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _mean(values: list[float]) -> float | None:
    return fmean(values) if values else None


def _timestamp_from_name(path: Path) -> tuple[str, float]:
    match = EVAL_RE.match(path.name)
    value = match.group(1) if match else ""
    parsed: datetime | None = None
    if value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            parsed = None
    if parsed is None:
        mtime = path.stat().st_mtime
        parsed = datetime.fromtimestamp(mtime, timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"), parsed.timestamp()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    return value if isinstance(value, dict) else {}


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    return value


def normalized_config(config: dict[str, Any]) -> dict[str, Any]:
    """Remove run identity fields while preserving data/scoring identity."""

    return _canonical({key: value for key, value in config.items() if key not in RUNTIME_CONFIG_KEYS})


def config_identity(config: dict[str, Any]) -> str:
    return json.dumps(normalized_config(config), ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def config_label(config: dict[str, Any]) -> str:
    """Make a short, stable label that still exposes meaningful differences."""

    parts: list[str] = []
    workload = config.get("workload")
    if workload:
        parts.append(str(workload))
    milliseconds = None
    if config.get("observation_window_ms") is not None:
        milliseconds = float(config["observation_window_ms"])
        window = f"{milliseconds / 1000:g}s" if milliseconds % 1000 == 0 else f"{milliseconds:g}ms"
        parts.append(f"窗口 {window}")
    snapshot = config.get("snapshot_digest") or config.get("profiling_database")
    if snapshot:
        snapshot_name = Path(str(snapshot)).name
        if snapshot_name.startswith("sha256:"):
            snapshot_name = snapshot_name[:20] + "…"
        elif len(snapshot_name) > 30:
            snapshot_name = snapshot_name[:27] + "…"
        parts.append(f"快照 {snapshot_name}")
    database = config.get("profiling_database")
    if database and snapshot and Path(str(database)).name != Path(str(snapshot)).name:
        parts.append(f"库 {Path(str(database)).stem}")
    scoring = config.get("scoring_context")
    if isinstance(scoring, dict):
        weights = [scoring.get(name) for name in ("accuracy_weight", "latency_weight", "resource_weight")]
        if all(value is not None for value in weights):
            parts.append("权重 " + "/".join(f"{float(value):g}" for value in weights))
        details = [
            ("β", scoring.get("beta")),
            ("L", scoring.get("latency_scale_ms")),
            ("R", scoring.get("resource_scale_mib")),
        ]
        details = [f"{label}{float(value):g}" for label, value in details if value is not None]
        if details:
            parts.append("尺度 " + "/".join(details))
    if not parts:
        retained = normalized_config(config)
        text = json.dumps(retained, ensure_ascii=False, sort_keys=True)
        parts.append(text[:100] + ("…" if len(text) > 100 else ""))
    return " · ".join(parts)


def _resource(record: dict[str, Any]) -> float | None:
    timeline = record.get("node_timeline")
    if not isinstance(timeline, list):
        return None
    total = 0.0
    for node in timeline:
        if not isinstance(node, dict) or _json_number(node.get("gpu_memory_mib")) is None:
            return None
        total += float(node["gpu_memory_mib"])
    return total


def _is_completed(record: dict[str, Any]) -> bool:
    return str(record.get("status", "")).lower() == "completed"


def _read_results(path: Path) -> tuple[dict[str, list[float]], int, int, int]:
    values: dict[str, list[float]] = {"quality": [], "resource": [], "latency": []}
    request_count = completed_count = failed_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                continue
            request_count += 1
            if not _is_completed(record):
                failed_count += 1
                continue
            completed_count += 1
            quality = _json_number(record.get("quality"))
            latency = _json_number(record.get("latency_ms"))
            resource = _resource(record)
            if quality is not None:
                values["quality"].append(float(quality))
            if latency is not None:
                values["latency"].append(float(latency) / 1000.0)
            if resource is not None:
                values["resource"].append(resource)
    return values, request_count, completed_count, failed_count


def _metric(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": _mean(values),
        "p95": percentile(values),
    }


def _run_path_metadata(root: Path, directory: Path, config: dict[str, Any]) -> dict[str, Any]:
    relative = directory.relative_to(root)
    parts = relative.parts
    rate_index = next((index for index, part in enumerate(parts) if RATE_RE.match(part)), None)
    seed_index = next((index for index, part in enumerate(parts) if SEED_RE.match(part)), None)
    baseline = parts[0] if parts else str(config.get("method", "unknown"))
    rate_text = RATE_RE.match(parts[rate_index]).group(1) if rate_index is not None else str(config.get("manifest_request_rate_per_second", ""))
    seed_text = SEED_RE.match(parts[seed_index]).group(1) if seed_index is not None else str(config.get("resolved_seed", config.get("seed", "")))
    try:
        rate: float | int = float(rate_text)
        rate = int(rate) if rate.is_integer() else rate
    except ValueError:
        rate = rate_text
    try:
        seed: int | str = int(seed_text)
    except ValueError:
        seed = seed_text
    return {"scheduler": baseline, "rate": rate, "seed": seed, "path": str(relative)}


def discover_runs(root: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Discover completed runs and report directories that are still incomplete."""

    candidates = sorted(root.rglob("config.yaml")) if root.is_dir() else []
    runs: list[dict[str, Any]] = []
    stats = {"candidate_count": len(candidates), "completed_count": 0, "incomplete_count": 0, "error_count": 0}
    for config_path in candidates:
        directory = config_path.parent
        result_path = directory / "results.jsonl"
        summary_path = directory / "summary.json"
        if not result_path.is_file() or not summary_path.is_file():
            stats["incomplete_count"] += 1
            continue
        try:
            config = _load_yaml(config_path)
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            values, request_count, completed_count, failed_count = _read_results(result_path)
            timestamp, timestamp_value = _timestamp_from_name(directory)
        except (OSError, ValueError, TypeError, json.JSONDecodeError, UnicodeError, RuntimeError):
            stats["error_count"] += 1
            continue
        identity_config = dict(config)
        # These values live in summary.json for older evaluator versions but
        # change the experiment population and therefore belong to config
        # identity.  Keep them visible in the detail shown by the UI.
        for key in ("snapshot_digest", "observation_window_ms"):
            if summary.get(key) is not None:
                identity_config[key] = summary[key]
        path_meta = _run_path_metadata(root, directory, identity_config)
        identity = config_identity(identity_config)
        run = {
            "id": str(directory.relative_to(root)),
            "scheduler": path_meta["scheduler"],
            "rate": path_meta["rate"],
            "seed": path_meta["seed"],
            "timestamp": timestamp,
            "timestamp_value": timestamp_value,
            "config_id": identity,
            "config_label": config_label(identity_config),
            "config": normalized_config(identity_config),
            "request_count": request_count,
            "completed_count": completed_count,
            "failed_count": failed_count,
            "metrics": {name: _metric(items) for name, items in values.items()},
            "_values": values,
            "summary": {
                "snapshot_digest": summary.get("snapshot_digest"),
                "actual_arrivals_per_second": summary.get("actual_arrivals_per_second"),
                "backlog_at_window_end": summary.get("backlog_at_window_end"),
            },
        }
        runs.append(run)
        stats["completed_count"] += 1
    return runs, stats


def select_runs(runs: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    if mode == "history":
        return list(runs)
    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    for run in runs:
        key = (run["scheduler"], run["config_id"], str(run["rate"]), str(run["seed"]))
        previous = latest.get(key)
        if previous is None or (run["timestamp_value"], run["id"]) > (previous["timestamp_value"], previous["id"]):
            latest[key] = run
    return sorted(latest.values(), key=lambda item: (str(item["scheduler"]), str(item["config_label"]), str(item["rate"]), str(item["seed"])))


def _matches(value: Any, selected: set[str]) -> bool:
    return not selected or str(value) in selected


def _aggregate_group(group: list[dict[str, Any]]) -> dict[str, Any]:
    values = {name: [value for run in group for value in run["_values"][name]] for name in ("quality", "resource", "latency")}
    metric_values = {name: _metric(items) for name, items in values.items()}
    return {
        "scheduler": group[0]["scheduler"],
        "rate": group[0]["rate"],
        "config_id": group[0]["config_id"],
        "config": group[0]["config_label"],
        "config_detail": group[0]["config"],
        "seeds": sorted({str(run["seed"]) for run in group}),
        "run_count": len(group),
        "request_count": sum(run["request_count"] for run in group),
        "completed_count": sum(run["completed_count"] for run in group),
        "failed_count": sum(run["failed_count"] for run in group),
        "quality_count": metric_values["quality"]["count"],
        "quality_mean": metric_values["quality"]["mean"],
        "quality_p95": metric_values["quality"]["p95"],
        "resource_count": metric_values["resource"]["count"],
        "resource_mean_mib": metric_values["resource"]["mean"],
        "resource_p95_mib": metric_values["resource"]["p95"],
        "latency_count": metric_values["latency"]["count"],
        "latency_mean_s": metric_values["latency"]["mean"],
        "latency_p95_s": metric_values["latency"]["p95"],
        "latest_timestamp": max(run["timestamp"] for run in group),
        "run_ids": [run["id"] for run in group],
    }


def _public_run(run: dict[str, Any]) -> dict[str, Any]:
    value = {key: item for key, item in run.items() if key not in {"_values", "timestamp_value", "config"}}
    value["config_detail"] = run["config"]
    return value


def make_payload(
    runs: list[dict[str, Any]],
    stats: dict[str, int],
    mode: str = "latest",
    scheduler: set[str] | None = None,
    rates: set[str] | None = None,
    configs: set[str] | None = None,
    seeds: set[str] | None = None,
) -> dict[str, Any]:
    selected_runs = select_runs(runs, mode)
    scheduler = scheduler or set()
    rates = rates or set()
    configs = configs or set()
    seeds = seeds or set()
    selected_runs = [
        run for run in selected_runs
        if _matches(run["scheduler"], scheduler)
        and _matches(run["rate"], rates)
        and _matches(run["config_id"], configs)
        and _matches(run["seed"], seeds)
    ]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for run in selected_runs:
        groups[(run["scheduler"], run["config_id"], str(run["rate"]))].append(run)
    rows = [_aggregate_group(group) for group in groups.values()]
    rows.sort(key=lambda row: (str(row["scheduler"]), str(row["config"]), float(row["rate"]) if isinstance(row["rate"], (int, float)) else str(row["rate"])))

    all_runs = select_runs(runs, mode)
    option_runs = all_runs
    options = {
        "schedulers": sorted({str(run["scheduler"]) for run in option_runs}),
        "rates": sorted({run["rate"] for run in option_runs}, key=lambda value: float(value) if isinstance(value, (int, float)) else str(value)),
        "seeds": sorted({str(run["seed"]) for run in option_runs}),
        "configs": [
            {"id": config_id, "label": label}
            for config_id, label in sorted({(run["config_id"], run["config_label"]) for run in option_runs}, key=lambda item: item[1])
        ],
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "runs_root": None,
        "mode": mode,
        "discovery": stats | {"selected_runs": len(selected_runs), "available_runs": len(all_runs)},
        "options": options,
        "rows": rows,
        "runs": [_public_run(run) for run in selected_runs],
        "notes": [
            "p95 按筛选后的原始请求合并计算，未平均各 run 的 p95。",
            "Resource 为每个请求 node_timeline.gpu_memory_mib 的求和；缺失数据显示为 —，0 保留为真实值。",
            "失败请求不参与质量、资源和时延指标，但保留在失败数中。",
        ],
    }


def _query_set(query: dict[str, list[str]], name: str) -> set[str]:
    values = query.get(name, [])
    # config_id is canonical JSON.  It may contain commas, so unlike the
    # short scalar filters it must remain one URL parameter value.
    if name == "config":
        return set(values)
    return {item for value in values for item in value.split(",") if item}


def _csv_bytes(payload: dict[str, Any]) -> bytes:
    fields = [
        ("scheduler", "scheduler"), ("config", "config"), ("rate", "请求速率"),
        ("quality_mean", "质量均值"), ("quality_p95", "质量 p95"),
        ("resource_mean_mib", "资源均值（MiB）"), ("resource_p95_mib", "资源 p95（MiB）"),
        ("latency_mean_s", "时延均值（秒）"), ("latency_p95_s", "时延 p95（秒）"),
        ("completed_count", "完成数"), ("failed_count", "失败数"), ("run_count", "run 数"),
    ]
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([label for _, label in fields])
    for row in payload["rows"]:
        writer.writerow([row.get(key, "") if row.get(key) is not None else "" for key, _ in fields])
    return output.getvalue().encode("utf-8-sig")


class Handler(BaseHTTPRequestHandler):
    runs_root: Path = DEFAULT_RUNS_ROOT

    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, status: int = 200) -> None:
        self._send(json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

    def do_GET(self) -> None:  # noqa: N802
        route = urlsplit(self.path).path
        if route in {"/api/data", "/api/data.csv"}:
            runs, stats = discover_runs(self.runs_root)
            query = parse_qs(urlsplit(self.path).query)
            mode = query.get("mode", ["latest"])[0]
            if mode not in {"latest", "history"}:
                mode = "latest"
            payload = make_payload(
                runs, stats, mode,
                _query_set(query, "scheduler"), _query_set(query, "rate"),
                _query_set(query, "config"), _query_set(query, "seed"),
            )
            if route.endswith(".csv"):
                self._send(_csv_bytes(payload), "text/csv; charset=utf-8")
            else:
                payload["runs_root"] = str(self.runs_root)
                self._json(payload)
            return
        if route == "/api/health":
            self._json({"ok": self.runs_root.is_dir(), "runs_root": str(self.runs_root)})
            return
        static_path = STATIC_FILES.get(route)
        if static_path is None:
            self.send_error(404, "Not found")
            return
        try:
            body = static_path.read_bytes()
        except OSError:
            self.send_error(404, "Static file not found")
            return
        content_type = "text/html; charset=utf-8" if static_path.suffix == ".html" else "text/css; charset=utf-8" if static_path.suffix == ".css" else "text/javascript; charset=utf-8"
        self._send(body, content_type)

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve scheduler evaluation aggregates")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT, help="evaluation runs root")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    runs_root = args.runs_root.expanduser().resolve()
    handler = type("EvalHandler", (Handler,), {"runs_root": runs_root})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Evaluation Explorer: http://{args.host}:{args.port}/")
    print(f"Runs root: {runs_root}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
