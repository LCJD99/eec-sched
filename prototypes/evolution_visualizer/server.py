"""Serve the Scheduler Evolution Explorer and read-only bottleneck traces.

The trace viewer is deliberately a small, dependency-free server. Trace files
are exposed only as direct ``.jsonl`` children of ``--trace-dir``; the server
never follows a symlink supplied through the API.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit


PROTOTYPE_DIR = Path(__file__).resolve().parent
REPO_DIR = PROTOTYPE_DIR.parent.parent
DEFAULT_TRACE_DIR = REPO_DIR / "outputs" / "bottleneck-analysis"
RESULT_ROUTES = {"/result", "/result.json"}
STATIC_ROUTES = {
    "/": PROTOTYPE_DIR / "index.html",
    "/index.html": PROTOTYPE_DIR / "index.html",
    "/session-trace": PROTOTYPE_DIR / "session_trace.html",
    "/session-trace.html": PROTOTYPE_DIR / "session_trace.html",
    "/session_trace.css": PROTOTYPE_DIR / "session_trace.css",
    "/session_trace.js": PROTOTYPE_DIR / "session_trace.js",
    "/session_trace_parser.mjs": PROTOTYPE_DIR / "session_trace_parser.mjs",
}


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _event_type(record: dict[str, Any]) -> str:
    value = record.get("event_type", record.get("type", record.get("event", "unknown")))
    return str(value) if value is not None else "unknown"


def summarize_trace(path: Path) -> dict[str, Any]:
    """Return a lightweight sidebar summary while tolerating malformed lines."""

    events: list[dict[str, Any]] = []
    error_count = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    error_count += 1
                    continue
                if isinstance(value, dict):
                    events.append(value)
                else:
                    error_count += 1
    except (OSError, UnicodeError):
        error_count += 1

    timestamps = [_timestamp(item.get("timestamp")) for item in events]
    timestamps = [value for value in timestamps if value is not None]
    start_time = min(timestamps) if timestamps else None
    end_time = max(timestamps) if timestamps else None
    event_types = [_event_type(item) for item in events]
    round_values = {
        item.get("round_number", item.get("round"))
        for item in events
        if item.get("round_number", item.get("round")) is not None
    }
    statuses = [str(item.get("status")) for item in events if item.get("status")]
    status = statuses[-1] if statuses else "unknown"
    if any(value in {"run_failed", "session_failed", "agent_failed", "model_failed"} for value in event_types):
        status = "failed"

    def iso(value: datetime | None) -> str | None:
        return value.isoformat().replace("+00:00", "Z") if value else None

    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0
    return {
        "filename": path.name,
        "status": status,
        "start_time": iso(start_time),
        "end_time": iso(end_time),
        "mtime": mtime,
        "event_count": len(events),
        "round_count": len(round_values),
        "tool_call_count": sum(item == "tool_call" for item in event_types),
        "model_output_count": sum(item in {"llm_response", "model_output"} for item in event_types),
        "error_count": error_count,
    }


def list_traces(trace_dir: Path) -> list[dict[str, Any]]:
    """List safe, direct JSONL files ordered newest-first."""

    if not trace_dir.is_dir() or trace_dir.is_symlink():
        return []
    summaries: list[dict[str, Any]] = []
    try:
        entries = list(trace_dir.iterdir())
    except OSError:
        return []
    for entry in entries:
        if entry.is_symlink() or not entry.is_file() or entry.suffix.lower() != ".jsonl":
            continue
        summaries.append(summarize_trace(entry))
    summaries.sort(key=lambda item: (item["start_time"] or "", item["mtime"]), reverse=True)
    return summaries


class Handler(SimpleHTTPRequestHandler):
    """HTTP handler for the two prototype pages and read-only trace APIs."""

    def __init__(
        self,
        *args: object,
        result_path: Path | None = None,
        trace_dir: Path = DEFAULT_TRACE_DIR,
        **kwargs: object,
    ) -> None:
        self.result_path = result_path
        self.trace_dir = Path(trace_dir)
        super().__init__(*args, directory=str(PROTOTYPE_DIR), **kwargs)

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _trace_path(self, route: str) -> Path | None:
        prefix = "/api/traces/"
        if not route.startswith(prefix):
            return None
        name = unquote(route[len(prefix) :])
        if not name or name in {".", ".."} or "/" in name or "\\" in name:
            return None
        if Path(name).name != name or not name.lower().endswith(".jsonl"):
            return None
        if not self.trace_dir.is_dir() or self.trace_dir.is_symlink():
            return None
        candidate = self.trace_dir / name
        if candidate.is_symlink() or not candidate.is_file():
            return None
        try:
            if candidate.resolve().parent != self.trace_dir.resolve():
                return None
        except OSError:
            return None
        return candidate

    def _handle_api(self, route: str) -> bool:
        if route == "/api/traces":
            self._send_json({"traces": list_traces(self.trace_dir)})
            return True
        if route.startswith("/api/traces/"):
            candidate = self._trace_path(route)
            if candidate is None:
                self.send_error(404, "Trace not found")
                return True
            try:
                body = candidate.read_bytes()
            except OSError:
                self.send_error(404, "Trace not found")
                return True
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return True
        return False

    def do_GET(self) -> None:  # noqa: N802
        route = urlsplit(self.path).path
        if self._handle_api(route):
            return
        if route in RESULT_ROUTES and self.result_path is None:
            self.send_error(404, "No evolution result configured")
            return
        if route in STATIC_ROUTES or route in RESULT_ROUTES:
            super().do_GET()
            return
        self.send_error(404, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        self.send_error(405, "Only GET is supported")

    def translate_path(self, path: str) -> str:
        route = urlsplit(path).path
        if route in STATIC_ROUTES:
            return str(STATIC_ROUTES[route])
        if route in RESULT_ROUTES:
            return str(self.result_path) if self.result_path is not None else str(PROTOTYPE_DIR / "missing-result.json")
        return str(PROTOTYPE_DIR / "missing-static-file")

    def end_headers(self) -> None:
        """Prevent stale UI, modules, and result data from browser caches."""

        route = urlsplit(self.path).path
        if route.startswith("/api/") or route in STATIC_ROUTES or route in RESULT_ROUTES:
            self.send_header("Cache-Control", "no-store")
        if route in RESULT_ROUTES and self.result_path is not None:
            self.send_header("X-Evolution-Result-Path", str(self.result_path))
        super().end_headers()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Scheduler Evolution Explorer prototype")
    parser.add_argument("--result-path", type=Path, help="optional evolution-result JSON written by scripts/run_evolution.py")
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR, help="directory containing bottleneck-analysis JSONL traces")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    result_path = args.result_path.resolve() if args.result_path is not None else None
    if result_path is not None and not result_path.is_file():
        parser.error(f"result path does not exist: {result_path}")
    trace_dir_arg = args.trace_dir
    if trace_dir_arg.is_symlink() or (trace_dir_arg.exists() and not trace_dir_arg.is_dir()):
        parser.error(f"trace dir is not a directory: {trace_dir_arg}")
    trace_dir = trace_dir_arg.resolve()
    handler = lambda *request_args, **request_kwargs: Handler(  # noqa: E731
        *request_args, result_path=result_path, trace_dir=trace_dir, **request_kwargs
    )
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"Scheduler Evolution Explorer: http://127.0.0.1:{args.port}/")
    print(f"Session Trace Viewer: http://127.0.0.1:{args.port}/session-trace")
    print(f"Trace directory: {trace_dir}")
    if result_path is not None:
        print(f"Evolution result: {result_path}")
    server.serve_forever()


if __name__ == "__main__":
    main()
