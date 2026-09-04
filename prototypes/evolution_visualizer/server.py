"""Serve the Scheduler Evolution Explorer prototype."""

from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


PROTOTYPE_DIR = Path(__file__).resolve().parent
RESULT_ROUTES = {"/result", "/result.json"}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args: object, result_path: Path | None = None, **kwargs: object) -> None:
        self.result_path = result_path
        super().__init__(*args, directory=str(PROTOTYPE_DIR), **kwargs)

    def translate_path(self, path: str) -> str:
        route = urlsplit(path).path
        if route == "/":
            return str(PROTOTYPE_DIR / "index.html")
        if route in RESULT_ROUTES:
            return str(self.result_path) if self.result_path is not None else str(PROTOTYPE_DIR / "missing-result.json")
        return super().translate_path(path)

    def end_headers(self) -> None:
        """Prevent stale UI and result data from being reused by the browser."""
        route = self.path.split("?", 1)[0]
        if route in {"/", *RESULT_ROUTES}:
            self.send_header("Cache-Control", "no-store")
        if route in RESULT_ROUTES and self.result_path is not None:
            self.send_header("X-Evolution-Result-Path", str(self.result_path))
        super().end_headers()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Scheduler Evolution Explorer prototype")
    parser.add_argument("--result-path", type=Path, required=True, help="evolution-result JSON written by scripts/run_evolution.py")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    result_path = args.result_path.resolve()
    if not result_path.is_file():
        parser.error(f"result path does not exist: {result_path}")
    handler = lambda *request_args, **request_kwargs: Handler(  # noqa: E731
        *request_args, result_path=result_path, **request_kwargs
    )
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"Scheduler Evolution Explorer: http://127.0.0.1:{args.port}")
    print(f"Evolution result: {result_path}")
    server.serve_forever()


if __name__ == "__main__":
    main()
