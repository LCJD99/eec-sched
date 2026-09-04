"""PROTOTYPE — serve the profiling database visualizer locally.

Question: which representation makes a complete profiling snapshot easiest to
inspect? Three variants are available with ``?variant=overview|tool|network``.
"""

from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class Handler(SimpleHTTPRequestHandler):
    def translate_path(self, path: str) -> str:
        if path == "/" or path.startswith("/?"):
            return str(Path(__file__).parent / "index.html")
        if path == "/fake-data":
            return str(ROOT / "docs/examples/profiling-database.fake.json")
        if path == "/schema":
            return str(ROOT / "docs/schemas/profiling-database.schema.json")
        return super().translate_path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the profiling database UI prototype")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Profiling database prototype: http://127.0.0.1:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
