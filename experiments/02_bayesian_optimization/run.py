"""One-command entry point for all selected MnMS model campaigns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eec_sched.profiling.campaign import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.mnms.example.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("campaigns"))
    args = parser.parse_args()
    paths = run(json.loads(args.config.read_text(encoding="utf-8")), args.output)
    print("\n".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
