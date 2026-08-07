"""Validate one profiling database snapshot against schema and semantics."""

from __future__ import annotations

import argparse
from pathlib import Path

from eec_sched.profiling_database import load_profiling_database


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("--schema", type=Path, default=Path("docs/schemas/profiling-database.schema.json"))
    args = parser.parse_args()
    snapshot = load_profiling_database(args.database, args.schema)
    print(f"valid {snapshot.snapshot_id} {snapshot.snapshot_digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
