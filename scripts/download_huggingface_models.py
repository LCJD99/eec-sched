#!/usr/bin/env python3
"""Download the Hugging Face repositories used by the MnMS tool catalog.

This script deliberately leaves Hugging Face cache configuration untouched.
``snapshot_download`` therefore uses the caller's normal Hugging Face cache
location (or any location the caller has configured before invoking it).
"""

from __future__ import annotations

import argparse
import sys

from eec_sched.mnms_tools import HUGGINGFACE_MODEL_IDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print repositories without downloading them.")
    parser.add_argument("--force", action="store_true", help="Re-download snapshots even when they are already cached.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dry_run:
        print("Hugging Face repositories that would be downloaded:")
        print(*HUGGINGFACE_MODEL_IDS, sep="\n")
        return 0
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Missing dependency: install huggingface-hub (or transformers) before running this script.", file=sys.stderr)
        return 2

    failures: list[str] = []
    for model_id in HUGGINGFACE_MODEL_IDS:
        print(f"Downloading {model_id}...", flush=True)
        try:
            path = snapshot_download(repo_id=model_id, force_download=args.force)
        except Exception as error:  # Continue so a single gated/failed model is visible.
            failures.append(f"{model_id}: {type(error).__name__}: {error}")
            print(f"FAILED {failures[-1]}", file=sys.stderr)
        else:
            print(f"Cached at {path}")
    if failures:
        print(f"{len(failures)} download(s) failed.", file=sys.stderr)
        return 1
    print("All MnMS Hugging Face model snapshots are available locally.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
