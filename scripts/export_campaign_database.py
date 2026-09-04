#!/usr/bin/env python3
"""Export a Bayesian campaign directory for one measured device."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eec_sched.profiling.export import export_campaign_database


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_directory", type=Path)
    parser.add_argument("--device", type=Path, required=True, help="JSON file with device_id, hardware_class, description")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = json.loads(args.device.read_text(encoding="utf-8"))
    payload = export_campaign_database(args.campaign_directory, device, args.output)
    count = sum(tool["configuration_count"] for tool in payload["tools"])
    print(f"wrote {args.output} with {len(payload['tools'])} tools and {count} configurations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
