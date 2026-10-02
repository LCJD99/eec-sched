"""Filter, convert, and augment MnMS requests for the offline planner.

The source file is the human-verified MnMS JSONL split.  Only rows whose GT
``plan_str`` calls tools from ``scripts.profile_mnms.SUITES`` or
``text_generation`` are retained.
Each output planner request contains three sampled ``user_request`` values.
Sampling is with
replacement, so one MnMS request may be reused across groups and may appear
more than once within a group.  The random seed makes the generated 1000
records reproducible, while a set of source-index triples prevents wasting
groups on an identical ordered triple.

The raw dataset is kept at ``data/benchmakrs/mnms`` (the directory spelling is
intentional).  The first run downloads it there; later runs read that local
copy.  The tool catalog is exported from the repository's MnMS tool
definitions so it stays in sync with the planner validator.

The default invocation writes ``requests.jsonl`` and ``tool_catalog.json`` in
this directory::

    uv run python experiments/09_offline_tool_planning/data_process.py

Use ``--dataset`` with a local JSONL path when testing with a fixture.  A
dataset row is expected to contain ``id`` and ``user_request``.  The original
request text is kept verbatim; only the ``first``, ``next``, and ``last``
connectors are added around each group of three requests.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import sys
from collections.abc import Collection, Iterable, Iterator, Mapping
from pathlib import Path
from urllib.request import urlopen

DEFAULT_DATASET_URL = (
    "https://huggingface.co/datasets/zixianma/mnms/resolve/"
    "e9cb0112a0a88788fc7559e27cdf40646f30ad3f/"
    "test_human_verified_filtered.json"
)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data/benchmakrs/mnms/test_human_verified_filtered.json"
DEFAULT_REQUESTS_PATH = SCRIPT_DIR / "requests.jsonl"
DEFAULT_CATALOG_PATH = SCRIPT_DIR / "tool_catalog.json"
DEFAULT_TARGET_COUNT = 1000
DEFAULT_SEED = 42

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eec_sched.mnms_tools import mnms_tool_specs
from scripts.profile_mnms import SUITES


ALLOWED_TOOL_IDS = frozenset(set(SUITES) | {"text_generation"})


def _rows_from_stream(lines: Iterable[str]) -> Iterator[dict[str, object]]:
    """Yield MnMS objects from a JSONL stream."""
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"dataset line {line_number} must be a JSON object")
        if "id" not in value or "user_request" not in value:
            raise ValueError(f"dataset line {line_number} must contain id and user_request")
        if not isinstance(value["user_request"], str) or not value["user_request"].strip():
            raise ValueError(f"dataset line {line_number} user_request must be non-empty text")
        yield value


def download_dataset(destination: Path = DEFAULT_DATASET_PATH, url: str = DEFAULT_DATASET_URL) -> Path:
    """Download the pinned MnMS JSONL file to the repository data directory."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=60) as response:  # noqa: S310 - URL is an explicit source constant/CLI input.
        with destination.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
    return destination


def ensure_dataset(source: str | Path = DEFAULT_DATASET_PATH) -> str | Path:
    """Return a dataset source, downloading the default local copy if needed."""
    source_path = Path(source)
    if source_path == DEFAULT_DATASET_PATH and not source_path.exists():
        return download_dataset()
    return source


def iter_dataset(source: str | Path) -> Iterator[dict[str, object]]:
    """Read a local MnMS JSONL file or an explicit HTTP dataset URL."""
    source_text = str(source)
    if source_text.startswith(("http://", "https://")):
        with urlopen(source_text, timeout=60) as response:  # noqa: S310 - URL is an explicit CLI input.
            for row in _rows_from_stream(
                line.decode("utf-8") for line in response
            ):
                yield row
        return

    with Path(source_text).open(encoding="utf-8") as dataset:
        yield from _rows_from_stream(dataset)


def _plan_tool_ids(raw_plan: object) -> tuple[str, ...] | None:
    """Parse MnMS ``plan_str`` and return normalized tool IDs."""
    if not isinstance(raw_plan, str):
        return None
    try:
        plan = ast.literal_eval(raw_plan)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(plan, list) or not plan:
        return None
    tool_ids: list[str] = []
    for node in plan:
        if not isinstance(node, dict) or not isinstance(node.get("name"), str):
            return None
        tool_ids.append(node["name"].replace(" ", "_"))
    return tuple(tool_ids)


def filter_suite_rows(
    rows: Iterable[Mapping[str, object]],
    suite_tool_ids: Collection[str] = ALLOWED_TOOL_IDS,
) -> tuple[list[Mapping[str, object]], int]:
    """Keep rows whose every GT plan tool belongs to the allowed set."""
    allowed = set(suite_tool_ids)
    selected: list[Mapping[str, object]] = []
    skipped = 0
    for row in rows:
        tool_ids = _plan_tool_ids(row.get("plan_str"))
        if tool_ids is None or any(tool_id not in allowed for tool_id in tool_ids):
            skipped += 1
            continue
        selected.append(row)
    return selected, skipped


def _combined_request(rows: list[Mapping[str, object]], ordinal: int) -> dict[str, str]:
    """Build one ``id``/``request`` record from exactly three source rows."""
    if len(rows) != 3:
        raise ValueError("a combined request requires exactly three rows")
    identifiers = [str(row["id"]) for row in rows]
    requests = [str(row["user_request"]).strip() for row in rows]
    return {
        "id": f"mnms-{ordinal:04d}-" + "-".join(identifiers),
        "request": f"first {requests[0]}, next {requests[1]}, last {requests[2]}",
    }


def convert_requests(
    rows: Iterable[Mapping[str, object]],
    target_count: int = DEFAULT_TARGET_COUNT,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, str]]:
    """Sample and combine exactly ``target_count`` groups of three rows.

    Sampling uses replacement both between and within groups.  Only an exact
    ordered source-index triple is excluded after it has been emitted while
    unused triples remain.  If a tiny fixture exhausts all possible triples,
    reuse is allowed so the requested count can still be produced.  The source
    itself is never modified.
    """
    if target_count < 0:
        raise ValueError("target_count must be non-negative")
    source_rows = list(rows)
    if not source_rows and target_count:
        raise ValueError("the MnMS dataset must contain at least one row")
    if target_count == 0:
        return []

    rng = random.Random(seed)
    seen_triples: set[tuple[int, int, int]] = set()
    records: list[dict[str, str]] = []
    source_count = len(source_rows)
    max_unique_triples = source_count**3
    while len(records) < target_count:
        triple = tuple(rng.randrange(source_count) for _ in range(3))
        if len(seen_triples) < max_unique_triples:
            if triple in seen_triples:
                continue
            seen_triples.add(triple)
        records.append(
            _combined_request(
                [source_rows[index] for index in triple],
                len(records),
            )
        )
    return records


def build_tool_catalog(
    suite_tool_ids: Collection[str] = ALLOWED_TOOL_IDS,
) -> dict[str, list[dict[str, object]]]:
    """Serialize only allowed planner tools in planner catalog shape."""
    allowed = set(suite_tool_ids)
    tools: list[dict[str, object]] = []
    for spec in mnms_tool_specs():
        if spec.tool_id not in allowed:
            continue
        tools.append(
            {
                "tool_id": spec.tool_id,
                "description": spec.description,
                "inputs": {
                    name: {"modality": port.modality, "required": port.required}
                    for name, port in spec.inputs.items()
                },
                "outputs": {
                    name: {"modality": port.modality}
                    for name, port in spec.outputs.items()
                },
            }
        )
    return {"tools": tools}


def _write_jsonl(path: Path, records: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def process_dataset(
    dataset: str | Path = DEFAULT_DATASET_PATH,
    requests_path: Path = DEFAULT_REQUESTS_PATH,
    catalog_path: Path = DEFAULT_CATALOG_PATH,
    target_count: int = DEFAULT_TARGET_COUNT,
    seed: int = DEFAULT_SEED,
) -> int:
    """Write planner request and catalog files and return request count."""
    source_rows = list(iter_dataset(ensure_dataset(dataset)))
    suite_rows, _ = filter_suite_rows(source_rows)
    records = convert_requests(suite_rows, target_count, seed)
    requests_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(requests_path, records)
    catalog = build_tool_catalog()
    catalog_path.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return len(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET_PATH, help="MnMS JSONL path")
    parser.add_argument("--requests-output", type=Path, default=DEFAULT_REQUESTS_PATH)
    parser.add_argument("--catalog-output", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--target-count", type=int, default=DEFAULT_TARGET_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    count = process_dataset(
        args.dataset,
        args.requests_output,
        args.catalog_output,
        args.target_count,
        args.seed,
    )
    print(
        f"Wrote {count} combined requests and {len(build_tool_catalog()['tools'])} tools "
        f"from allowed source rows (seed {args.seed})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
