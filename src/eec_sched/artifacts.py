"""Persistence adapters for reproducible experiment artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Mapping, Protocol


class ArtifactStore(Protocol):
    def append_event(self, event_type: str, payload: Mapping[str, object]) -> None: ...

    def write_result(self, payload: Mapping[str, object]) -> Path | None: ...


@dataclass
class InMemoryArtifactStore:
    events: list[dict[str, object]] = field(default_factory=list)
    result: dict[str, object] | None = None

    def append_event(self, event_type: str, payload: Mapping[str, object]) -> None:
        self.events.append({"type": event_type, **dict(payload)})

    def write_result(self, payload: Mapping[str, object]) -> None:
        self.result = dict(payload)


class JsonlArtifactStore:
    """Store one immutable run's events and final result in a timestamped pair."""

    def __init__(self, output_dir: str | Path, run_id: str | None = None) -> None:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.events_path = root / f"evolution-trace_{self.run_id}.jsonl"
        self.result_path = root / f"evolution-result_{self.run_id}.json"
        self.events_path.touch(exist_ok=False)

    def append_event(self, event_type: str, payload: Mapping[str, object]) -> None:
        with self.events_path.open("a", encoding="utf-8") as target:
            target.write(json.dumps({"type": event_type, **dict(payload)}, ensure_ascii=False) + "\n")

    def write_result(self, payload: Mapping[str, object]) -> Path:
        self.result_path.write_text(
            json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return self.result_path
