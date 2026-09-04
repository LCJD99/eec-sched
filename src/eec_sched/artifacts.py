"""Persistence adapters for reproducible experiment artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Mapping, Protocol

import yaml


class ArtifactStore(Protocol):
    @property
    def run_dir(self) -> Path | None: ...

    def write_config(self, config: Mapping[str, object]) -> Path | None: ...

    def append_event(self, event_type: str, payload: Mapping[str, object]) -> None: ...

    def write_result(self, payload: Mapping[str, object]) -> Path | None: ...


@dataclass
class InMemoryArtifactStore:
    events: list[dict[str, object]] = field(default_factory=list)
    result: dict[str, object] | None = None
    config: dict[str, object] | None = None

    @property
    def run_dir(self) -> None:
        return None

    def write_config(self, config: Mapping[str, object]) -> None:
        self.config = dict(config)

    def append_event(self, event_type: str, payload: Mapping[str, object]) -> None:
        self.events.append({"type": event_type, **dict(payload)})

    def write_result(self, payload: Mapping[str, object]) -> None:
        self.result = dict(payload)


class JsonlArtifactStore:
    """Store every artifact from one execution in a fresh run directory.

    ``output_dir`` is the parent directory for run records.  ``run_id`` is kept
    as a compatibility escape hatch for tests and callers that need a
    deterministic directory name; normal experiment runs derive it from the
    experiment directory name and the current timestamp.
    """

    def __init__(
        self,
        output_dir: str | Path,
        run_id: str | None = None,
        experiment_name: str = "run",
    ) -> None:
        for name, value in (("run_id", run_id), ("experiment_name", experiment_name)):
            if value is not None and (not value or Path(value).name != value):
                raise ValueError(f"{name} must be a single path component")
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or self._new_run_id(root, experiment_name)
        self.run_dir = root / self.run_id
        if run_id is not None:
            self.run_dir.mkdir(exist_ok=False)
        self.config_path = self.run_dir / "config.yaml"
        self.events_path = self.run_dir / "evolution-trace.jsonl"
        self.result_path = self.run_dir / "evolution-result.json"
        self.events_path.touch(exist_ok=False)

    @staticmethod
    def _new_run_id(root: Path, experiment_name: str) -> str:
        """Create a unique second-resolution run id without reusing a record."""
        while True:
            run_id = f"{experiment_name}_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
            try:
                (root / run_id).mkdir(exist_ok=False)
            except FileExistsError:
                # A concurrent launch can share a timestamp.  Wait for the
                # clock to advance instead of silently overwriting that run.
                time.sleep(0.05)
                continue
            return run_id

    def write_config(self, config: Mapping[str, object]) -> Path:
        if self.config_path.exists():
            raise FileExistsError(f"run configuration already exists: {self.config_path}")
        self.config_path.write_text(
            yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return self.config_path

    def append_event(self, event_type: str, payload: Mapping[str, object]) -> None:
        with self.events_path.open("a", encoding="utf-8") as target:
            target.write(json.dumps({"type": event_type, **dict(payload)}, ensure_ascii=False) + "\n")

    def write_result(self, payload: Mapping[str, object]) -> Path:
        if self.result_path.exists():
            raise FileExistsError(f"run result already exists: {self.result_path}")
        self.result_path.write_text(
            json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return self.result_path


# Descriptive name for new callers; keep the original import stable for
# existing scripts and notebooks.
RunArtifactStore = JsonlArtifactStore
