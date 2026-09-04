"""Experience memory adapters for evolution experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Experience:
    summary: str
    strategy_version: int | None = None
    metadata: dict[str, object] | None = None


class Memory(Protocol):
    def retrieve(self, query: str = "", *, limit: int = 10) -> tuple[Experience, ...]: ...

    def record(self, experience: Experience) -> None: ...


class EmptyMemory:
    """No-op memory adapter used for clean ablations and stateless runs."""

    def retrieve(self, query: str = "", *, limit: int = 10) -> tuple[Experience, ...]:
        return ()

    def record(self, experience: Experience) -> None:
        return None


class InMemoryMemory:
    """Simple run-local memory implementation for tests and experiments."""

    def __init__(self) -> None:
        self._experiences: list[Experience] = []

    def retrieve(self, query: str = "", *, limit: int = 10) -> tuple[Experience, ...]:
        if limit <= 0:
            return ()
        matches = (item for item in reversed(self._experiences) if not query or query.lower() in item.summary.lower())
        return tuple(matches)[:limit]

    def record(self, experience: Experience) -> None:
        self._experiences.append(experience)


class JsonlMemory:
    """Persistent cross-run memory with an append-only, inspectable format."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def retrieve(self, query: str = "", *, limit: int = 10) -> tuple[Experience, ...]:
        if limit <= 0 or not self.path.exists():
            return ()
        experiences: list[Experience] = []
        with self.path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    experience = Experience(**value)
                except (json.JSONDecodeError, TypeError) as exc:
                    raise ValueError(f"invalid evolution memory record at line {line_number}") from exc
                if not query or query.lower() in experience.summary.lower():
                    experiences.append(experience)
        return tuple(reversed(experiences[-limit:]))

    def record(self, experience: Experience) -> None:
        with self.path.open("a", encoding="utf-8") as target:
            target.write(
                json.dumps(
                    {
                        "summary": experience.summary,
                        "strategy_version": experience.strategy_version,
                        "metadata": experience.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
