"""Small persistence and execution helpers for evolving diagnosis tools.

The diagnosis agent owns the decision to create a composite tool.  This module
keeps the storage contract deliberately small: a composite is a declarative
sequence of calls to the three diagnosis meta-tools, and its Python file is a
deterministically rendered wrapper around that sequence.  It is not a general
plugin or code execution framework.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pprint
import re


META_TOOLS = frozenset(
    {"profile_trace", "intervene_assignment", "validate_intervention"}
)
DEFAULT_RECENT_LIMIT = 5
_PLACEHOLDER = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class ToolStep:
    """One call in a composite tool specification."""

    tool: str
    arguments: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.tool not in META_TOOLS:
            allowed = ", ".join(sorted(META_TOOLS))
            raise ValueError(f"unknown meta-tool {self.tool!r}; allowed: {allowed}")
        if not isinstance(self.arguments, Mapping):
            raise TypeError("tool step arguments must be a mapping")

    def to_dict(self) -> dict[str, object]:
        return {"tool": self.tool, "arguments": _jsonable(self.arguments)}


@dataclass(frozen=True)
class CompositeToolSpec:
    """Minimal, stable format persisted for a generated diagnosis tool.

    ``parameters`` is a list of parameter names used by ``$name`` placeholders.
    """

    name: str
    description: str
    parameters: tuple[str, ...]
    steps: tuple[ToolStep, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("tool name must be a non-empty string")
        if not _NAME.fullmatch(self.name):
            raise ValueError(
                "tool name must be a Python identifier (letters, digits, underscore)"
            )
        if self.name in META_TOOLS:
            raise ValueError("generated tool name must not shadow a meta-tool")
        if len(set(self.parameters)) != len(self.parameters):
            raise ValueError("tool parameters must be unique")
        for parameter in self.parameters:
            if not isinstance(parameter, str) or not _NAME.fullmatch(parameter):
                raise ValueError(f"invalid tool parameter {parameter!r}")
        if not self.steps:
            raise ValueError("a composite tool needs at least one step")
        # A specification is one flat sequence.  Placeholder validation here
        # catches misspellings before a persisted tool can be loaded.
        for step in self.steps:
            for value in _walk_values(step.arguments):
                if isinstance(value, str):
                    unknown = {
                        match.group(1)
                        for match in _PLACEHOLDER.finditer(value)
                        if match.group(1) not in self.parameters
                    }
                    if unknown:
                        names = ", ".join(sorted(unknown))
                        raise ValueError(f"unknown parameter placeholder(s): {names}")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "CompositeToolSpec":
        if not isinstance(value, Mapping):
            raise TypeError("tool specification must be a mapping")
        raw_parameters = value.get("parameters", ())
        if isinstance(raw_parameters, Sequence) and not isinstance(
            raw_parameters, str | bytes
        ):
            parameters = raw_parameters
        else:
            raise TypeError("tool parameters must be a list")

        raw_steps = value.get("steps", ())
        if not isinstance(raw_steps, Sequence) or isinstance(raw_steps, str | bytes):
            raise TypeError("tool steps must be a list")
        steps: list[ToolStep] = []
        for raw_step in raw_steps:
            if not isinstance(raw_step, Mapping):
                raise TypeError("each tool step must be a mapping")
            tool = raw_step.get("tool")
            arguments = raw_step.get("arguments", {})
            if not isinstance(tool, str):
                raise TypeError("each tool step needs a string 'tool'")
            if not isinstance(arguments, Mapping):
                raise TypeError("tool step arguments must be a mapping")
            steps.append(ToolStep(tool, dict(arguments)))
        raw_name = value.get("name", "")
        raw_description = value.get("description", "")
        name = raw_name if isinstance(raw_name, str) else ""
        description = (
            raw_description
            if isinstance(raw_description, str)
            else str(raw_description)
        )
        return cls(
            name=name,
            description=description,
            parameters=tuple(str(parameter) for parameter in parameters),
            steps=tuple(steps),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": list(self.parameters),
            "steps": [step.to_dict() for step in self.steps],
        }


def coerce_tool_spec(
    spec: CompositeToolSpec | Mapping[str, object],
) -> CompositeToolSpec:
    """Validate and normalize a model-produced specification."""

    if isinstance(spec, CompositeToolSpec):
        return spec
    return CompositeToolSpec.from_dict(spec)


def substitute_placeholders(value: object, parameters: Mapping[str, object]) -> object:
    """Recursively replace ``$parameter`` values in a step's arguments.

    A string consisting solely of a placeholder retains the parameter's type;
    embedded placeholders are interpolated as strings.  Unknown or missing
    placeholders raise ``KeyError``.
    """

    if isinstance(value, Mapping):
        return {
            key: substitute_placeholders(child, parameters)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [substitute_placeholders(child, parameters) for child in value]
    if isinstance(value, tuple):
        return tuple(substitute_placeholders(child, parameters) for child in value)
    if not isinstance(value, str):
        return value

    exact = _PLACEHOLDER.fullmatch(value)
    if exact:
        name = exact.group(1)
        if name not in parameters:
            raise KeyError(name)
        return parameters[name]

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in parameters:
            raise KeyError(name)
        return str(parameters[name])

    return _PLACEHOLDER.sub(replace, value)


MetaToolDispatcher = Callable[[str, Mapping[str, object]], object]


def execute_tool_spec(
    spec: CompositeToolSpec | Mapping[str, object],
    parameters: Mapping[str, object],
    dispatcher: MetaToolDispatcher,
) -> list[dict[str, object]]:
    """Execute a flat composite sequence through an injected dispatcher.

    Errors are represented as strings in the corresponding result and do not
    stop later steps.  This intentionally leaves tool lifecycle simple: a
    failed generated tool remains available for a later diagnosis run.
    """

    normalized = coerce_tool_spec(spec)
    missing = [name for name in normalized.parameters if name not in parameters]
    if missing:
        raise ValueError(f"missing tool parameter(s): {', '.join(missing)}")
    results: list[dict[str, object]] = []
    for step in normalized.steps:
        try:
            arguments = substitute_placeholders(step.arguments, parameters)
            if not isinstance(arguments, Mapping):  # pragma: no cover - by construction
                raise TypeError("resolved tool arguments must be a mapping")
            result = dispatcher(step.tool, arguments)
            results.append(
                {
                    "tool": step.tool,
                    "arguments": _jsonable(arguments),
                    "result": _jsonable(result),
                }
            )
        except Exception as exc:  # generated tool failures are data, not control flow
            results.append(
                {
                    "tool": step.tool,
                    "arguments": _jsonable(locals().get("arguments", step.arguments)),
                    "result": f"{type(exc).__name__}: {exc}",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return results


def render_tool_source(spec: CompositeToolSpec | Mapping[str, object]) -> str:
    """Render a deterministic Python wrapper for a declarative tool spec."""

    normalized = coerce_tool_spec(spec)
    # ``json.dumps`` is ideal for storage, but its lowercase true/false/null
    # tokens are not valid Python literals.  ``pformat`` keeps the generated
    # wrapper directly compilable for every JSON-compatible argument value.
    payload = pprint.pformat(normalized.to_dict(), sort_dicts=True, width=88)
    return (
        '"""Generated diagnosis composite tool.\n'
        "This file is a wrapper around a declarative meta-tool sequence.\n"
        '"""\n\n'
        "from eec_sched.diagnosis.tool_evolution import execute_tool_spec\n\n\n"
        f"TOOL_SPEC = {payload}\n\n\n"
        f"def {normalized.name}(dispatcher, **kwargs):\n"
        "    return execute_tool_spec(TOOL_SPEC, kwargs, dispatcher)\n"
    )


class DiagnosisToolStore:
    """Per-experiment JSONL history and generated-tool store."""

    def __init__(
        self, state_dir: str | Path, *, recent_limit: int = DEFAULT_RECENT_LIMIT
    ) -> None:
        if recent_limit < 1:
            raise ValueError("recent_limit must be positive")
        self.state_dir = Path(state_dir)
        self.recent_limit = recent_limit
        self.history_path = self.state_dir / "history.jsonl"
        self.tools_dir = self.state_dir / "generated_tools"
        self.index_path = self.tools_dir / "index.jsonl"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.tools_dir.mkdir(parents=True, exist_ok=True)

    def append_history(
        self,
        *,
        tool_calls: Iterable[object],
        diagnosis: object,
        snapshot_digest: str,
    ) -> dict[str, object]:
        """Append one compact diagnosis trajectory and return its JSON row."""

        row = {
            "tool_calls": [_normalize_tool_call(call) for call in tool_calls],
            "diagnosis": _jsonable(diagnosis),
            "snapshot_digest": snapshot_digest,
            "timestamp": _now(),
        }
        _append_jsonl(self.history_path, row)
        return row

    def read_recent_history(self, limit: int | None = None) -> list[dict[str, object]]:
        """Read the last ``limit`` trajectory rows, in chronological order."""

        rows = _read_jsonl(self.history_path)
        count = self.recent_limit if limit is None else limit
        if count < 1:
            return []
        return rows[-count:]

    def save_tool_spec(
        self, spec: CompositeToolSpec | Mapping[str, object]
    ) -> CompositeToolSpec:
        """Persist a spec and its generated source, then append a simple index row."""

        normalized = coerce_tool_spec(spec)
        encoded = json.dumps(normalized.to_dict(), ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]
        tool_id = f"{_slug(normalized.name)}-{digest}"
        spec_name = f"{tool_id}.json"
        source_name = f"{tool_id}.py"
        (self.tools_dir / spec_name).write_text(
            json.dumps(normalized.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (self.tools_dir / source_name).write_text(
            render_tool_source(normalized), encoding="utf-8"
        )
        now = _now()
        _append_jsonl(
            self.index_path,
            {
                "event": "save",
                "id": tool_id,
                "name": normalized.name,
                "spec": spec_name,
                "source": source_name,
                "created_at": now,
            },
        )
        return normalized

    def load_recent_tool_specs(
        self, limit: int | None = None
    ) -> list[CompositeToolSpec]:
        """Load the most recently generated tools, newest names winning."""

        count = self.recent_limit if limit is None else limit
        if count < 1:
            return []
        records = self._tool_records()
        selected: list[CompositeToolSpec] = []
        seen: set[str] = set()
        seen_names: set[str] = set()
        for record in reversed(records):
            tool_id = str(record.get("id", ""))
            tool_name = str(record.get("name", ""))
            if not tool_id or tool_id in seen or tool_name in seen_names:
                continue
            spec_file = record.get("spec")
            if not isinstance(spec_file, str):
                continue
            path = self.tools_dir / spec_file
            if not path.is_file():
                continue
            try:
                spec = CompositeToolSpec.from_dict(
                    json.loads(path.read_text(encoding="utf-8"))
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
            selected.append(spec)
            seen.add(tool_id)
            seen_names.add(tool_name)
            if len(selected) >= count:
                break
        selected.reverse()
        return selected

    def execute(
        self,
        spec: CompositeToolSpec | Mapping[str, object],
        parameters: Mapping[str, object],
        dispatcher: MetaToolDispatcher,
    ) -> list[dict[str, object]]:
        return execute_tool_spec(spec, parameters, dispatcher)

    def _tool_records(self) -> list[dict[str, object]]:
        records: dict[str, dict[str, object]] = {}
        for row in _read_jsonl(self.index_path):
            tool_id = row.get("id")
            if not isinstance(tool_id, str):
                continue
            if row.get("event") == "save":
                records[tool_id] = dict(row)
        return list(records.values())


def _normalize_tool_call(call: object) -> dict[str, object]:
    if isinstance(call, Mapping):
        name = call.get("tool", call.get("name", "unknown"))
        arguments = call.get("arguments", call.get("params", {}))
        result = call.get("result_summary", call.get("result", ""))
    elif isinstance(call, Sequence) and not isinstance(call, str | bytes):
        values = list(call)
        name = values[0] if values else "unknown"
        arguments = values[1] if len(values) > 1 else {}
        result = values[2] if len(values) > 2 else ""
    else:
        name, arguments, result = str(call), {}, ""
    return {
        "tool": str(name),
        "arguments": _jsonable(arguments),
        "result_summary": _summarize(result),
    }


def _summarize(value: object, max_chars: int = 2_000) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(_jsonable(value), ensure_ascii=False, default=str)
    return text if len(text) <= max_chars else text[:max_chars] + "..."


def _walk_values(value: object) -> Iterable[object]:
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _walk_values(child)
    elif isinstance(value, list | tuple):
        for child in value:
            yield from _walk_values(child)
    else:
        yield value


def _jsonable(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_jsonable(child) for child in value]
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return str(value)
    return value


def _append_jsonl(path: Path, row: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(row), ensure_ascii=False, default=str) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-") or "tool"
