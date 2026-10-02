"""Run one Scheduler Candidate through trusted evaluation and diagnosis.

This is intentionally a small composition root for interactive debugging.  It
loads exactly one Candidate, evaluates it on one deterministic dataset split,
and then invokes only :class:`SelfEvolvingDiagnosis`; it never enters the
Candidate coding/evolution loop.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, fields, is_dataclass, replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence, cast

from dotenv import load_dotenv
from omegaconf import OmegaConf

from ..candidate import (
    CandidateEvaluation,
    EvaluationTrace,
    SchedulerCandidate,
    SchedulerCandidateRegistry,
    SchedulerProposal,
    SchedulerView,
)
from ..diagnosis.agents import SelfEvolvingDiagnosis
from ..diagnosis.models import DiagnosisResult
from ..diagnosis.tool_evolution import DiagnosisToolStore
from ..diagnosis.tools import BottleneckMetaTools
from ..evolution import evaluate_scheduler_candidate
from ..evolution.engine import _complete_trace_event
from ..evolution.models import ScoringContext
from ..llm import OpenAICompatibleChatModel
from ..profiling.snapshot import load_profiling_database
from ..workflow import ToolCallPlanDataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = PROJECT_ROOT / "data/mnms-ground-truth-dags.jsonl"
DEFAULT_PROFILING_DATABASE = PROJECT_ROOT / "docs/examples/profiling-database.fake.json"
DEFAULT_PROFILING_SCHEMA = PROJECT_ROOT / "docs/schemas/profiling-database.schema.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/bottleneck-analysis"
DEFAULT_STATE_DIR = PROJECT_ROOT / "outputs/diagnosis-tools/bottleneck-analysis"
DEFAULT_MODEL_BASE_URL = "http://localhost:9888/v1"
DEFAULT_MODEL_NAME = "qwen3.8-27b"
DEFAULT_TOKEN_ENV = "EEC_SCHED_REFLECTION_TOKEN"

_SECRET_KEYS = {
    "token",
    "api_key",
    "password",
    "secret",
    "authorization",
}


@dataclass(frozen=True)
class BottleneckAnalysisOptions:
    """Inputs for :func:`run_analysis`, kept separate from argument parsing."""

    scheduler_source: Path
    scheduler_version: int = 1
    dataset: Path = DEFAULT_DATASET
    split: str = "test"
    profiling_database: Path = DEFAULT_PROFILING_DATABASE
    profiling_schema: Path = DEFAULT_PROFILING_SCHEMA
    output_dir: Path = DEFAULT_OUTPUT_DIR
    state_dir: Path = DEFAULT_STATE_DIR
    model_config: Path | None = None
    model_base_url: str = DEFAULT_MODEL_BASE_URL
    model_name: str = DEFAULT_MODEL_NAME
    model_token: str | None = None
    model_token_env: str = DEFAULT_TOKEN_ENV
    model_timeout_seconds: float = 360.0
    _model_config_loaded: bool = field(default=False, repr=False, compare=False)
    accuracy_weight: float = 1.0 / 3.0
    latency_weight: float = 1.0 / 3.0
    resource_weight: float = 1.0 / 3.0
    latency_scale_ms: float = 100.0
    resource_scale_mib: float = 8_192.0
    history_limit: int = 5
    generated_tool_limit: int = 5
    # Bound the trusted evaluation input before scoring.  ``trace_limit`` is
    # retained as the separate diagnosis-context bound for compatibility.
    dag_limit: int = 3
    trace_limit: int = 3
    candidate_timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.scheduler_version, int)
            or isinstance(self.scheduler_version, bool)
            or self.scheduler_version < 1
        ):
            raise ValueError("scheduler_version must be a positive integer")
        if self.split not in {"train", "validation", "test", "all"}:
            raise ValueError("split must be train, validation, test, or all")
        for name in ("accuracy_weight", "latency_weight", "resource_weight"):
            if (
                not math.isfinite(float(getattr(self, name)))
                or float(getattr(self, name)) < 0
            ):
                raise ValueError(f"{name} must be a finite non-negative number")
        if self.accuracy_weight + self.latency_weight + self.resource_weight <= 0:
            raise ValueError("at least one scoring weight must be positive")
        for name in ("latency_scale_ms", "resource_scale_mib"):
            if (
                not math.isfinite(float(getattr(self, name)))
                or float(getattr(self, name)) <= 0
            ):
                raise ValueError(f"{name} must be a finite positive number")
        for name in (
            "history_limit",
            "generated_tool_limit",
            "dag_limit",
            "trace_limit",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be positive")
        if self.candidate_timeout_seconds is not None and (
            not math.isfinite(self.candidate_timeout_seconds)
            or self.candidate_timeout_seconds <= 0
        ):
            raise ValueError("candidate_timeout_seconds must be finite and positive")


# Short alias for callers that prefer a generic orchestration name.
AnalysisOptions = BottleneckAnalysisOptions


@dataclass(frozen=True)
class BottleneckAnalysisResult:
    output_path: Path
    evaluation: CandidateEvaluation
    diagnosis: DiagnosisResult


class JsonlEventRecorder:
    """Append-only, flush-on-every-event recorder for live debugging."""

    def __init__(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        candidate = output_dir / f"{stem}.jsonl"
        suffix = 1
        while True:
            try:
                self._handle = candidate.open("x", encoding="utf-8")
                break
            except FileExistsError:
                candidate = output_dir / f"{stem}-{suffix:02d}.jsonl"
                suffix += 1
        self.path = candidate
        self._sequence = 0

    def emit(self, event: Mapping[str, object]) -> None:
        event_type = event.get("event_type", event.get("type"))
        if not isinstance(event_type, str) or not event_type:
            raise ValueError("event must contain a non-empty string event_type or type")
        self._sequence += 1
        row: dict[str, object] = {
            "sequence": self._sequence,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "event_type": event_type,
            "type": event_type,
        }
        for key, value in event.items():
            if key not in {"sequence", "timestamp", "type", "event_type"}:
                row[str(key)] = _event_safe(value, key=str(key))
        self._handle.write(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def _event_safe(value: object, *, key: str | None = None) -> object:
    """Make event payloads JSON-safe and guard against accidental credentials."""
    if key is not None and key.lower() in _SECRET_KEYS:
        return "<redacted>"
    if key is not None and key.lower() in {"source_code", "scheduler_code"}:
        return "<redacted>"
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _event_safe(getattr(value, item.name), key=item.name)
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(child_key): _event_safe(child, key=str(child_key))
            for child_key, child in value.items()
        }
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_event_safe(child) for child in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _load_candidate(source_path: Path, scheduler_version: int) -> SchedulerCandidate:
    """Load a Candidate through the same SchedulerCandidate contract as eval.py."""
    source_code = source_path.read_text(encoding="utf-8")
    compiled = compile(source_code, str(source_path), "exec")

    def propose(view: SchedulerView) -> SchedulerProposal:
        namespace: dict[str, object] = {"__builtins__": __builtins__}
        exec(compiled, namespace)  # noqa: S102 - trusted evaluator boundary
        loaded = namespace.get("propose")
        if not callable(loaded):
            raise ValueError(f"{source_path} must define callable propose(view)")
        return cast(Callable[[SchedulerView], SchedulerProposal], loaded)(view)

    return SchedulerCandidate(
        scheduler_version=scheduler_version,
        propose=propose,
        source_code=source_code,
        strategy_description=f"loaded from {source_path}",
    )


def _split(dataset: ToolCallPlanDataset, name: str) -> Sequence[EvaluationTrace]:
    if name == "all":
        return dataset.traces
    splits = dataset.split()
    return {
        "train": splits.train,
        "validation": splits.validation,
        "test": splits.test,
    }[name]


def _diagnosis_evidence(
    candidate: SchedulerCandidate,
    evaluation: CandidateEvaluation,
    scoring_context: ScoringContext | None = None,
) -> dict[str, object]:
    traces = [
        {
            key: value
            for key, value in _complete_trace_event(
                record, candidate.scheduler_version
            ).items()
            if key != "type"
        }
        for record in evaluation.traces
    ]
    digest = traces[0].get("snapshot_digest") if traces else None
    evidence: dict[str, object] = {
        "schema_version": evaluation.schema_version,
        "scheduler_version": candidate.scheduler_version,
        "strategy_description": candidate.strategy_description,
        "source_code": candidate.source_code,
        "snapshot_digest": digest,
        "candidate_score": evaluation.candidate_score,
        "traces": traces,
    }
    if scoring_context is not None:
        evidence["scoring_context"] = {
            "accuracy_weight": scoring_context.accuracy_weight,
            "latency_weight": scoring_context.latency_weight,
            "resource_weight": scoring_context.resource_weight,
            "latency_scale_ms": scoring_context.latency_scale_ms,
            "resource_scale_mib": scoring_context.resource_scale_mib,
        }
    return evidence


def _options_event(options: BottleneckAnalysisOptions) -> dict[str, object]:
    # Deliberately omit model_token and model_token_env from the persisted
    # configuration.  Paths and model identity are useful for reproducing a
    # run, while credentials are not.
    return {
        "scheduler_source": str(options.scheduler_source),
        "scheduler_version": options.scheduler_version,
        "dataset": str(options.dataset),
        "split": options.split,
        "profiling_database": str(options.profiling_database),
        "profiling_schema": str(options.profiling_schema),
        "model": {
            "config": str(options.model_config) if options.model_config else None,
            "base_url": options.model_base_url,
            "name": options.model_name,
        },
        "scoring": {
            "accuracy_weight": options.accuracy_weight,
            "latency_weight": options.latency_weight,
            "resource_weight": options.resource_weight,
            "latency_scale_ms": options.latency_scale_ms,
            "resource_scale_mib": options.resource_scale_mib,
        },
        "history_limit": options.history_limit,
        "generated_tool_limit": options.generated_tool_limit,
        "dag_limit": options.dag_limit,
        "trace_limit": options.trace_limit,
        "state_dir": str(options.state_dir),
    }


def run_analysis(
    options: BottleneckAnalysisOptions,
    *,
    model: Any | None = None,
) -> BottleneckAnalysisResult:
    """Run one trusted evaluation followed by SelfEvolvingDiagnosis.

    ``model`` is an explicit seam for offline tests and local debugging.  When
    omitted, the OpenAI-compatible adapter reads its token from the configured
    environment variable at call time.
    """
    options = _coerce_options(options)
    recorder = JsonlEventRecorder(Path(options.output_dir))
    recorder.emit({"type": "run_started", "config": _options_event(options)})
    try:
        candidate = _load_candidate(
            Path(options.scheduler_source), options.scheduler_version
        )
        recorder.emit(
            {
                "type": "scheduler_loaded",
                "scheduler_version": candidate.scheduler_version,
                "scheduler_source": str(options.scheduler_source),
            }
        )

        snapshot = load_profiling_database(
            Path(options.profiling_database), Path(options.profiling_schema)
        )
        recorder.emit(
            {
                "type": "snapshot_loaded",
                "snapshot_id": snapshot.snapshot_id,
                "snapshot_digest": snapshot.snapshot_digest,
                "schema_version": snapshot.schema_version,
            }
        )

        dataset = ToolCallPlanDataset.load(Path(options.dataset))
        split_traces = tuple(_split(dataset, options.split))
        traces = split_traces[: options.dag_limit]
        recorder.emit(
            {
                "type": "dataset_loaded",
                "dataset": str(options.dataset),
                "split": options.split,
                "trace_count": len(traces),
                "selected_trace_count": len(traces),
                "split_trace_count": len(split_traces),
                "dataset_trace_count": len(dataset),
            }
        )

        scoring_context = ScoringContext(
            accuracy_weight=options.accuracy_weight,
            latency_weight=options.latency_weight,
            resource_weight=options.resource_weight,
            latency_scale_ms=options.latency_scale_ms,
            resource_scale_mib=options.resource_scale_mib,
        )
        recorder.emit(
            {
                "type": "evaluation_started",
                "scheduler_version": candidate.scheduler_version,
                "split": options.split,
                "trace_count": len(traces),
                "selected_trace_count": len(traces),
            }
        )
        if options.candidate_timeout_seconds is not None:
            import eec_sched.evolution as evolution_package

            evolution_package._CANDIDATE_TIMEOUT_SECONDS = (
                options.candidate_timeout_seconds
            )
        evaluation = evaluate_scheduler_candidate(
            snapshot,
            traces,
            candidate.scheduler_version,
            SchedulerCandidateRegistry({candidate.scheduler_version: candidate}),
            scoring_context=scoring_context,
        )
        if not isinstance(
            evaluation, CandidateEvaluation
        ):  # pragma: no cover - contract guard
            raise TypeError("candidate evaluation returned an unexpected result")
        for trace in evaluation.traces:
            recorder.emit(_complete_trace_event(trace, candidate.scheduler_version))
        recorder.emit(
            {
                "type": "evaluation_completed",
                "scheduler_version": evaluation.scheduler_version,
                "candidate_score": evaluation.candidate_score,
                "trace_count": len(evaluation.traces),
                "evaluation": evaluation.concise_projection(),
            }
        )

        evidence = _diagnosis_evidence(candidate, evaluation, scoring_context)
        recorder.emit(
            {
                "type": "diagnosis_started",
                "scheduler_version": candidate.scheduler_version,
                "trace_count": len(evaluation.traces),
                "candidate_score": evaluation.candidate_score,
                "snapshot_digest": snapshot.snapshot_digest,
            }
        )
        diagnosis_model = model
        if diagnosis_model is None:
            load_dotenv(Path.cwd() / ".env")
            token = options.model_token
            if token is None:
                token = os.environ.get(options.model_token_env, "")
            diagnosis_model = OpenAICompatibleChatModel(
                base_url=options.model_base_url,
                token=token,
                model=options.model_name,
                timeout_seconds=options.model_timeout_seconds,
            )
        tool_store = DiagnosisToolStore(
            Path(options.state_dir),
            recent_limit=max(options.history_limit, options.generated_tool_limit),
        )

        def provider(document: Mapping[str, object]) -> BottleneckMetaTools:
            return BottleneckMetaTools(
                document=document,
                snapshot=snapshot,
                scoring_context=scoring_context,
            )

        diagnosis_agent = SelfEvolvingDiagnosis(
            diagnosis_model,
            tool_store=tool_store,
            tool_provider=provider,
            history_limit=options.history_limit,
            generated_tool_limit=options.generated_tool_limit,
            trace_limit=options.trace_limit,
            event_recorder=recorder.emit,
        )
        diagnosis = diagnosis_agent.diagnose(evidence)
        recorder.emit(
            {
                "type": "diagnosis_completed",
                "scheduler_version": candidate.scheduler_version,
                "result": diagnosis,
                "evaluation": evaluation.concise_projection(),
            }
        )
        recorder.emit(
            {
                "type": "run_completed",
                "scheduler_version": candidate.scheduler_version,
                "result": {
                    "diagnosis": diagnosis,
                    "evaluation": evaluation.concise_projection(),
                },
            }
        )
        result = BottleneckAnalysisResult(recorder.path, evaluation, diagnosis)
        recorder.close()
        return result
    except Exception as exc:
        # Keep the failure event as the final durable record, then propagate so
        # library callers can handle it and ``main`` can return non-zero.
        try:
            recorder.emit(
                {
                    "type": "run_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        finally:
            recorder.close()
        raise


def _coerce_options(options: BottleneckAnalysisOptions) -> BottleneckAnalysisOptions:
    if isinstance(options, BottleneckAnalysisOptions):
        coerced = options
    else:
        # A light compatibility seam for callers passing argparse.Namespace or a
        # test SimpleNamespace without making the CLI parser part of orchestration.
        values = {
            item.name: getattr(options, item.name)
            for item in fields(BottleneckAnalysisOptions)
            if hasattr(options, item.name)
        }
        coerced = BottleneckAnalysisOptions(**values)

    if coerced.model_config is None or coerced._model_config_loaded:
        return coerced

    config_values = _model_config_values(Path(coerced.model_config))
    defaults = BottleneckAnalysisOptions.__dataclass_fields__
    values = {item.name: getattr(coerced, item.name) for item in fields(coerced)}
    for key, value in config_values.items():
        field_default = defaults[key].default
        if values[key] == field_default:
            values[key] = value
    return replace(BottleneckAnalysisOptions(**values), _model_config_loaded=True)


def _load_model_config(path: Path) -> dict[str, object]:
    """Load a Hydra/OmegaConf model YAML into plain Python values.

    Model configs used by the project are deliberately small mappings, for
    example ``configs/model/local_qwen.yaml``.  Resolving through OmegaConf
    also supports the ``${oc.env:...}`` token syntax used by the other model
    configs, while the optional ``_target_`` metadata is ignored here because
    the analysis command constructs the model itself.
    """
    load_dotenv(Path.cwd() / ".env")
    try:
        loaded = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    except Exception as exc:
        raise ValueError(f"failed to load model config {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"model config {path} must contain a mapping")
    return {str(key): value for key, value in loaded.items() if key != "_target_"}


def _model_config_values(path: Path) -> dict[str, object]:
    config = _load_model_config(path)
    aliases = {
        "base_url": "model_base_url",
        "token": "model_token",
        "api_key": "model_token",
        "model": "model_name",
        "timeout_seconds": "model_timeout_seconds",
    }
    values: dict[str, object] = {}
    for key, value in config.items():
        destination = aliases.get(key)
        if destination is not None:
            values[destination] = value
    return values


def _options_from_args(args: argparse.Namespace) -> BottleneckAnalysisOptions:
    values = vars(args).copy()
    model_config = values.get("model_config")
    if model_config is not None:
        config_values = _model_config_values(Path(model_config))
        for key, value in config_values.items():
            # ``None`` is used by the parser to distinguish an omitted CLI
            # option from the dataclass default.  Explicit CLI values win.
            if values.get(key) is None:
                values[key] = value
    values.pop("model_config", None)
    values = {key: value for key, value in values.items() if value is not None}
    values["model_config"] = Path(model_config) if model_config is not None else None
    options = BottleneckAnalysisOptions(**values)
    if model_config is not None:
        options = replace(options, _model_config_loaded=True)
    return options


class _SingleValueAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):  # type: ignore[no-untyped-def]
        if getattr(namespace, self.dest, None) is not None:
            parser.error(
                f"{option_string or self.option_strings[0]} may be provided only once"
            )
        setattr(namespace, self.dest, values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scheduler-source",
        type=Path,
        required=True,
        action=_SingleValueAction,
        help="one Candidate Python file defining propose(view)",
    )
    parser.add_argument("--scheduler-version", type=int, default=1)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--split", choices=("train", "validation", "test", "all"), default="test"
    )
    parser.add_argument(
        "--profiling-database", type=Path, default=DEFAULT_PROFILING_DATABASE
    )
    parser.add_argument(
        "--profiling-schema", type=Path, default=DEFAULT_PROFILING_SCHEMA
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument(
        "--model-config",
        "--llm-config",
        type=Path,
        default=None,
        action=_SingleValueAction,
        help="YAML model config, such as configs/model/local_qwen.yaml",
    )
    parser.add_argument("--model-base-url", default=None)
    parser.add_argument("--model", dest="model_name", default=None)
    parser.add_argument("--model-token", default=None)
    parser.add_argument("--model-token-env", default=None)
    parser.add_argument("--model-timeout-seconds", type=float, default=None)
    parser.add_argument("--accuracy-weight", type=float, default=1.0 / 3.0)
    parser.add_argument("--latency-weight", type=float, default=1.0 / 3.0)
    parser.add_argument("--resource-weight", type=float, default=1.0 / 3.0)
    parser.add_argument("--latency-scale-ms", type=float, default=100.0)
    parser.add_argument("--resource-scale-mib", type=float, default=8_192.0)
    parser.add_argument("--history-limit", type=int, default=5)
    parser.add_argument("--generated-tool-limit", type=int, default=5)
    parser.add_argument(
        "--dag-limit",
        "--trace-count",
        dest="dag_limit",
        type=int,
        default=3,
        help="number of DAG traces to evaluate and provide to diagnosis (default: 3)",
    )
    parser.add_argument("--trace-limit", type=int, default=3)
    parser.add_argument("--candidate-timeout-seconds", type=float, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_analysis(_options_from_args(args))
    except Exception as exc:
        print(
            f"bottleneck analysis failed: {type(exc).__name__}: {exc}", file=sys.stderr
        )
        return 1
    print(result.output_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
