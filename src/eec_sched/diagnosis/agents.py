"""Small diagnosis adapters; tool-using agents can be added behind the same interface."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import json
import inspect
import math
from typing import Any, Callable, Literal, Mapping, Sequence, cast

from .models import DiagnosisResult
from .tool_evolution import CompositeToolSpec, DiagnosisToolStore
from .tools import TraceAnalysisTools


_AGENT_OMITTED_FIELDS = (
    "task_input",
    "dag",
    "assignments",
    "nodes",
    "transfers",
    "raw_metrics",
)


@dataclass(frozen=True)
class EmptyDiagnosis:
    """Ablation adapter that deliberately contributes no diagnostic evidence."""

    advice: str = (
        "Improve the Scheduler Candidate using the supplied evaluation evidence."
    )

    def diagnose(self, evidence: Mapping[str, object]) -> DiagnosisResult:
        return DiagnosisResult(advice=self.advice)


@dataclass(frozen=True)
class OneShotDiagnosis:
    """Adapt one model call to the structured diagnosis interface."""

    complete: Callable[[Mapping[str, object]], str]

    def diagnose(self, evidence: Mapping[str, object]) -> DiagnosisResult:
        advice = self.complete(evidence).strip()
        if not advice:
            raise ValueError("diagnosis model returned empty advice")
        return DiagnosisResult(advice=advice)


@dataclass(frozen=True)
class ReactDiagnosis:
    """Tool-using diagnosis adapter backed by the optional OpenAI Agents SDK."""

    base_url: str
    token: str
    model: str

    def diagnose(self, evidence: Mapping[str, object]) -> DiagnosisResult:
        try:
            from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool  # type: ignore[import-not-found]
            from openai import AsyncOpenAI  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "install the agentic-reflection dependency group for ReAct diagnosis"
            ) from exc

        tools = TraceAnalysisTools(evidence)

        @function_tool
        def inspect_node_dimension(
            dimension: str, trace_id: str | None = None, limit: int = 100
        ) -> str:
            """Inspect one dimension for every matching Scheduler Trace node."""
            return tools.inspect_node_dimension(dimension, trace_id, limit)

        @function_tool
        def summarize_node_dimension(
            dimension: str, group_by: str = "device_id", trace_id: str | None = None
        ) -> str:
            """Summarize one numeric node dimension by device, tool, or Configuration."""
            return tools.summarize_node_dimension(dimension, group_by, trace_id)

        @function_tool
        def compare_assignments(left_version: int, right_version: int) -> str:
            """Compare differing assignments from two Scheduler Candidate versions."""
            return tools.compare_assignments(left_version, right_version)

        client = AsyncOpenAI(base_url=self.base_url.rstrip("/"), api_key=self.token)
        agent = Agent(
            name="Scheduler bottleneck analyst",
            model=OpenAIChatCompletionsModel(model=self.model, openai_client=client),
            instructions=(
                "Use at least one supplied tool to locate an End-Edge-Cloud scheduling bottleneck. "
                "Return concise advice containing the observed evidence, the scheduling behavior "
                "to change, and a testable expected effect. Never modify evaluator rules."
            ),
            tools=[
                inspect_node_dimension,
                summarize_node_dimension,
                compare_assignments,
            ],
        )
        output = Runner.run_sync(
            agent, "Diagnose the supplied Candidate Evaluation Evidence."
        ).final_output
        advice = str(output).strip()
        if not advice:
            raise ValueError("ReAct diagnosis returned empty advice")
        return DiagnosisResult(advice=advice)


@dataclass
class SelfEvolvingDiagnosis:
    """A small tool-using diagnosis loop with persistent experiment memory.

    The class deliberately keeps orchestration here and delegates profiling,
    counterfactual evaluation, and generated-tool persistence to injected
    collaborators.  This makes the composition root the authority for the
    trusted snapshot/evaluator while keeping this adapter usable with a fake
    model and fake tools in tests.

    ``tool_provider`` may be an object with the three meta-tool methods or a
    factory accepting the current evidence. Generated tools and histories use
    the concrete, deliberately small :class:`DiagnosisToolStore` contract.
    """

    model: Any
    tool_store: DiagnosisToolStore
    tool_provider: Any = None
    history_limit: int = 5
    generated_tool_limit: int = 5
    trace_limit: int = 3
    runner: Any = None
    # Optional lifecycle sink used by the standalone bottleneck-analysis CLI.
    # The callback is deliberately structural (rather than introducing a
    # dependency on the CLI's recorder) so existing callers remain unchanged.
    event_recorder: Callable[[Mapping[str, object]], None] | None = None
    _active_evidence: Mapping[str, object] | None = field(
        default=None, init=False, repr=False
    )
    _active_provider: Any = field(default=None, init=False, repr=False)
    _tool_calls: list[object] = field(default_factory=list, init=False, repr=False)
    _session_number: int = field(default=0, init=False, repr=False)
    _active_session_id: str | None = field(default=None, init=False, repr=False)
    _round_number: int = field(default=0, init=False, repr=False)
    _active_round: int | None = field(default=None, init=False, repr=False)
    _tool_call_number: int = field(default=0, init=False, repr=False)
    _tool_failure_count: int = field(default=0, init=False, repr=False)

    TOOL_NAMES = ("profile_trace", "intervene_assignment", "validate_intervention")

    def __post_init__(self) -> None:
        if self.history_limit < 1:
            raise ValueError("history_limit must be positive")
        if self.generated_tool_limit < 1:
            raise ValueError("generated_tool_limit must be positive")
        if self.trace_limit < 1:
            raise ValueError("trace_limit must be positive")

    def diagnose(self, evidence: Mapping[str, object]) -> DiagnosisResult:
        self._session_number += 1
        self._active_session_id = f"diagnosis-session-{self._session_number}"
        self._round_number = 0
        self._active_round = None
        self._tool_call_number = 0
        self._tool_failure_count = 0
        self._emit("session_start", session_id=self._active_session_id, phase="diagnosis")
        self._emit("diagnosis_agent_started")
        try:
            # Keep the trusted provider on the complete Candidate Evaluation
            # Evidence.  ``trace_limit`` bounds only the model's initial
            # trace_index; profile_trace is intentionally higher privilege and
            # must be able to inspect every retained Trace.
            complete_evidence = dict(evidence)
            bounded_evidence = dict(evidence)
            traces = evidence.get("traces")
            if isinstance(traces, Sequence) and not isinstance(traces, str | bytes):
                bounded_evidence["traces"] = list(traces[: self.trace_limit])
            self._active_evidence = complete_evidence
            self._active_provider = (
                self.tool_provider(complete_evidence)
                if callable(self.tool_provider)
                and not isinstance(self.tool_provider, Mapping)
                else self.tool_provider
            )
            self._tool_calls = []
            history = self._history_projection(
                self._recent_history(),
                snapshot_digest=str(bounded_evidence.get("snapshot_digest") or ""),
            )
            generated = self._recent_tools()
            names = list(self.TOOL_NAMES) + [item.name for item in generated]
            agent_context = _agent_context_projection(bounded_evidence)
            payload = {
                "evidence": _jsonable(agent_context),
                "history": _jsonable(history),
                "available_tools": names,
            }
            trace_values = agent_context.get("trace_index")
            context_size = len(json.dumps(_jsonable(payload), ensure_ascii=False, separators=(",", ":")))
            self._emit(
                "agent_context_prepared",
                trace_count=(
                    len(cast(Sequence[object], trace_values))
                    if isinstance(trace_values, list | tuple)
                    else 0
                ),
                history_count=len(history),
                generated_tool_count=len(generated),
                available_tools=names,
                context_chars=context_size,
                context_schema_version=agent_context.get("context_schema_version", 1),
                omitted_fields=list(_AGENT_OMITTED_FIELDS),
                context=_jsonable(payload),
            )
            raw = self._run_agent(payload, generated)
            result = _parse_diagnosis(raw)
            self._record_history(result)
            self._maybe_generate_tool(result)
            self._emit(
                "diagnosis_agent_completed",
                diagnosis=_jsonable(result),
            )
        except Exception as exc:
            self._emit(
                "diagnosis_agent_failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            self._emit(
                "session_end",
                session_id=self._active_session_id,
                phase="diagnosis",
                status="failed",
                error_type=type(exc).__name__,
            )
            raise
        self._emit(
            "session_end",
            session_id=self._active_session_id,
            phase="diagnosis",
            status="completed",
        )
        return result

    def _recent_history(self) -> list[dict[str, object]]:
        return self.tool_store.read_recent_history(self.history_limit)

    def _recent_tools(self) -> list[CompositeToolSpec]:
        return self.tool_store.load_recent_tool_specs(self.generated_tool_limit)

    def _history_projection(
        self, rows: Sequence[Mapping[str, object]], *, snapshot_digest: str
    ) -> list[dict[str, object]]:
        """Keep prior diagnoses useful without replaying long tool results."""
        projected: list[dict[str, object]] = []
        for row in rows:
            digest = str(row.get("snapshot_digest") or "")
            diagnosis = row.get("diagnosis", {})
            projected_diagnosis: dict[str, object] = {}
            if isinstance(diagnosis, Mapping):
                for key in ("bottlenecks", "evidence", "impact_path", "confidence"):
                    value = diagnosis.get(key)
                    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
                        projected_diagnosis[key] = [str(item)[:240] for item in value[:8]]
                    elif key == "confidence" and isinstance(value, (int, float)):
                        projected_diagnosis[key] = value
            tool_calls: list[dict[str, object]] = []
            raw_calls = row.get("tool_calls", ())
            if isinstance(raw_calls, Sequence) and not isinstance(raw_calls, str | bytes):
                for call in raw_calls[:12]:
                    if not isinstance(call, Mapping):
                        continue
                    arguments = call.get("arguments", {})
                    trace_ids = _find_trace_ids(arguments)
                    tool_calls.append(
                        {
                            "tool": str(call.get("tool", call.get("name", "unknown"))),
                            "trace_ids": trace_ids,
                        }
                    )
            projected.append(
                {
                    "snapshot_compatibility": (
                        "current" if digest and digest == snapshot_digest else "stale" if digest else "unknown"
                    ),
                    "snapshot_digest": digest,
                    "diagnosis": projected_diagnosis,
                    "tool_calls": tool_calls,
                }
            )
        return projected

    def _record_history(self, result: DiagnosisResult) -> None:
        raw_digest = (
            self._active_evidence.get("snapshot_digest")
            if self._active_evidence is not None
            else ""
        )
        row = self.tool_store.append_history(
            tool_calls=tuple(self._tool_calls),
            diagnosis=result,
            snapshot_digest=str(raw_digest or ""),
        )
        self._emit(
            "history_recorded",
            snapshot_digest=str(raw_digest or ""),
            tool_call_count=len(self._tool_calls),
            history_timestamp=row.get("timestamp"),
        )

    def _run_agent(
        self, payload: Mapping[str, object], generated: Sequence[CompositeToolSpec]
    ) -> object:
        """Run Agents SDK when available, with a deterministic model fallback."""
        self._emit("agent_started", phase="diagnosis")
        # The fallback is only a test seam for small fake models. A configured
        # OpenAI-compatible model must use the tool-capable SDK.
        try:
            if self.runner is None and not all(
                isinstance(getattr(self.model, field_name, None), str)
                and getattr(self.model, field_name)
                for field_name in ("base_url", "token", "model")
            ):
                result = self._complete(payload, phase="diagnosis")
            else:
                result = self._run_openai_agents(payload, generated)
        except Exception as exc:
            self._emit(
                "agent_failed",
                phase="diagnosis",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        self._emit("agent_completed", phase="diagnosis")
        return result

    def _complete(
        self,
        payload: Mapping[str, object],
        *,
        system: str | None = None,
        phase: str = "diagnosis",
    ) -> str:
        complete = getattr(self.model, "complete", None)
        if not callable(complete):
            raise RuntimeError("self-evolving diagnosis model must expose complete()")
        effective_system = system or (
            "Diagnose the supplied scheduling evidence. You may call exactly the "
            "three analysis tools and any listed generated tools. Return JSON with "
            "bottlenecks (non-empty array), evidence (array), impact_path (array), "
            "advice (string), and confidence (number from 0 to 1)."
        )
        user_payload = dict(payload)
        round_number = self._start_round(phase)
        self._emit("model_started", phase=phase)
        self._emit_llm_request(
            round_number,
            phase=phase,
            system_prompt=effective_system,
            input_value=user_payload,
            json_output=True,
        )
        try:
            result = str(
                complete(
                    system=effective_system,
                    user=user_payload,
                    json_output=True,
                )
            )
        except Exception as exc:
            self._emit_llm_response(
                round_number,
                phase=phase,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            self._emit(
                "model_failed",
                phase=phase,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            self._end_round(round_number, phase=phase, status="failed")
            raise
        self._emit_llm_response(
            round_number, phase=phase, status="completed", output=result
        )
        self._emit("model_completed", phase=phase, output=result)
        self._end_round(round_number, phase=phase, status="completed")
        return result

    def _run_openai_agents(
        self, payload: Mapping[str, object], generated: Sequence[CompositeToolSpec]
    ) -> object:
        try:
            from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool  # type: ignore[import-not-found]
            from agents.lifecycle import RunHooksBase  # type: ignore[import-not-found]
            from openai import AsyncOpenAI  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "install the agentic-reflection dependency group for self_evolving diagnosis"
            ) from exc
        if self.runner is not None:
            runner = self.runner
        else:
            base_url, token, model_name = (
                getattr(self.model, "base_url", None),
                getattr(self.model, "token", None),
                getattr(self.model, "model", None),
            )
            if not all(
                isinstance(item, str) and item for item in (base_url, token, model_name)
            ):
                raise RuntimeError("OpenAI Agents requires base_url, token and model")
            base_url = cast(str, base_url)
            token = cast(str, token)
            model_name = cast(str, model_name)
            client = AsyncOpenAI(base_url=base_url.rstrip("/"), api_key=token)
            runner = Runner
            model = OpenAIChatCompletionsModel(model=model_name, openai_client=client)

        @function_tool
        def profile_trace(
            trace_id: str,
            focus: Literal[
                "latency", "resource", "quality", "communication", "placement"
            ]
            | None = None,
            top_k: int = 5,
        ) -> str:
            """Analyze one complete Trace and return a bounded TraceProfile.

            The profile joins observed execution evidence with the immutable
            Profiling Database Snapshot and is the required first diagnostic
            step. Use focus for a second, targeted profile when needed.
            """
            return self._invoke_meta(
                "profile_trace",
                trace_id=trace_id,
                focus=focus,
                top_k=top_k,
            )

        @function_tool
        def intervene_assignment(trace_id: str, assignment_patch: str) -> str:
            return self._invoke_meta(
                "intervene_assignment",
                trace_id=trace_id,
                assignment_patch=self._parse_object(assignment_patch, {}),
            )

        @function_tool
        def validate_intervention(interventions: str) -> str:
            return self._invoke_meta(
                "validate_intervention",
                interventions=self._parse_object(interventions, {}),
            )

        wrapped = [profile_trace, intervene_assignment, validate_intervention]
        for item in generated:
            callable_item = self._make_composite_callable(item)
            name = item.name
            description = item.description or "Generated diagnosis tool"
            if item.parameters:
                description += "; arguments JSON keys: " + ", ".join(item.parameters)

            def _make_generated_tool(
                tool_name: str, tool_description: str, function: Callable[..., object]
            ):
                @function_tool(
                    name_override=tool_name,
                    description_override=tool_description,
                )
                def generated_tool(arguments: str = "") -> str:
                    self._emit(
                        "generated_tool_started",
                        tool=tool_name,
                        arguments=arguments,
                    )
                    try:
                        value = json.loads(arguments) if arguments else {}
                        if not isinstance(value, Mapping):
                            raise ValueError(
                                "generated tool arguments must be a JSON object"
                            )
                    except (
                        Exception
                    ) as exc:  # generated-tool failures are evidence for the Agent
                        call_id = self._start_tool(
                            tool_name, tool_kind="evolved", arguments=arguments
                        )
                        self._tool_failure_count += 1
                        self._tool_calls.append(
                            {
                                "call_id": call_id,
                                "tool": tool_name,
                                "arguments": arguments,
                                "result": f"{type(exc).__name__}: {exc}",
                                "status": "failed",
                            }
                        )
                        self._finish_tool(
                            call_id,
                            name=tool_name,
                            tool_kind="evolved",
                            status="failed",
                            error=f"{type(exc).__name__}: {exc}",
                        )
                        self._emit(
                            "generated_tool_completed",
                            tool=tool_name,
                            arguments=arguments,
                            status="failed",
                            error_type=type(exc).__name__,
                            error=str(exc),
                        )
                        return f"generated tool error: {type(exc).__name__}: {exc}"
                    try:
                        result = function(**value)
                    except Exception as exc:  # generated-tool failures are evidence for the Agent
                        self._emit(
                            "generated_tool_completed",
                            tool=tool_name,
                            arguments=value,
                            status="failed",
                            error_type=type(exc).__name__,
                            error=str(exc),
                        )
                        return f"generated tool error: {type(exc).__name__}: {exc}"
                    self._emit(
                        "generated_tool_completed",
                        tool=tool_name,
                        arguments=value,
                        result=result,
                    )
                    return str(result)

                return generated_tool

            wrapped.append(_make_generated_tool(name, description, callable_item))

        instructions = (
            "You are a self-evolving scheduling bottleneck analyst. The initial context includes "
            "the complete source code of the Scheduler Candidate being diagnosed plus a "
            "lightweight trace_index, not raw execution evidence. Use the source code to "
            "understand the strategy's scheduling decisions before investigating their effects. "
            "First choose one main Trace "
            "and call profile_trace(trace_id) at least once; it joins complete Candidate "
            "Evaluation Evidence with the Profiling Database Snapshot and Scoring Context. "
            "Use a focused profile_trace call with focus=latency, resource, quality, "
            "communication, or placement only when the combined profile needs more detail. "
            "Use concrete node_id values from the TraceProfile for intervene_assignment, then "
            "validate_intervention for explicit per-trace patches. Do not state an unverified "
            "association as causal. Stop when evidence is sufficient and return only the "
            "requested JSON diagnosis; bottlenecks must contain one item."
        )
        agent_kwargs = {
            "name": "Self-evolving scheduler bottleneck analyst",
            "instructions": instructions,
            "tools": wrapped,
        }
        if self.runner is None:
            agent_kwargs["model"] = model
        agent = Agent(**agent_kwargs)
        call = getattr(runner, "run_sync", None)
        if not callable(call):
            raise RuntimeError("Agents Runner has no run_sync")

        owner = self

        class _LifecycleHooks(RunHooksBase):
            async def on_llm_start(self, context, agent, system_prompt, input_items):  # type: ignore[no-untyped-def]
                round_number = owner._start_round("diagnosis")
                owner._emit_llm_request(
                    round_number,
                    phase="diagnosis",
                    system_prompt=system_prompt,
                    input_value=input_items,
                )

            async def on_llm_end(self, context, agent, response):  # type: ignore[no-untyped-def]
                round_number = owner._active_round or owner._round_number
                owner._emit_llm_response(
                    round_number,
                    phase="diagnosis",
                    status="completed",
                    output=response,
                )
                owner._end_active_round(phase="diagnosis", status="completed")

        hooks = _LifecycleHooks()
        self._emit("model_started", phase="diagnosis")
        before_rounds = self._round_number
        supports_hooks = False
        try:
            parameters = inspect.signature(call).parameters
            supports_hooks = "hooks" in parameters or any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
        except (TypeError, ValueError):
            # Some test doubles and extension runners do not expose a Python
            # signature.  The SDK's concrete Runner does, so omit hooks only
            # when introspection genuinely cannot decide.
            supports_hooks = False
        run_kwargs = {"hooks": hooks} if supports_hooks else {}
        if not supports_hooks:
            round_number = self._start_round("diagnosis")
            self._emit_llm_request(
                round_number,
                phase="diagnosis",
                system_prompt=instructions,
                input_value=payload,
            )
        try:
            result = call(agent, json.dumps(payload, ensure_ascii=False), **run_kwargs)
        except Exception as exc:
            if self._active_round is not None:
                self._emit_llm_response(
                    self._active_round,
                    phase="diagnosis",
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            self._end_active_round(phase="diagnosis", status="failed")
            self._emit(
                "model_failed",
                phase="diagnosis",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        if supports_hooks and self._round_number == before_rounds:
            # Keep old/simple runner doubles observable even if they accept a
            # hooks keyword but do not invoke it.
            number = self._start_round("diagnosis")
            self._emit_llm_request(
                number,
                phase="diagnosis",
                system_prompt=instructions,
                input_value=payload,
            )
            self._emit_llm_response(
                number,
                phase="diagnosis",
                status="completed",
                output=getattr(result, "final_output", result),
            )
            self._end_round(number, phase="diagnosis", status="completed")
        elif not supports_hooks:
            self._emit_llm_response(
                self._active_round or self._round_number,
                phase="diagnosis",
                status="completed",
                output=getattr(result, "final_output", result),
            )
            self._end_active_round(phase="diagnosis", status="completed")
        output = getattr(result, "final_output", result)
        if self.runner is None and not any(
            isinstance(call_item, Mapping)
            and call_item.get("tool") == "profile_trace"
            for call_item in self._tool_calls
        ):
            raise RuntimeError("diagnosis agent must call profile_trace before concluding")
        self._emit("model_completed", phase="diagnosis", output=output)
        return output

    def _make_composite_callable(
        self, spec: CompositeToolSpec
    ) -> Callable[..., object]:
        """Bind one persisted spec without sharing the loop's closure cell."""

        def call(**kwargs: object) -> object:
            return self._execute_evolved_tool(
                spec.name,
                kwargs,
                lambda: self.tool_store.execute(
                    spec,
                    kwargs,
                    lambda tool, arguments: self._invoke_meta(
                        tool, **dict(arguments)
                    ),
                ),
            )

        return call

    def _next_tool_call_id(self) -> str:
        self._tool_call_number += 1
        session = self._active_session_id or "diagnosis-session-0"
        return f"{session}:tool-{self._tool_call_number}"

    def _start_tool(
        self, name: str, *, tool_kind: str, arguments: object
    ) -> str:
        call_id = self._next_tool_call_id()
        self._emit(
            "tool_call",
            call_id=call_id,
            tool=name,
            name=name,
            tool_kind=tool_kind,
            arguments=arguments,
        )
        return call_id

    def _finish_tool(
        self,
        call_id: str,
        *,
        name: str,
        tool_kind: str,
        status: str,
        result: object = None,
        error: str | None = None,
    ) -> None:
        result_event: dict[str, object] = {
            "call_id": call_id,
            "tool": name,
            "name": name,
            "tool_kind": tool_kind,
            "status": status,
        }
        if status == "success":
            result_event["result"] = result
        elif error is not None:
            result_event["error"] = error
        self._emit("tool_result", **result_event)
        evaluation_event: dict[str, object] = {
            "call_id": call_id,
            "tool": name,
            "name": name,
            "tool_kind": tool_kind,
            "status": status,
            "execution_outcome": status,
        }
        if error is not None:
            evaluation_event["error"] = error
        self._emit("tool_evaluation", **evaluation_event)

    def _execute_evolved_tool(
        self,
        name: str,
        arguments: Mapping[str, object],
        operation: Callable[[], object],
    ) -> object:
        call_id = self._start_tool(
            name, tool_kind="evolved", arguments=arguments
        )
        failures_before = self._tool_failure_count
        try:
            result = operation()
            status = "failed" if self._tool_failure_count > failures_before else "success"
            self._tool_calls.append(
                {
                    "call_id": call_id,
                    "tool": name,
                    "arguments": arguments,
                    "result": result,
                    "status": status,
                }
            )
            self._finish_tool(
                call_id,
                name=name,
                tool_kind="evolved",
                status=status,
                result=result,
                error=("one or more composite steps failed" if status == "failed" else None),
            )
            return result
        except Exception as exc:
            self._tool_failure_count += 1
            self._tool_calls.append(
                {
                    "call_id": call_id,
                    "tool": name,
                    "arguments": arguments,
                    "result": f"{type(exc).__name__}: {exc}",
                    "status": "failed",
                }
            )
            self._finish_tool(
                call_id,
                name=name,
                tool_kind="evolved",
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise

    def _invoke_meta(self, name: str, **kwargs: object) -> str:
        call_id = self._start_tool(name, tool_kind="meta", arguments=kwargs)
        self._emit("meta_tool_started", tool=name, arguments=kwargs)
        try:
            provider = self._active_provider
            callable_item = None
            if isinstance(provider, Mapping):
                callable_item = provider.get(name)
            elif provider is not None:
                callable_item = getattr(provider, name, None)
            if callable_item is None:
                raise RuntimeError(f"meta tool {name!r} is not configured")
            if callable(callable_item):
                value = callable_item(**kwargs)
                self._tool_calls.append(
                    {
                        "call_id": call_id,
                        "tool": name,
                        "arguments": kwargs,
                        "result": value,
                        "status": "success",
                    }
                )
                self._emit(
                    "meta_tool_completed",
                    tool=name,
                    arguments=kwargs,
                    result=value,
                )
                self._finish_tool(
                    call_id,
                    name=name,
                    tool_kind="meta",
                    status="success",
                    result=value,
                )
                return str(value)
            value = str(callable_item)
            self._emit("meta_tool_completed", tool=name, arguments=kwargs, result=value)
            self._tool_calls.append(
                {
                    "call_id": call_id,
                    "tool": name,
                    "arguments": kwargs,
                    "result": value,
                    "status": "success",
                }
            )
            self._finish_tool(
                call_id,
                name=name,
                tool_kind="meta",
                status="success",
                result=value,
            )
            return value
        except Exception as exc:
            self._tool_failure_count += 1
            self._tool_calls.append(
                {
                    "call_id": call_id,
                    "tool": name,
                    "arguments": kwargs,
                    "result": f"{type(exc).__name__}: {exc}",
                    "status": "failed",
                }
            )
            self._emit(
                "meta_tool_completed",
                tool=name,
                arguments=kwargs,
                status="failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            self._finish_tool(
                call_id,
                name=name,
                tool_kind="meta",
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            return f"{name} error: {type(exc).__name__}: {exc}"

    def _start_round(self, phase: str) -> int:
        self._round_number += 1
        self._active_round = self._round_number
        self._emit(
            "round_start",
            session_id=self._active_session_id,
            round_number=self._round_number,
            phase=phase,
        )
        return self._round_number

    def _end_round(self, round_number: int, *, phase: str, status: str) -> None:
        self._emit(
            "round_end",
            session_id=self._active_session_id,
            round_number=round_number,
            phase=phase,
            status=status,
        )
        if self._active_round == round_number:
            self._active_round = None

    def _end_active_round(self, *, phase: str, status: str) -> None:
        round_number = self._active_round
        if round_number is not None:
            self._end_round(round_number, phase=phase, status=status)

    def _emit_llm_request(
        self,
        round_number: int,
        *,
        phase: str,
        system_prompt: object,
        input_value: object,
        json_output: bool | None = None,
    ) -> None:
        payload: dict[str, object] = {
            "session_id": self._active_session_id,
            "round_number": round_number,
            "phase": phase,
            "system_prompt": _jsonable(system_prompt),
            "input": _jsonable(input_value),
        }
        if json_output is not None:
            payload["json_output"] = json_output
        self._emit("llm_request", **payload)

    def _emit_llm_response(
        self,
        round_number: int,
        *,
        phase: str,
        status: str,
        output: object = None,
        error: str | None = None,
    ) -> None:
        payload: dict[str, object] = {
            "session_id": self._active_session_id,
            "round_number": round_number,
            "phase": phase,
            "status": status,
        }
        if status == "completed":
            payload["output"] = _jsonable(output)
        elif error is not None:
            payload["error"] = error
        self._emit("llm_response", **payload)

    @staticmethod
    def _parse_object(value: str | Mapping[str, object], default: object) -> object:
        if isinstance(value, Mapping):
            return value
        if not value:
            return default
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default

    def _maybe_generate_tool(self, result: DiagnosisResult) -> None:
        self._emit("tool_evolution_started")
        try:
            history = self._history_projection(
                self._recent_history(),
                snapshot_digest=str(
                    (self._active_evidence or {}).get("snapshot_digest") or ""
                ),
            )
            raw = self._complete(
                {
                    "recent_history": _jsonable(history),
                    "latest_diagnosis": _jsonable(result),
                    "allowed_tools": list(self.TOOL_NAMES),
                },
                system=(
                    "Review the recent diagnosis history. Decide whether one reusable composite "
                    'tool is justified. Return JSON either {"tool": null} or {"tool": '
                    '{"name": string, "description": string, "parameters": array, '
                    '"steps": array}}. Steps may only call the three allowed tools. Keep it '
                    "small and parameterize trace and node identifiers."
                ),
                phase="tool_evolution",
            )
            value = _parse_json(raw)
            spec = value.get("tool") if isinstance(value, Mapping) else None
            if isinstance(spec, Mapping) and spec.get("name") and spec.get("steps"):
                saved = self.tool_store.save_tool_spec(spec)
                self._emit(
                    "tool_evolution_completed",
                    status="completed",
                    tool_name=saved.name,
                    parameter_count=len(saved.parameters),
                    step_count=len(saved.steps),
                )
            else:
                self._emit(
                    "tool_evolution_completed", status="skipped", reason="no_tool"
                )
        except Exception as exc:
            self._emit(
                "tool_evolution_completed",
                status="failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            # Tool evolution is opportunistic; a malformed proposal must not
            # make the trusted diagnosis disappear.
            return

    def _emit(self, event_type: str, **payload: object) -> None:
        """Best-effort lifecycle notification; observers cannot break diagnosis."""
        recorder = self.event_recorder
        if recorder is None:
            return
        try:
            recorder({"type": event_type, **payload})
        except Exception:
            # A debugging sink is intentionally non-authoritative.  In
            # particular, a full disk or a recorder serialization bug must not
            # change the diagnosis result.
            return


def _agent_context_projection(evidence: Mapping[str, object]) -> dict[str, object]:
    """Project complete trusted evidence into the model's initial index.

    The provider receives the original evidence.  This projection is the only
    evidence representation put in the first model request or its lifecycle
    audit event.  The complete Scheduler Candidate source is included because
    bottleneck analysis must be able to connect observed outcomes to the
    strategy's actual decisions; raw DAG and execution details remain available
    through ``profile_trace`` without diluting the initial context.
    """
    context: dict[str, object] = {
        "context_schema_version": 2,
        "scheduler_version": evidence.get("scheduler_version"),
        "snapshot_digest": evidence.get("snapshot_digest"),
        "trace_index": [],
    }
    source_code = evidence.get("source_code")
    if isinstance(source_code, str):
        context["source_code"] = source_code
    strategy_description = evidence.get("strategy_description")
    if isinstance(strategy_description, str) and strategy_description:
        context["strategy_description"] = strategy_description
    for name in ("candidate_score", "scoring_context"):
        value = evidence.get(name)
        if value is not None:
            context[name] = _scalar_or_mapping(value)
    traces = evidence.get("traces")
    if not isinstance(traces, Sequence) or isinstance(traces, str | bytes):
        return context
    index: list[dict[str, object]] = []
    for trace in traces:
        if not isinstance(trace, Mapping):
            continue
        metrics = trace.get("metrics", {})
        if not isinstance(metrics, Mapping):
            metrics = {}
        row: dict[str, object] = {
            "trace_id": trace.get("trace_id"),
            "status": trace.get("status"),
            "node_count": _trace_node_count(trace),
            "edge_count": _trace_edge_count(trace),
        }
        for metric in ("accuracy", "latency", "resource", "composite_score"):
            value = metrics.get(metric)
            if _finite_number(value):
                row[metric] = value
        index.append(row)
    context["trace_index"] = index
    return context


def _trace_node_count(trace: Mapping[str, object]) -> int:
    dag = trace.get("dag")
    if isinstance(dag, Mapping) and isinstance(dag.get("nodes"), Sequence) and not isinstance(dag.get("nodes"), str | bytes):
        return len(dag["nodes"])
    nodes = trace.get("nodes")
    return len(nodes) if isinstance(nodes, Sequence) and not isinstance(nodes, str | bytes) else 0


def _trace_edge_count(trace: Mapping[str, object]) -> int:
    dag = trace.get("dag")
    nodes = dag.get("nodes") if isinstance(dag, Mapping) else None
    if not isinstance(nodes, Sequence) or isinstance(nodes, str | bytes):
        return 0
    count = 0
    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, Mapping):
            continue
        for source in inputs.values():
            if isinstance(source, Mapping) and source.get("kind") == "node":
                count += 1
    return count


def _scalar_or_mapping(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): child
            for key, child in value.items()
            if isinstance(child, (str, int, float, bool)) or child is None
        }
    return value if isinstance(value, (str, int, float, bool)) or value is None else str(value)


def _finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _find_trace_ids(value: object) -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) == "trace_id" and isinstance(child, str):
                found.append(child)
            else:
                found.extend(_find_trace_ids(child))
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for child in value:
            found.extend(_find_trace_ids(child))
    return list(dict.fromkeys(found))[:8]


def _parse_json(value: object) -> object:
    if isinstance(value, Mapping):
        return value
    text = str(value).strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.startswith("json"):
            text = text[4:].lstrip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("self-evolving diagnosis model returned invalid JSON") from exc


def _parse_diagnosis(value: object) -> DiagnosisResult:
    parsed = _parse_json(value)
    if not isinstance(parsed, Mapping):
        parsed = {"advice": str(parsed)}
    advice = str(parsed.get("advice", "")).strip()
    if not advice:
        raise ValueError("self-evolving diagnosis returned empty advice")

    def _strings(key: str) -> tuple[str, ...]:
        item = parsed.get(key, ())
        if isinstance(item, str):
            return (item,)
        if isinstance(item, Sequence):
            return tuple(str(value) for value in item)
        return ()

    bottlenecks = _strings("bottlenecks")
    evidence = _strings("evidence")
    impact_path = _strings("impact_path")
    if not bottlenecks or not all(item.strip() for item in bottlenecks):
        raise ValueError("self-evolving diagnosis must identify a bottleneck")
    if not evidence or not all(item.strip() for item in evidence):
        raise ValueError("self-evolving diagnosis must include evidence")
    if not impact_path or not all(item.strip() for item in impact_path):
        raise ValueError("self-evolving diagnosis must include an impact_path")
    confidence = parsed.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, int | float | str):
        raise ValueError("self-evolving diagnosis confidence must be numeric")
    try:
        confidence_value = float(confidence)
    except ValueError:
        raise ValueError("self-evolving diagnosis confidence must be numeric") from None
    if not math.isfinite(confidence_value) or not 0.0 <= confidence_value <= 1.0:
        raise ValueError("self-evolving diagnosis confidence must be between 0 and 1")
    return DiagnosisResult(
        advice=advice,
        bottlenecks=bottlenecks,
        evidence=evidence,
        confidence=confidence_value,
        impact_path=impact_path,
    )


def _jsonable(value: object) -> object:
    if isinstance(value, DiagnosisResult):
        return {
            "advice": value.advice,
            "bottlenecks": list(value.bottlenecks),
            "evidence": list(value.evidence),
            "confidence": value.confidence,
            "impact_path": list(value.impact_path),
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__dict__"):
        return {
            str(key): _jsonable(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
