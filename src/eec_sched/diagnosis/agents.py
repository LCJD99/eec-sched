"""Small diagnosis adapters; tool-using agents can be added behind the same interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from .models import DiagnosisResult
from .tools import TraceAnalysisTools


@dataclass(frozen=True)
class EmptyDiagnosis:
    """Ablation adapter that deliberately contributes no diagnostic evidence."""

    advice: str = "Improve the Scheduler Candidate using the supplied evaluation evidence."

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
            raise RuntimeError("install the agentic-reflection dependency group for ReAct diagnosis") from exc

        tools = TraceAnalysisTools(evidence)

        @function_tool
        def inspect_node_dimension(dimension: str, trace_id: str | None = None, limit: int = 100) -> str:
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
            tools=[inspect_node_dimension, summarize_node_dimension, compare_assignments],
        )
        output = Runner.run_sync(agent, "Diagnose the supplied Candidate Evaluation Evidence.").final_output
        advice = str(output).strip()
        if not advice:
            raise ValueError("ReAct diagnosis returned empty advice")
        return DiagnosisResult(advice=advice)
