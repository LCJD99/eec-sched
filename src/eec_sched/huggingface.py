"""Optional real Hugging Face runners with PIL/string port boundaries."""

from __future__ import annotations

from typing import Any, Callable, Mapping, cast

from .domain import Configuration, Port, ToolRegistry, ToolRunner, ToolSpec
from .mnms_tools import mnms_tool_specs, register_mnms_tools


class TransformersPipelineRunner(ToolRunner):
    """Lazily loads a transformers pipeline, keeping metadata-only planning fast."""

    def __init__(self, task: str, input_port: str, output_port: str, device: int = -1, pipeline_factory: Callable[..., Any] | None = None) -> None:
        self.task, self.input_port, self.output_port, self.device, self._factory = task, input_port, output_port, device, pipeline_factory
        self._pipelines: dict[str, Any] = {}

    def prepare(self, configuration: Configuration) -> None:
        if configuration.configuration_id not in self._pipelines:
            factory = self._factory
            if factory is None:
                from transformers import pipeline  # type: ignore[import-not-found]
                factory = cast(Callable[..., Any], pipeline)
            model = configuration.parameters.get("model")
            self._pipelines[configuration.configuration_id] = factory(self.task, model=model, device=self.device)

    def run(self, inputs: Mapping[str, object], configuration: Configuration) -> Mapping[str, object]:
        inference_kwargs = configuration.parameters.get("inference_kwargs", {})
        result = self._pipelines[configuration.configuration_id](inputs[self.input_port], **inference_kwargs)
        if isinstance(result, list) and result:
            result = result[0]
        if isinstance(result, dict):
            for key in ("generated_text", "summary_text", "text", "image"):
                if key in result:
                    result = result[key]
                    break
        if not isinstance(result, str) and self.output_port == "text":
            result = str(result)
        return {self.output_port: result}


def representative_tool_specs() -> tuple[ToolSpec, ...]:
    """Compatibility name for the complete, reference-aligned MnMS catalog."""
    return mnms_tool_specs()


def register_representative_huggingface_tools(registry: ToolRegistry, device: int = -1) -> None:
    """Compatibility registration entry point for all MnMS reference tools."""
    register_mnms_tools(registry, "cpu" if device == -1 else f"cuda:{device}")
