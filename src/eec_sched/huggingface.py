"""Optional real Hugging Face runners with PIL/string port boundaries."""

from __future__ import annotations

from typing import Any, Callable, Mapping, cast

from .domain import Configuration, Port, ToolRegistry, ToolRunner, ToolSpec


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
    """Three concrete, optional adapters; models can be profiled independently."""
    return (
        ToolSpec("image_caption", "Describe an image.", {"image": Port("image", "image")}, {"text": Port("text", "text")}, (Configuration("default", {"model": "nlpconnect/vit-gpt2-image-captioning", "inference_kwargs": {"max_new_tokens": 32}}),), True, 0.0),
        ToolSpec("summarize", "Summarize text.", {"text": Port("text", "text")}, {"text": Port("text", "text")}, (Configuration("default", {"model": "facebook/bart-large-cnn", "inference_kwargs": {"min_length": 1, "max_length": 32, "do_sample": False}}),), True, 0.0),
        ToolSpec("super_resolve", "Upscale an image.", {"image": Port("image", "image")}, {"image": Port("image", "image")}, (Configuration("base", {"model": "caidas/swin2SR-classical-sr-x2-64"}),), None, None),
    )


def register_representative_huggingface_tools(registry: ToolRegistry, device: int = -1) -> None:
    """Register executable image-captioning, summarization, and upscaling tools."""
    task_by_tool = {"image_caption": ("image-to-text", "image", "text"), "summarize": ("summarization", "text", "text"), "super_resolve": ("image-to-image", "image", "image")}
    for spec in representative_tool_specs():
        task, input_port, output_port = task_by_tool[spec.tool_id]
        registry.register(spec, lambda task=task, input_port=input_port, output_port=output_port: TransformersPipelineRunner(task, input_port, output_port, device=device))
