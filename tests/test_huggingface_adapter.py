from __future__ import annotations

from eec_sched import Configuration
from eec_sched.huggingface import TransformersPipelineRunner


def test_pipeline_runner_preserves_model_device_and_generation_configuration() -> None:
    captured: dict[str, object] = {}

    class FakePipeline:
        def __call__(self, value: object, **kwargs: object) -> list[dict[str, str]]:
            captured["input"] = value
            captured["inference_kwargs"] = kwargs
            return [{"generated_text": "caption"}]

    def factory(task: str, model: str, device: int) -> FakePipeline:
        captured.update(task=task, model=model, device=device)
        return FakePipeline()

    configuration = Configuration("default", {"model": "example/model", "inference_kwargs": {"max_new_tokens": 12}})
    runner = TransformersPipelineRunner("image-to-text", "image", "text", device=0, pipeline_factory=factory)
    runner.prepare(configuration)

    assert runner.run({"image": "image-value"}, configuration) == {"text": "caption"}
    assert captured == {"task": "image-to-text", "model": "example/model", "device": 0, "input": "image-value", "inference_kwargs": {"max_new_tokens": 12}}
