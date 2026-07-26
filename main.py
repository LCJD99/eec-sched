"""Run a real image-captioning then summarization scheduling demonstration."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlretrieve

from eec_sched import (
    AccuracyProfile,
    FakePlannerClient,
    FinalOutput,
    InMemoryProfileRepository,
    InputSource,
    LatencyProfile,
    PlanningRequest,
    ToolCallPlan,
    ToolNode,
    ToolRegistry,
    execute_request,
)
from eec_sched.huggingface import TransformersPipelineRunner, representative_tool_specs
from eec_sched.profiles import profile_warm_latency

IMAGE_URL = "https://ankur3107.github.io/assets/images/image-captioning-example.png"
IMAGE_PATH = Path(".cache/image-captioning-example.png")


def load_demo_image() -> object:
    from PIL import Image

    IMAGE_PATH.parent.mkdir(exist_ok=True)
    if not IMAGE_PATH.exists():
        urlretrieve(IMAGE_URL, IMAGE_PATH)
    with Image.open(IMAGE_PATH) as source:
        return source.convert("RGB")


def main() -> None:
    registry = ToolRegistry()
    specs = {spec.tool_id: spec for spec in representative_tool_specs()}
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    pipeline_device = 0 if device == "cuda" else -1
    caption_runner = TransformersPipelineRunner("image-to-text", "image", "text", pipeline_device)
    summarizer_runner = TransformersPipelineRunner("summarization", "text", "text", pipeline_device)
    registry.register(specs["image_caption"], lambda: caption_runner)
    registry.register(specs["summarize"], lambda: summarizer_runner)
    image = load_demo_image()
    caption_config = specs["image_caption"].configurations[0]
    summary_config = specs["summarize"].configurations[0]
    caption_latency = profile_warm_latency("image_caption", "nlpconnect/vit-gpt2-image-captioning", caption_config, device, caption_runner, {"image": image}, samples=3)
    caption = caption_runner.run({"image": image}, caption_config)["text"]
    summary_latency = profile_warm_latency("summarize", "facebook/bart-large-cnn", summary_config, device, summarizer_runner, {"text": caption}, samples=3)
    profiles = InMemoryProfileRepository(
        accuracy=(
            AccuracyProfile("image_caption", "default", 1.0, "nlpconnect/vit-gpt2-image-captioning", dict(caption_config.parameters)),
            AccuracyProfile("summarize", "default", 1.0, "facebook/bart-large-cnn", dict(summary_config.parameters)),
        ),
        latency=(caption_latency, summary_latency),
    )
    plan = ToolCallPlan(
        nodes=(
            ToolNode("caption", "image_caption", {"image": InputSource.request("image")}),
            ToolNode("summary", "summarize", {"text": InputSource.node("caption", "text")}),
        ),
        final_outputs=(FinalOutput("summary", "text"),),
    )
    result = execute_request(
        PlanningRequest("Caption this image, then summarize the caption.", {"image": image}, 1.0, (caption_latency.p95_ms + summary_latency.p95_ms) * 1.1, 0.5),
        registry=registry,
        planner=FakePlannerClient([plan]),
        profiles=profiles,
        device=device,
    )
    print(json.dumps({
        "status": result.status,
        "device": device,
        "outputs": result.outputs,
        "selected_configurations": {node.node_id: node.configuration_id for node in result.schedule.nodes} if result.schedule else {},
        "predicted_accuracy": result.schedule.predicted_accuracy if result.schedule else None,
        "predicted_latency_ms": result.schedule.predicted_latency_ms if result.schedule else None,
        "utility": result.schedule.utility if result.schedule else None,
    }, indent=2))


if __name__ == "__main__":
    main()
