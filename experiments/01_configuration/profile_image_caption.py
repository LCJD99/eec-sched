"""Profile image-captioning configuration trade-offs on one local image.

This experiment uses one-factor-at-a-time configurations: all settings other
than the parameter being studied remain at their baseline values.  It records
the generated captions as qualitative evidence; this is deliberately not an
accuracy evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import transformers
from PIL import Image
from transformers import pipeline


MODEL_ID = "nlpconnect/vit-gpt2-image-captioning"
BASELINE = {"image_size": 224, "num_beams": 1, "max_new_tokens": 32}
CONFIGURATIONS = (
    ("image_size_224", {**BASELINE, "image_size": 224}),
    ("image_size_384", {**BASELINE, "image_size": 384}),
    ("image_size_512", {**BASELINE, "image_size": 512}),
    ("num_beams_1", {**BASELINE, "num_beams": 1}),
    ("num_beams_2", {**BASELINE, "num_beams": 2}),
    ("num_beams_4", {**BASELINE, "num_beams": 4}),
    ("max_new_tokens_16", {**BASELINE, "max_new_tokens": 16}),
    ("max_new_tokens_32", {**BASELINE, "max_new_tokens": 32}),
    ("max_new_tokens_64", {**BASELINE, "max_new_tokens": 64}),
)


@dataclass(frozen=True)
class Measurement:
    configuration_id: str
    varied_parameter: str
    parameters: dict[str, int]
    generated_captions: list[str]
    latency_ms: list[float]
    p50_ms: float
    p95_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=Path("img1.jpg"))
    parser.add_argument("--output", type=Path, default=Path("experiments/01_configuration/results.json"))
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def selected_device(requested: str) -> tuple[str, int]:
    if requested == "cpu":
        return "cpu", -1
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()):
        return "cuda", 0
    return "cpu", -1


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def prepared_image(image: Image.Image, image_size: int) -> Image.Image:
    """Make image_size part of measured preprocessing, preserving RGB input."""
    return image.convert("RGB").resize((image_size, image_size), Image.Resampling.LANCZOS)


def caption(captioner: Any, image: Image.Image, parameters: dict[str, int]) -> str:
    result = captioner(
        prepared_image(image, parameters["image_size"]),
        max_new_tokens=parameters["max_new_tokens"],
        generate_kwargs={"num_beams": parameters["num_beams"], "do_sample": False},
    )
    return str(result[0]["generated_text"])


def percentile_95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(len(ordered) * 0.95) - 1]


def varied_parameter(parameters: dict[str, int]) -> str:
    changed = [key for key, value in parameters.items() if BASELINE[key] != value]
    return changed[0] if changed else "baseline"


def measure(captioner: Any, image: Image.Image, device: str, configuration_id: str, parameters: dict[str, int], samples: int, warmup: int) -> Measurement:
    for _ in range(warmup):
        caption(captioner, image, parameters)
    synchronize(device)

    latencies: list[float] = []
    captions: list[str] = []
    for _ in range(samples):
        synchronize(device)
        started = time.perf_counter()
        captions.append(caption(captioner, image, parameters))
        synchronize(device)
        latencies.append((time.perf_counter() - started) * 1000)

    return Measurement(
        configuration_id=configuration_id,
        varied_parameter=varied_parameter(parameters),
        parameters=parameters,
        generated_captions=captions,
        latency_ms=latencies,
        p50_ms=statistics.median(latencies),
        p95_ms=percentile_95(latencies),
    )


def main() -> None:
    args = parse_args()
    if args.samples < 1 or args.warmup < 0:
        raise ValueError("--samples must be at least 1 and --warmup cannot be negative")
    if not args.image.is_file():
        raise FileNotFoundError(f"Input image does not exist: {args.image}")

    device, pipeline_device = selected_device(args.device)
    image = Image.open(args.image)
    captioner = pipeline("image-to-text", model=MODEL_ID, device=pipeline_device)
    measurements = [
        measure(captioner, image, device, configuration_id, parameters, args.samples, args.warmup)
        for configuration_id, parameters in CONFIGURATIONS
    ]

    payload = {
        "experiment": "image-captioning-configuration-profile",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model_id": MODEL_ID,
        "input_image": str(args.image),
        "input_image_size": {"width": image.width, "height": image.height},
        "device": device,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "samples_per_configuration": args.samples,
        "warmup_runs_per_configuration": args.warmup,
        "baseline_parameters": BASELINE,
        "method_notes": [
            "Each configuration changes at most one parameter from the baseline.",
            "Latency includes the experiment's RGB conversion and resize, pipeline preprocessing, generation, and postprocessing; model loading is excluded.",
            "image_size is an external pre-resize. This ViT-GPT2 model's image processor may resize it again to its model-required resolution, so it is not evidence that encoder resolution changed.",
            "Generated captions are qualitative artifacts for manual accuracy comparison, not a measured accuracy metric.",
        ],
        "measurements": [asdict(measurement) for measurement in measurements],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(measurements)} configuration profiles to {args.output}")


if __name__ == "__main__":
    main()
