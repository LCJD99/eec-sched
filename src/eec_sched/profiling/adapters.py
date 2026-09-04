"""Evaluation adapters used by the MnMS Bayesian campaign.

Adapters intentionally consume only the fixed records supplied by the
campaign. They do not sample a second dataset or silently substitute a proxy
metric. Projects can replace an adapter with a stricter local implementation
while keeping the campaign engine and artifact format unchanged.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from .campaign import Candidate


_PIPELINES: dict[tuple[str, str], Any] = {}
_OCR_READERS: dict[tuple[tuple[str, ...], bool], Any] = {}


def _require_fields(samples: list[dict[str, Any]], *fields: str) -> None:
    missing = [field for field in fields if any(field not in sample for sample in samples)]
    if missing:
        raise RuntimeError("fixed evaluation manifest is missing required fields: " + ", ".join(missing))


def _pipeline(model: dict[str, Any], task: str):
    try:
        from transformers import pipeline
    except ImportError as error:
        raise RuntimeError("install transformers before running a model campaign") from error
    key = (task, str(model["model"]))
    if key not in _PIPELINES:
        runner = pipeline(task, model=model["model"])
        tokenizer = getattr(runner, "tokenizer", None)
        if tokenizer is not None and task == "summarization":
            tokenizer.model_max_length = 1024
        _PIPELINES[key] = runner
    return _PIPELINES[key]


def _gpu_memory_start() -> None:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to measure peak_gpu_memory_mib")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def _gpu_memory_mib() -> float:
    import torch
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def _pipeline_arguments(candidate: Candidate, task: str) -> dict[str, Any]:
    """Translate search-space names to the public pipeline API.

    Search-space names describe the experiment, while pipeline arguments are
    task-specific.  Passing every candidate key through blindly makes valid
    configurations fail at runtime (notably Whisper's beam count).
    """
    arguments = candidate.as_dict()
    if task == "automatic-speech-recognition":
        generate_kwargs = {
            "language": "en",
            "task": "transcribe",
        }
        if "beam_size" in arguments:
            generate_kwargs["num_beams"] = arguments.pop("beam_size")
        if "max_new_tokens" in arguments:
            generate_kwargs["max_new_tokens"] = arguments.pop("max_new_tokens")
        arguments["generate_kwargs"] = generate_kwargs
    if task == "question-answering" and "max_length" in arguments:
        arguments["max_seq_len"] = arguments.pop("max_length")
    if task == "visual-question-answering":
        # VisualQuestionAnsweringPipeline currently accepts only preprocessing
        # and top_k kwargs; generation settings are applied to its config by
        # _candidate_generation below.
        arguments = {}
    if task == "image-to-text" and "num_beams" in arguments:
        arguments["generate_kwargs"] = {"num_beams": arguments.pop("num_beams")}
    if task in {"summarization", "text-classification"}:
        # CNN/DailyMail contains articles longer than BART's context window.
        # Without explicit truncation, the CUDA path can fail asynchronously
        # with an index assertion instead of returning a useful exception.
        arguments["truncation"] = True
    return arguments


def _run_pipeline(candidate: Candidate, samples: list[dict[str, Any]], model: dict[str, Any], task: str, input_field: str, target_field: str) -> tuple[float, float, float, dict[str, Any]]:
    _require_fields(samples, input_field, target_field)
    runner = _pipeline(model, task)
    _gpu_memory_start()
    started = time.perf_counter()
    pipeline_arguments = _pipeline_arguments(candidate, task)
    predictions = [runner(sample[input_field], **pipeline_arguments) for sample in samples]
    latency = (time.perf_counter() - started) * 1000 / len(samples)
    # Metric scoring belongs to the adopted evaluator. A manifest must provide
    # its scorer name and targets; never infer quality from runtime output.
    scorer = model.get("scorer")
    if not scorer:
        raise RuntimeError(f"{model['tool']} requires an official scorer implementation")
    module_name, function_name = scorer.rsplit(":", 1)
    module = __import__(module_name, fromlist=[function_name])
    quality = float(getattr(module, function_name)([_prediction_value(item) for item in predictions], [sample[target_field] for sample in samples]))
    return quality, latency, _gpu_memory_mib(), {"batch_size": 1, "sample_count": len(samples), "metric": model["metric"], "resource_metric": "peak_gpu_memory_mib"}


def _prediction_value(value: Any) -> Any:
    """Convert common Transformers pipeline outputs to scorer inputs."""
    if isinstance(value, list):
        return _prediction_value(value[0]) if value else ""
    if isinstance(value, dict):
        for key in ("generated_text", "text", "answer", "label"):
            if key in value:
                return value[key]
    return value


def evaluate_summarization(candidate, samples, model):
    normalized = [{**sample, "text": sample.get("text", sample.get("article")), "target": sample.get("target", sample.get("highlights"))} for sample in samples]
    return _run_pipeline(candidate, normalized, model, "summarization", "text", "target")


def evaluate_text_classification(candidate, samples, model):
    normalized = [
        {
            **sample,
            "text": sample.get("text", sample.get("sentence")),
            # The fine-tuned SST-2 pipeline returns POSITIVE/NEGATIVE while
            # the dataset stores the official integer labels.
            "label": {0: "NEGATIVE", 1: "POSITIVE"}.get(sample["label"], sample["label"]),
        }
        for sample in samples
    ]
    result = _run_pipeline(candidate, normalized, model, "text-classification", "text", "label")
    return result


def evaluate_question_answering(candidate, samples, model):
    _require_fields(samples, "question", "context", "answers")
    runner = _pipeline(model, "question-answering")
    _gpu_memory_start()
    started = time.perf_counter()
    predictions = [
        runner(
            question=sample["question"],
            context=sample["context"],
            **_pipeline_arguments(candidate, "question-answering"),
        )
        for sample in samples
    ]
    latency = (time.perf_counter() - started) * 1000 / len(samples)
    quality = _score(
        model,
        [_prediction_value(item) for item in predictions],
        [{"id": sample.get("id", index), "answers": sample["answers"]} for index, sample in enumerate(samples)],
    )
    return quality, latency, _gpu_memory_mib(), {"batch_size": 1, "sample_count": len(samples), "metric": model["metric"], "resource_metric": "peak_gpu_memory_mib", "scorer": "evaluate.squad_v2"}


def evaluate_asr(candidate, samples, model):
    _require_fields(samples, "audio", "text")
    missing = [sample["audio"] for sample in samples if not Path(str(sample["audio"])).is_file()]
    if missing:
        raise FileNotFoundError(f"ASR manifest references missing audio: {missing[0]}")
    return _run_pipeline(candidate, samples, model, "automatic-speech-recognition", "audio", "text")


def evaluate_captioning(candidate, samples, model):
    _require_fields(samples, "image")
    references = [sample.get("references", sample.get("captions")) for sample in samples]
    if any(not isinstance(value, list) or not value for value in references):
        raise RuntimeError("COCO caption manifest must provide non-empty references per image")
    runner = _pipeline(model, "image-to-text")
    _gpu_memory_start()
    started = time.perf_counter()
    predictions = [runner(sample["image"], **_pipeline_arguments(candidate, "image-to-text")) for sample in samples]
    latency = (time.perf_counter() - started) * 1000 / len(samples)
    quality = _score(model, [_prediction_value(item) for item in predictions], references)
    return quality, latency, _gpu_memory_mib(), {"batch_size": 1, "sample_count": len(samples), "metric": model["metric"], "resource_metric": "peak_gpu_memory_mib", "scorer": "pycocoevalcap.cider.Cider"}


def evaluate_image_classification(candidate, samples, model):
    _require_fields(samples, "image", "label")
    runner = _pipeline(model, "image-classification")
    size = int(candidate.as_dict().get("image_size", 224))
    arguments = {"top_k": int(candidate.as_dict().get("top_k", 1))}
    processor = getattr(runner, "image_processor", None)
    original_size = getattr(processor, "size", None)
    if processor is None or original_size is None:
        raise RuntimeError("ImageNet pipeline does not expose an image processor")
    # ImageClassificationPipeline currently invokes image_processor without
    # per-call kwargs, so set the processor's size for this candidate and
    # restore it after the measurement. This makes image_size real.
    processor.size = {"height": size, "width": size}
    _gpu_memory_start()
    started = time.perf_counter()
    original_forward = runner.model.forward

    def forward_with_interpolated_positions(*args, **kwargs):
        kwargs.setdefault("interpolate_pos_encoding", True)
        return original_forward(*args, **kwargs)

    runner.model.forward = forward_with_interpolated_positions
    try:
        outputs = [runner(sample["image"], **arguments) for sample in samples]
    finally:
        runner.model.forward = original_forward
        processor.size = original_size
    latency = (time.perf_counter() - started) * 1000 / len(samples)
    predictions = [_image_label_to_index(items[0], runner) for items in outputs]
    targets = [_image_label_to_index(sample["label"], runner) for sample in samples]
    quality = _score(model, predictions, targets)
    return quality, latency, _gpu_memory_mib(), {"batch_size": 1, "sample_count": len(samples), "metric": model["metric"], "resource_metric": "peak_gpu_memory_mib", "image_processor_size": {"height": size, "width": size}}


def evaluate_vqa(candidate, samples, model):
    _require_fields(samples, "image", "question", "answers")
    runner = _pipeline(model, "visual-question-answering")
    _gpu_memory_start()
    started = time.perf_counter()
    with _candidate_generation(runner, candidate):
        predictions = [
            runner(sample["image"], question=sample["question"], **_pipeline_arguments(candidate, "visual-question-answering"))
            for sample in samples
        ]
    latency = (time.perf_counter() - started) * 1000 / len(samples)
    quality = _score(model, [_prediction_value(item) for item in predictions], [sample["answers"] for sample in samples])
    return quality, latency, _gpu_memory_mib(), {"batch_size": 1, "sample_count": len(samples), "metric": model["metric"], "resource_metric": "peak_gpu_memory_mib", "scorer": "GT-Vision-Lab/VQA VQAEval"}


def evaluate_coco_detection(candidate, samples, model):
    """Evaluate DETR predictions with the official COCO box AP scorer."""
    _require_fields(samples, "image", "image_id", "annotations")
    try:
        import torch
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        from transformers import pipeline
    except ImportError as error:
        raise RuntimeError("install the profiling group for COCO evaluation") from error

    annotation_path = samples[0]["annotations"]
    coco = COCO(annotation_path)
    category_ids = {item["name"]: item["id"] for item in coco.loadCats(coco.getCatIds())}
    runner = _pipeline(model, "object-detection")
    _gpu_memory_start()
    started = time.perf_counter()
    predictions: list[dict[str, Any]] = []
    with _candidate_processor_size(runner, candidate):
        for sample in samples:
            outputs = runner(sample["image"], threshold=float(candidate.as_dict()["score_threshold"]))
            for output in outputs:
                label = str(output["label"])
                category_id = _category_id_from_label(label, runner, category_ids)
                if category_id is None:
                    continue
                box = output["box"]
                x = float(box["xmin"])
                y = float(box["ymin"])
                predictions.append({"image_id": int(sample["image_id"]), "category_id": category_id, "bbox": [x, y, float(box["xmax"]) - x, float(box["ymax"]) - y], "score": float(output["score"])})
    latency = (time.perf_counter() - started) * 1000 / len(samples)
    result = COCOeval(coco, coco.loadRes(predictions), "bbox")
    result.params.imgIds = [int(sample["image_id"]) for sample in samples]
    result.evaluate()
    result.accumulate()
    result.summarize()
    return float(result.stats[0]), latency, _gpu_memory_mib(), {"batch_size": 1, "sample_count": len(samples), "metric": "box_AP", "resource_metric": "peak_gpu_memory_mib", "coco_evaluator": "pycocotools.COCOeval"}


def evaluate_coco_segmentation(candidate, samples, model):
    _require_fields(samples, "image", "image_id", "annotations")
    try:
        import numpy as np
        from pycocotools.coco import COCO
        from pycocotools import mask as mask_utils
    except ImportError as error:
        raise RuntimeError("install the profiling group for COCO mask evaluation") from error
    annotation_path = samples[0]["annotations"]
    coco = COCO(annotation_path)
    category_ids = {item["name"]: item["id"] for item in coco.loadCats(coco.getCatIds())}
    runner = _pipeline(model, "image-segmentation")
    _gpu_memory_start()
    started = time.perf_counter()
    predictions: list[dict[str, Any]] = []
    threshold = float(candidate.as_dict().get("score_threshold", 0.0))
    with _candidate_processor_size(runner, candidate):
        for sample in samples:
            outputs = runner(sample["image"], threshold=threshold)
            for output in outputs:
                score = float(output.get("score", 1.0))
                if score < threshold:
                    continue
                category_id = _category_id_from_label(str(output.get("label", "")), runner, category_ids)
                if category_id is None:
                    continue
                mask = output.get("mask")
                if mask is None:
                    raise RuntimeError("MaskFormer returned no mask for a segmentation prediction")
                if hasattr(mask, "convert"):
                    mask = np.asarray(mask.convert("L")) > 0
                else:
                    mask = np.asarray(mask)
                    if mask.ndim == 3:
                        mask = mask[..., 0]
                    mask = mask > 0
                encoded = mask_utils.encode(np.asfortranarray(mask.astype("uint8")))
                if isinstance(encoded["counts"], bytes):
                    encoded["counts"] = encoded["counts"].decode("ascii")
                predictions.append({"image_id": int(sample["image_id"]), "category_id": category_id, "segmentation": encoded, "score": score})
    latency = (time.perf_counter() - started) * 1000 / len(samples)
    from pycocotools.cocoeval import COCOeval
    if not predictions:
        return 0.0, latency, _gpu_memory_mib(), {"batch_size": 1, "sample_count": len(samples), "metric": "mask_AP", "resource_metric": "peak_gpu_memory_mib", "coco_evaluator": "pycocotools.COCOeval", "prediction_count": 0}
    result = COCOeval(coco, coco.loadRes(predictions), "segm")
    result.params.imgIds = [int(sample["image_id"]) for sample in samples]
    result.evaluate(); result.accumulate(); result.summarize()
    return float(result.stats[0]), latency, _gpu_memory_mib(), {"batch_size": 1, "sample_count": len(samples), "metric": "mask_AP", "resource_metric": "peak_gpu_memory_mib", "coco_evaluator": "pycocotools.COCOeval"}


def evaluate_ocr(candidate, samples, model):
    _require_fields(samples, "image", "transcription", "boxes")
    try:
        import torch
        import easyocr
    except ImportError as error:
        raise RuntimeError("install easyocr and the profiling dependency group for ICDAR evaluation") from error
    gpu = bool(torch.cuda.is_available())
    language_groups = _ocr_language_groups(samples)
    readers = []
    for languages in language_groups:
        key = (languages, gpu)
        if key not in _OCR_READERS:
            _OCR_READERS[key] = easyocr.Reader(list(languages), gpu=gpu)
        readers.append(_OCR_READERS[key])
    options = candidate.as_dict()
    _gpu_memory_start()
    started = time.perf_counter()
    predictions = []
    for sample in samples:
        sample_predictions = []
        for reader in readers:
            sample_predictions.extend(
                {
                    "box": box,
                    "text": text,
                    "confidence": confidence,
                }
                for box, text, confidence in reader.readtext(
                    sample["image"], detail=1, paragraph=False,
                    canvas_size=int(options.get("canvas_size", 1280)),
                    text_threshold=float(options.get("text_threshold", 0.7)),
                    width_ths=float(options.get("width_ths", 0.7)),
                )
            )
        predictions.append(_deduplicate_ocr_predictions(sample_predictions))
    latency = (time.perf_counter() - started) * 1000 / len(samples)
    quality = _score(model, predictions, [{"boxes": sample["boxes"], "transcription": sample["transcription"]} for sample in samples])
    return quality, latency, _gpu_memory_mib(), {"batch_size": 1, "sample_count": len(samples), "metric": model["metric"], "resource_metric": "peak_gpu_memory_mib", "evaluator": "ICDAR 2019 MLT Task 4 end-to-end Hmean"}


def _ocr_language_groups(samples: list[dict[str, Any]]) -> tuple[tuple[str, ...], ...]:
    """Return EasyOCR-compatible script readers for the fixed MLT slice."""
    groups = set()
    for sample in samples:
        for script in sample.get("script", []):
            groups.add({
                "Arabic": ("ar", "en"),
                "Bengali": ("bn", "en"),
                "Chinese": ("ch_sim", "en"),
                "Japanese": ("ja", "en"),
                "Korean": ("ko", "en"),
                "Devanagari": ("hi", "en"),
                "Latin": ("en",),
            }.get(script, ("en",)))
    return tuple(sorted(groups)) or (("en",),)


def _deduplicate_ocr_predictions(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge overlapping outputs from the compatible per-script readers."""
    kept: list[dict[str, Any]] = []
    for prediction in sorted(predictions, key=lambda item: float(item["confidence"]), reverse=True):
        if all(_ocr_box_iou(prediction["box"], item["box"]) <= 0.5 for item in kept):
            kept.append(prediction)
    return kept


def _ocr_box_iou(first: list[list[float]], second: list[list[float]]) -> float:
    def bounds(box: list[list[float]]) -> tuple[float, float, float, float]:
        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]
        return min(xs), min(ys), max(xs), max(ys)

    ax1, ay1, ax2, ay2 = bounds(first)
    bx1, by1, bx2, by2 = bounds(second)
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union else 0.0


def _score(model: dict[str, Any], predictions: list[Any], targets: list[Any]) -> float:
    scorer = model.get("scorer")
    if not scorer:
        raise RuntimeError(f"{model['tool']} requires an official scorer implementation")
    module_name, function_name = str(scorer).rsplit(":", 1)
    module = __import__(module_name, fromlist=[function_name])
    return float(getattr(module, function_name)(predictions, targets))


class _candidate_generation:
    """Temporarily apply generation-only candidate fields to a HF pipeline."""

    def __init__(self, runner: Any, candidate: Candidate) -> None:
        self.config = getattr(getattr(runner, "model", None), "generation_config", None)
        self.changes = {
            name: value
            for name, value in candidate.as_dict().items()
            if name in {"max_length", "num_beams"}
        }
        self.original = {name: getattr(self.config, name, None) for name in self.changes} if self.config is not None else {}

    def __enter__(self) -> None:
        if self.config is None:
            raise RuntimeError("VQA pipeline does not expose a generation config")
        for name, value in self.changes.items():
            setattr(self.config, name, value)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        for name, value in self.original.items():
            setattr(self.config, name, value)


class _candidate_processor_size:
    """Temporarily apply the COCO resize candidate to a HF image processor."""

    def __init__(self, runner: Any, candidate: Candidate) -> None:
        self.processor = getattr(runner, "image_processor", None)
        self.original = getattr(self.processor, "size", None)
        self.size = candidate.as_dict().get("shortest_edge")

    def __enter__(self) -> None:
        if self.processor is None or self.original is None:
            raise RuntimeError("COCO pipeline does not expose an image processor")
        if self.size is not None and isinstance(self.original, dict):
            updated = dict(self.original)
            updated["shortest_edge"] = int(self.size)
            self.processor.size = updated

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self.processor is not None and self.original is not None:
            self.processor.size = self.original


def _image_label_to_index(value: Any, runner: Any) -> int:
    if isinstance(value, dict):
        value = value.get("index", value.get("label"))
    if isinstance(value, int):
        return value
    text = str(value).strip()
    match = re.search(r"(?:label[_ -]?)?(\d+)$", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    labels = getattr(getattr(runner, "model", None), "config", None)
    id2label = getattr(labels, "id2label", {}) or {}
    normalized = text.casefold()
    for index, label in id2label.items():
        if str(label).casefold() == normalized:
            return int(index)
    raise ValueError(f"cannot map ImageNet label to a class index: {value!r}")


def _category_id_from_label(label: str, runner: Any, category_ids: dict[str, int]) -> int | None:
    if label in category_ids:
        return int(category_ids[label])
    match = re.fullmatch(r"(?:label[_ -]?)?(\d+)", label, re.IGNORECASE)
    if match:
        id2label = getattr(getattr(getattr(runner, "model", None), "config", None), "id2label", {}) or {}
        label = str(id2label.get(int(match.group(1)), label))
    normalized = label.casefold().replace(" ", "_")
    for name, category_id in category_ids.items():
        if name.casefold().replace(" ", "_") == normalized:
            return int(category_id)
    return None
