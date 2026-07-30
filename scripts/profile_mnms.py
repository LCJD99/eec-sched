#!/usr/bin/env python3
"""Run one MnMS configuration-profile suite and persist its JSON artifact.

The input file is deliberately local: some adopted public corpora require
separate credentials/licenses (ImageNet, COCO and ICDAR), and this project must
not download them on import.  The script nevertheless runs the real model,
configuration, five fixed inputs, standard scorer, and 50 latency observations.

Example:
    uv run --group profiling python scripts/profile_mnms.py \
      --tool text_classification --samples profiles/sst2-five.json \
      --output profiles/text-classification.json --device cuda

``samples`` is a JSON array (or ``{"samples": [...]}``) of at least five
records in the pinned split order.  Every record has ``id``, ``inputs``, and
``target``.  Image/audio values in ``inputs`` are filesystem paths.  On a
rerun, an existing output's saved IDs are replayed rather than reselected.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from itertools import islice
from pathlib import Path
from typing import Callable, Mapping, Sequence

from eec_sched.mnms_tools import MNMS_MODELS, MnmsToolRunner, mnms_tool_specs
from eec_sched.profile_evaluation import (
    EvaluationSample,
    EvaluationSuite,
    load_persisted_samples,
    load_profile_artifact,
    profile_suite,
    save_profile_artifact,
    select_fixed_samples,
)


SUITES: dict[str, EvaluationSuite] = {
    "text_summarization": EvaluationSuite("text_summarization", "abisee/cnn_dailymail", "3.0.0", "test", "rougeL", True, "evaluate/rouge"),
    "text_classification": EvaluationSuite("text_classification", "glue/sst2", "main", "validation", "accuracy", True, "GLUE accuracy"),
    "question_answering": EvaluationSuite("question_answering", "squad_v2", "main", "validation", "F1", True, "evaluate/squad_v2"),
    "automatic_speech_recognition": EvaluationSuite("automatic_speech_recognition", "librispeech_asr/clean", "main", "test", "WER", False, "evaluate/wer"),
    "image_captioning": EvaluationSuite("image_captioning", "COCO 2014", "2014", "validation", "CIDEr", True, "coco-caption"),
    "image_classification": EvaluationSuite("image_classification", "ImageNet-1k", "2012", "validation", "top-1 accuracy", True, "ImageNet top-1"),
    "visual_question_answering": EvaluationSuite("visual_question_answering", "VQAv2", "v2", "validation", "VQA accuracy", True, "official VQA evaluator"),
    "object_detection": EvaluationSuite("object_detection", "COCO 2017", "2017", "validation", "box AP", True, "pycocotools COCOeval"),
    "image_segmentation": EvaluationSuite("image_segmentation", "COCO 2017", "2017", "validation", "mask AP", True, "pycocotools COCOeval"),
    "optical_character_recognition": EvaluationSuite("optical_character_recognition", "ICDAR 2015 RRC Challenge 4", "2015", "test", "Hmean", True, "official ICDAR RRC evaluator"),
}


def _samples(path: Path) -> tuple[EvaluationSample, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload["samples"] if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("samples JSON must be an array or an object with a 'samples' array")
    return tuple(EvaluationSample(str(row["id"]), row["inputs"], row["target"]) for row in records)


def _download_public_samples(tool_id: str) -> tuple[EvaluationSample, ...]:
    """Fetch the directly redistributable text suites in their specified split."""
    from datasets import load_dataset  # type: ignore[import-not-found]
    if tool_id == "text_summarization":
        rows = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test")
        return tuple(EvaluationSample(str(row["id"]), {"text": row["article"]}, row["highlights"]) for row in rows.select(range(5)))
    if tool_id == "text_classification":
        rows = load_dataset("glue", "sst2", split="validation")
        return tuple(EvaluationSample(str(row["idx"]), {"text": row["sentence"]}, "POSITIVE" if row["label"] else "NEGATIVE") for row in rows.select(range(5)))
    if tool_id == "question_answering":
        rows = load_dataset("squad_v2", split="validation")
        return tuple(EvaluationSample(str(row["id"]), {"question": row["question"], "text": row["context"]}, {"id": row["id"], "answers": row["answers"]}) for row in rows.select(range(5)))
    if tool_id == "automatic_speech_recognition":
        import soundfile as sf  # type: ignore[import-not-found]
        rows = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
        audio_directory = Path("profiles/datasets/librispeech-test-clean-five")
        audio_directory.mkdir(parents=True, exist_ok=True)
        selected: list[EvaluationSample] = []
        for index, row in enumerate(islice(rows, 5)):
            sample_id = str(row["id"])
            path = audio_directory / f"{sample_id}.wav"
            if not path.exists():
                sf.write(path, row["audio"]["array"], row["audio"]["sampling_rate"])
            selected.append(EvaluationSample(sample_id, {"audio": str(path)}, row["text"]))
        return tuple(selected)
    if tool_id == "image_classification":
        from transformers import AutoConfig  # type: ignore[import-not-found]
        rows = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)
        image_directory = Path("profiles/datasets/imagenet-validation-five")
        image_directory.mkdir(parents=True, exist_ok=True)
        labels = AutoConfig.from_pretrained(MNMS_MODELS[tool_id]).id2label
        selected = []
        for index, row in enumerate(islice(rows, 5)):
            path = image_directory / f"validation-{index:05d}.jpg"
            if not path.exists():
                row["image"].convert("RGB").save(path, format="JPEG")
            selected.append(EvaluationSample(str(index), {"image": str(path)}, labels[int(row["label"])]))
        return tuple(selected)
    raise ValueError(f"{tool_id}'s official corpus has media, access, or evaluator prerequisites; provide its prepared five-record --samples file")


def _text_predictions(predictions: Sequence[object]) -> list[str]:
    return [str(prediction["output"]["text"]) for prediction in predictions]  # type: ignore[index]


def _scorer(tool_id: str) -> Callable[[Sequence[object], Sequence[object]], float]:
    if tool_id in {"text_classification", "image_classification"}:
        return lambda predictions, targets: sum(prediction == str(target) for prediction, target in zip(_text_predictions(predictions), targets)) / len(targets)
    if tool_id == "text_summarization":
        def rouge(predictions: Sequence[object], targets: Sequence[object]) -> float:
            from evaluate import load  # type: ignore[import-not-found]
            return float(load("rouge").compute(predictions=_text_predictions(predictions), references=[str(target) for target in targets])["rougeL"])
        return rouge
    if tool_id == "automatic_speech_recognition":
        def wer(predictions: Sequence[object], targets: Sequence[object]) -> float:
            from evaluate import load  # type: ignore[import-not-found]
            return float(load("wer").compute(predictions=_text_predictions(predictions), references=[str(target) for target in targets]))
        return wer
    if tool_id == "question_answering":
        def squad_v2(predictions: Sequence[object], targets: Sequence[object]) -> float:
            from evaluate import load  # type: ignore[import-not-found]
            answers = []
            for index, target in enumerate(targets):
                if not isinstance(target, Mapping) or "answers" not in target:
                    raise ValueError("SQuAD targets must contain {'id': ..., 'answers': {'text': [...], 'answer_start': [...]}}")
                answers.append({"id": str(target.get("id", index)), "answers": target["answers"]})
            predicted = [{"id": answer["id"], "prediction_text": text, "no_answer_probability": 0.0} for answer, text in zip(answers, _text_predictions(predictions))]
            return float(load("squad_v2").compute(predictions=predicted, references=answers)["f1"])
        return squad_v2
    raise ValueError(f"{tool_id} requires its official evaluator; pass --scorer module:function")


def _load_scorer(reference: str | None, tool_id: str) -> Callable[[Sequence[object], Sequence[object]], float]:
    if reference is None:
        return _scorer(tool_id)
    module_name, separator, function_name = reference.partition(":")
    if not separator:
        raise ValueError("--scorer must use module:function syntax")
    scorer = getattr(importlib.import_module(module_name), function_name)
    if not callable(scorer):
        raise ValueError(f"{reference} is not callable")
    return scorer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", choices=sorted(SUITES))
    parser.add_argument("--samples", type=Path, help="Pinned/local public-corpus records in split order")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--scorer", help="Official evaluator adapter as importable module:function")
    args = parser.parse_args()

    samples = _samples(args.samples) if args.samples else _download_public_samples(args.tool)
    selected = load_persisted_samples(samples, load_profile_artifact(args.output).sample_ids) if args.output.exists() else select_fixed_samples(samples)
    specs = {spec.tool_id: spec for spec in mnms_tool_specs()}
    artifact = profile_suite(
        suite=SUITES[args.tool], model_name=MNMS_MODELS[args.tool], configurations=specs[args.tool].configurations,
        runner_factory=lambda: MnmsToolRunner(args.tool, args.device), samples=selected,
        scorer=_load_scorer(args.scorer, args.tool), device=args.device,
    )
    save_profile_artifact(args.output, artifact)
    print(f"wrote {len(artifact.accuracy)} accuracy and {len(artifact.latency)} latency profiles to {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError) as error:
        print(f"profile failed: {error}", file=sys.stderr)
        raise SystemExit(2)
