"""Official-compatible scorer entry points used by campaign configurations.

The campaign config stores import paths, so these functions intentionally keep
their public interface small: ``(predictions, targets) -> float``.  Dataset
specific adapters normalize their records before calling them.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


def accuracy(predictions: Sequence[object], targets: Sequence[object]) -> float:
    if not targets:
        raise ValueError("cannot score an empty evaluation set")
    values = [str(item).strip().lower() for item in predictions]
    expected = [str(item).strip().lower() for item in targets]
    return sum(actual == target for actual, target in zip(values, expected)) / len(expected)


def rouge_l(predictions: Sequence[object], targets: Sequence[object]) -> float:
    try:
        from rouge_score import rouge_scorer
    except ImportError as error:
        raise RuntimeError("install the profiling dependency group for ROUGE-L") from error
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return sum(scorer.score(str(target), str(pred))["rougeL"].fmeasure for pred, target in zip(predictions, targets)) / len(targets)


def token_f1(predictions: Sequence[object], targets: Sequence[object]) -> float:
    scores = []
    for prediction, target in zip(predictions, targets):
        left, right = str(prediction).lower().split(), str(target).lower().split()
        overlap = len(set(left) & set(right))
        scores.append(0.0 if not left or not right or not overlap else 2 * overlap / (len(left) + len(right)))
    return sum(scores) / len(scores)


def squad_v2_f1(predictions: Sequence[object], targets: Sequence[object]) -> float:
    """Return the official SQuAD v2 F1 score.

    ``evaluate`` is the maintained packaging of the official SQuAD v2
    evaluator.  Unlike the old generic token-F1 helper, it handles answer
    aliases, punctuation/articles, and unanswerable questions.
    """
    try:
        from evaluate import load
    except ImportError as error:
        raise RuntimeError("install the profiling dependency group for SQuAD v2") from error
    if len(predictions) != len(targets) or not targets:
        raise ValueError("SQuAD predictions and targets must be non-empty and aligned")
    references: list[dict[str, Any]] = []
    formatted_predictions: list[dict[str, Any]] = []
    for index, (prediction, target) in enumerate(zip(predictions, targets)):
        if not isinstance(target, Mapping):
            raise ValueError("SQuAD targets must be mappings")
        target_id = str(target.get("id", index))
        answer = _prediction_text(prediction)
        references.append({"id": target_id, "answers": _squad_answers(target.get("answers"))})
        formatted_predictions.append(
            {"id": target_id, "prediction_text": answer, "no_answer_probability": 0.0}
        )
    result = load("squad_v2").compute(
        predictions=formatted_predictions, references=references
    )
    return float(result["f1"])


def _squad_answers(value: object) -> dict[str, list[object]]:
    if isinstance(value, Mapping):
        return {"text": list(value.get("text", [])), "answer_start": list(value.get("answer_start", []))}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        answers = [item for item in value if isinstance(item, Mapping)]
        return {"text": [item.get("text", "") for item in answers], "answer_start": [item.get("answer_start", 0) for item in answers]}
    raise ValueError("SQuAD targets must contain an answers mapping or list")


def wer(predictions: Sequence[object], targets: Sequence[object]) -> float:
    try:
        from jiwer import wer as score
    except ImportError as error:
        raise RuntimeError("install the profiling dependency group for WER") from error
    return float(score([str(target) for target in targets], [str(prediction) for prediction in predictions]))


def normalize_answer(value: object) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", str(value).lower())).strip()


def _prediction_text(value: object) -> str:
    if isinstance(value, Mapping):
        for key in ("answer", "text", "generated_text", "prediction_text"):
            if key in value:
                return str(value[key])
    return str(value)


def cider(predictions: Sequence[object], targets: Sequence[object]) -> float:
    """Compute COCO-caption CIDEr with the official ``pycocoevalcap`` code."""
    try:
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    except ImportError as error:
        raise RuntimeError("install the profiling dependency group for COCO CIDEr") from error
    if len(predictions) != len(targets) or not targets:
        raise ValueError("CIDEr predictions and targets must be non-empty and aligned")
    gts: dict[str, list[dict[str, str]]] = {}
    res: dict[str, list[dict[str, str]]] = {}
    for index, (prediction, target) in enumerate(zip(predictions, targets)):
        if isinstance(target, Mapping):
            references = target.get("references", target.get("captions"))
        else:
            references = target
        if isinstance(references, str) or not isinstance(references, Sequence):
            raise ValueError("COCO caption targets must be a sequence of reference strings")
        image_id = str(index)
        gts[image_id] = [{"caption": str(reference)} for reference in references]
        res[image_id] = [{"caption": _prediction_text(prediction)}]
    # Cider.compute_score consumes PTB-tokenized strings; COCOEvalCap applies
    # this exact tokenizer before invoking the metric implementation.
    tokenizer = PTBTokenizer()
    tokenized_gts = tokenizer.tokenize(gts)
    tokenized_res = tokenizer.tokenize(res)
    score, _ = Cider().compute_score(tokenized_gts, tokenized_res)
    return float(score)


_VQA_ARTICLES = {"a", "an", "the"}
_VQA_NUMBER_MAP = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
    "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
}
_VQA_PUNCTUATION = {
    ";", "/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\",
    "_", "-", ">", "<", "@", "`", ",", "?", "!",
}


def vqa_normalize_answer(value: object) -> str:
    """Apply the normalization from the official VQAv2 ``VQAEval``."""
    text = str(value).lower().replace("\n", " ").replace("\t", " ").strip()
    result: list[str] = []
    for char in text:
        if char in _VQA_PUNCTUATION:
            # The official evaluator retains commas inside numbers (e.g. 1,000)
            # and otherwise turns punctuation into a separator.
            if (
                char == ","
                and result
                and result[-1].isdigit()
                and len(text) > len(result)
                and text[len(result)].isdigit()
            ):
                result.append(char)
            else:
                result.append(" ")
        else:
            result.append(char)
    tokens = []
    for token in "".join(result).split():
        token = _VQA_NUMBER_MAP.get(token, token)
        if token not in _VQA_ARTICLES:
            tokens.append(token)
    return " ".join(tokens)


def vqa_accuracy(predictions: Sequence[object], targets: Sequence[object]) -> float:
    """Compute official VQAv2 soft accuracy (min matching annotators / 3)."""
    if len(predictions) != len(targets) or not targets:
        raise ValueError("VQAv2 predictions and targets must be non-empty and aligned")
    scores: list[float] = []
    for prediction, target in zip(predictions, targets):
        if isinstance(target, Mapping):
            answers = target.get("answers")
        else:
            answers = target
        if not isinstance(answers, Sequence) or isinstance(answers, (str, bytes)):
            raise ValueError("VQAv2 targets must contain an answers sequence")
        predicted = vqa_normalize_answer(_prediction_text(prediction))
        normalized = [vqa_normalize_answer(answer.get("answer", "") if isinstance(answer, Mapping) else answer) for answer in answers]
        matches = sum(candidate == predicted for candidate in normalized)
        scores.append(min(1.0, matches / 3.0))
    return sum(scores) / len(scores)


def imagenet_top1(predictions: Sequence[object], targets: Sequence[object]) -> float:
    """Score integer ImageNet class IDs, avoiding label-name/id mismatches."""
    if len(predictions) != len(targets) or not targets:
        raise ValueError("ImageNet predictions and targets must be non-empty and aligned")
    predicted = []
    for value in predictions:
        if isinstance(value, Mapping):
            value = value.get("index", value.get("label"))
        match = re.search(r"(?:label[_ -]?)?(\d+)$", str(value), re.IGNORECASE)
        if match is None:
            raise ValueError(f"ImageNet prediction is not a class index: {value!r}")
        predicted.append(int(match.group(1)))
    return sum(int(actual == int(target)) for actual, target in zip(predicted, targets)) / len(targets)


def mlt19_hmean(predictions: Sequence[object], targets: Sequence[object]) -> float:
    """Compute the official ICDAR 2019 MLT Task-4 H-mean.

    The adapter supplies quadrilateral detections and transcriptions.  A match
    requires polygon IoU > 0.5 and a case-insensitive exact transcription.
    MLT-19 ``###`` entries are don't-care regions and are excluded from both
    the ground-truth and detection counts when overlapped by a prediction.
    """
    if len(predictions) != len(targets) or not targets:
        raise ValueError("ICDAR predictions and targets must be non-empty and aligned")
    total_gt = total_det = total_matches = 0
    for prediction, target in zip(predictions, targets):
        detections = prediction if isinstance(prediction, Sequence) else []
        gt = target.get("boxes", []) if isinstance(target, Mapping) else []
        gt_text = target.get("transcription", []) if isinstance(target, Mapping) else []
        cared_for = [index for index, value in enumerate(gt_text) if str(value).strip() != "###"]
        dont_care = [index for index, value in enumerate(gt_text) if str(value).strip() == "###"]
        cared_detections = []
        for detection in detections:
            det_box, _ = _ocr_item(detection)
            det_area = max(_polygon_area(det_box), 1e-12)
            overlaps_dont_care = any(
                _polygon_intersection_area(det_box, gt[gt_index]) / det_area > 0.5
                for gt_index in dont_care
            )
            if not overlaps_dont_care:
                cared_detections.append(detection)
        detections = cared_detections
        total_gt += len(cared_for)
        total_det += len(detections)
        candidates: list[tuple[float, int, int]] = []
        for det_index, detection in enumerate(detections):
            det_box, det_value = _ocr_item(detection)
            for gt_index in cared_for:
                gt_value = gt_text[gt_index] if gt_index < len(gt_text) else ""
                if _mlt19_normalize_transcription(det_value) != _mlt19_normalize_transcription(gt_value):
                    continue
                intersection = _polygon_intersection_area(det_box, gt[gt_index])
                union = _polygon_area(det_box) + _polygon_area(gt[gt_index]) - intersection
                if intersection / max(union, 1e-12) > 0.5:
                    candidates.append((intersection, det_index, gt_index))
        used_det: set[int] = set()
        used_gt: set[int] = set()
        for _, det_index, gt_index in sorted(candidates, reverse=True):
            if det_index not in used_det and gt_index not in used_gt:
                used_det.add(det_index)
                used_gt.add(gt_index)
                total_matches += 1
    precision = total_matches / total_det if total_det else 0.0
    recall = total_matches / total_gt if total_gt else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def icdar2015_hmean(predictions: Sequence[object], targets: Sequence[object]) -> float:
    """Backward-compatible alias for older local scorer references."""
    return mlt19_hmean(predictions, targets)


def _ocr_item(value: object) -> tuple[Sequence[Sequence[float]], str]:
    if isinstance(value, Mapping):
        return value["box"], str(value.get("text", value.get("transcription", "")))
    box, text, *_ = value  # EasyOCR's (quad, text, confidence) tuple
    return box, str(text)


def _icdar_normalize_transcription(value: object) -> str:
    special = "!?.:,*\"()·[]/'"
    text = str(value).strip().upper()
    while text and text[0] in special:
        text = text[1:]
    while text and text[-1] in special:
        text = text[:-1]
    return text


def _mlt19_normalize_transcription(value: object) -> str:
    """MLT-19 uses case-insensitive exact matching, without punctuation stripping."""
    return str(value).strip().casefold()


def _polygon_area(polygon: Sequence[Sequence[float]]) -> float:
    points = [(float(point[0]), float(point[1])) for point in polygon]
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])) / 2)


def _polygon_intersection_area(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> float:
    # Sutherland-Hodgman is sufficient because ICDAR boxes are convex quads.
    subject = [(float(x), float(y)) for x, y in left]
    clip = [(float(x), float(y)) for x, y in right]
    if not subject or not clip:
        return 0.0
    orientation = 1 if _signed_polygon_area(clip) >= 0 else -1
    for start, end in zip(clip, clip[1:] + clip[:1]):
        def inside(point: tuple[float, float]) -> bool:
            return orientation * ((end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (point[0] - start[0])) >= -1e-9
        def intersection(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
            ax, ay = a; bx, by = b; sx, sy = start; ex, ey = end
            denominator = (ax - bx) * (sy - ey) - (ay - by) * (sx - ex)
            if abs(denominator) < 1e-12:
                return b
            t = ((ax - sx) * (sy - ey) - (ay - sy) * (sx - ex)) / denominator
            return (ax + t * (bx - ax), ay + t * (by - ay))
        output: list[tuple[float, float]] = []
        for current, previous in zip(subject, subject[-1:] + subject[:-1]):
            if inside(current):
                if not inside(previous):
                    output.append(intersection(previous, current))
                output.append(current)
            elif inside(previous):
                output.append(intersection(previous, current))
        subject = output
        if not subject:
            break
    return _polygon_area(subject)


def _signed_polygon_area(polygon: Sequence[Sequence[float]]) -> float:
    points = [(float(point[0]), float(point[1])) for point in polygon]
    return sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])) / 2
