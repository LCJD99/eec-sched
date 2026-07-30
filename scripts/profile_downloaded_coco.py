#!/usr/bin/env python3
"""Run official COCO caption, box-AP, or mask-AP profiles from five-sample packs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Sequence

from eec_sched.mnms_tools import MNMS_MODELS, MnmsToolRunner, mnms_tool_specs
from eec_sched.profile_evaluation import EvaluationSample, profile_suite, save_profile_artifact
from profile_mnms import SUITES


ROOT = Path("profiles/datasets")


def caption_samples() -> tuple[EvaluationSample, ...]:
    data = json.loads((ROOT / "coco-caption-2014-five/captions.json").read_text())
    captions: dict[int, list[str]] = {}
    for annotation in data["annotations"]:
        captions.setdefault(annotation["image_id"], []).append(annotation["caption"])
    return tuple(EvaluationSample(str(image["id"]), {"image": str(ROOT / "coco-caption-2014-five/images" / f"{image['id']:012d}.jpg")}, {"image_id": image["id"], "captions": captions[image["id"]]}) for image in data["images"])


def coco_samples() -> tuple[EvaluationSample, ...]:
    data = json.loads((ROOT / "coco-2017-five/instances.json").read_text())
    return tuple(EvaluationSample(str(image["id"]), {"image": str(ROOT / "coco-2017-five/images" / f"{image['id']:012d}.jpg")}, {"image_id": image["id"]}) for image in data["images"])


def cider(predictions: Sequence[object], targets: Sequence[object]) -> float:
    from pycocoevalcap.cider.cider import Cider
    gts = {target["image_id"]: target["captions"] for target in targets}  # type: ignore[index]
    res = {target["image_id"]: [str(prediction["output"]["text"])] for prediction, target in zip(predictions, targets)}  # type: ignore[index]
    return float(Cider().compute_score(gts, res)[0])


def coco_ap(kind: str) -> Callable[[Sequence[object], Sequence[object]], float]:
    def score(predictions: Sequence[object], targets: Sequence[object]) -> float:
        import numpy as np
        from pycocotools import mask as mask_utils
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        ground_truth = COCO(str(ROOT / "coco-2017-five/instances.json"))
        results = []
        for prediction, target in zip(predictions, targets):
            image_id = target["image_id"]  # type: ignore[index]
            if kind == "bbox":
                for item in prediction["evaluation"]["detections"]:  # type: ignore[index]
                    x1, y1, x2, y2 = item["bbox"]
                    results.append({"image_id": image_id, "category_id": item["label_id"], "bbox": [x1, y1, x2 - x1, y2 - y1], "score": item["score"]})
            else:
                for item in prediction["evaluation"]["segments"]:  # type: ignore[index]
                    encoded = mask_utils.encode(np.asfortranarray(item["mask"].astype("uint8")))
                    encoded["counts"] = encoded["counts"].decode("ascii")
                    results.append({"image_id": image_id, "category_id": item["label_id"], "segmentation": encoded, "score": item["score"]})
        evaluation = COCOeval(ground_truth, ground_truth.loadRes(results), kind)
        evaluation.params.imgIds = [target["image_id"] for target in targets]  # type: ignore[index]
        evaluation.evaluate(); evaluation.accumulate(); evaluation.summarize()
        return float(evaluation.stats[0])
    return score


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", choices=("image_captioning", "object_detection", "image_segmentation"), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    samples, scorer = (caption_samples(), cider) if args.tool == "image_captioning" else (coco_samples(), coco_ap("bbox" if args.tool == "object_detection" else "segm"))
    spec = next(spec for spec in mnms_tool_specs() if spec.tool_id == args.tool)
    artifact = profile_suite(suite=SUITES[args.tool], model_name=MNMS_MODELS[args.tool], configurations=spec.configurations, runner_factory=lambda: MnmsToolRunner(args.tool, args.device), samples=samples, scorer=scorer, device=args.device)
    output = Path("profiles") / f"{args.tool.replace('_', '-')}.json"
    save_profile_artifact(output, artifact)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
