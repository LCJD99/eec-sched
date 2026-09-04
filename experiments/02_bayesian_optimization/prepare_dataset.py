"""Materialize the fixed COCO subsets inside the portable benchmark bundle."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any, Iterable


def _subset_coco(payload: dict[str, Any], image_ids: Iterable[int]) -> dict[str, Any]:
    selected = set(image_ids)
    result = {key: value for key, value in payload.items() if key not in {"images", "annotations"}}
    result["images"] = [image for image in payload["images"] if image["id"] in selected]
    result["annotations"] = [annotation for annotation in payload["annotations"] if annotation["image_id"] in selected]
    return result


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def _copy_images(source: Path, destination: Path, image_ids: Iterable[int], prefix: str = "") -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for image_id in image_ids:
        source_path = source / f"{prefix}{image_id:012d}.jpg"
        if not source_path.is_file():
            raise FileNotFoundError(f"COCO image is missing: {source_path}")
        shutil.copy2(source_path, destination / source_path.name)


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def coco_subset(root: Path, destination: Path, count: int, seed: int) -> None:
    """Copy exactly ``count`` COCO samples and their official targets."""
    instance_annotation = root / "annotations" / "instances_val2017.json"
    instance_payload = json.loads(instance_annotation.read_text(encoding="utf-8"))
    instance_ids = sorted({int(image["id"]) for image in instance_payload["images"]})
    selected_instances = sorted(random.Random(seed).sample(instance_ids, count))

    coco2017 = destination / "coco2017"
    _copy_images(root / "val2017", coco2017 / "images", selected_instances)
    _write_json(coco2017 / "instances.json", _subset_coco(instance_payload, selected_instances))
    instance_image = lambda image_id: f"coco2017/images/{image_id:012d}.jpg"
    instance_annotation_path = "coco2017/instances.json"
    for tool in ("object_detection", "image_segmentation"):
        rows = [
            {
                "id": str(image_id),
                "image_id": image_id,
                "image": instance_image(image_id),
                "annotations": instance_annotation_path,
            }
            for image_id in selected_instances
        ]
        _write_manifest(destination / tool / "samples.jsonl", rows)

    caption_annotation = root / "annotations" / "captions_val2014.json"
    if not caption_annotation.is_file():
        raise FileNotFoundError(f"COCO caption references are required: {caption_annotation}")
    caption_payload = json.loads(caption_annotation.read_text(encoding="utf-8"))
    caption_ids = sorted({int(row["image_id"]) for row in caption_payload["annotations"]})
    selected_captions = sorted(random.Random(seed).sample(caption_ids, count))
    coco2014 = destination / "coco2014"
    _copy_images(root / "val2014", coco2014 / "images", selected_captions, prefix="COCO_val2014_")
    caption_subset = _subset_coco(caption_payload, selected_captions)
    _write_json(coco2014 / "captions.json", caption_subset)
    references: dict[int, list[str]] = {image_id: [] for image_id in selected_captions}
    for row in caption_payload["annotations"]:
        image_id = int(row["image_id"])
        if image_id in references:
            references[image_id].append(str(row["caption"]))
    _write_manifest(
        destination / "image_captioning" / "samples.jsonl",
        [
            {
                "id": str(image_id),
                "image_id": image_id,
                "image": f"coco2014/images/COCO_val2014_{image_id:012d}.jpg",
                "annotations": "coco2014/captions.json",
                "references": references[image_id],
            }
            for image_id in selected_captions
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-root", type=Path, default=Path("external/coco"))
    parser.add_argument("--output", type=Path, default=Path("data/benchmarks"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    coco_subset(args.coco_root, args.output, args.count, args.seed)
    print(f"wrote portable COCO subsets to {args.output}")


if __name__ == "__main__":
    main()
