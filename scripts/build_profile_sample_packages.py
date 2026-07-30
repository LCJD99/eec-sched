#!/usr/bin/env python3
"""Build five-sample official annotation packages without extracting full zips.

The resulting files are sufficient for the scorer adapters used by
``profile_mnms.py``.  Source archives stay untouched unless
``--delete-source-zips`` is specified after a successful build.

Example:
    uv run python scripts/build_profile_sample_packages.py \
      --coco2014-captions /data/captions_trainval2014.zip \
      --coco2017-instances /data/annotations_trainval2017.zip \
      --vqav2-questions /data/v2_OpenEnded_mscoco_val2014_questions.zip \
      --vqav2-annotations /data/v2_mscoco_val2014_annotations.zip
"""

from __future__ import annotations

import argparse
import json
import shutil
import ssl
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Iterable


COCO_2017_IDS = (397133, 37777, 252219, 87038, 174482)
COCO_2014_IMAGE_URL = "https://images.cocodataset.org/val2014/COCO_val2014_{image_id:012d}.jpg"
COCO_2017_IMAGE_URL = "https://images.cocodataset.org/val2017/{image_id:012d}.jpg"


def _read_zip_json(path: Path, filename: str) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        member = next((name for name in archive.namelist() if name.endswith(filename)), None)
        if member is None:
            raise ValueError(f"{filename} is absent from {path}")
        with archive.open(member) as file:
            return json.load(file)


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _subset_coco(payload: dict[str, Any], image_ids: Iterable[int]) -> dict[str, Any]:
    selected = set(image_ids)
    result = {key: value for key, value in payload.items() if key not in {"images", "annotations"}}
    result["images"] = [image for image in payload["images"] if image["id"] in selected]
    result["annotations"] = [annotation for annotation in payload["annotations"] if annotation["image_id"] in selected]
    if len(result["images"]) != len(selected):
        found = {image["id"] for image in result["images"]}
        raise ValueError(f"some selected image IDs are absent: {sorted(selected - found)}")
    return result


def _download_images(directory: Path, image_ids: Iterable[int], url_template: str, *, insecure: bool) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for image_id in image_ids:
        destination = directory / f"{image_id:012d}.jpg"
        if not destination.exists():
            context = ssl._create_unverified_context() if insecure else None
            with urllib.request.urlopen(url_template.format(image_id=image_id), context=context) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)


def build_coco_caption(source: Path, output: Path, *, insecure_images: bool) -> None:
    payload = _read_zip_json(source, "captions_val2014.json")
    # Split order is the official annotation order, not a random sample.
    image_ids = tuple(dict.fromkeys(annotation["image_id"] for annotation in payload["annotations"]))[:5]
    _write(output / "captions.json", _subset_coco(payload, image_ids))
    _download_images(output / "images", image_ids, COCO_2014_IMAGE_URL, insecure=insecure_images)


def build_coco_detection(source: Path, output: Path, *, insecure_images: bool) -> None:
    payload = _read_zip_json(source, "instances_val2017.json")
    _write(output / "instances.json", _subset_coco(payload, COCO_2017_IDS))
    _download_images(output / "images", COCO_2017_IDS, COCO_2017_IMAGE_URL, insecure=insecure_images)


def build_vqa(questions_zip: Path, annotations_zip: Path, output: Path, *, insecure_images: bool) -> None:
    questions = _read_zip_json(questions_zip, "v2_OpenEnded_mscoco_val2014_questions.json")
    annotations = _read_zip_json(annotations_zip, "v2_mscoco_val2014_annotations.json")
    selected_questions = questions["questions"][:5]
    selected_ids = {question["question_id"] for question in selected_questions}
    selected_annotations = [annotation for annotation in annotations["annotations"] if annotation["question_id"] in selected_ids]
    if len(selected_annotations) != len(selected_questions):
        raise ValueError("VQAv2 question and annotation IDs did not match")
    _write(output / "questions.json", {key: value for key, value in questions.items() if key != "questions"} | {"questions": selected_questions})
    _write(output / "annotations.json", {key: value for key, value in annotations.items() if key != "annotations"} | {"annotations": selected_annotations})
    _download_images(output / "images", dict.fromkeys(question["image_id"] for question in selected_questions), COCO_2014_IMAGE_URL, insecure=insecure_images)


def build_icdar(images_zip: Path, ground_truth_zip: Path, output: Path) -> None:
    """Copy the first five test images and their matching Challenge 4 GT files."""
    output_images = output / "images"
    output_gt = output / "gt"
    output_images.mkdir(parents=True, exist_ok=True)
    output_gt.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(images_zip) as images, zipfile.ZipFile(ground_truth_zip) as ground_truth:
        names = sorted(name for name in images.namelist() if name.lower().endswith((".jpg", ".png")))[:5]
        if len(names) != 5:
            raise ValueError("ICDAR archive has fewer than five images")
        gt_by_stem = {Path(name).stem.removeprefix("gt_"): name for name in ground_truth.namelist() if name.lower().endswith(".txt")}
        manifest = []
        for name in names:
            stem = Path(name).stem
            gt_name = gt_by_stem.get(stem)
            if gt_name is None:
                raise ValueError(f"ground truth for {name} is absent")
            image_path = output_images / Path(name).name
            gt_path = output_gt / Path(gt_name).name
            with images.open(name) as source, image_path.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            with ground_truth.open(gt_name) as source, gt_path.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            manifest.append({"id": stem, "image": image_path.name, "ground_truth": gt_path.name})
    _write(output / "manifest.json", {"samples": manifest})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("profiles/datasets"))
    parser.add_argument("--coco2014-captions", type=Path)
    parser.add_argument("--coco2017-instances", type=Path)
    parser.add_argument("--vqav2-questions", type=Path)
    parser.add_argument("--vqav2-annotations", type=Path)
    parser.add_argument("--icdar-images", type=Path)
    parser.add_argument("--icdar-ground-truth", type=Path)
    parser.add_argument("--delete-source-zips", action="store_true")
    parser.add_argument("--insecure-image-downloads", action="store_true", help="Disable TLS verification only for image downloads when a trusted corporate proxy rewrites certificates")
    args = parser.parse_args()

    sources: list[Path] = []
    if args.coco2014_captions:
        build_coco_caption(args.coco2014_captions, args.output_root / "coco-caption-2014-five", insecure_images=args.insecure_image_downloads)
        sources.append(args.coco2014_captions)
    if args.coco2017_instances:
        build_coco_detection(args.coco2017_instances, args.output_root / "coco-2017-five", insecure_images=args.insecure_image_downloads)
        sources.append(args.coco2017_instances)
    if bool(args.vqav2_questions) != bool(args.vqav2_annotations):
        parser.error("VQAv2 requires both --vqav2-questions and --vqav2-annotations")
    if args.vqav2_questions and args.vqav2_annotations:
        build_vqa(args.vqav2_questions, args.vqav2_annotations, args.output_root / "vqav2-2014-five", insecure_images=args.insecure_image_downloads)
        sources.extend((args.vqav2_questions, args.vqav2_annotations))
    if bool(args.icdar_images) != bool(args.icdar_ground_truth):
        parser.error("ICDAR requires both --icdar-images and --icdar-ground-truth")
    if args.icdar_images and args.icdar_ground_truth:
        build_icdar(args.icdar_images, args.icdar_ground_truth, args.output_root / "icdar2015-five")
        sources.extend((args.icdar_images, args.icdar_ground_truth))
    if not sources:
        parser.error("provide at least one official source zip")
    if args.delete_source_zips:
        for source in sources:
            source.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
