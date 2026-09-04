"""Download 100-record slices of the adopted non-COCO benchmarks.

Most suites are read through Hugging Face ``datasets`` in streaming mode, so
the script does not download an entire split.  ImageNet is gated and needs a
Hugging Face token.  MLT-19 is downloaded one Kaggle file at a time, retaining
only the first 100 training images and their matching ground-truth files.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import ssl
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


BENCHMARKS: dict[str, dict[str, Any]] = {
    "cnn_dailymail": {"dataset": "abisee/cnn_dailymail", "config": "3.0.0", "split": "test", "kind": "hf"},
    "sst2": {"dataset": "glue", "config": "sst2", "split": "validation", "kind": "hf"},
    "squad_v2": {"kind": "squad"},
    "librispeech_test_clean": {"dataset": "librispeech_asr", "config": "clean", "split": "test", "kind": "hf"},
    "imagenet_1k": {"dataset": "ILSVRC/imagenet-1k", "config": None, "split": "validation", "kind": "hf"},
    "vqav2": {"kind": "vqa"},
    "mlt19": {
        "kind": "kaggle_mlt19",
        "dataset": "zubairalibhutto/mlt-19-ocr-dataset",
        "split": "train",
    },
}


def _json_value(value: Any, media_dir: Path, sample_id: str) -> Any:
    """Persist PIL/audio-like values and return JSON-safe manifest values."""
    bundle_root = media_dir.parents[1]

    def bundle_path(path: Path) -> str:
        return path.relative_to(bundle_root).as_posix()

    if isinstance(value, Path):
        return bundle_path(media_dir / value.name)
    if isinstance(value, dict):
        if "array" in value and "sampling_rate" in value:
            try:
                import soundfile as sf
            except ImportError as error:
                raise RuntimeError("install the profiling group to save LibriSpeech audio") from error
            path = media_dir / f"{sample_id}.wav"
            if not path.is_file():
                sf.write(path, value["array"], value["sampling_rate"])
            return bundle_path(path)
        return {str(key): _json_value(item, media_dir, sample_id) for key, item in value.items() if key != "array"}
    if hasattr(value, "save"):
        path = media_dir / f"{sample_id}.png"
        if not path.is_file():
            value.save(path)
        return bundle_path(path)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item, media_dir, sample_id) for item in value]
    return str(value)


def _row(row: dict[str, Any], index: int, media_dir: Path) -> dict[str, Any]:
    sample_id = str(row.get("id", row.get("image_id", row.get("file", index))))
    # ``file`` is a transient Hugging Face cache path.  The materialized
    # ``audio`` field is the portable copy used by the evaluator.
    return {"id": sample_id, **{key: _json_value(value, media_dir, sample_id) for key, value in row.items() if key not in {"id", "file"}}}


def download_huggingface(name: str, output: Path, count: int, token: str | None, seed: int) -> Path:
    spec = BENCHMARKS[name]
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("install datasets first: uv sync --group profiling") from error
    kwargs: dict[str, Any] = {"split": spec["split"], "streaming": True}
    if spec["config"] is not None:
        dataset = load_dataset(spec["dataset"], spec["config"], token=token, **kwargs)
    else:
        dataset = load_dataset(spec["dataset"], token=token, **kwargs)
    # The benchmark split order is frozen in the generated manifest. We take
    # the first N records rather than scanning an entire remote split.
    destination = output / name
    media_dir = destination / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    iterator = iter(dataset.take(count))
    try:
        for index, item in enumerate(iterator):
            rows.append(_row(dict(item), index, media_dir))
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
    if len(rows) != count:
        raise RuntimeError(f"{name}: requested {count} records but received {len(rows)}")
    path = destination / "samples.jsonl"
    path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows), encoding="utf-8")
    (destination / "provenance.json").write_text(json.dumps({"dataset": spec["dataset"], "config": spec["config"], "split": spec["split"], "count": count, "selection": "first records in pinned split order", "seed": seed}, indent=2) + "\n", encoding="utf-8")
    return path


def _download_json(url: str, destination: Path, ssl_context: ssl.SSLContext | None = None) -> Any:
    if not destination.is_file():
        partial = destination.with_suffix(destination.suffix + ".part")
        opener = urllib.request.urlopen(url, context=ssl_context) if ssl_context else urllib.request.urlopen(url)
        with opener as response, partial.open("wb") as file:
            shutil.copyfileobj(response, file)
        partial.replace(destination)
    return json.loads(destination.read_text(encoding="utf-8"))


def _download_json_from_zip(url: str, destination: Path, member_name: str, ssl_context: ssl.SSLContext | None = None) -> Any:
    if destination.is_file():
        return json.loads(destination.read_text(encoding="utf-8"))
    archive_path = destination.with_suffix(".zip")
    if not archive_path.is_file():
        partial = archive_path.with_suffix(".zip.part")
        opener = urllib.request.urlopen(url, context=ssl_context) if ssl_context else urllib.request.urlopen(url)
        with opener as response, partial.open("wb") as file:
            shutil.copyfileobj(response, file)
        partial.replace(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(member_name) as file:
            value = json.load(file)
    destination.write_text(json.dumps(value), encoding="utf-8")
    return value


def download_vqav2(output: Path, count: int, seed: int, ssl_context: ssl.SSLContext | None = None) -> Path:
    """Download 100 VQAv2 validation questions and only their images.

    The official VQA JSON files are small metadata files. COCO val2014 images
    are fetched individually, avoiding the multi-GB val2014 archive.
    """
    destination = output / "vqav2"
    media_dir = destination / "images"
    destination.mkdir(parents=True, exist_ok=True)
    media_dir.mkdir(parents=True, exist_ok=True)
    questions = _download_json_from_zip(
        "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
        destination / "questions.json",
        "v2_OpenEnded_mscoco_val2014_questions.json",
        ssl_context,
    )
    annotations = _download_json_from_zip(
        "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
        destination / "annotations.json",
        "v2_mscoco_val2014_annotations.json",
        ssl_context,
    )
    annotation_by_question = {int(item["question_id"]): item for item in annotations["annotations"]}
    rows: list[dict[str, Any]] = []
    for item in questions["questions"][:count]:
        question_id = int(item["question_id"])
        image_id = int(item["image_id"])
        image_path = media_dir / f"COCO_val2014_{image_id:012d}.jpg"
        if not image_path.is_file():
            url = f"https://images.cocodataset.org/val2014/COCO_val2014_{image_id:012d}.jpg"
            opener = urllib.request.urlopen(url, context=ssl_context) if ssl_context else urllib.request.urlopen(url)
            with opener as response, image_path.open("wb") as file:
                shutil.copyfileobj(response, file)
        answer_record = annotation_by_question.get(question_id)
        if answer_record is None:
            raise RuntimeError(f"VQAv2 annotation missing question_id={question_id}")
        rows.append({"id": str(question_id), "question_id": question_id, "image_id": image_id, "image": f"vqav2/images/{image_path.name}", "question": item["question"], "answers": answer_record["answers"], "multiple_choice_answer": answer_record["multiple_choice_answer"]})
    if len(rows) != count:
        raise RuntimeError(f"vqav2: requested {count} records but received {len(rows)}")
    path = destination / "samples.jsonl"
    path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows), encoding="utf-8")
    (destination / "provenance.json").write_text(json.dumps({"source": "GT-Vision-Lab/VQA v2.0", "split": "val2014", "count": count, "selection": "first questions in official JSON order", "seed": seed}, indent=2) + "\n", encoding="utf-8")
    return path


def download_squad_v2(output: Path, count: int, ssl_context: ssl.SSLContext | None = None) -> Path:
    """Download the official SQuAD v2 validation JSON and retain 100 QA rows."""
    destination = output / "squad_v2"
    destination.mkdir(parents=True, exist_ok=True)
    source = _download_json(
        "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
        destination / "dev-v2.0.json",
        ssl_context,
    )
    rows: list[dict[str, Any]] = []
    for article in source["data"]:
        for paragraph in article["paragraphs"]:
            for question in paragraph["qas"]:
                rows.append({"id": str(question["id"]), "question": question["question"], "context": paragraph["context"], "answers": question["answers"], "is_impossible": question["is_impossible"]})
                if len(rows) == count:
                    break
            if len(rows) == count:
                break
        if len(rows) == count:
            break
    if len(rows) != count:
        raise RuntimeError(f"squad_v2: requested {count} records but received {len(rows)}")
    path = destination / "samples.jsonl"
    path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows), encoding="utf-8")
    (destination / "provenance.json").write_text(json.dumps({"source": "SQuAD v2.0 official dev-v2.0.json", "split": "validation", "count": count, "selection": "first QA records in official file order"}, indent=2) + "\n", encoding="utf-8")
    return path


def _read_mlt19_ground_truth(path: Path) -> tuple[list[list[list[float]]], list[str], list[str]]:
    """Read the official MLT-19 ``x1,...,x4,script,transcription`` format."""
    boxes: list[list[list[float]]] = []
    scripts: list[str] = []
    transcriptions: list[str] = []
    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.reader(file):
            if len(row) < 10:
                continue
            coordinates = [float(value.strip()) for value in row[:8]]
            boxes.append([[coordinates[index], coordinates[index + 1]] for index in range(0, 8, 2)])
            scripts.append(row[8].strip())
            transcriptions.append(",".join(row[9:]).strip())
    if not boxes:
        raise RuntimeError(f"MLT-19 ground-truth file is empty or malformed: {path}")
    return boxes, scripts, transcriptions


def download_mlt19(output: Path, count: int, dataset: str) -> Path:
    """Download exactly ``count`` MLT-19 train image/GT pairs via Kaggle API."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as error:
        raise RuntimeError("install kaggle first: uv sync --group profiling") from error
    api = KaggleApi()
    api.authenticate()
    destination = output / "mlt19"
    image_dir = destination / "images"
    ground_truth_dir = destination / "gt"
    image_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index in range(1, count + 1):
        sample_id = f"tr_img_{index:05d}"
        image_path = next(
            (path for path in image_dir.iterdir() if path.stem == sample_id),
            None,
        )
        if image_path is None:
            image_path = _download_mlt19_file(
                api,
                dataset,
                [f"TrainImages/TrainImages/{sample_id}{suffix}" for suffix in (".jpg", ".JPG", ".png", ".PNG")],
                image_dir,
                sample_id,
            )
        ground_truth_name = f"{sample_id}.txt"
        ground_truth_path = ground_truth_dir / ground_truth_name
        if not ground_truth_path.is_file():
            ground_truth_path = _download_mlt19_file(
                api,
                dataset,
                [f"TrainGT/TrainGT/{sample_id}.txt"],
                ground_truth_dir,
                sample_id,
            )
        if not image_path.is_file() or not ground_truth_path.is_file():
            raise RuntimeError(f"Kaggle did not materialize the MLT-19 pair for {sample_id}")
        boxes, scripts, transcriptions = _read_mlt19_ground_truth(ground_truth_path)
        rows.append({
            "id": sample_id,
            "image": f"mlt19/images/{image_path.name}",
            "ground_truth": f"mlt19/gt/{ground_truth_name}",
            "boxes": boxes,
            "script": scripts,
            "transcription": transcriptions,
        })
    path = destination / "samples.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (destination / "provenance.json").write_text(
        json.dumps({
            "source": "ICDAR 2019 MLT via Kaggle",
            "kaggle_dataset": dataset,
            "split": "train",
            "count": count,
            "selection": "tr_img_00001 through tr_img_00100",
            "ground_truth": "MLT-19 quadrilateral, script, transcription files",
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _download_mlt19_file(api: Any, dataset: str, candidates: list[str], destination: Path, stem: str) -> Path:
    """Download the first existing Kaggle file, tolerating extension variants."""
    for remote_name in candidates:
        try:
            api.dataset_download_file(dataset, remote_name, path=str(destination), quiet=True)
        except Exception as error:
            if "404" not in str(error):
                raise
            continue
        downloaded = next((path for path in destination.iterdir() if path.stem == stem), None)
        if downloaded is not None:
            return downloaded
    raise RuntimeError(f"none of the Kaggle files exists: {', '.join(candidates)}")


def manifest_is_complete(path: Path, count: int, *, root: Path | None = None, requires_ground_truth: bool = False) -> bool:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError):
        return False
    if len(rows) != count:
        return False
    for row in rows:
        image = row.get("image")
        if image and not _manifest_path(image, root).is_file():
            return False
        ground_truth = row.get("ground_truth")
        if ground_truth and not _manifest_path(ground_truth, root).is_file():
            return False
        if requires_ground_truth and (
            not row.get("ground_truth")
            or not row.get("boxes")
            or not row.get("transcription")
        ):
            return False
    return True


def _manifest_path(value: str, root: Path | None) -> Path:
    path = Path(value)
    return path if path.is_absolute() or root is None else root / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/benchmarks"))
    parser.add_argument("--benchmark", choices=[*BENCHMARKS, "all"], action="append")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--hf-token", help="Hugging Face token; alternatively use `huggingface-cli login`")
    parser.add_argument("--kaggle-dataset", default="zubairalibhutto/mlt-19-ocr-dataset", help="Kaggle dataset ref for the MLT-19 OCR slice")
    parser.add_argument("--insecure", action="store_true", help="Disable TLS verification for VQAv2 downloads (use only with a trusted TLS-intercepting network)")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    selected = list(BENCHMARKS) if not args.benchmark or "all" in args.benchmark else args.benchmark

    # datasets' streaming loaders may own native/background threads. Run each
    # benchmark in a short-lived worker so a broken remote split cannot leave
    # a thread behind in the process that is coordinating all downloads.
    if not args._worker and len(selected) > 1:
        failures: list[str] = []
        for name in selected:
            command = [sys.executable, str(Path(__file__).resolve()), "--output", str(args.output), "--benchmark", name, "--count", str(args.count), "--seed", str(args.seed), "--_worker"]
            if args.hf_token:
                command.extend(["--hf-token", args.hf_token])
            if args.kaggle_dataset:
                command.extend(["--kaggle-dataset", args.kaggle_dataset])
            if args.insecure:
                command.append("--insecure")
            result = subprocess.run(command)
            if result.returncode:
                failures.append(name)
        if failures:
            raise SystemExit("failed benchmarks: " + ", ".join(failures))
        return

    args.output.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    ssl_context = ssl._create_unverified_context() if args.insecure else None
    if args.insecure:
        print("WARNING: TLS certificate verification is disabled for this run.")
    for name in selected:
        target = args.output / name / "samples.jsonl"
        if target.is_file():
            line_count = sum(1 for _ in target.open(encoding="utf-8"))
            if line_count == args.count and manifest_is_complete(target, args.count, root=args.output, requires_ground_truth=name == "mlt19"):
                print(f"Already present, skip: {target}")
            else:
                print(f"Existing manifest is incomplete ({line_count} records or missing media), repairing without redownloading existing media: {target}")
                target.unlink()
                if not target.exists():
                    pass
                else:
                    continue
        try:
            if BENCHMARKS[name]["kind"] == "kaggle_mlt19":
                print(download_mlt19(args.output, args.count, args.kaggle_dataset))
            elif BENCHMARKS[name]["kind"] == "vqa":
                print(download_vqav2(args.output, args.count, args.seed, ssl_context))
            elif BENCHMARKS[name]["kind"] == "squad":
                print(download_squad_v2(args.output, args.count, ssl_context))
            else:
                print(download_huggingface(name, args.output, args.count, args.hf_token, args.seed))
        except Exception as error:
            print(f"ERROR {name}: {type(error).__name__}: {error}")
            failures.append(name)
    if failures:
        message = "failed benchmarks: " + ", ".join(failures)
        if args._worker:
            print(message, flush=True)
            os._exit(1)
        raise SystemExit(message)
    if args._worker:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
