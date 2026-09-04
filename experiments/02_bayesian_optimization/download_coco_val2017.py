"""Download the exact COCO files required by the Bayesian-optimization demo.

The resulting root has the layout expected by ``config.example.json``:

    <root>/val2017/
    <root>/annotations/instances_val2017.json
    <root>/annotations/panoptic_val2017.json

With ``--with-captions`` it also contains COCO val2014 images and
``annotations/captions_val2014.json``.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import ssl
import urllib.request
import zipfile
from pathlib import Path


ARTIFACTS = (
    (
        "val2017 images",
        "https://images.cocodataset.org/zips/val2017.zip",
        ("val2017",),
    ),
    (
        "COCO instance annotations",
        "https://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        ("annotations/instances_val2017.json",),
    ),
    (
        "COCO panoptic annotations",
        "https://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip",
        ("annotations/panoptic_val2017.json",),
    ),
)

CAPTION_ARTIFACTS = (
    (
        "COCO val2014 images",
        "https://images.cocodataset.org/zips/val2014.zip",
        ("val2014",),
    ),
    (
        "COCO caption annotations",
        "https://images.cocodataset.org/annotations/annotations_trainval2014.zip",
        ("annotations/captions_val2014.json",),
    ),
)

CAPTION_ANNOTATION_ARTIFACT = (
    "COCO caption annotations",
    "https://images.cocodataset.org/annotations/annotations_trainval2014.zip",
    ("annotations/captions_val2014.json",),
)


def required_path_exists(root: Path, member: str) -> bool:
    path = root / member
    return path.is_dir() if member in {"val2017", "val2014"} else path.is_file()


def download(
    url: str, destination: Path, ssl_context: ssl.SSLContext | None = None
) -> None:
    """Download to a sibling temporary file, then atomically publish it."""
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    try:
        expected_size: int | None = None
        received_size = 0
        while True:
            request = (
                urllib.request.Request(url, headers={"Range": f"bytes={received_size}-"})
                if received_size
                else url
            )
            opener = (
                urllib.request.urlopen(request)
                if ssl_context is None
                else urllib.request.urlopen(request, context=ssl_context)
            )
            with opener as response:
                status = getattr(response, "status", None)
                if received_size and status != 206:
                    raise RuntimeError(
                        f"Server did not honor range request for {url} (status {status})"
                    )
                headers = getattr(response, "headers", {})
                content_range = headers.get("Content-Range", "")
                range_match = re.search(r"/([0-9]+)$", content_range)
                if range_match:
                    expected_size = int(range_match.group(1))
                elif not received_size:
                    content_length = headers.get("Content-Length")
                    if content_length:
                        expected_size = int(content_length)
                with partial.open("ab" if received_size else "wb") as output:
                    shutil.copyfileobj(response, output)
            received_size = partial.stat().st_size
            if expected_size is None or received_size >= expected_size:
                break
            if received_size == 0:
                raise RuntimeError(f"Empty response while downloading {url}")
        if expected_size is not None and received_size != expected_size:
            raise RuntimeError(
                f"Incomplete download for {url}: {received_size} of {expected_size} bytes"
            )
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def extract_members(archive: Path, root: Path, members: tuple[str, ...]) -> None:
    with zipfile.ZipFile(archive) as source:
        available = set(source.namelist())
        missing = [member for member in members if member not in available and not any(name.startswith(member + "/") for name in available)]
        if missing:
            raise RuntimeError(f"Archive {archive.name} does not contain: {', '.join(missing)}")
        for member in members:
            if member in {"val2017", "val2014"}:
                for info in source.infolist():
                    if info.filename.startswith(member + "/"):
                        source.extract(info, root)
            else:
                source.extract(member, root)


def ensure_artifact(
    root: Path,
    name: str,
    url: str,
    members: tuple[str, ...],
    keep_archives: bool,
    ssl_context: ssl.SSLContext | None,
) -> None:
    if all(required_path_exists(root, member) for member in members):
        print(f"Already present: {name}")
        return
    archive_dir = root / ".downloads"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive = archive_dir / url.rsplit("/", maxsplit=1)[-1]
    if not archive.is_file():
        print(f"Downloading: {name}")
        download(url, archive, ssl_context)
    else:
        print(f"Using cached archive: {archive.name}")
    print(f"Extracting: {name}")
    extract_members(archive, root, members)
    if not keep_archives:
        archive.unlink()


def download_caption_subset(
    root: Path,
    count: int,
    seed: int,
    keep_archive: bool,
    ssl_context: ssl.SSLContext | None,
) -> None:
    """Download only the fixed COCO-caption image subset needed by profiling."""
    name, url, members = CAPTION_ANNOTATION_ARTIFACT
    ensure_artifact(root, name, url, members, keep_archive, ssl_context)
    annotation_path = root / "annotations" / "captions_val2014.json"
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    image_ids = sorted({int(item["image_id"]) for item in payload["annotations"]})
    if count > len(image_ids):
        raise ValueError(f"requested {count} caption images but COCO has only {len(image_ids)}")
    selected = sorted(random.Random(seed).sample(image_ids, count))
    image_directory = root / "val2014"
    image_directory.mkdir(parents=True, exist_ok=True)
    for image_id in selected:
        destination = image_directory / f"COCO_val2014_{image_id:012d}.jpg"
        if destination.is_file():
            continue
        download(
            f"https://images.cocodataset.org/val2014/COCO_val2014_{image_id:012d}.jpg",
            destination,
            ssl_context,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("external/coco"),
        help="COCO root directory (default: external/coco)",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep downloaded ZIP files under <root>/.downloads for reuse.",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help=(
            "Disable TLS certificate validation. Use only for a trusted network "
            "whose TLS interception causes certificate hostname failures."
        ),
    )
    parser.add_argument(
        "--with-captions",
        action="store_true",
        help="Also download COCO val2014 images and captions for image-captioning profiling.",
    )
    parser.add_argument(
        "--caption-count",
        type=int,
        help="Download only this many fixed val2014 caption images plus caption metadata.",
    )
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()
    if args.caption_count is not None and args.caption_count < 1:
        parser.error("--caption-count must be positive")
    if args.caption_count is not None and args.with_captions:
        parser.error("use either --caption-count or --with-captions, not both")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    ssl_context = ssl._create_unverified_context() if args.insecure else None
    if args.insecure:
        print("WARNING: TLS certificate validation is disabled for this download.")
    artifacts = () if args.caption_count is not None else ARTIFACTS + (CAPTION_ARTIFACTS if args.with_captions else ())
    for name, url, members in artifacts:
        ensure_artifact(
            root, name, url, members, args.keep_archives, ssl_context
        )
    if args.caption_count is not None:
        download_caption_subset(root, args.caption_count, args.seed, args.keep_archives, ssl_context)
    print(f"COCO val2017 is ready at {root}")


if __name__ == "__main__":
    main()
