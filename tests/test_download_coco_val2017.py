"""Tests for the COCO downloader's local, network-free extraction seam."""

from __future__ import annotations

import importlib.util
import io
import sys
import zipfile
from pathlib import Path


def load_downloader():
    path = Path("experiments/02_bayesian_optimization/download_coco_val2017.py")
    spec = importlib.util.spec_from_file_location("coco_downloader", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_extract_members_writes_only_requested_annotation(tmp_path: Path) -> None:
    downloader = load_downloader()
    archive = tmp_path / "annotations.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("annotations/instances_val2017.json", "{}")
        output.writestr("annotations/captions_val2017.json", "not requested")

    destination = tmp_path / "coco"
    downloader.extract_members(archive, destination, ("annotations/instances_val2017.json",))

    assert (destination / "annotations/instances_val2017.json").is_file()
    assert not (destination / "annotations/captions_val2017.json").exists()


def test_extract_members_supports_optional_val2014_caption_images(tmp_path: Path) -> None:
    downloader = load_downloader()
    archive = tmp_path / "val2014.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("val2014/example.jpg", b"image")

    destination = tmp_path / "coco"
    downloader.extract_members(archive, destination, ("val2014",))

    assert (destination / "val2014/example.jpg").read_bytes() == b"image"


def test_download_passes_an_explicit_ssl_context_to_urlopen(
    tmp_path: Path, monkeypatch
) -> None:
    downloader = load_downloader()
    seen: dict[str, object] = {}

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *unused):
            self.close()

    def fake_urlopen(url, *, context):
        seen["url"] = url
        seen["context"] = context
        return Response(b"fixture")

    monkeypatch.setattr(downloader.urllib.request, "urlopen", fake_urlopen)
    context = object()
    destination = tmp_path / "archive.zip"

    downloader.download("https://example.test/archive.zip", destination, context)

    assert destination.read_bytes() == b"fixture"
    assert seen == {"url": "https://example.test/archive.zip", "context": context}
