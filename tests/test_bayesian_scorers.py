from __future__ import annotations

import pytest

from eec_sched.profiling.scorers import (
    cider,
    icdar2015_hmean,
    imagenet_top1,
    mlt19_hmean,
    squad_v2_f1,
    vqa_accuracy,
    vqa_normalize_answer,
)


def test_squad_v2_f1_handles_unanswerable_questions_and_aliases() -> None:
    pytest.importorskip("evaluate")
    score = squad_v2_f1(
        ["France", ""],
        [
            {"id": "answerable", "answers": {"text": ["France"], "answer_start": [0]}},
            {"id": "unanswerable", "answers": {"text": [], "answer_start": []}},
        ],
    )
    assert score == pytest.approx(100.0)


def test_cider_uses_reference_sets() -> None:
    pytest.importorskip("pycocoevalcap")
    # Multiple documents avoid the degenerate all-document IDF=0 case.
    score = cider(
        ["a black cat sits on the mat", "a red car drives on the road"],
        [
            ["a black cat sits on the mat", "a cat sits on a mat"],
            ["a red car drives on the road", "a car is driving down the road"],
        ],
    )
    assert score > 0


def test_vqa_normalization_and_soft_accuracy() -> None:
    assert vqa_normalize_answer("The, two cats!") == "2 cats"
    assert vqa_accuracy(
        ["two cats"],
        [[{"answer": "2 cats"}, {"answer": "two cats"}, {"answer": "2 cats"}, {"answer": "dogs"}]],
    ) == pytest.approx(1.0)
    assert vqa_accuracy(
        ["cat"], [[{"answer": "cat"}, {"answer": "cat"}, {"answer": "dog"}]]
    ) == pytest.approx(2 / 3)


def test_imagenet_top1_scores_class_ids_not_display_names() -> None:
    assert imagenet_top1([{"index": 5}, "LABEL_7"], [5, 7]) == 1.0


def test_icdar_hmean_requires_text_and_polygon_overlap() -> None:
    target = {"boxes": [[[0, 0], [10, 0], [10, 10], [0, 10]]], "transcription": ["Hello"]}
    assert icdar2015_hmean(
        [[{"box": [[0, 0], [10, 0], [10, 10], [0, 10]], "text": "hello"}]], [target]
    ) == 1.0
    assert icdar2015_hmean(
        [[{"box": [[0, 0], [10, 0], [10, 10], [0, 10]], "text": "wrong"}]], [target]
    ) == 0.0


def test_mlt19_hmean_uses_case_insensitive_exact_transcription() -> None:
    target = {"boxes": [[[0, 0], [10, 0], [10, 10], [0, 10]]], "transcription": ["Hello!"]}
    assert mlt19_hmean(
        [[{"box": [[0, 0], [10, 0], [10, 10], [0, 10]], "text": "hello!"}]], [target]
    ) == 1.0
    assert mlt19_hmean(
        [[{"box": [[0, 0], [10, 0], [10, 10], [0, 10]], "text": "hello"}]], [target]
    ) == 0.0
