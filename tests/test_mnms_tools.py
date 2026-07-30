from __future__ import annotations

from pathlib import Path

from PIL import Image

from eec_sched import Configuration, InputSource, PlanningRequest, ToolCallPlan, ToolNode, ToolRegistry
from eec_sched.mnms_tools import HUGGINGFACE_MODEL_IDS, MNMS_MODELS, MnmsToolRunner, mnms_tool_specs, register_mnms_tools
from eec_sched.planning import validate_plan


def test_mnms_catalog_registers_all_reference_tools_without_loading_models() -> None:
    specs = mnms_tool_specs()

    assert len(specs) == 28
    assert {spec.tool_id for spec in specs} >= set(MNMS_MODELS)
    configured = {spec.tool_id: spec.configurations[0].parameters.get("model") for spec in specs if spec.configurations}
    assert {tool_id: configured[tool_id] for tool_id in MNMS_MODELS if tool_id not in {"text_generation", "image_generation", "image_editing"}} == {tool_id: model for tool_id, model in MNMS_MODELS.items() if tool_id not in {"text_generation", "image_generation", "image_editing"}}
    assert all(not next(spec for spec in specs if spec.tool_id == tool_id).configurations for tool_id in {"text_generation", "image_generation", "image_editing"})
    assert len(HUGGINGFACE_MODEL_IDS) == 10
    assert set(HUGGINGFACE_MODEL_IDS).issubset(set(MNMS_MODELS.values()))
    assert "stabilityai/stable-diffusion-xl-base-1.0" not in HUGGINGFACE_MODEL_IDS

    registry = ToolRegistry()
    register_mnms_tools(registry)
    assert len(registry.catalog()) == 28


def test_active_profile_configurations_match_the_adopted_finite_choices() -> None:
    specs = {spec.tool_id: spec for spec in mnms_tool_specs()}

    assert [(config.configuration_id, config.parameters["num_beams"]) for config in specs["automatic_speech_recognition"].configurations] == [(str(beams), beams) for beams in range(1, 6)]
    assert [config.parameters["input_size"] for config in specs["image_classification"].configurations] == [128, 192, 224]
    assert [config.parameters["shortest_edge"] for config in specs["object_detection"].configurations] == [480, 640, 800]
    assert [config.parameters["shortest_edge"] for config in specs["image_segmentation"].configurations] == [384, 512, 800]
    assert [config.parameters["decoder"] for config in specs["optical_character_recognition"].configurations] == ["greedy", "beamsearch", "beamsearch"]


def test_audio_path_is_a_valid_audio_request_value() -> None:
    registry = ToolRegistry()
    register_mnms_tools(registry)
    plan = ToolCallPlan((ToolNode("asr", "automatic_speech_recognition", {"audio": InputSource.request("audio")}),), ())

    errors = validate_plan(plan, PlanningRequest("transcribe", {"audio": Path("sample.wav")}, 0, 1, 0.5), registry)

    assert [error.code for error in errors] == ["missing_final_output"]


def test_local_image_and_structured_text_tools_preserve_reference_semantics() -> None:
    image = Image.new("RGB", (10, 6), "red")
    crop = MnmsToolRunner("image_crop")
    count = MnmsToolRunner("count")
    selected = MnmsToolRunner("select_object")
    configuration = Configuration("reference")
    for runner in (crop, count, selected):
        runner.prepare(configuration)

    assert crop.run({"image": image, "object": '{"bbox":[0.0,0.0,0.5,0.99]}'}, configuration)["image"].size == (5, 6)
    assert count.run({"objects": '[{"label":"dog"},{"label":"cat"}]'}, configuration) == {"text": "2"}
    assert selected.run({"objects": '[{"label":"small dog"},{"label":"cat"}]', "object_name": "dog"}, configuration) == {"object": '{"label":"small dog"}'}
