"""MnMS-compatible tool catalog and lazily loaded runners.

The public seam remains :class:`ToolRunner`: model libraries are imported only
inside ``prepare``/``run``.  Consequently catalog construction and planning do
not download weights, initialise caches, or require optional MnMS dependencies.
"""

from __future__ import annotations

import ast
import io
import json
import os
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, cast, overload

from PIL import Image

from .domain import Configuration, Port, ToolRegistry, ToolRunner, ToolSpec


MNMS_MODELS: dict[str, str] = {
    "text_generation": "gpt-3.5-turbo-0125",
    "text_summarization": "facebook/bart-large-cnn",
    "text_classification": "distilbert-base-uncased-finetuned-sst-2-english",
    "question_answering": "deepset/roberta-base-squad2",
    "automatic_speech_recognition": "openai/whisper-small",
    "image_generation": "stabilityai/stable-diffusion-xl-base-1.0",
    "image_captioning": "Salesforce/blip-image-captioning-large",
    "image_editing": "stabilityai/stable-diffusion-xl-refiner-1.0",
    "image_classification": "google/vit-base-patch16-224",
    "visual_question_answering": "Salesforce/blip-vqa-base",
    "object_detection": "facebook/detr-resnet-101",
    "image_segmentation": "facebook/maskformer-swin-base-coco",
    "optical_character_recognition": "easyOCR",
}

# OpenAI models and EasyOCR are intentionally excluded: they are not Hugging
# Face repositories. SDXL Base is intentionally excluded from the optional
# pre-download workflow because image generation is not prepared in advance.
HUGGINGFACE_MODEL_IDS: tuple[str, ...] = (
    "facebook/bart-large-cnn",
    "distilbert-base-uncased-finetuned-sst-2-english",
    "deepset/roberta-base-squad2",
    "openai/whisper-small",
    "Salesforce/blip-image-captioning-large",
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    "google/vit-base-patch16-224",
    "Salesforce/blip-vqa-base",
    "facebook/detr-resnet-101",
    "facebook/maskformer-swin-base-coco",
)


def _configuration(tool_id: str) -> tuple[Configuration, ...]:
    model = MNMS_MODELS.get(tool_id)
    profiles: dict[str, tuple[tuple[str, dict[str, object]], ...]] = {
        "text_summarization": (("fast", {"max_length": 64, "min_length": 16, "num_beams": 1}), ("balanced", {"max_length": 96, "min_length": 24, "num_beams": 2}), ("quality", {"max_length": 130, "min_length": 30, "num_beams": 4})),
        "text_classification": (("fast", {"max_tokens": 32}), ("balanced", {"max_tokens": 64}), ("quality", {"max_tokens": 128})),
        "question_answering": (("fast", {"max_context_tokens": 128, "doc_stride": 64}), ("balanced", {"max_context_tokens": 256, "doc_stride": 96}), ("quality", {"max_context_tokens": 384, "doc_stride": 128})),
        "automatic_speech_recognition": tuple((str(beams), {"num_beams": beams}) for beams in range(1, 6)),
        "image_captioning": (("fast", {"max_new_tokens": 16, "num_beams": 1}), ("balanced", {"max_new_tokens": 32, "num_beams": 3}), ("quality", {"max_new_tokens": 64, "num_beams": 5})),
        "image_classification": (("fast", {"input_size": 128}), ("balanced", {"input_size": 192}), ("quality", {"input_size": 224})),
        "visual_question_answering": (("fast", {"max_new_tokens": 5, "num_beams": 1}), ("balanced", {"max_new_tokens": 10, "num_beams": 3}), ("quality", {"max_new_tokens": 20, "num_beams": 5})),
        "object_detection": (("fast", {"shortest_edge": 480}), ("balanced", {"shortest_edge": 640}), ("quality", {"shortest_edge": 800})),
        "image_segmentation": (("fast", {"shortest_edge": 384}), ("balanced", {"shortest_edge": 512}), ("quality", {"shortest_edge": 800})),
        "optical_character_recognition": (("fast", {"canvas_size": 1280, "mag_ratio": 1.0, "decoder": "greedy"}), ("balanced", {"canvas_size": 1920, "mag_ratio": 1.5, "decoder": "beamsearch", "beam_width": 3}), ("quality", {"canvas_size": 2560, "mag_ratio": 2.0, "decoder": "beamsearch", "beam_width": 5})),
    }
    if tool_id in {"text_generation", "image_generation", "image_editing"}:
        return ()
    if tool_id in profiles:
        return tuple(Configuration(name, {"model": model, **parameters}) for name, parameters in profiles[tool_id])
    return (Configuration("reference", {"model": model}),) if model else (Configuration("reference"),)


def mnms_tool_specs() -> tuple[ToolSpec, ...]:
    """Return the complete MnMS catalog without importing any model runtime."""
    image = Port("image", "image")
    text = Port("text", "text")
    audio = Port("audio", "audio")
    object_text = Port("object", "text")
    objects_text = Port("objects", "text")
    def spec(tool_id: str, description: str, inputs: dict[str, Port], outputs: dict[str, Port]) -> ToolSpec:
        return ToolSpec(tool_id, description, inputs, outputs, _configuration(tool_id), True, 0.0)
    return (
        spec("text_generation", "Generate text from text.", {"text": text}, {"text": text}),
        spec("text_summarization", "Summarize text.", {"text": text}, {"text": text}),
        spec("text_classification", "Classify text sentiment.", {"text": text}, {"text": text}),
        spec("question_answering", "Answer a question using supplied text.", {"question": text, "text": text}, {"text": text}),
        spec("automatic_speech_recognition", "Transcribe audio.", {"audio": audio}, {"text": text}),
        spec("image_generation", "Generate an image from text.", {"text": text}, {"image": image}),
        spec("image_captioning", "Describe an image.", {"image": image}, {"text": text}),
        spec("image_editing", "Edit an image according to text.", {"image": image, "prompt": text}, {"image": image}),
        spec("image_classification", "Classify an image.", {"image": image}, {"text": text}),
        spec("visual_question_answering", "Answer a question about an image.", {"image": image, "question": text}, {"text": text}),
        spec("object_detection", "Detect objects in an image.", {"image": image}, {"image": image, "objects": objects_text}),
        spec("image_segmentation", "Segment objects in an image.", {"image": image}, {"image": image, "objects": objects_text}),
        spec("optical_character_recognition", "Read text in an image.", {"image": image}, {"text": text}),
        spec("image_crop", "Crop an image to a bounding box.", {"image": image, "object": object_text}, {"image": image}),
        spec("image_crop_left", "Crop the left half of an image.", {"image": image}, {"image": image}),
        spec("image_crop_right", "Crop the right half of an image.", {"image": image}, {"image": image}),
        spec("image_crop_top", "Crop the top half of an image.", {"image": image}, {"image": image}),
        spec("image_crop_bottom", "Crop the bottom half of an image.", {"image": image}, {"image": image}),
        spec("background_blur", "Blur an image background.", {"image": image, "object": Port("object", "text", False)}, {"image": image}),
        spec("color_pop", "Keep an object coloured and desaturate its background.", {"image": image, "object": Port("object", "text", False)}, {"image": image}),
        spec("count", "Count objects.", {"objects": objects_text}, {"text": text}),
        spec("tag", "Draw object labels on an image.", {"image": image, "objects": objects_text}, {"image": image}),
        spec("emoji", "Overlay an emoji on an object.", {"image": image, "object": object_text, "emoji": text}, {"image": image}),
        spec("select_object", "Select a named object.", {"objects": objects_text, "object_name": text}, {"object": object_text}),
        spec("get_date_fact", "Get a date fact.", {"date": text}, {"text": text}),
        spec("get_year_fact", "Get a year fact.", {"year": text}, {"text": text}),
        spec("get_math_fact", "Get a mathematical number fact.", {"number": text}, {"text": text}),
        spec("get_trivia_fact", "Get a trivia number fact.", {"number": text}, {"text": text}),
    )


def _text(value: object) -> str:
    value = str(value)
    path = Path(value)
    try:
        exists = path.is_file()
    except OSError:
        # Long natural-language values are never viable filesystem paths.
        exists = False
    if exists and path.suffix.lower() == ".txt":
        return path.read_text()
    if exists and path.suffix.lower() in {".doc", ".docx"}:
        import textract  # type: ignore[import-not-found]
        return textract.process(str(path)).decode("utf-8")
    return value


def _image(value: object) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    path = Path(str(value))
    if path.is_file():
        return Image.open(path).convert("RGB")
    raise FileNotFoundError(str(value))


@overload
def _decode(value: object, *, many: Literal[False] = False) -> dict[str, Any]: ...


@overload
def _decode(value: object, *, many: Literal[True]) -> list[dict[str, Any]]: ...


def _decode(value: object, *, many: bool = False) -> dict[str, Any] | list[dict[str, Any]]:
    if isinstance(value, (dict, list)):
        result = value
    else:
        try:
            result = json.loads(str(value))
        except json.JSONDecodeError:
            result = ast.literal_eval(str(value))
    if many:
        if not isinstance(result, list):
            raise ValueError("objects must be a list")
        return result
    if not isinstance(result, dict):
        raise ValueError("object must be a mapping")
    return result


def _encode(value: object) -> str:
    return json.dumps(value, separators=(",", ":"))


class MnmsToolRunner(ToolRunner):
    """One deep runner implementation for every MnMS tool.

    The runner hides model-specific processors, optional dependencies, and
    structured-object text codecs behind the normal registry runner seam.
    """

    def __init__(self, tool_id: str, device: str = "auto") -> None:
        self.tool_id = tool_id
        self.device = device
        self._prepared: dict[str, Any] = {}
        # This is deliberately not part of ``run``'s public mapping: masks and
        # OCR quadrilaterals are evaluator inputs, not connectable DAG values.
        self._evaluation_output: Mapping[str, object] = {}

    def evaluation_output(self) -> Mapping[str, object]:
        """Return evaluator-only details produced by the most recent run."""
        return self._evaluation_output

    def _device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    def prepare(self, configuration: Configuration) -> None:
        key = configuration.configuration_id
        if key in self._prepared:
            return
        model = str(configuration.parameters.get("model", MNMS_MODELS.get(self.tool_id, "")))
        if self.tool_id in {"text_summarization", "text_classification", "question_answering", "image_captioning"}:
            from transformers import pipeline  # type: ignore[import-not-found]
            task = {"text_summarization": "summarization", "text_classification": "text-classification", "question_answering": "question-answering", "image_captioning": "image-to-text"}[self.tool_id]
            pipeline_device = 0 if self._device().startswith("cuda") else -1
            self._prepared[key] = cast(Callable[..., Any], pipeline)(task, model=model, device=pipeline_device)
        elif self.tool_id == "automatic_speech_recognition":
            from transformers import WhisperForConditionalGeneration, WhisperProcessor  # type: ignore[import-not-found]
            processor = WhisperProcessor.from_pretrained(model)
            loaded = cast(Any, WhisperForConditionalGeneration.from_pretrained(model)).to(self._device())
            loaded.eval()
            setattr(loaded, "forced_decoder_ids", None)
            self._prepared[key] = (processor, loaded)
        elif self.tool_id == "image_editing":
            import torch
            from diffusers import StableDiffusionXLImg2ImgPipeline  # type: ignore[import-not-found]
            dtype = torch.float16 if self._device().startswith("cuda") else torch.float32
            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(model, torch_dtype=dtype, variant="fp16" if dtype == torch.float16 else None, use_safetensors=True)
            self._prepared[key] = pipe.to(self._device())
        elif self.tool_id == "image_classification":
            from transformers import ViTForImageClassification, ViTImageProcessor  # type: ignore[import-not-found]
            loaded = cast(Any, ViTForImageClassification.from_pretrained(model)).to(self._device())
            loaded.eval()
            self._prepared[key] = (ViTImageProcessor.from_pretrained(model), loaded)
        elif self.tool_id == "visual_question_answering":
            import torch
            from transformers import BlipForQuestionAnswering, BlipProcessor  # type: ignore[import-not-found]
            dtype = torch.float16 if self._device().startswith("cuda") else torch.float32
            loaded = cast(Any, BlipForQuestionAnswering.from_pretrained(model, torch_dtype=dtype)).to(self._device())
            self._prepared[key] = (BlipProcessor.from_pretrained(model), loaded, dtype)
        elif self.tool_id == "object_detection":
            from transformers import DetrForObjectDetection, DetrImageProcessor  # type: ignore[import-not-found]
            loaded = cast(Any, DetrForObjectDetection.from_pretrained(model, revision="no_timm")).to(self._device())
            loaded.eval()
            self._prepared[key] = (DetrImageProcessor.from_pretrained(model, revision="no_timm"), loaded)
        elif self.tool_id == "image_segmentation":
            import transformers  # type: ignore[import-not-found]
            extractor = transformers.MaskFormerFeatureExtractor.from_pretrained(model)
            loaded = cast(Any, transformers.MaskFormerForInstanceSegmentation.from_pretrained(model)).to(self._device())
            loaded.eval()
            self._prepared[key] = (extractor, loaded)
        elif self.tool_id == "optical_character_recognition":
            import easyocr  # type: ignore[import-not-found]
            self._prepared[key] = easyocr.Reader(["en"], gpu=self._device().startswith("cuda"))
        else:
            self._prepared[key] = None

    def run(self, inputs: Mapping[str, object], configuration: Configuration) -> Mapping[str, object]:
        prepared = self._prepared[configuration.configuration_id]
        self._evaluation_output = {}
        text = lambda name: _text(inputs[name])
        if self.tool_id == "text_generation":
            from openai import OpenAI  # type: ignore[import-not-found]
            response = OpenAI().chat.completions.create(model=MNMS_MODELS[self.tool_id], messages=[{"role": "user", "content": [{"type": "text", "text": text("text")}]}], max_tokens=300)
            return {"text": response.choices[0].message.content or ""}
        if self.tool_id == "text_summarization":
            return {"text": prepared(text("text"), max_length=configuration.parameters.get("max_length", 130), min_length=configuration.parameters.get("min_length", 30), num_beams=configuration.parameters.get("num_beams", 4), do_sample=False)[0]["summary_text"]}
        if self.tool_id == "text_classification":
            return {"text": prepared(text("text"), truncation=True, max_length=configuration.parameters.get("max_tokens", 128))[0]["label"]}
        if self.tool_id == "question_answering":
            return {"text": prepared(question=text("question"), context=text("text"), max_seq_len=configuration.parameters.get("max_context_tokens", 384), doc_stride=configuration.parameters.get("doc_stride", 128))["answer"]}
        if self.tool_id == "automatic_speech_recognition":
            import librosa  # type: ignore[import-not-found]
            processor, model = prepared
            samples, rate = librosa.load(str(inputs["audio"]), sr=16_000)
            features = processor(samples, sampling_rate=rate, return_tensors="pt").input_features.to(self._device())
            generated = model.generate(features, num_beams=configuration.parameters.get("num_beams", 5))
            return {"text": processor.batch_decode(generated, skip_special_tokens=True)[0]}
        if self.tool_id == "image_generation":
            prompt = text("text")
            if len(prompt) >= 75:
                import requests  # type: ignore[import-not-found]
                from openai import OpenAI  # type: ignore[import-not-found]
                response = OpenAI().images.generate(model="dall-e-3", prompt=prompt, size="1024x1024", quality="hd", n=1)
                downloaded = requests.get(response.data[0].url, timeout=60)
                downloaded.raise_for_status()
                return {"image": Image.open(io.BytesIO(downloaded.content)).convert("RGB")}
            import torch
            from diffusers import DiffusionPipeline  # type: ignore[import-not-found]
            pipe = DiffusionPipeline.from_pretrained(MNMS_MODELS[self.tool_id], torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to("cuda")
            return {"image": pipe(prompt=prompt).images[0]}
        if self.tool_id == "image_captioning":
            return {"text": prepared(_image(inputs["image"]), generate_kwargs={"max_new_tokens": configuration.parameters.get("max_new_tokens", 64), "num_beams": configuration.parameters.get("num_beams", 5), "do_sample": False})[0]["generated_text"]}
        if self.tool_id == "image_editing":
            return {"image": prepared(text("prompt"), image=_image(inputs["image"])).images[0]}
        if self.tool_id == "image_classification":
            processor, model = prepared
            size = int(configuration.parameters.get("input_size", 224))
            encoded = processor(images=_image(inputs["image"]), return_tensors="pt", size={"height": size, "width": size})
            encoded = {name: value.to(self._device()) for name, value in encoded.items()}
            outputs = model(**encoded, interpolate_pos_encoding=size != 224)
            return {"text": model.config.id2label[outputs.logits.argmax(-1).item()]}
        if self.tool_id == "visual_question_answering":
            processor, model, dtype = prepared
            image = _image(inputs["image"])
            encoded = processor(image, text("question"), return_tensors="pt").to(self._device(), dtype)
            generated = model.generate(**encoded, max_new_tokens=configuration.parameters.get("max_new_tokens", 20), num_beams=configuration.parameters.get("num_beams", 5), do_sample=False)
            return {"text": processor.decode(generated[0], skip_special_tokens=True)}
        if self.tool_id == "object_detection":
            import torch
            processor, model = prepared
            image = _image(inputs["image"])
            shortest_edge = int(configuration.parameters.get("shortest_edge", 800))
            encoded = processor(images=image, return_tensors="pt", size={"shortest_edge": shortest_edge, "longest_edge": 1333})
            encoded = {name: value.to(self._device()) for name, value in encoded.items()}
            results = processor.post_process_object_detection(model(**encoded), target_sizes=torch.tensor([image.size[::-1]]), threshold=0.5)[0]
            objects = [{"bbox": [round(item, 2) for item in box.tolist()], "label": model.config.id2label[label.item()]} for label, box in zip(results["labels"], results["boxes"])]
            self._evaluation_output = {"detections": [{"bbox": box.tolist(), "label_id": label.item(), "score": score.item()} for label, box, score in zip(results["labels"], results["boxes"], results["scores"])]}
            return {"image": image, "objects": _encode(objects)}
        if self.tool_id == "image_segmentation":
            import numpy as np  # type: ignore[import-not-found]
            import torch
            extractor, model = prepared
            image = _image(inputs["image"])
            encoded = {name: value.to(self._device()) for name, value in extractor(images=image, return_tensors="pt", size={"shortest_edge": int(configuration.parameters.get("shortest_edge", 800)), "longest_edge": 1333}).items()}
            with torch.no_grad():
                output = extractor.post_process_panoptic_segmentation(model(**encoded))[0]
            instance_map = output["segmentation"].cpu().numpy()
            objects: list[dict[str, object]] = []
            for segment in output["segments_info"]:
                mask = (instance_map == segment["id"]).astype(float)
                resized = np.array(Image.fromarray(mask).resize(image.size, resample=Image.Resampling.BILINEAR))
                y, x = np.where(resized > 0.5)
                if len(x):
                    objects.append({"bbox": [int(x.min()), int(y.min()), int(x.max()), int(y.max())], "label": model.config.id2label[segment["label_id"]], "inst_id": segment["id"]})
            self._evaluation_output = {"segments": [{"mask": (instance_map == segment["id"]), "label_id": segment["label_id"], "score": segment.get("score", 1.0)} for segment in output["segments_info"]]}
            return {"image": image, "objects": _encode(objects)}
        if self.tool_id == "optical_character_recognition":
            image = _image(inputs["image"])
            buffer = io.BytesIO(); image.save(buffer, format="JPEG"); buffer.seek(0)
            details = prepared.readtext(buffer, decoder=configuration.parameters.get("decoder", "beamsearch"), beamWidth=configuration.parameters.get("beam_width", 5), canvas_size=configuration.parameters.get("canvas_size", 2560), mag_ratio=configuration.parameters.get("mag_ratio", 2.0))
            self._evaluation_output = {"detections": [{"box": box, "text": value, "confidence": confidence} for box, value, confidence in details]}
            return {"text": ", ".join(value for _, value, _ in details)}
        return self._run_image_or_utility(inputs)

    def _run_image_or_utility(self, inputs: Mapping[str, object]) -> Mapping[str, object]:
        tool = self.tool_id
        if tool.startswith("image_crop"):
            image = _image(inputs["image"]); width, height = image.size
            boxes = {"image_crop_left": (0, 0, int(width / 2), height - 1), "image_crop_right": (int(width / 2), 0, width - 1, height - 1), "image_crop_top": (0, 0, width - 1, int(height / 2)), "image_crop_bottom": (0, int(height / 2), width - 1, height - 1)}
            if tool == "image_crop":
                box = _decode(inputs["object"])["bbox"]
                if isinstance(box, str): box = ast.literal_eval(box)
                if len(box) == 4 and all(float(value) < 1.0 for value in box): box = [float(box[0]) * width, float(box[1]) * height, float(box[2]) * width, float(box[3]) * height]
                if len(box) == 4:
                    crop_box = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
                    return {"image": image.crop(crop_box)}
                return {"image": image}
            return {"image": image.crop(boxes[tool])}
        if tool in {"background_blur", "color_pop"}:
            import cv2  # type: ignore[import-not-found]
            import numpy as np  # type: ignore[import-not-found]
            from PIL import ImageFilter
            image = _image(inputs["image"]); obj = _decode(inputs.get("object", "{}"))
            original = np.array(image).astype(float)
            alternate = np.array(image.filter(ImageFilter.GaussianBlur(radius=2)) if tool == "background_blur" else image.convert("L").convert("RGB")).astype(float)
            if "mask" in obj:
                mask, _, _ = cv2.grabCut(original.astype("uint8"), np.array(obj["mask"]).astype("uint8"), None, np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64), 5, cv2.GC_INIT_WITH_MASK)
                selected = np.tile(mask[:, :, None].astype(float), (1, 1, 3))
                if tool == "background_blur":
                    selected = np.array(Image.fromarray(255 * selected.astype("uint8")).filter(ImageFilter.GaussianBlur(radius=5))).astype(float) / 255
                alternate = selected * original + (1 - selected) * alternate
            return {"image": Image.fromarray(alternate.astype("uint8"))}
        if tool == "count": return {"text": str(len(_decode(inputs["objects"], many=True)))}
        if tool == "select_object":
            needle = _text(inputs["object_name"]); chosen: dict[str, Any] = {}
            for obj in _decode(inputs["objects"], many=True):
                label = str(obj.get("label", ""))
                if needle in label or set(label.split()).intersection(needle.split()): chosen = obj; break
            return {"object": _encode(chosen)}
        if tool == "tag":
            from PIL import ImageDraw, ImageFont
            image = _image(inputs["image"]).copy(); draw = ImageDraw.Draw(image); font = ImageFont.load_default(); width, height = image.size
            for obj in _decode(inputs["objects"], many=True):
                x1, y1, x2, y2 = obj["bbox"]; label = str(obj["label"]); draw.rectangle((x1, y1, x2, y2), outline="green", width=4)
                font_box = font.getbbox(label); text_width, text_height = font_box[2] - font_box[0], font_box[3] - font_box[1]
                if x1 + text_width > width: x1 -= text_width
                if y1 + text_height > height: y1 -= text_height
                draw.rectangle((x1, y1 - text_height, x1 + text_width, y1), fill="green")
                draw.text((x1, y1 - text_height), label, fill="white", font=font)
            return {"image": image}
        if tool == "emoji":
            import augly.image as imaugs  # type: ignore[import-not-found]
            from augly.utils.base_paths import EMOJI_DIR  # type: ignore[import-not-found]
            image = _image(inputs["image"]); x1, y1, x2, y2 = _decode(inputs["object"])["bbox"]; name = _text(inputs["emoji"]); path = os.path.join(EMOJI_DIR, f"smileys/{name}.png")
            if not os.path.exists(path): path = os.path.join(EMOJI_DIR, "smileys/smiling_face.png")
            width, height = image.size; size = (y2-y1)/1.5; return {"image": imaugs.OverlayEmoji(emoji_path=path, emoji_size=size/height, x_pos=(((x1+x2)/2)-.5*size)/width, y_pos=(((y1+y2)/2)-.5*size)/height)(image)}
        if tool.startswith("get_"):
            import requests  # type: ignore[import-not-found]
            kind = {"get_date_fact": "date", "get_year_fact": "year", "get_math_fact": "math", "get_trivia_fact": "trivia"}[tool]
            value = _text(inputs["date" if kind == "date" else "year" if kind == "year" else "number"])
            if kind == "date":
                from dateutil import parser  # type: ignore[import-not-found]
                value = parser.parse(value).strftime("%m/%d")
            response = requests.get(f"https://numbersapi.p.rapidapi.com/{value}/{kind}", headers={"X-RapidAPI-Key": os.environ["RAPID_API_KEY"], "X-RapidAPI-Host": "numbersapi.p.rapidapi.com"}, params={"fragment": "true", "json": "true"}, timeout=30)
            response.raise_for_status(); return {"text": str(response.json().get("text", ""))}
        raise ValueError(f"unsupported MnMS tool: {tool}")


def register_mnms_tools(registry: ToolRegistry, device: str = "auto") -> None:
    """Register all 28 reference tools; registration itself performs no I/O."""
    for spec in mnms_tool_specs():
        registry.register(spec, lambda tool_id=spec.tool_id: MnmsToolRunner(tool_id, device))
