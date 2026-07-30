# MnMS Tool Catalog

`eec_sched.mnms_tools.register_mnms_tools(registry)` registers all 28 tools
from `external/mnms/mnms/execution/tool_api.py`. Registration only creates
`ToolSpec` metadata and runner factories. Model imports, model acquisition, and
remote client creation occur only when a scheduled node is prepared or run.
The module does not set `HF_HOME`, `TORCH_HOME`, or any other cache variable.

## Model-backed tools

| Tool | Reference selection |
| --- | --- |
| `text_generation` | `gpt-3.5-turbo-0125` |
| `text_summarization` | `facebook/bart-large-cnn` |
| `text_classification` | `distilbert-base-uncased-finetuned-sst-2-english` |
| `question_answering` | `deepset/roberta-base-squad2` |
| `automatic_speech_recognition` | `openai/whisper-large-v2` |
| `image_generation` | `stabilityai/stable-diffusion-xl-base-1.0`; long prompts use DALL·E 3 as in MnMS |
| `image_captioning` | `Salesforce/blip-image-captioning-large` |
| `image_editing` | `stabilityai/stable-diffusion-xl-refiner-1.0` |
| `image_classification` | `google/vit-base-patch16-224` |
| `visual_question_answering` | `Salesforce/blip-vqa-base` |
| `object_detection` | `facebook/detr-resnet-101` |
| `image_segmentation` | `facebook/maskformer-swin-base-coco` |
| `optical_character_recognition` | EasyOCR English reader |

The remaining tools are the MnMS Pillow/OpenCV/AugLy utilities and the four
Numbers API fact tools, with their reference processing retained.

To download the 10 Hugging Face repositories that are eligible for
pre-download later, run:

```bash
uv run python scripts/download_huggingface_models.py
```

Use `--dry-run` to inspect the list and `--force` to retrieve snapshots again.
The script neither configures nor requires `HF_HOME`.
`stabilityai/stable-diffusion-xl-base-1.0` is deliberately excluded.

## Framework adaptations

- The `audio` modality accepts a `pathlib.Path` request input. Text paths stay
  text, avoiding ambiguity in plans.
- Object lists, selected objects, and bounding boxes cross the DAG as compact
  JSON text. This keeps the existing small composition model rather than
  adding a general JSON/mask port modality. Detection and segmentation still
  expose their original image plus object descriptions; segmentation masks are
  used internally to calculate reference-compatible boxes, then discarded
  because they are not connectable in the current framework.
- Optional libraries are intentionally not project-wide required dependencies:
  `openai`, `diffusers`, `easyocr`, `librosa`, `textract`, `opencv-python`,
  `augly`, `requests`, and `python-dateutil` are imported only by the tools
  that use them. Missing dependencies therefore fail the selected node with
  the executor's normal structured failure result.
- `OPENAI_API_KEY` and `RAPID_API_KEY` are read only by the tool invocation
  that requires them.
