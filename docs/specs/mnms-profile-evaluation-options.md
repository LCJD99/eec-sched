# MnMS model profiling: public evaluation options

This note proposes a **single, fixed public accuracy corpus per tool** for the
first configuration profiles.  The model is fixed by the MnMS catalog (with
`openai/whisper-small` selected for ASR); a configuration changes inference
parameters only.  Every latency measurement remains a separate warm,
end-to-end, batch-one measurement on the actual target device.

The recommendation column records the alternatives considered during design.
The adopted contract below selects the recommended first profile for each
active tool. Use the named official scorer, not an approximate replacement;
all reported quality scores are higher-is-better except WER/CER, FID, and
LPIPS.

## Adopted coarse profiling contract

The following decisions are adopted for the first profiling pass.  This is a
coarse scheduler input, not a full benchmark run.

- `text_generation`, `image_generation`, and `image_editing` are disabled and
  omitted from the profile candidate catalog.
- The active fixed-model tools use the **recommended first profile** in the
  table below: CNN/DailyMail/ROUGE-L, SST-2/accuracy, SQuAD v2/F1,
  LibriSpeech test-clean/WER, COCO-caption/CIDEr, ImageNet/top-1, VQAv2
  official accuracy, COCO box AP, COCO mask AP, and ICDAR 2015 Hmean.
- Every configuration is deterministic. Text uses `do_sample=False`; the
  diffusion tools are disabled; any remaining random seed is fixed and stored.
- Execution uses one GPU, batch size one, and FP16 where the selected model
  supports it. Each latency record includes the discovered GPU model, memory,
  CUDA version, and PyTorch version.
- Quality is evaluated on five fixed samples selected from the chosen test
  split. The persisted accuracy profile stores the raw standard score only.
- For latency, each configuration runs each of those five inputs ten times
  after preparation: 50 warm end-to-end observations produce p50 and p95.
- Segmentation and OCR keep their masks/detection boxes in an evaluation-only
  path. They remain non-connectable: DAG outputs are still image/text values.

### Adopted configuration sets

All identifiers are finite scheduler choices. They are ordered only for
readability; the scheduler must use recorded profiles rather than infer a
quality or latency order from their names.

| Tool | `fast` | `balanced` | `quality` |
| --- | --- | --- | --- |
| `text_summarization` | `max_length=64`, `min_length=16`, `num_beams=1` | `96`, `24`, `2` | `130`, `30`, `4` |
| `text_classification` | `max_tokens=32` | `64` | `128` |
| `question_answering` | `max_context_tokens=128`, `doc_stride=64` | `256`, `96` | `384`, `128` |
| `image_captioning` | `max_new_tokens=16`, `num_beams=1` | `32`, `3` | `64`, `5` |
| `image_classification` | `input_size=128` | `192` | `224` |
| `visual_question_answering` | `max_new_tokens=5`, `num_beams=1` | `10`, `3` | `20`, `5` |
| `object_detection` | `shortest_edge=480` | `640` | `800` |
| `image_segmentation` | `shortest_edge=384` | `512` | `800` |
| `optical_character_recognition` | `canvas_size=1280`, `mag_ratio=1.0`, greedy decoder | `1920`, `1.5`, beam decoder width 3 | `2560`, `2.0`, beam decoder width 5 |

`automatic_speech_recognition` uses five configurations: `num_beams` 1, 2,
3, 4, and 5. Audio preprocessing is fixed at 16 kHz. `image_classification`
uses the ViT model's supported position-encoding interpolation for the 128 and
192 input-size configurations.

## Selectable evaluation suites

| Tool and fixed model | Recommended first profile | Alternative to select instead | Raw metric recorded | Why it is compatible / source |
| --- | --- | --- | --- | --- |
| `text_generation` — `gpt-3.5-turbo-0125` | **ELI5 validation**, prompt = question plus its supplied evidence; **ROUGE-L** | ELI5 validation, ROUGE-1/2/L triplet; select `ROUGE-L` only as scheduler quality | ROUGE-L ↑ | ELI5 is a public long-form QA/generation corpus and publishes Full ROUGE evaluation. [ELI5 project](https://facebookresearch.github.io/ELI5/) / [paper](https://aclanthology.org/P19-1346/) |
| `text_summarization` — `facebook/bart-large-cnn` | **CNN/DailyMail test**, `3.0.0` configuration; **ROUGE-L** | CNN/DailyMail test, report ROUGE-1/2/L and use ROUGE-L | ROUGE-L ↑ | This is the summarization benchmark associated with the model's task; its public dataset card describes it as abstractive summarization. [Dataset card](https://huggingface.co/datasets/abisee/cnn_dailymail) / [dataset paper](https://aclanthology.org/P17-1099/) |
| `text_classification` — `distilbert-base-uncased-finetuned-sst-2-english` | **GLUE SST-2 validation**, **accuracy** | SST-2 validation, macro-F1 (only if later class balance changes) | accuracy ↑ | The model card identifies its SST-2 fine-tuning; GLUE provides the public SST-2 task and official evaluation materials. [Model card](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) / [GLUE data](https://github.com/nyu-mll/GLUE-baselines) |
| `question_answering` — `deepset/roberta-base-squad2` | **SQuAD v2 validation**, **F1** | SQuAD v2 validation, exact match (EM) | token F1 ↑ | The model card is for SQuAD 2; the official-compatible scorer defines both F1 and EM, including unanswerable questions. [Model card](https://huggingface.co/deepset/roberta-base-squad2) / [metric implementation](https://github.com/huggingface/evaluate/tree/main/metrics/squad_v2) |
| `automatic_speech_recognition` — `openai/whisper-small` | **LibriSpeech `test-clean`**, **WER** | LibriSpeech `test-other`, WER (robustness suite, slower and noisier) | WER ↓ | Whisper's own evaluation-data instructions use both LibriSpeech test splits; WER is its English transcription measure. [Whisper evaluation data](https://github.com/openai/whisper/blob/main/data/README.md) / [Whisper paper](https://cdn.openai.com/papers/whisper.pdf) |
| `image_generation` — SDXL Base (or MnMS DALL·E 3 long-prompt route) | **GenEval**, fixed four seeded images per prompt; **macro overall score** | T2I-CompBench validation splits and their task-specific score; use this only if compositional coverage is the priority | GenEval macro overall ↑ | GenEval directly tests prompt fidelity (single/two objects, counting, colour, position, colour-attribute); unlike FID, it measures whether an individual generation obeys its prompt. Do **not** combine SDXL and DALL·E results in one profile family: they are different execution routes. [GenEval data and evaluator](https://github.com/djghosh13/geneval) / [paper](https://arxiv.org/abs/2310.11513) / [T2I-CompBench](https://github.com/Karine-Huang/T2I-CompBench) |
| `image_captioning` — `Salesforce/blip-image-captioning-large` | **COCO 2014 validation**, **CIDEr** using official COCO-caption | COCO 2014 validation, SPICE; retain BLEU-4/METEOR/ROUGE-L as diagnostics | CIDEr ↑ | COCO Caption has reference captions and its official evaluation reports BLEU, METEOR, ROUGE, and CIDEr. [COCO caption evaluation](https://aclanthology.org/P15-1051/) / [official scorer](https://github.com/tylin/coco-caption) |
| `image_editing` — `stabilityai/stable-diffusion-xl-refiner-1.0` | **PIE-Bench**, all 700 images / 10 edit types; **target-prompt CLIP similarity** plus mask-outside LPIPS | MagicBrush test (requires access) with its instruction-editing protocol | target-prompt CLIP ↑; preservation LPIPS ↓ | PIE-Bench provides source/target images, prompts, instructions, and masks, so it separately measures requested edit and background preservation. This must first be confirmed as instruction editing: the selected SDXL Refiner checkpoint is an img2img refiner, not an instruction-editing model. [PIE-Bench paper](https://arxiv.org/abs/2310.01506) / [authors' code and data](https://github.com/cure-lab/pnpinversion) / [Refiner card](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0) / [MagicBrush](https://github.com/OSU-NLP-Group/MagicBrush) |
| `image_classification` — `google/vit-base-patch16-224` | **ImageNet-1k validation**, **top-1 accuracy** | ImageNet-1k validation, top-5 accuracy | top-1 accuracy ↑ | The model card reports ImageNet-1k use and gives the required 224-pixel pre-processing. [Model card](https://huggingface.co/google/vit-base-patch16-224) / [ImageNet site](https://www.image-net.org/) |
| `visual_question_answering` — `Salesforce/blip-vqa-base` | **VQAv2 validation**, official VQA accuracy | VQAv2 test-dev via EvalAI submission (only after local validation profile is stable) | VQA accuracy ↑ | VQAv2 is open-ended VQA; use its official evaluator/answer normalisation rather than exact string match. [BLIP model card](https://huggingface.co/Salesforce/blip-vqa-base) / [VQAv2 dataset card](https://github.com/salesforce/LAVIS/blob/main/dataset_card/vqav2.md) / [official evaluation](https://github.com/GT-Vision-Lab/VQA) |
| `object_detection` — `facebook/detr-resnet-101` | **COCO 2017 validation**, official COCO `AP@[.50:.95]` | COCO 2017 validation, AP50 (diagnostic only) | box AP ↑ | DETR is COCO-trained and returns boxes/labels. Use COCO API rather than threshold-specific F1, because AP evaluates ranked detections. [Model card](https://huggingface.co/facebook/detr-resnet-101) / [COCO API](https://github.com/cocodataset/cocoapi) |
| `image_segmentation` — `facebook/maskformer-swin-base-coco` | **COCO 2017 validation instance annotations**, official mask `AP@[.50:.95]` | COCO 2017 panoptic validation, PQ, **only if** the runner emits panoptic masks/categories | mask AP ↑ | The checkpoint is trained for COCO panoptic segmentation and its runner calls panoptic post-processing; its public output presently discards masks and emits boxes. It cannot honestly report mask AP or PQ until masks are preserved for the evaluator. [Model card](https://huggingface.co/facebook/maskformer-swin-base-coco) / [COCO API](https://github.com/cocodataset/cocoapi) / [panoptic API](https://github.com/cocodataset/panopticapi) |
| `optical_character_recognition` — EasyOCR English | **ICDAR 2015 RRC Challenge 4**, end-to-end transcription Hmean | IIIT5K test, case-insensitive word accuracy, **after supplying ground-truth word crops** | Hmean ↑ | The public tool takes a whole image and returns a transcript, so the end-to-end scene-text task is the closer contract. IIIT5K is useful only for isolated word recognition, not the current whole-image port. [ICDAR 2015 challenge description](https://rrc.cvc.uab.es/files/short_rrc_2015.pdf) / [IIIT5K project](https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset) / [EasyOCR repository](https://github.com/JaidedAI/EasyOCR) |

## Required profile protocol, regardless of selected row

1. Pin the dataset revision, split, sample IDs/order, preprocessing, prompt
   template, scorer version, model revision, and the exact configuration
   parameters.  Never tune against the held-out split after its quality floor
   has been adopted.
2. Generate every stochastic output with a recorded fixed seed.  Text remains
   deterministic (`temperature=0`, `do_sample=False`); for diffusion, record
   seed, scheduler, guidance scale, width/height, steps, and strength.
3. Profile each configuration with batch size one.  Record warm end-to-end
   p50/p95 (preprocess + inference + postprocess + necessary transfers), plus
   sample count and input bucket.  Exclude download and model loading.
4. A metric whose evaluator cannot consume the runner's public output is a
   blocker, not a proxy opportunity.  In particular, segmentation needs masks,
   and editing needs a declared instruction-following evaluation path.
5. Keep quality datasets separate from latency input buckets.  For example,
   ASR latency buckets should be audio-duration ranges even though WER is
   evaluated over the selected corpus.

## Implementation prerequisites

- Select and persist the five fixed sample identifiers, in test-split order,
  for every active evaluation suite.
- Add configuration-aware runner support and profile drivers that execute the
  adopted parameters rather than the current single reference configuration.
- Add evaluation-only raw output retention for segmentation masks and OCR
  detection boxes without widening the DAG's connectable modalities.
- Install the tool-specific evaluation libraries and data dependencies before
  executing a profile; none are downloaded by this design document.
