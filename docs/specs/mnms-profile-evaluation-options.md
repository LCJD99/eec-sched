# MnMS model profiling: evaluation and Configuration discovery contract

This note fixes the model and public quality-evaluation contract for each MnMS
tool and defines how its Configuration catalog is discovered. The model
is fixed by the MnMS catalog (with `openai/whisper-small` selected for ASR);
only inference-time parameters, runtime choices, and other declared
Configuration dimensions are searched.

Configuration discovery is an **offline multi-objective Bayesian-optimization
campaign**, following the method demonstrated in
[`experiments/02_bayesian_optimization`](../../experiments/02_bayesian_optimization/).
There are no manually assigned `fast`, `balanced`, or `quality` parameter tiers.
After discovery, every eligible tool exposes exactly **100 measured candidate
Configurations** to the Profiling Database. The scheduler sees only that
immutable catalog; it never runs Bayesian optimization itself.

Use the adopted dataset and named official scorer below, not an approximate
replacement. All reported quality scores are higher-is-better except metrics
explicitly marked lower-is-better, such as WER, CER, FID, and LPIPS.

## Adopted profiling contract

The following decisions apply to every profiling campaign.

- `text_generation`, `image_generation`, and `image_editing` are disabled and
  omitted from the profile candidate catalog.
- The active fixed-model tools use the **adopted evaluation** in the table
  below: CNN/DailyMail/ROUGE-L, SST-2/accuracy, SQuAD v2/F1,
  LibriSpeech test-clean/WER, COCO-caption/CIDEr, ImageNet/top-1, VQAv2
  official accuracy, COCO box AP, COCO mask AP, and ICDAR 2019 MLT Task-4
  Hmean.
- Every stochastic Configuration is reproducible. Deterministic text uses
  `do_sample=False`; diffusion tools are disabled; any remaining random seed is
  fixed and stored.
- Every campaign freezes the dataset revision, evaluation sample IDs, search
  space, canonicalization rules, model revision, runtime, random seed, campaign
  budget, and Bayesian-optimization settings before adaptive sampling starts.
- Quality, warm end-to-end latency, and mean incremental execution energy are
  measured as separate objectives. Metric direction comes from the tool's
  quality contract; latency and energy are minimized. The campaign must freeze
  the target device or the deterministic cross-device aggregation used by its
  execution objectives.
- Formal measurements use batch size one on each compatible target device.
  They record the hardware and runtime environment required by
  [`profiling-database.schema.json`](../schemas/profiling-database.schema.json),
  including the observations needed for quality confidence intervals and warm
  latency p95. A single warm observation or peak GPU memory may be used as a
  BO demo signal, but cannot be published as the schema's p95 latency or mean
  incremental execution energy.
- Segmentation and OCR keep their masks/detection boxes in an evaluation-only
  path. They remain non-connectable: DAG outputs are still image/text values.

## Bayesian-optimization Configuration discovery

### Search-space contract

Each eligible tool declares a joint search space over only the parameters that
can change quality, execution cost, or device compatibility. The declaration
must contain each parameter's type, continuous minimum and maximum, precision
for canonical serialization, fixed parameters, illegal combinations, and the
canonical serialization rule. BO proposes points in the continuous joint
domain; canonicalization converts them into the stored Configuration form.
Parameters are not independently schedulable: one canonical joint assignment,
including backend and numeric format, is one Configuration.

The search-space bounds replace the old hand-authored parameter tiers. They
must be justified and frozen before profiling, but they do not imply an order
such as fast/balanced/quality. Duplicate canonical assignments, illegal points,
OOM results, and evaluator failures are recorded and never promoted as
candidates.

### Offline optimization

1. Select an initial set of distinct points in the continuous joint domain with
   a seeded space-filling design such as Sobol sampling, then canonicalize them.
2. Measure the adopted raw quality metric and execution objectives for every
   proposed joint Configuration. Lower-is-better quality metrics are modeled
   with their direction reversed for acquisition only; the database retains
   the original raw metric and direction.
3. Fit surrogate models to the successful measured observations and use a
   multi-objective acquisition function, such as constrained qNEHVI, to propose
   new continuous points. Do not fabricate objective values for failed runs.
4. Continue until the frozen campaign budget is exhausted. Continuous domains
   have no finite unmeasured-grid termination condition; duplicate canonical
   points are recorded as duplicate proposals and are not retained.

The campaign artifact records every proposal and whether it came from initial
coverage or adaptive acquisition, all measured objectives and failures, model
and acquisition settings, random seeds, software/hardware provenance, and the
measured Pareto fronts.

### Retaining exactly 100 candidates per tool

Only successful, distinct canonical Configurations with all campaign objectives measured
are eligible for the final catalog. Rank them by successive non-dominated
fronts over normalized quality, warm latency, and mean incremental execution
energy. Preserve the quality, latency, and energy extreme points, then select
for deterministic coverage in normalized objective space until exactly 100
points remain. When a front must be truncated, use greedy hypervolume
contribution; break remaining ties by the canonical Configuration
serialization.

The retained 100 are a trade-off coverage set, not parameter presets and not a
scalar `quality / cost` top-100 ranking. Exploration points that are not
retained remain in the campaign artifact but are omitted from the published
Profiling Database Snapshot. Each retained Configuration must then satisfy the
schema's complete Quality Profile and all required compatible-device Execution
Profiles before publication.

## Adopted evaluation suites

| Tool and fixed model | Adopted evaluation | Non-adopted alternative considered | Raw metric recorded | Why it is compatible / source |
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
| `optical_character_recognition` — EasyOCR multilingual | **ICDAR 2019 MLT train subset**, Task-4 end-to-end transcription Hmean using the official polygon-IoU and case-insensitive exact-transcription rule | ICDAR 2019 MLT test set (its GT is not public), or isolated-word recognition | Hmean ↑ | The 100-record portable bundle uses matching `TrainImages`/`TrainGT` files downloaded individually through Kaggle. MLT Task-4 counts a match only when polygon IoU exceeds 0.5 and the transcription is exactly correct ignoring case; `###` is don't-care. [MLT-2019 task definition](https://rrc.cvc.uab.es/?ch=15&com=tasks) / [MLT-2019 paper](https://arxiv.org/abs/1907.00945) / [EasyOCR repository](https://github.com/JaidedAI/EasyOCR) |

## Required profile protocol

1. Pin the dataset revision, split, sample IDs/order, preprocessing, prompt
   template, scorer version, model revision, search space, and all campaign
   settings. Never change these during a campaign or tune against a held-out
   split after its quality floor has been adopted.
2. Generate every stochastic output with a recorded fixed seed.  Text remains
   deterministic (`temperature=0`, `do_sample=False`); for diffusion, record
   seed, scheduler, guidance scale, width/height, steps, and strength.
3. Profile each retained Configuration with batch size one. Record warm
   end-to-end p95 (preprocess + inference + postprocess + necessary transfers),
   mean incremental execution energy, sample counts, and input bucket. Exclude
   download and model loading. BO exploration measurements that do not meet
   this protocol are evidence for candidate discovery only and must be repeated
   before the Configuration is published.
4. A metric whose evaluator cannot consume the runner's public output is a
   blocker, not a proxy opportunity.  In particular, segmentation needs masks,
   and editing needs a declared instruction-following evaluation path.
5. Keep quality datasets separate from latency input buckets.  For example,
   ASR latency buckets should be audio-duration ranges even though WER is
   evaluated over the selected corpus.

## Implementation prerequisites

- Define and review one legal joint search space for every active tool; persist
  it as part of the campaign provenance.
- Select and persist the fixed evaluation sample identifiers, in split order,
  for every active evaluation suite. Sample counts and confidence methods are
  tool-specific and must be sufficient for the schema's quality interval; the
  number of evaluation samples is independent of the 100 retained
  Configurations.
- Add Configuration-aware runner support and profile drivers that execute BO
  proposals rather than a single reference Configuration or named parameter
  tiers.
- Generalize the campaign method in `experiments/02_bayesian_optimization/`
  from its YOLO11m demo to each active tool and change final retention from the
  demo's at-most-10 coverage set to exactly 100 candidates.
- Add evaluation-only raw output retention for segmentation masks and OCR
  detection boxes without widening the DAG's connectable modalities.
- Install the tool-specific evaluation libraries and data dependencies before
  executing a profile; none are downloaded by this design document.
