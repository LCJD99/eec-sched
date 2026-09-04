# MnMS Bayesian optimization

`run.py` is the single entry point. It loads `config.mnms.example.json` (or a
config supplied with `--config`), selects the models
for this campaign, runs each model's continuous joint search space, and writes one
artifact per model under `campaigns/<tool>/campaign.json`.

The default contract follows `docs/specs/mnms-profile-evaluation-options.md`:
text generation, image generation, and image editing are disabled; the other
10 active model tools are available. The example campaign evaluates the first
10 records from each fixed manifest. The manifests and required media remain
materialized under one portable `data/benchmarks` bundle, which contains 100
records for each benchmark. COCO records include their 100 selected images and
subsetted official annotations; profiling does not depend on `external/coco`
after preparation.

```bash
uv run python experiments/02_bayesian_optimization/prepare_dataset.py \
  --coco-root external/coco --output data/benchmarks
uv run python experiments/02_bayesian_optimization/run.py \
  --config experiments/02_bayesian_optimization/config.mnms.example.json
```

For profiling only 100 caption samples, reuse the existing 2017 checkout and
download only the caption metadata plus the selected 100 `val2014` images:

```bash
uv run python experiments/02_bayesian_optimization/download_coco_val2017.py \
  --root external/coco --caption-count 100 --seed 20260816 --insecure
```

For a clean COCO checkout, include the full caption corpus when downloading COCO:

```bash
uv run python experiments/02_bayesian_optimization/download_coco_val2017.py \
  --root external/coco --with-captions
```

Download the non-COCO benchmark slices separately. The command uses streaming
and writes only 100 records per benchmark:

```bash
uv run python experiments/02_bayesian_optimization/download_benchmarks.py \
  --output data/benchmarks --benchmark all --count 100
```

The downloader is incremental: an existing `samples.jsonl` is never replaced,
existing audio/images are reused, and VQAv2 metadata is read from existing
local JSON files when present. All media paths in the benchmark manifests are
relative to `data/benchmarks`, so the directory can be copied to another
device and evaluated without another preparation step.

ImageNet requires a Hugging Face token. The MLT-19 OCR slice uses the Kaggle
Python API and the credentials already configured for Kaggle. It downloads
only the first 100 `TrainImages/TrainImages/tr_img_*.jpg` files and matching
`TrainGT/TrainGT/tr_img_*.txt` files. The downloader writes polygons, scripts,
and transcriptions into `data/benchmarks/mlt19/samples.jsonl`; the campaign
uses the official MLT Task-4 end-to-end Hmean rule.

To run a subset, copy the example config and change `models`, for example to
`["text_classification", "object_detection"]`. Set `dataset_dir` to an
existing prepared manifest directory. Each model override declares its
adapter and its legal joint search space; failed/OOM evaluations are recorded
and are never converted into fabricated objective values.

The campaign budget is the number of continuous BO proposals measured for each
model, not the number of evaluation examples. `min`/`max` define continuous
bounds; integer dimensions are rounded and float dimensions use their declared
`precision` during canonicalization. Each proposal is stored in the campaign,
and up to 100 successful distinct canonical configurations are copied to
`selected_results`. The budget is deliberately larger than 100 to leave room
for failed evaluations. The configured examples are reused for every
configuration. The checked-in example uses 10 examples to keep a local smoke
campaign practical; increase `evaluation_samples` only when the manifests
contain at least that many records.

Export a completed campaign for one measured device:

```bash
uv run python scripts/export_campaign_database.py \
  experiments/02_bayesian_optimization/campaigns/<campaign-id> \
  --device path/to/device.json \
  --output outputs/profiling/<device-id>.json
```

The device JSON contains `device_id`, `hardware_class`, and `description`.
The export contains successful configurations and their quality, latency, and
`gpu_memory_mib` observations. It is a device-specific campaign database; the
existing v1 scheduler snapshot additionally requires all three placement
devices and transfer profiles.

Run the complete 10-tool campaign on the local RTX 5060 Ti (CUDA device 0):

```bash
CUDA_VISIBLE_DEVICES=0 uv run python \
  experiments/02_bayesian_optimization/run.py \
  --config experiments/02_bayesian_optimization/config.mnms.example.json \
  --output experiments/02_bayesian_optimization/campaigns
```

After the run prints its campaign ID, export this device's profiling database:

```bash
uv run python scripts/export_campaign_database.py \
  experiments/02_bayesian_optimization/campaigns/<campaign-id> \
  --device experiments/02_bayesian_optimization/devices/local-rtx-5060-ti.json \
  --output outputs/profiling/local-nvidia-geforce-rtx-5060-ti.json
```
