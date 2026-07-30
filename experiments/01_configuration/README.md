# Image-captioning configuration profile

This experiment profiles `nlpconnect/vit-gpt2-image-captioning` using
`img1.jpg`. It changes one of `image_size`, `num_beams`, and
`max_new_tokens` at a time, measures warm end-to-end latency, and saves every
generated caption for manual quality comparison.

Run it from the repository root:

```bash
rtk uv run python experiments/01_configuration/profile_image_caption.py
```

The default result is `experiments/01_configuration/results.json`. It contains
per-run latencies, p50 and p95, the exact configuration, generated captions,
and environment metadata. Model loading and download time are excluded.

`image_size` is implemented as an external resize before the Hugging Face
pipeline. The selected ViT-GPT2 model may then resize again to its fixed model
input size. Consequently, this experiment can measure the end-to-end cost of
that preprocessing choice, but it must not be interpreted as changing the
vision encoder's native resolution.
