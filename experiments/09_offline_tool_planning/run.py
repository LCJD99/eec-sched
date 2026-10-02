"""Generate canonical three-DAG workload records with one LLM call per item.

Example::

    uv run python experiments/09_offline_tool_planning/run.py \
      input_path=data/requests.jsonl \
      catalog_path=data/tool-catalog.json

The default model is configs/model/deepseekv4.yaml.  Select another shared
model config with ``model=local_qwen``.  Use ``limit=2`` or ``item_id=some-id``
for a smoke run.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from eec_sched.llm import OpenAICompatibleChatModel
from eec_sched.offline_tool_planner import (
    PlannerBatchError,
    load_tool_catalog,
    load_work_items,
    run_batch,
)


def _path(root: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else root / candidate


def _load_system_prompt(path: Path) -> str:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PlannerBatchError(f"could not read system prompt {path}") from exc
    if path.suffix.lower() != ".json":
        prompt = raw
    else:
        value = json.loads(raw)
        if isinstance(value, str):
            prompt = value
        elif isinstance(value, dict) and isinstance(value.get("system_prompt"), str):
            prompt = value["system_prompt"]
        else:
            raise PlannerBatchError("system prompt JSON must be a string or {system_prompt: string}")
    if not prompt.strip():
        raise PlannerBatchError("system prompt must not be empty")
    return prompt


def run_from_config(config: DictConfig) -> object:
    root = Path(get_original_cwd())
    input_path = _path(root, str(config.input_path))
    catalog_path = _path(root, str(config.catalog_path))
    prompt_path = _path(root, str(config.system_prompt_path))
    schema_path = _path(root, str(config.schema_path)) if config.get("schema_path") else None
    configured_output_dir = config.get("output_dir")
    if configured_output_dir:
        output_dir = _path(root, str(configured_output_dir))
        if output_dir.exists():
            raise PlannerBatchError(f"refusing to use existing output directory: {output_dir}")
    else:
        run_root = root / "experiments/09_offline_tool_planning/runs"
        stamp = datetime.now(timezone.utc).strftime("tool-planner-%Y%m%dT%H%M%SZ")
        output_dir = run_root / stamp
        suffix = 1
        while output_dir.exists():
            output_dir = run_root / f"{stamp}-{suffix:02d}"
            suffix += 1
    items = load_work_items(input_path)
    specs = load_tool_catalog(catalog_path)
    prompt = _load_system_prompt(prompt_path)
    model_config = config.model
    token = str(model_config.token)
    model = OpenAICompatibleChatModel(
        base_url=str(model_config.base_url),
        token=token,
        model=str(model_config.model),
        timeout_seconds=float(model_config.get("timeout_seconds", 360)),
    )
    return run_batch(
        items=items,
        specs=specs,
        system_prompt=prompt,
        model=model,
        output_path=output_dir / "dags.jsonl",
        failures_path=output_dir / "failures.jsonl",
        manifest_path=output_dir / "manifest.json",
        trace_path=output_dir / "trace.jsonl",
        input_path=input_path,
        catalog_path=catalog_path,
        model_name=str(model_config.model),
        endpoint=str(model_config.base_url),
        prompt_version=str(config.prompt_version),
        limit=int(config.limit) if config.get("limit") is not None else None,
        item_id=str(config.item_id) if config.get("item_id") else None,
        schema_path=schema_path,
        api_token=token,
    )


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config: DictConfig) -> None:
    try:
        summary = run_from_config(config)
    except (PlannerBatchError, OSError, ValueError) as exc:
        raise SystemExit(f"tool planner configuration error: {exc}") from exc
    # Keep this short: model responses and tokens never enter stdout.
    print(OmegaConf.to_yaml({
        "selected": summary.selected,
        "succeeded": summary.succeeded,
        "failed": summary.failed,
        "dags": str(summary.output_path),
        "failures": str(summary.failures_path),
        "trace": str(summary.trace_path),
        "manifest": str(summary.manifest_path),
    }))


if __name__ == "__main__":
    main()
