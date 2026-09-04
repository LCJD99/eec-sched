from __future__ import annotations

import json
from pathlib import Path

from eec_sched.profiling.export import export_campaign_database


def test_exports_selected_results_for_one_device(tmp_path: Path) -> None:
    campaign = tmp_path / "campaigns" / "text_classification"
    campaign.mkdir(parents=True)
    (campaign / "campaign.json").write_text(
        json.dumps(
            {
                "tool": "text_classification",
                "model": "example/model",
                "campaign_status": "ready_for_candidate_discovery",
                "results": [{"configuration": {"max_length": 128}, "quality": 0.9, "quality_metric": "accuracy", "latency_ms": 4.2, "gpu_memory_mib": 200.0, "status": "success"}],
                "selected_results": [{"configuration": {"max_length": 128}, "quality": 0.9, "quality_metric": "accuracy", "latency_ms": 4.2, "gpu_memory_mib": 200.0, "status": "success"}],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "database.json"
    payload = export_campaign_database(
        campaign.parent,
        {"device_id": "device", "hardware_class": "rtx", "description": "test GPU"},
        output,
    )
    configuration = payload["tools"][0]["configurations"][0]
    assert configuration["quality"] == {"metric": "accuracy", "value": 0.9}
    assert configuration["execution"] == {"device_id": "device", "latency_ms": 4.2, "gpu_memory_mib": 200.0}
    assert output.is_file()
