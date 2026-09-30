"""Tests for src/status.py on synthetic runs (no API calls)."""

import json

from src import status
from src.config import load_config
from src.run_experiment import plan_runs
from tests.test_analyze import make_row


def test_status_progress_and_eta(tmp_path, monkeypatch):
    cfg = {**load_config(), "results_dir": "res"}
    (tmp_path / "res").mkdir()
    plan = plan_runs(cfg)
    rows = [make_row(s, n_llm_calls=2, error="tool_use_failed: x" if i == 0 else None)
            for i, s in enumerate(plan[:40])]
    (tmp_path / "res" / "raw_runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    monkeypatch.setattr(status, "ROOT", tmp_path)
    text = status.status(cfg)
    assert "Progress: 40/660" in text
    assert "Errors so far: 1/40" in text
    # 3200 tokens/run -> 62 runs/day by tokens (requests allow 500) -> 620 remaining runs = 10 days
    assert "~10 (62 runs/day)" in text
    assert "observed pace, 40 runs per active day" in text
