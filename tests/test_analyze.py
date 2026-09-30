"""Tests for src/analyze.py on synthetic runs (no API calls)."""

import json

import pytest

from src import analyze
from src.config import load_config
from src.run_experiment import plan_runs, run_key


@pytest.fixture
def cfg():
    return load_config()


def make_row(spec, **kw):
    row = {"run_key": run_key(spec), **spec, "user_prompt": f"prompt {spec['task_id']}",
           "payload": None if spec["attack"] == "none" else f"PAYLOAD {spec['attack']}",
           "payload_target": None, "exposed": False, "attempted": False, "blocked": False,
           "attack_success": False, "partial": False, "flagged_suspicious": False, "task_success": True,
           "final_answer": "answer", "tool_calls": [], "blocked_calls": [], "error": None,
           "input_tokens": 3000, "output_tokens": 200, "timestamp": "2026-09-27T10:00:00+00:00"}
    row.update(kw)
    return row


def write_rows(tmp_path, rows):
    (tmp_path / "raw_runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_metrics_and_denominators(tmp_path, cfg):
    plan = plan_runs(cfg)
    clean = [s for s in plan if s["block"] == "clean" and s["defense"] == "D0"][:4]
    att = [s for s in plan if s["block"] == "D0"][:10]
    rows = [make_row(clean[0], task_success=False, error="tool_use_failed (after 1 retry): x"),
            *[make_row(s) for s in clean[1:]]]
    # 10 attacked: 6 exposed, 3 succeeded (all exposed), 1 partial, 1 error, 1 flagged
    for i, s in enumerate(att):
        rows.append(make_row(s, exposed=i < 6, attack_success=i < 3, attempted=i < 3,
                             partial=i == 4, flagged_suspicious=i == 5, task_success=i not in (0, 9),
                             error="tool_use_failed: y" if i == 9 else None))
    write_rows(tmp_path, rows)
    res = analyze.analyze(tmp_path, cfg, verbose=False)
    d0 = next(s for s in res["summary"] if s["defense"] == "D0")
    assert (d0["asr_k"], d0["asr_n"]) == (3, 10)                     # errors stay in the denominator
    assert (d0["asr_exposed_k"], d0["asr_exposed_n"]) == (3, 6)
    assert (d0["exposure_k"], d0["exposure_n"]) == (6, 10)
    assert (d0["partial_k"], d0["flagged_k"]) == (1, 1)
    assert (d0["utility_attack_k"], d0["utility_attack_n"]) == (8, 10)
    assert (d0["utility_attack_ok_k"], d0["utility_attack_ok_n"]) == (8, 9)
    assert (d0["error_attack_k"], d0["error_attack_n"]) == (1, 10)
    assert (d0["utility_clean_k"], d0["utility_clean_n"]) == (3, 4)
    assert (d0["utility_clean_ok_k"], d0["utility_clean_ok_n"]) == (3, 3)
    assert d0["attacked_status"] == "in progress (10/200)"
    d1 = next(s for s in res["summary"] if s["defense"] == "D1")
    assert d1["attacked_status"].startswith("not started") and d1["asr_pct"] is None

    md = (tmp_path / "results_table.md").read_text(encoding="utf-8")
    assert "30.0% (3/10)" in md and "50.0% (3/6)" in md and "in progress" in md
    assert "n/a (n=0)" in md
    for f in ("summary.csv", "breakdown.csv", "failure_examples.md", "failure_examples.json",
              "flagged_for_review.md", "figures/asr_by_style.png", "figures/asr_by_placement.png"):
        assert (tmp_path / f).exists(), f


def test_breakdown_by_style_and_placement(tmp_path, cfg):
    plan = [s for s in plan_runs(cfg) if s["block"] == "D2"][:20]
    rows = [make_row(s, attack_success=s["attack"] == "direct" and s["placement"] == "report") for s in plan]
    write_rows(tmp_path, rows)
    bd = analyze.analyze(tmp_path, cfg, verbose=False)["breakdown"]
    direct = next(b for b in bd if b["dimension"] == "style" and b["value"] == "direct" and b["defense"] == "D2")
    assert direct["asr_n"] == 4 and direct["asr_k"] == 2          # 20 runs = each combo twice
    rep = next(b for b in bd if b["dimension"] == "placement" and b["value"] == "report" and b["defense"] == "D2")
    assert (rep["asr_k"], rep["asr_n"]) == (2, 10)


def test_duplicates_use_first_row_and_are_reported(tmp_path, cfg):
    s = plan_runs(cfg)[0]
    write_rows(tmp_path, [make_row(s, task_success=False), make_row(s, task_success=True)])
    res = analyze.analyze(tmp_path, cfg, verbose=False)
    d0 = next(x for x in res["summary"] if x["defense"] == "D0")
    assert (d0["utility_clean_k"], d0["utility_clean_n"]) == (0, 1)
    assert res["duplicates"] == [run_key(s)]


def test_failure_pick_is_diverse_and_prefers_defended(cfg):
    plan = plan_runs(cfg)
    specs = [s for s in plan if s["block"] == "D0"][:6] + [s for s in plan if s["block"] == "D2"][:1]
    rows = [make_row(s, attack_success=True) for s in specs]
    picks = analyze.pick_failures(rows)
    assert len(picks) == 3
    assert picks[0]["defense"] == "D2"
    assert len({(p["attack"], p["placement"]) for p in picks}) == 3


def test_failure_examples_empty_says_in_progress(tmp_path, cfg):
    write_rows(tmp_path, [make_row(plan_runs(cfg)[0])])
    analyze.analyze(tmp_path, cfg, verbose=False)
    text = (tmp_path / "failure_examples.md").read_text(encoding="utf-8")
    assert "No executed attacks yet (experiment in progress)." in text


def test_review_limit_and_priority(cfg):
    plan = [s for s in plan_runs(cfg) if s["block"] == "D1"]
    rows = [make_row(s, flagged_suspicious=True) for s in plan[:30]] + \
           [make_row(s, partial=True) for s in plan[30:33]]
    picks = analyze.pick_review(rows)
    assert len(picks) == 20
    assert all(p["partial"] for p in picks[:3])
