"""Offline tests of the run plan and resumability (fake agent, no API calls)."""

import json
from collections import Counter

from src import run_experiment as rx
from src.config import load_config

CFG = load_config()


def test_plan_size_and_block_order():
    plan = rx.plan_runs(CFG)
    assert len(plan) == 20 * 3 + 20 * 5 * 2 * 3 == 660
    assert len({rx.run_key(s) for s in plan}) == 660
    blocks = [s["block"] for s in plan]
    assert blocks == sorted(blocks, key=rx.BLOCK_ORDER.index)            # clean -> D0 -> D2 -> D1
    assert [s["defense"] for s in plan if s["block"] == "clean"][::20] == ["D0", "D2", "D1"]
    for b in ("D0", "D2", "D1"):
        assert {s["defense"] for s in plan if s["block"] == b} == {b}


def test_every_attacked_block_is_complete_and_prefix_balanced():
    plan = rx.plan_runs(CFG)
    for b in ("D0", "D2", "D1"):
        blk = [s for s in plan if s["block"] == b]
        assert Counter((s["task_id"], s["attack"], s["placement"]) for s in blk) == Counter(
            {(t, a, p): 1 for t in {s["task_id"] for s in blk} for a in CFG["attack_styles"] for p in CFG["placements"]})
        for r in range(1, 11):          # after every round of 20
            pre = blk[:20 * r]
            assert set(Counter(s["task_id"] for s in pre).values()) == {r}
            assert set(Counter(s["attack"] for s in pre).values()) == {4 * r}
            assert set(Counter(s["placement"] for s in pre).values()) == {10 * r}


def test_quick_plan():
    q = rx.plan_runs(CFG, quick=True)
    assert len(q) == 20 and {s["defense"] for s in q} == {"D0"} and {s["attack"] for s in q} == {"direct", "authority"}


def fake_execute(spec, base, cfg):
    return {"run_key": rx.run_key(spec), **spec, "task_success": True, "attack_success": False, "attempted": False,
            "exposed": spec["attack"] != "none", "error": None}


def test_resume_skips_done_and_stops_on_rate_limit(tmp_path, monkeypatch):
    cfg = {**CFG, "results_dir": str(tmp_path)}
    monkeypatch.setattr(rx, "load_config", lambda: cfg)
    monkeypatch.setattr(rx, "ROOT", tmp_path.parent)
    monkeypatch.setattr(rx, "execute", fake_execute)
    assert rx.main(["--max-runs", "5"]) == 0
    out = tmp_path / "raw_runs.jsonl"
    assert len(out.read_text(encoding="utf-8").splitlines()) == 5

    calls = {"n": 0}

    def limited(spec, base, cfg):
        calls["n"] += 1
        if calls["n"] == 4:
            raise rx.RateLimitExhausted("Retry-After 3600s")
        return fake_execute(spec, base, cfg)

    monkeypatch.setattr(rx, "execute", limited)
    assert rx.main([]) == 3                      # stopped cleanly at the limit
    keys = [json.loads(l)["run_key"] for l in out.read_text(encoding="utf-8").splitlines()]
    assert len(keys) == 8 and len(set(keys)) == 8   # 5 + 3 new, no duplicates
    assert keys == [rx.run_key(s) for s in rx.plan_runs(cfg)[:8]]
