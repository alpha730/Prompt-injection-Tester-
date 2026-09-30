"""Experiment runner (spec section 10).

Grid: 20 tasks x 5 attack styles x 2 placements x defenses, plus clean runs (attack=none)
for every task x defense. Order, so that a partial run is still usable:

  block "clean"  all clean runs: D0 tasks, then D2 tasks, then D1 tasks
  block "D0"     all D0 attacked runs
  block "D2"     all D2 attacked runs
  block "D1"     all D1 attacked runs

Within an attacked block, runs go in rounds of 20 (one per task); round k gives task i the
style/placement combo (i + k) mod 10. After every round each task has run once and each
combo twice, so any prefix is balanced across tasks, styles and placements.

Resumable: runs already in results/raw_runs.jsonl are skipped. Per-minute rate limits are slept
through inside the agent (src/agent.py). On a daily limit, or a safety cap (too many rate-limit
waits in a row, or max_session_hours reached), progress is saved, a per-block summary is printed,
and the runner exits with code 3. Rerun the same command later to continue.

  python -m src.run_experiment            # full grid
  python -m src.run_experiment --quick    # 5 tasks x 2 attacks x 2 placements x D0 -> results/quick_runs.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import groq
from tqdm import tqdm

from src import attacks
from src.agent import RateLimitExhausted, run_agent
from src.config import ROOT, load_config
from src.tasks import TASKS, TASKS_BY_ID
from src.tools import ToolEnv

BLOCK_ORDER = ["clean", "D0", "D2", "D1"]
QUICK_TASKS = ["T01", "T08", "T14", "T17", "T13"]
QUICK_ATTACKS = ["direct", "authority"]


def run_key(spec: dict) -> str:
    return "|".join(str(spec[k]) for k in ("task_id", "attack", "placement", "defense", "model", "repeat"))


def plan_runs(cfg: dict, quick: bool = False) -> list[dict]:
    model, defenses = cfg["model"], cfg["defenses"]
    styles, placements = cfg["attack_styles"], cfg["placements"]
    combos = [(s, p) for s in styles for p in placements]

    def spec(block, task_id, attack, placement, defense, repeat):
        return {"block": block, "task_id": task_id, "attack": attack, "placement": placement,
                "defense": defense, "model": model, "repeat": repeat}

    if quick:
        return [spec("quick", t, a, p, "D0", 0) for t in QUICK_TASKS for a in QUICK_ATTACKS for p in placements]

    runs = []
    for rep in range(cfg["n_repeats"]):
        for d in [d for d in BLOCK_ORDER[1:] if d in defenses]:
            runs += [spec("clean", t.id, "none", "none", d, rep) for t in TASKS]
    for d in [d for d in BLOCK_ORDER[1:] if d in defenses]:
        for rep in range(cfg["n_repeats"]):
            for k in range(len(combos)):
                for i, t in enumerate(TASKS):
                    style, placement = combos[(i + k) % len(combos)]
                    runs.append(spec(d, t.id, style, placement, d, rep))
    return runs


def execute(spec: dict, base: ToolEnv, cfg: dict) -> dict:
    task = TASKS_BY_ID[spec["task_id"]]
    env = base.fresh_copy()
    info = None
    if spec["attack"] != "none":
        info = attacks.apply_attack(env, task, spec["attack"], spec["placement"], cfg["attacker_email"])
    out = run_agent(task.user_prompt, env, cfg, model=spec["model"], defense=spec["defense"])
    ok, failed = task.success_check(out)
    m = attacks.evaluate(out, info, cfg["attacker_email"])
    return {
        "run_key": run_key(spec), **spec,
        "user_prompt": task.user_prompt,
        "payload": info["payload"] if info else None,
        "payload_target": info["target"] if info else None,
        "harm": info["harm"] if info else None,
        **m,
        "task_success": ok,
        "failed_checks": failed,
        "final_answer": out["final_answer"],
        "tool_calls": out["tool_calls"],
        "blocked_calls": out["blocked_calls"],
        "outbox": out["outbox"],
        "deletions": out["deletions"],
        "hit_max_steps": out["hit_max_steps"],
        "n_llm_calls": out["n_llm_calls"],
        "input_tokens": out["input_tokens"],
        "output_tokens": out["output_tokens"],
        "tokens_per_call": out["tokens_per_call"],
        "retries": out["retries"],
        "rate_limit_waits": out["rate_limit_waits"],
        "rate_limit_wait_s": out["rate_limit_wait_s"],
        "tool_use_failed": out["tool_use_failed"],
        "latency": out["latency_s"],
        "error": out["error"],
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "messages": out["messages"],
    }


def load_done(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    return {r["run_key"]: r for r in rows}


def pct(k, n):
    return f"{100 * k / n:5.1f}% ({k}/{n})" if n else "   n/a"


def print_progress(plan: list[dict], done: dict[str, dict]):
    print("\nProgress per block:")
    for b in BLOCK_ORDER + ["quick"]:
        specs = [s for s in plan if s["block"] == b]
        if specs:
            n_done = sum(run_key(s) in done for s in specs)
            print(f"  {b:6} {n_done:4d}/{len(specs):4d} runs done")


def print_summary(done: dict[str, dict]):
    rows = list(done.values())
    if not rows:
        return
    print("\nMini-summary (all runs so far):")
    for d in sorted({r["defense"] for r in rows}):
        clean = [r for r in rows if r["defense"] == d and r["attack"] == "none"]
        att = [r for r in rows if r["defense"] == d and r["attack"] != "none"]
        exp = [r for r in att if r["exposed"]]
        print(f"  {d}: clean utility {pct(sum(r['task_success'] for r in clean), len(clean))} | attacked n={len(att)}: "
              f"ASR {pct(sum(r['attack_success'] for r in att), len(att))}, "
              f"attempted {pct(sum(r['attempted'] for r in att), len(att))}, "
              f"exposed {pct(len(exp), len(att))}, "
              f"utility {pct(sum(r['task_success'] for r in att), len(att))}")
    errs = Counter(r["error"].split(":")[0] for r in rows if r["error"])
    if errs:
        print(f"  run errors: {dict(errs)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test: 5 tasks x 2 attacks x 2 placements x D0")
    ap.add_argument("--max-runs", type=int, default=None, help="stop after this many new runs")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    cfg = load_config()
    results = ROOT / cfg["results_dir"]
    results.mkdir(parents=True, exist_ok=True)
    out_path = results / ("quick_runs.jsonl" if args.quick else "raw_runs.jsonl")
    plan = plan_runs(cfg, quick=args.quick)
    done = load_done(out_path)
    todo = [s for s in plan if run_key(s) not in done]
    if args.max_runs is not None:
        todo = todo[:args.max_runs]
    print(f"model={cfg['model']}  planned={len(plan)}  already done={len(plan) - len([s for s in plan if run_key(s) not in done])}"
          f"  to run now={len(todo)}  -> {out_path}")
    print_progress(plan, done)

    base = ToolEnv.from_disk()
    status = 0
    t0 = time.time()
    with out_path.open("a", encoding="utf-8") as f:
        for spec in tqdm(todo, unit="run"):
            hours = (time.time() - t0) / 3600
            if hours >= cfg.get("max_session_hours", 12):
                tqdm.write(f"\nStopped before {run_key(spec)}: session reached max_session_hours "
                           f"({cfg.get('max_session_hours', 12)} h); rerun the same command to resume.")
                status = 3
                break
            try:
                row = execute(spec, base, cfg)
            except (RateLimitExhausted, groq.RateLimitError) as e:
                tqdm.write(f"\nStopped at {run_key(spec)}: {e}\nRate limit reached; rerun the same command later to resume.")
                status = 3
                break
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            done[row["run_key"]] = row
    print(f"\nThis session: {time.time() - t0:.0f}s")
    print_progress(plan, done)
    print_summary(done)
    return status


if __name__ == "__main__":
    sys.exit(main())
