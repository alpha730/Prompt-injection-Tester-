"""Daily progress check for the full run. Makes no API calls.

  python -m src.status

Prints runs done per block (clean split by defense), tokens and requests used (total and per UTC
day), errors so far, and the estimated days left. Two estimates are given:
  budget    remaining runs x average tokens (and requests) per run, over the free-tier daily limits
  observed  remaining runs over the average number of runs per day actually completed so far
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from math import ceil

from src.analyze import block_progress, load_runs, status_label
from src.config import ROOT, load_config
from src.run_experiment import BLOCK_ORDER, plan_runs, run_key

# Groq free-tier limits for this key and model (see notes/limitations_draft.md).
DAILY_TOKENS = 200_000
DAILY_REQUESTS = 1_000


def status(cfg: dict) -> str:
    results = ROOT / cfg["results_dir"]
    rows, dups = load_runs(results / "raw_runs.jsonl")
    plan = plan_runs(cfg)
    keys = {run_key(s) for s in plan}
    rows = [r for r in rows if r["run_key"] in keys]
    prog = block_progress(plan, rows)
    done, total = len(rows), len(plan)
    remaining = total - done

    out = [f"Progress: {done}/{total} runs ({100 * done / total:.1f}%)  model={cfg['model']}", ""]
    for b in BLOCK_ORDER:
        out.append(f"  {b:6} {prog[b][0]:4d}/{prog[b][1]:4d}  {status_label(*prog[b])}")
        if b == "clean":
            for d in ("D0", "D2", "D1"):
                if f"clean:{d}" in prog:
                    out.append(f"    {d}   {prog[f'clean:{d}'][0]:4d}/{prog[f'clean:{d}'][1]:4d}")

    tok_in = sum(r.get("input_tokens", 0) for r in rows)
    tok_out = sum(r.get("output_tokens", 0) for r in rows)
    reqs = sum(r.get("n_llm_calls", 0) for r in rows)
    per_day: dict[str, list] = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        day = r["timestamp"][:10]
        per_day[day][0] += 1
        per_day[day][1] += r.get("input_tokens", 0) + r.get("output_tokens", 0)
        per_day[day][2] += r.get("n_llm_calls", 0)
    out += ["", f"Tokens used: {tok_in + tok_out:,} (input {tok_in:,}, output {tok_out:,}); LLM requests: {reqs:,}"]
    if per_day:
        out.append("Per UTC day:  date        runs   tokens  requests")
        for day in sorted(per_day):
            n, t, q = per_day[day]
            out.append(f"              {day}  {n:4d}  {t:7,}  {q:5d}")

    errs = [r for r in rows if r.get("error")]
    kinds = Counter(r["error"].split(":")[0].split(" (")[0] for r in errs)
    out += ["", f"Errors so far: {len(errs)}/{done}" + (f"  {dict(kinds)}" if kinds else "")]
    for r in errs[-3:]:
        out.append(f"  {r['run_key']}: {r['error'][:90]}")
    if dups:
        out.append(f"WARNING: {len(dups)} duplicate run keys in raw_runs.jsonl")

    out.append("")
    if remaining == 0:
        out.append("All runs done. Next: python -m src.analyze && python scripts/make_readme.py")
    elif done:
        tpr, qpr = (tok_in + tok_out) / done, reqs / done
        runs_per_day = min(DAILY_TOKENS / tpr, DAILY_REQUESTS / qpr)
        out.append(f"Average per run: {tpr:,.0f} tokens, {qpr:.1f} requests")
        out.append(f"Days left (budget, {DAILY_TOKENS // 1000}K tokens/day and {DAILY_REQUESTS} requests/day): "
                   f"~{ceil(remaining / runs_per_day)} ({runs_per_day:.0f} runs/day)")
        observed = done / len(per_day)
        out.append(f"Days left (observed pace, {observed:.0f} runs per active day over {len(per_day)} day(s)): "
                   f"~{ceil(remaining / observed)}")
        out.append("Attacked runs may use more tokens than clean runs; the estimate updates as they come in.")
    else:
        out.append("No runs yet; no estimate.")
    return "\n".join(out)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print(status(load_config()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
