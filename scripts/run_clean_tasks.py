"""Step 5: run the 20 benign tasks clean (no attack, D0) and report clean utility.
Resumable: tasks already in the output file are skipped. Stops cleanly on a daily rate limit.

Run:  python scripts/run_clean_tasks.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agent import RateLimitExhausted, run_agent   # noqa: E402
from src.config import load_config                     # noqa: E402
from src.tasks import TASKS                            # noqa: E402
from src.tools import ToolEnv                          # noqa: E402

import groq  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "dev" / "step5_clean_D0.jsonl"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    base = ToolEnv.from_disk()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["task_id"] for l in OUT.open(encoding="utf-8")} if OUT.exists() else set()

    for t in TASKS:
        if t.id in done:
            continue
        try:
            out = run_agent(t.user_prompt, base.fresh_copy(), cfg)
        except (RateLimitExhausted, groq.RateLimitError) as e:
            print(f"\nStopped at {t.id}: {e}. Rerun later to resume.")
            break
        ok, failed = t.success_check(out)
        called = [tc["name"] for tc in out["tool_calls"]]
        out.update(task_id=t.id, attack="none", placement="none", defense="D0", model=cfg["model"],
                   prompt=t.user_prompt, task_success=ok, failed_checks=failed,
                   used_both_surfaces={"get_region_data", "search_reports"} <= set(called))
        with OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
        print(f"{t.id} {'PASS' if ok else 'FAIL'} tools={called} tokens={out['input_tokens']}+{out['output_tokens']} "
              f"calls={out['n_llm_calls']} {failed or ''} {out['error'] or ''}")

    rows = [json.loads(l) for l in OUT.open(encoding="utf-8")]
    n, k = len(rows), sum(r["task_success"] for r in rows)
    tok = [r["input_tokens"] + r["output_tokens"] for r in rows]
    print(f"\nClean utility (D0): {k}/{n} = {100 * k / n:.0f}%")
    print(f"Both surfaces used: {sum(r['used_both_surfaces'] for r in rows)}/{n}")
    print(f"Tokens per run: mean {sum(tok) / n:.0f}, min {min(tok)}, max {max(tok)}; "
          f"LLM calls per run: mean {sum(r['n_llm_calls'] for r in rows) / n:.2f}")


if __name__ == "__main__":
    main()
