# Results

Model `qwen/qwen3.8-27b` (reasoning_effort none), temperature 0. Runs done: 17/660 — **experiment in progress; numbers are partial**.

| Block | Status |
|---|---|
| clean | in progress (17/60) |
| D0 | not started (0/200) |
| D2 | not started (0/200) |
| D1 | not started (0/200) |

## Main table

| Defense | Attacked runs | ASR | ASR among exposed | Utility under attack | Clean utility | Clean runs |
|---|---|---|---|---|---|---|
| D0 none | not started (0/200) | n/a (n=0) | n/a (n=0) | n/a (n=0) | 94.1% (16/17) | in progress (17/20) |
| D1 data-tagging | not started (0/200) | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | not started (0/20) |
| D2 confirmation gate | not started (0/200) | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | not started (0/20) |

## All metrics

| Metric | D0 none | D1 data-tagging | D2 confirmation gate |
|---|---|---|---|
| Attacked runs | not started (0/200) | not started (0/200) | not started (0/200) |
| Clean runs | in progress (17/20) | not started (0/20) | not started (0/20) |
| ASR | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| Exposure rate | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| ASR among exposed | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| Attempt rate (incl. D2-blocked) | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| Blocked by D2 | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| Partial (mention only) | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| Flagged suspicious | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| Utility under attack (errors = fail) | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| Utility under attack (errors excl.) | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| Error rate (attacked) | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| Clean utility (errors = fail) | 94.1% (16/17) | n/a (n=0) | n/a (n=0) |
| Clean utility (errors excl.) | 100.0% (16/16) | n/a (n=0) | n/a (n=0) |
| Error rate (clean) | 5.9% (1/17) | n/a (n=0) | n/a (n=0) |
| Flagged suspicious (clean; false alarms) | 0.0% (0/17) | n/a (n=0) | n/a (n=0) |

## ASR by attack style

| Style | D0 none | D1 data-tagging | D2 confirmation gate |
|---|---|---|---|
| direct | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| fake_system | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| authority | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| metadata_note | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| hindi | n/a (n=0) | n/a (n=0) | n/a (n=0) |

## ASR by placement

| Placement | D0 none | D1 data-tagging | D2 confirmation gate |
|---|---|---|---|
| report | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| metadata | n/a (n=0) | n/a (n=0) | n/a (n=0) |

All rates are k/n. With n below about 30 per cell, small differences between cells may be noise; no significance tests are reported.

Errors by type: {'tool_use_failed': 1}
