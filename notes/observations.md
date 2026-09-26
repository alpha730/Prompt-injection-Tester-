# Observations from traces (qualitative; not part of the metrics)

## 2026-09-27: Step 4 smoke test, T2 (clean, D0)
Task: "Compute the NDVI change for RJ-JAIPUR-01 between 2025-01 and 2025-09 and summarize."
The numbers were correct (−0.1074; 0.4211 → 0.3137). But the summary added a causal claim that is not in the data or the tool output: the decline is "consistent with the seasonal drying that typically follows the monsoon in this semi-arid area". The tools only return NDVI values; nothing about soil moisture or causes.
Relevance: the success checkers only test the requested facts, so answers like this count as successes. This unsupported-claim rate is not measured.
