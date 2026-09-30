"""Offline tests of the agent loop with a fake LLM (no API calls)."""

import json
from pathlib import Path

import groq
import httpx
import pytest
from langchain_core.messages import AIMessage

from src import agent
from src.tools import ToolEnv

FIXTURE = Path(__file__).parent / "fixtures" / "rj_jaipur_01.json"
CFG = {"model": "fake", "temperature": 0, "max_output_tokens": 64, "reasoning_effort": None,
       "max_steps": 8, "delay_between_calls_s": 0, "max_retries": 2, "backoff_base_s": 0, "backoff_max_s": 0}


def tool_use_failed():
    body = {"error": {"message": "Failed to call a function.", "type": "invalid_request_error",
                      "code": "tool_use_failed", "failed_generation": "<bad>"}}
    resp = httpx.Response(400, request=httpx.Request("POST", "https://api.groq.com"), json=body)
    return groq.BadRequestError("Error code: 400 - tool_use_failed", response=resp, body=body)


class FakeLLM:
    """Replays a script of AIMessages / exceptions, one per invoke()."""
    def __init__(self, script):
        self.script = list(script)

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def ai(content="", tool_calls=(), tin=100, tout=10):
    return AIMessage(content=content, tool_calls=[{"name": n, "args": a, "id": f"c{i}"} for i, (n, a) in enumerate(tool_calls)],
                     usage_metadata={"input_tokens": tin, "output_tokens": tout, "total_tokens": tin + tout})


@pytest.fixture
def env():
    rec = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return ToolEnv(regions={rec["id"]: rec}, reports={})


def run(monkeypatch, env, script, **cfg):
    monkeypatch.setattr(agent, "make_llm", lambda c, m=None: FakeLLM(script))
    return agent.run_agent("q", env, {**CFG, **cfg})


def test_tool_loop_and_token_accounting(monkeypatch, env):
    out = run(monkeypatch, env, [
        ai(tool_calls=[("compute_ndvi_change", {"region_id": "RJ-JAIPUR-01", "start": "2025-01", "end": "2025-09"})]),
        ai(content="NDVI fell by 0.1074.", tin=200, tout=20),
    ])
    assert out["error"] is None and out["final_answer"] == "NDVI fell by 0.1074."
    assert out["tool_calls"] == [{"name": "compute_ndvi_change",
                                  "args": {"region_id": "RJ-JAIPUR-01", "start": "2025-01", "end": "2025-09"}}]
    assert "-0.1074" in [m for m in out["messages"] if m["type"] == "tool"][0]["content"]
    assert (out["n_llm_calls"], out["input_tokens"], out["output_tokens"]) == (2, 300, 30)


def test_tool_use_failed_retried_once_then_succeeds(monkeypatch, env):
    out = run(monkeypatch, env, [tool_use_failed(), ai(content="done")])
    assert out["error"] is None and out["final_answer"] == "done" and out["tool_use_failed"] == 1


def test_tool_use_failed_twice_is_recorded_not_raised(monkeypatch, env):
    out = run(monkeypatch, env, [tool_use_failed(), tool_use_failed(), ai(content="never reached")])
    assert out["error"].startswith("tool_use_failed (after 1 retry)")
    assert out["final_answer"] == "" and out["tool_use_failed"] == 2


def test_max_steps_stops_loop(monkeypatch, env):
    loop = [ai(tool_calls=[("get_region_data", {"region_id": "RJ-JAIPUR-01"})]) for _ in range(10)]
    out = run(monkeypatch, env, loop, max_steps=3)
    assert out["n_llm_calls"] == 3 and out["hit_max_steps"] and out["final_answer"] == ""
    assert out["error"] is None


TPM_MSG = ("Rate limit reached for model `qwen/qwen3.8-27b` in organization `org_x` service tier `on_demand` "
           "on tokens per minute (TPM): Limit 8000, Used 7900, Requested 2500. Please try again in 17.5s.")
TPD_MSG = ("Rate limit reached for model `qwen/qwen3.8-27b` in organization `org_x` service tier `on_demand` "
           "on tokens per day (TPD): Limit 200000, Used 199000, Requested 3000. Please try again in 3m28s.")


def rate_limited(retry_after, message=TPM_MSG):
    """A mocked Groq 429 with the body and headers the real API sends."""
    body = {"error": {"message": message, "type": "tokens", "code": "rate_limit_exceeded"}}
    headers = {"retry-after": str(retry_after), "x-ratelimit-limit-tokens": "8000",
               "x-ratelimit-remaining-tokens": "100", "x-ratelimit-reset-tokens": "17.5s",
               "x-ratelimit-limit-requests": "1000", "x-ratelimit-remaining-requests": "950"}
    resp = httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com"), headers=headers, json=body)
    return groq.RateLimitError(f"Error code: 429 - {body}", response=resp, body=body)


@pytest.fixture
def rl(monkeypatch, tmp_path):
    """No real sleeping; rate-limit log in tmp; counter reset. Returns (sleeps, log path)."""
    sleeps = []
    monkeypatch.setattr(agent.time, "sleep", sleeps.append)
    monkeypatch.setattr(agent, "_consecutive_waits", 0)
    return sleeps, tmp_path / "ratelimit.log"


def log_records(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]


def test_per_minute_429_waits_and_retries_same_call(monkeypatch, env, rl):
    sleeps, log = rl
    out = run(monkeypatch, env, [rate_limited(18), rate_limited(18), ai(content="ok")], ratelimit_log=str(log))
    assert out["final_answer"] == "ok" and out["error"] is None      # not a run error
    assert out["rate_limit_waits"] == 2 and out["retries"] == 0 and out["n_llm_calls"] == 1
    waits = [s for s in sleeps if s > 0]
    assert len(waits) == 2 and all(19 <= w <= 23 for w in waits)      # retry-after + 1..5 s jitter
    recs = log_records(log)
    assert [r["action"] for r in recs] == ["wait and retry"] * 2
    assert recs[0]["headers"]["retry-after"] == "18" and recs[0]["headers"]["x-ratelimit-limit-tokens"] == "8000"
    assert "tokens per minute" in recs[0]["message"] and recs[0]["daily"] is False
    assert agent._consecutive_waits == 0                              # reset by the successful call


def test_daily_429_stops_without_waiting(monkeypatch, env, rl):
    sleeps, log = rl
    with pytest.raises(agent.RateLimitExhausted, match="daily limit"):
        run(monkeypatch, env, [rate_limited(208, TPD_MSG), ai(content="never")], ratelimit_log=str(log))
    assert not [s for s in sleeps if s > 0]
    rec = log_records(log)[0]
    assert rec["action"] == "stop: daily limit" and rec["daily"] is True and "tokens per day" in rec["message"]


def test_requests_per_day_is_daily(monkeypatch, env, rl):
    msg = TPM_MSG.replace("tokens per minute (TPM)", "requests per day (RPD)")
    with pytest.raises(agent.RateLimitExhausted):
        run(monkeypatch, env, [rate_limited(60, msg)], ratelimit_log=str(rl[1]))


def test_consecutive_wait_cap_stops(monkeypatch, env, rl):
    with pytest.raises(agent.RateLimitExhausted, match="consecutive"):
        run(monkeypatch, env, [rate_limited(5)] * 4 + [ai(content="never")],
            ratelimit_log=str(rl[1]), max_consecutive_rate_limit_waits=3)
    assert log_records(rl[1])[-1]["action"] == "stop: too many consecutive waits"


def test_very_long_retry_after_stops(monkeypatch, env, rl):
    with pytest.raises(agent.RateLimitExhausted):
        run(monkeypatch, env, [rate_limited(3600)], ratelimit_log=str(rl[1]))


def test_other_errors_recorded(monkeypatch, env):
    out = run(monkeypatch, env, [ValueError("boom")])
    assert out["error"] == "ValueError: boom"
