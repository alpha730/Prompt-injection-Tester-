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


def test_other_errors_recorded(monkeypatch, env):
    out = run(monkeypatch, env, [ValueError("boom")])
    assert out["error"] == "ValueError: boom"
