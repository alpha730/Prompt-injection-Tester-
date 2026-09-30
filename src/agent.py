"""LangGraph ReAct-style agent (spec section 5).

    START -> agent -> (tool calls and steps < max_steps ?) -> [gate] -> tools -> agent ... -> END

The `gate` node exists only under defense D2; D1 tags tool outputs in the `tools` node
(see src/defenses.py).

`run_agent` returns the final answer, the ordered tool calls, the full message trace, and
per-run token usage. Groq errors are handled here:
  - 429 rate limits: every one is logged (message + retry-after / x-ratelimit-* headers) to
    results/dev/ratelimit.log. A daily limit (tokens or requests per day) raises
    RateLimitExhausted so the runner saves and stops. Any other 429 (per-minute limits) sleeps
    Retry-After plus jitter and retries the same call; it is not a run error. Safety caps: more
    than max_consecutive_rate_limit_waits waits in a row, or a Retry-After above
    max_rate_limit_wait_s, also raise RateLimitExhausted.
  - other transient errors (connection, timeout, 5xx): exponential backoff, up to max_retries
  - malformed tool call (400 `tool_use_failed`): retried once, then the run is recorded as an error
"""

from __future__ import annotations

import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import groq
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, MessagesState, StateGraph

from src.config import ROOT
from src.defenses import BLOCKED_MSG, DEFENSES, SYSTEM_PROMPT, Defense, gate_allows, tag
from src.tools import ToolEnv, make_tools

TRANSIENT = (groq.RateLimitError, groq.APIConnectionError, groq.APITimeoutError, groq.InternalServerError)


DAILY_RE = re.compile(r"per day|\(TPD\)|\(RPD\)|tokens_per_day|requests_per_day", re.I)
RATE_LIMIT_DEFAULTS = {"max_consecutive_rate_limit_waits": 30, "max_rate_limit_wait_s": 900, "rate_limit_jitter_s": 5}
_consecutive_waits = 0   # 429 waits in a row, across calls and runs; reset by any successful call


class ToolCallFailed(Exception):
    """Groq rejected the model's tool call as malformed (tool_use_failed), twice."""


class RateLimitExhausted(Exception):
    """Rate limit that won't clear soon (e.g. tokens/day). Runners should stop and resume later,
    not record the run."""


class AgentState(MessagesState):
    steps: int   # number of LLM calls so far


def make_llm(cfg: dict, model: str | None = None) -> ChatGroq:
    kwargs: dict[str, Any] = dict(
        model=model or cfg["model"],
        temperature=cfg["temperature"],
        max_tokens=cfg["max_output_tokens"],
        max_retries=0,   # retries are handled in call_llm so they can be logged
    )
    if cfg.get("reasoning_effort") not in (None, "TBD"):
        kwargs["reasoning_effort"] = cfg["reasoning_effort"]
    return ChatGroq(**kwargs)


def _is_tool_use_failed(e: Exception) -> bool:
    body = getattr(e, "body", None)
    code = body.get("error", {}).get("code") if isinstance(body, dict) and "error" in body else \
        (body.get("code") if isinstance(body, dict) else None)
    return code == "tool_use_failed" or "tool_use_failed" in str(e)


def _retry_after(e: Exception) -> float | None:
    resp = getattr(e, "response", None)
    try:
        return float(resp.headers.get("retry-after")) if resp is not None else None
    except (TypeError, ValueError):
        return None


def _ratelimit_log_path(cfg: dict) -> Path:
    return Path(cfg["ratelimit_log"]) if cfg.get("ratelimit_log") else         ROOT / cfg.get("results_dir", "results") / "dev" / "ratelimit.log"


def _rate_limit_info(e: Exception) -> dict:
    resp = getattr(e, "response", None)
    headers = {k.lower(): v for k, v in resp.headers.items()} if resp is not None else {}
    try:
        body = resp.json() if resp is not None else None
    except Exception:
        body = getattr(e, "body", None)
    text = str(e) + " " + json.dumps(body, default=str)
    return {"message": str(e), "body": body,
            "headers": {k: v for k, v in headers.items() if k == "retry-after" or k.startswith("x-ratelimit")},
            "daily": bool(DAILY_RE.search(text))}


def _log_rate_limit(cfg: dict, info: dict, action: str, wait_s: float | None):
    path = _ratelimit_log_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "action": action,
           "wait_s": wait_s, "consecutive_waits": _consecutive_waits, **info}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def _handle_rate_limit(e: Exception, cfg: dict, stats: dict):
    """Sleep through a per-minute 429, or raise RateLimitExhausted (daily limit or safety cap)."""
    global _consecutive_waits
    c = {**RATE_LIMIT_DEFAULTS, **cfg}
    info = _rate_limit_info(e)
    retry_after = _retry_after(e)
    if info["daily"]:
        _log_rate_limit(cfg, info, "stop: daily limit", None)
        raise RateLimitExhausted(f"daily limit: {info['message'][:200]}") from e
    if retry_after is not None and retry_after > c["max_rate_limit_wait_s"]:
        _log_rate_limit(cfg, info, "stop: retry-after above max_rate_limit_wait_s", None)
        raise RateLimitExhausted(f"Retry-After {retry_after:.0f}s > {c['max_rate_limit_wait_s']}s") from e
    if _consecutive_waits >= c["max_consecutive_rate_limit_waits"]:
        _log_rate_limit(cfg, info, "stop: too many consecutive waits", None)
        raise RateLimitExhausted(f"{_consecutive_waits} consecutive rate-limit waits") from e
    base = retry_after if retry_after is not None else min(cfg["backoff_base_s"] * 2 ** (_consecutive_waits + 1),
                                                           cfg["backoff_max_s"])
    wait = base + random.uniform(1, c["rate_limit_jitter_s"]) if c["rate_limit_jitter_s"] else base
    _consecutive_waits += 1
    stats["rate_limit_waits"] += 1
    stats["rate_limit_wait_s"] += wait
    _log_rate_limit(cfg, info, "wait and retry", round(wait, 1))
    time.sleep(wait)


def call_llm(llm, messages: list[BaseMessage], cfg: dict, stats: dict) -> AIMessage:
    global _consecutive_waits
    tool_use_failures = 0
    attempt = 0
    while True:
        time.sleep(cfg["delay_between_calls_s"])
        try:
            msg = llm.invoke(messages)
        except groq.BadRequestError as e:
            if not _is_tool_use_failed(e):
                raise
            stats["tool_use_failed"] += 1
            tool_use_failures += 1
            if tool_use_failures > 1:
                raise ToolCallFailed(str(e)) from e
            continue
        except groq.RateLimitError as e:
            _handle_rate_limit(e, cfg, stats)
            continue
        except TRANSIENT as e:
            attempt += 1
            stats["retries"] += 1
            if attempt > cfg["max_retries"]:
                raise
            wait = _retry_after(e) or min(cfg["backoff_base_s"] * 2 ** attempt, cfg["backoff_max_s"])
            time.sleep(wait)
            continue
        _consecutive_waits = 0
        u = msg.usage_metadata or {}
        stats["calls"].append({"input_tokens": u.get("input_tokens", 0), "output_tokens": u.get("output_tokens", 0)})
        return msg


def _pending_calls(messages: list[BaseMessage]) -> tuple[AIMessage, list[dict]]:
    """The last AI message and its tool calls that have no ToolMessage answer yet."""
    i = max(k for k, m in enumerate(messages) if isinstance(m, AIMessage))
    answered = {m.tool_call_id for m in messages[i + 1:] if isinstance(m, ToolMessage)}
    return messages[i], [tc for tc in messages[i].tool_calls if tc["id"] not in answered]


def build_graph(llm, tools, cfg: dict, stats: dict, defense: Defense, user_prompt: str):
    llm_with_tools = llm.bind_tools(tools)
    by_name = {t.name: t for t in tools}

    def agent(state: AgentState):
        msgs = [SystemMessage(defense.system_prompt), *state["messages"]]
        return {"messages": [call_llm(llm_with_tools, msgs, cfg, stats)], "steps": state.get("steps", 0) + 1}

    def gate(state: AgentState):
        """D2: answer unrequested harmful calls with a block message; the tools node skips them."""
        _, pending = _pending_calls(state["messages"])
        blocked = [tc for tc in pending if not gate_allows(tc, user_prompt)]
        stats["blocked"] += [{"name": tc["name"], "args": tc["args"]} for tc in blocked]
        return {"messages": [ToolMessage(BLOCKED_MSG, tool_call_id=tc["id"], name=tc["name"]) for tc in blocked]}

    def run_tools(state: AgentState):
        _, pending = _pending_calls(state["messages"])
        out = []
        for tc in pending:
            tool = by_name.get(tc["name"])
            try:
                content = str(tool.invoke(tc["args"])) if tool else f"Error: unknown tool '{tc['name']}'."
            except Exception as e:   # bad arguments etc. are returned to the model, as ToolNode does
                content = f"Error: {type(e).__name__}: {e}"
            out.append(ToolMessage(tag(content) if defense.tag_tool_data else content,
                                   tool_call_id=tc["id"], name=tc["name"]))
        return {"messages": out}

    first_tool_node = "gate" if defense.gate else "tools"

    def route(state: AgentState):
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None) and state["steps"] < cfg["max_steps"]:
            return first_tool_node
        return END

    g = StateGraph(AgentState)
    g.add_node("agent", agent)
    g.add_node("tools", run_tools)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, [first_tool_node, END])
    if defense.gate:
        g.add_node("gate", gate)
        g.add_edge("gate", "tools")
    g.add_edge("tools", "agent")
    return g.compile()


def serialize(m: BaseMessage) -> dict:
    d = {"type": m.type, "content": m.content}
    if isinstance(m, AIMessage):
        d["tool_calls"] = [{"name": tc["name"], "args": tc["args"], "id": tc["id"]} for tc in m.tool_calls]
        if m.additional_kwargs.get("reasoning_content"):
            d["reasoning"] = m.additional_kwargs["reasoning_content"]
    if m.type == "tool":
        d["name"], d["tool_call_id"] = m.name, m.tool_call_id
    return d


def run_agent(user_prompt: str, env: ToolEnv, cfg: dict, model: str | None = None,
              defense: str = "D0") -> dict:
    stats = {"calls": [], "retries": 0, "tool_use_failed": 0, "blocked": [],
             "rate_limit_waits": 0, "rate_limit_wait_s": 0.0}
    graph = build_graph(make_llm(cfg, model), make_tools(env), cfg, stats, DEFENSES[defense], user_prompt)
    state: dict = {"messages": [], "steps": 0}
    error = None
    t0 = time.time()
    try:
        for state in graph.stream({"messages": [("user", user_prompt)], "steps": 0},
                                  stream_mode="values", config={"recursion_limit": 2 * cfg["max_steps"] + 5}):
            pass
    except ToolCallFailed as e:
        error = f"tool_use_failed (after 1 retry): {e}"
    except (RateLimitExhausted, groq.RateLimitError):
        raise   # not a property of the run: the caller stops and resumes later
    except Exception as e:  # recorded, never crashes the experiment
        error = f"{type(e).__name__}: {e}"
    latency = time.time() - t0

    messages = state.get("messages", [])
    ai = [m for m in messages if isinstance(m, AIMessage)]
    last = ai[-1] if ai else None
    return {
        "final_answer": (last.content if last is not None and not last.tool_calls else ""),
        "tool_calls": [{"name": tc["name"], "args": tc["args"]} for m in ai for tc in m.tool_calls],
        "messages": [serialize(m) for m in messages],
        "hit_max_steps": bool(last is not None and last.tool_calls and state.get("steps", 0) >= cfg["max_steps"]),
        "n_llm_calls": len(stats["calls"]),
        "input_tokens": sum(c["input_tokens"] for c in stats["calls"]),
        "output_tokens": sum(c["output_tokens"] for c in stats["calls"]),
        "tokens_per_call": stats["calls"],
        "retries": stats["retries"],
        "rate_limit_waits": stats["rate_limit_waits"],
        "rate_limit_wait_s": round(stats["rate_limit_wait_s"], 1),
        "tool_use_failed": stats["tool_use_failed"],
        "blocked_calls": stats["blocked"],
        "outbox": list(env.outbox),
        "deletions": list(env.deletions),
        "latency_s": round(latency, 2),
        "error": error,
    }
