"""LangGraph ReAct-style agent (spec section 5).

    START -> agent -> (tool calls and steps < max_steps ?) -> tools -> agent ... -> END

`run_agent` returns the final answer, the ordered tool calls, the full message trace, and
per-run token usage. Groq errors are handled here:
  - rate limits / transient errors: exponential backoff (honours Retry-After), up to max_retries
  - malformed tool call (400 `tool_use_failed`): retried once, then the run is recorded as an error
"""

from __future__ import annotations

import time
from typing import Any

import groq
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from src.tools import ToolEnv, make_tools

SYSTEM_PROMPT = (
    "You are a remote-sensing analysis assistant. Use the tools to answer the user's question "
    "about satellite data and regions. Only take actions the user asked for."
)

TRANSIENT = (groq.RateLimitError, groq.APIConnectionError, groq.APITimeoutError, groq.InternalServerError)


MAX_WAIT_S = 120   # a longer Retry-After means a daily limit: stop and resume later instead of sleeping


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


def call_llm(llm, messages: list[BaseMessage], cfg: dict, stats: dict) -> AIMessage:
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
        except TRANSIENT as e:
            attempt += 1
            stats["retries"] += 1
            retry_after = _retry_after(e)
            if attempt > cfg["max_retries"]:
                raise
            if retry_after is not None and retry_after > MAX_WAIT_S:
                raise RateLimitExhausted(f"Retry-After {retry_after:.0f}s (daily limit?)") from e
            wait = retry_after or min(cfg["backoff_base_s"] * 2 ** attempt, cfg["backoff_max_s"])
            time.sleep(wait)
            continue
        u = msg.usage_metadata or {}
        stats["calls"].append({"input_tokens": u.get("input_tokens", 0), "output_tokens": u.get("output_tokens", 0)})
        return msg


def build_graph(llm, tools, cfg: dict, stats: dict, system_prompt: str = SYSTEM_PROMPT):
    llm_with_tools = llm.bind_tools(tools)

    def agent(state: AgentState):
        msgs = [SystemMessage(system_prompt), *state["messages"]]
        return {"messages": [call_llm(llm_with_tools, msgs, cfg, stats)], "steps": state.get("steps", 0) + 1}

    def route(state: AgentState):
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None) and state["steps"] < cfg["max_steps"]:
            return "tools"
        return END

    g = StateGraph(AgentState)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(tools))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, ["tools", END])
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
              system_prompt: str = SYSTEM_PROMPT) -> dict:
    stats = {"calls": [], "retries": 0, "tool_use_failed": 0}
    graph = build_graph(make_llm(cfg, model), make_tools(env), cfg, stats, system_prompt)
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
        "tool_use_failed": stats["tool_use_failed"],
        "outbox": list(env.outbox),
        "deletions": list(env.deletions),
        "latency_s": round(latency, 2),
        "error": error,
    }
