"""
Real-time event stream for the agent pipeline.

A tiny thread-safe pub/sub bus plus a LangChain callback handler.  The
orchestrator wraps every agent with `instrument(agent, name)` so each
agent's LLM turns, tool calls and tool results are published as events.
`server/live_server.py` subscribes and forwards them to the browser over
a WebSocket.

Nothing here imports tornado / flask, so importing this module from the
pipeline is free even when no GUI is running.
"""

from __future__ import annotations

import itertools
import queue
import threading
import time
from typing import Any

from langchain_core.callbacks.base import BaseCallbackHandler

# ---------------------------------------------------------------------------
# BUS
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_subscribers: list["queue.Queue[dict]"] = []
_history: list[dict] = []
_seq = itertools.count(1)

_MAX_HISTORY = 5000


def subscribe() -> "queue.Queue[dict]":
    """Register a new subscriber queue, pre-loaded with the event history."""
    q: "queue.Queue[dict]" = queue.Queue()
    with _lock:
        for ev in _history:
            q.put_nowait(ev)
        _subscribers.append(q)
    return q


def unsubscribe(q: "queue.Queue[dict]") -> None:
    with _lock:
        if q in _subscribers:
            _subscribers.remove(q)


def clear_history() -> None:
    with _lock:
        _history.clear()


def emit(kind: str, **data: Any) -> dict:
    """Publish an event to every subscriber and append it to the history."""
    ev = {"seq": next(_seq), "t": time.time(), "kind": kind, **data}
    with _lock:
        _history.append(ev)
        if len(_history) > _MAX_HISTORY:
            del _history[: len(_history) - _MAX_HISTORY]
        subs = list(_subscribers)
    for q in subs:
        try:
            q.put_nowait(ev)
        except Exception:
            pass
    return ev


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _text(content: Any) -> str:
    """Flatten LangChain message content (str | list[parts]) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict):
                out.append(part.get("text") or part.get("content") or "")
            else:
                out.append(str(part))
        return "".join(out)
    return str(content)


# ---------------------------------------------------------------------------
# CALLBACK HANDLER  (one instance per agent)
# ---------------------------------------------------------------------------

class AgentStream(BaseCallbackHandler):
    """Publishes bus events for one named agent's LLM / tool activity."""

    raise_error = False

    def __init__(self, agent: str):
        self.agent = agent

    # -- input the agent was handed -------------------------------------
    def on_chain_start(self, serialized, inputs, *, run_id=None, parent_run_id=None, **kw):
        if parent_run_id is not None:
            return  # only the top-level invocation, not inner graph nodes
        msgs = inputs.get("messages") if isinstance(inputs, dict) else None
        if not msgs:
            return
        last = msgs[-1]
        content = last.get("content") if isinstance(last, dict) else getattr(last, "content", "")
        text = _text(content)
        if text.strip():
            emit("input", agent=self.agent, content=text)

    # -- LLM turns ----------------------------------------------------------
    def on_chat_model_start(self, serialized, messages, *, run_id=None, **kw):
        emit("llm_start", agent=self.agent, run_id=str(run_id))

    def on_llm_start(self, serialized, prompts, *, run_id=None, **kw):
        emit("llm_start", agent=self.agent, run_id=str(run_id))

    def on_llm_end(self, response, *, run_id=None, **kw):
        text, tool_calls = "", []
        try:
            gen = response.generations[0][0]
            msg = getattr(gen, "message", None)
            text = _text(getattr(msg, "content", None) if msg is not None else getattr(gen, "text", ""))
            tool_calls = list(getattr(msg, "tool_calls", None) or [])
        except Exception:
            pass
        if text.strip():
            emit("message", agent=self.agent, role="ai", content=text, run_id=str(run_id))
        for tc in tool_calls:
            emit("tool_call", agent=self.agent, tool=tc.get("name"),
                 args=tc.get("args"), run_id=str(run_id))

    def on_llm_error(self, error, *, run_id=None, **kw):
        emit("error", agent=self.agent, error=str(error))

    # -- tool calls -------------------------------------------------------
    def on_tool_start(self, serialized, input_str, *, run_id=None, **kw):
        emit("tool_start", agent=self.agent,
             tool=(serialized or {}).get("name"), input=_text(input_str), run_id=str(run_id))

    def on_tool_end(self, output, *, run_id=None, **kw):
        emit("tool_end", agent=self.agent,
             output=_text(getattr(output, "content", output)), run_id=str(run_id))

    def on_tool_error(self, error, *, run_id=None, **kw):
        emit("error", agent=self.agent, error=str(error))


def instrument(agent, name: str):
    """Bind an AgentStream callback (and a run name) to a compiled agent."""
    return agent.with_config({"callbacks": [AgentStream(name)], "run_name": name})
