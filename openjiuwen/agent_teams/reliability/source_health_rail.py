# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Rail that shares source-health observations across TaskTool subagents."""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlparse

from openjiuwen.agent_teams.reliability.source_health import (
    SourceHealthKey,
    SourceHealthOutcome,
    SourceHealthProvenance,
    SourceHealthStore,
    is_threshold_hit,
    recheck_window_for,
)
from openjiuwen.core.foundation.llm.schema.message import ToolMessage
from openjiuwen.core.single_agent.rail.base import AgentCallbackContext
from openjiuwen.harness.rails.base import DeepAgentRail


_EMPTY_MARKERS = (
    "no results found",
    "[empty]",
    "no search results",
)


def _tool_args_as_dict(tool_args: Any) -> dict[str, Any]:
    if isinstance(tool_args, dict):
        return tool_args
    if isinstance(tool_args, str):
        try:
            parsed = json.loads(tool_args)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _registered_domain(host: str) -> str:
    """Best-effort eTLD+1 collapse without adding a dependency."""
    host = (host or "").split(":", 1)[0].strip(".").lower()
    if not host:
        return ""
    parts = [part for part in host.split(".") if part]
    if len(parts) <= 2:
        return host
    if parts[-2] in {"co", "com", "org", "net", "gov", "ac"} and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def _domain_from_url(url: str) -> str:
    parsed = urlparse(url if "://" in url else f"https://{url}")
    return _registered_domain(parsed.netloc)


def _query_signature(query: str) -> str:
    terms = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", (query or "").lower())
    return " ".join(terms[:12])


def _source_key(tool_name: str, tool_args: dict[str, Any]) -> SourceHealthKey | None:
    url = str(tool_args.get("url") or tool_args.get("link") or "").strip()
    if url:
        domain = _domain_from_url(url)
        if domain:
            return SourceHealthKey(tool_name=tool_name, source=domain)
    query = str(tool_args.get("query") or tool_args.get("q") or tool_args.get("task_description") or "").strip()
    if query:
        signature = _query_signature(query)
        if signature:
            return SourceHealthKey(tool_name=tool_name, source=signature)
    return None


def _result_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)
    return str(value)


def _classify_result(result: Any) -> tuple[SourceHealthOutcome, SourceHealthProvenance] | None:
    text = _result_text(result).strip()
    lowered = text.lower()
    if not text or any(marker in lowered for marker in _EMPTY_MARKERS):
        return SourceHealthOutcome.EMPTY, SourceHealthProvenance.BODY
    if "content quality:" in lowered:
        if "unrelated" in lowered or "garbage" in lowered:
            return SourceHealthOutcome.GARBAGE, SourceHealthProvenance.BODY
        if "empty" in lowered:
            return SourceHealthOutcome.EMPTY, SourceHealthProvenance.BODY
        if "landing" in lowered or "low_signal" in lowered:
            return SourceHealthOutcome.LOW_SIGNAL, SourceHealthProvenance.BODY
    if "status: 403" in lowered or "http 403" in lowered or "status_code\": 403" in lowered:
        return SourceHealthOutcome.HTTP_403, SourceHealthProvenance.TRANSPORT
    if "status: 429" in lowered or "http 429" in lowered or "status_code\": 429" in lowered:
        return SourceHealthOutcome.HTTP_429, SourceHealthProvenance.TRANSPORT
    if "blocked" in lowered or "access denied" in lowered or "captcha" in lowered:
        return SourceHealthOutcome.BLOCKED, SourceHealthProvenance.BODY
    if "getaddrinfo" in lowered or "name or service not known" in lowered or "dns" in lowered:
        return SourceHealthOutcome.DNS_FAIL, SourceHealthProvenance.TRANSPORT
    if "[error]:" in lowered or "connection timed out" in lowered or "timeout" in lowered:
        return SourceHealthOutcome.TRANSPORT, SourceHealthProvenance.TRANSPORT
    return None


def _current_turn(ctx: AgentCallbackContext) -> int:
    session = getattr(ctx, "session", None)
    if session is None:
        return 0
    for key in ("source_health_turn", "turn", "iteration"):
        try:
            value = session.get_state(key)
        except Exception:
            continue
        if isinstance(value, int):
            return value
    return 0


def _should_skip(entry: Any, turn: int) -> bool:
    if not is_threshold_hit(entry):
        return False
    if (
        entry.provenance == SourceHealthProvenance.BODY
        and not entry.has_transport_observation
    ):
        return False
    if entry.recheck_after_turn is not None and turn >= entry.recheck_after_turn:
        return False
    return True


class SourceHealthRail(DeepAgentRail):
    """Record source-health outcomes into a shared store.

    The rail instance is per-agent. The store is shared by the parent
    ``DeepAgent`` and TaskTool-spawned children, giving parallel subagents one
    run-scoped view of broken sources.
    """

    priority = 75

    def __init__(self, store: SourceHealthStore) -> None:
        super().__init__()
        self._store = store

    async def before_tool_call(self, ctx: AgentCallbackContext) -> None:
        inputs = getattr(ctx, "inputs", None)
        tool_name = str(getattr(inputs, "tool_name", "") or "").strip()
        if not tool_name:
            return
        key = _source_key(tool_name, _tool_args_as_dict(getattr(inputs, "tool_args", None)))
        if key is None:
            return
        entry = self._store.get(key)
        if entry is None:
            return
        turn = _current_turn(ctx)
        if not _should_skip(entry, turn):
            return

        message = (
            "Source health blocked this tool call: "
            f"{key.source} has {entry.fail_count} recent {entry.outcome.value} observations. "
            "Use another source or broaden the query."
        )
        tool_call = getattr(inputs, "tool_call", None)
        tool_call_id = str(getattr(tool_call, "id", "") or "") or "unknown-tool-call"
        if hasattr(ctx, "extra"):
            ctx.extra["_skip_tool"] = True
        if inputs is not None:
            inputs.tool_result = {"error": message, "source_health": entry.outcome.value}
            inputs.tool_msg = ToolMessage(content=message, tool_call_id=tool_call_id)
        self._emit_decision(ctx, "source.blocked", key, entry.outcome)

    async def after_tool_call(self, ctx: AgentCallbackContext) -> None:
        inputs = getattr(ctx, "inputs", None)
        tool_name = str(getattr(inputs, "tool_name", "") or "").strip()
        if not tool_name:
            return
        tool_args = _tool_args_as_dict(getattr(inputs, "tool_args", None))
        key = _source_key(tool_name, tool_args)
        if key is None:
            return
        verdict = _classify_result(getattr(inputs, "tool_result", None))
        if verdict is None:
            return
        outcome, provenance = verdict
        tool_call = getattr(inputs, "tool_call", None)
        logical_call_id = str(getattr(tool_call, "id", "") or "") or None
        turn = _current_turn(ctx)
        recheck_window = recheck_window_for(outcome)
        self._store.record(
            key,
            outcome=outcome,
            provenance=provenance,
            turn=turn,
            logical_call_id=logical_call_id,
            recheck_after_turn=turn + recheck_window if recheck_window is not None else None,
        )
        event = {
            SourceHealthOutcome.EMPTY: "source.empty",
            SourceHealthOutcome.GARBAGE: "source.garbage",
            SourceHealthOutcome.LOW_SIGNAL: "source.steered",
        }.get(outcome)
        if event:
            self._emit_decision(ctx, event, key, outcome)

    async def after_invoke(self, ctx: AgentCallbackContext) -> None:
        inputs = getattr(ctx, "inputs", None)
        result = getattr(inputs, "result", None)
        if not isinstance(result, dict):
            return
        candidates = [
            {
                "tool_name": key.tool_name,
                "source": key.source,
                "outcome": entry.outcome.value,
                "provenance": entry.provenance.value,
                "fail_count": entry.fail_count,
                "first_seen_turn": entry.first_seen_turn,
                "last_seen_turn": entry.last_seen_turn,
            }
            for key, entry in self._store.threshold_hits()
        ]
        if candidates:
            result["source_health_candidates"] = candidates

    @staticmethod
    def _emit_decision(
        ctx: AgentCallbackContext,
        event_name: str,
        key: SourceHealthKey,
        outcome: SourceHealthOutcome,
    ) -> None:
        extra = getattr(ctx, "extra", None)
        if isinstance(extra, dict):
            events = extra.setdefault("source_health_events", [])
            if isinstance(events, list):
                events.append(
                    {
                        "event": event_name,
                        "tool_name": key.tool_name,
                        "source": key.source,
                        "outcome": outcome.value,
                    }
                )
        try:
            from opentelemetry import trace

            span = trace.get_current_span()
            if span is not None and getattr(span, "is_recording", lambda: False)():
                span.add_event(
                    event_name,
                    {
                        "tool_name": key.tool_name,
                        "source": key.source,
                        "outcome": outcome.value,
                    },
                )
        except Exception:
            return


__all__ = ["SourceHealthRail"]
