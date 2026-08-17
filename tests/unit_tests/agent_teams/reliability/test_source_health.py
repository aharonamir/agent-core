# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

from types import SimpleNamespace

import pytest

from openjiuwen.agent_teams.reliability.source_health import (
    SourceHealthKey,
    SourceHealthOutcome,
    SourceHealthProvenance,
    SourceHealthStore,
)
from openjiuwen.agent_teams.reliability.source_health_rail import SourceHealthRail
from openjiuwen.core.foundation.llm.schema.tool_call import ToolCall
from openjiuwen.core.single_agent.rail.base import AgentCallbackContext, ToolCallInputs


def _ctx(tool_name: str, tool_args, tool_result=None, call_id: str = "tc1") -> AgentCallbackContext:
    return AgentCallbackContext(
        agent=None,
        inputs=ToolCallInputs(
            tool_call=ToolCall(id=call_id, type="function", name=tool_name, arguments="{}"),
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=tool_result,
        ),
    )


@pytest.mark.asyncio
async def test_empty_success_result_records_empty_not_ok() -> None:
    store = SourceHealthStore()
    rail = SourceHealthRail(store)

    await rail.after_tool_call(_ctx("ddg-search", {"query": "barcelona hotels"}, "No results found."))

    entries = store.entries()
    assert len(entries) == 1
    assert entries[0][1].outcome == SourceHealthOutcome.EMPTY
    assert entries[0][1].provenance == SourceHealthProvenance.BODY


@pytest.mark.asyncio
async def test_registered_domains_collapse_subdomains() -> None:
    store = SourceHealthStore()
    rail = SourceHealthRail(store)

    await rail.after_tool_call(_ctx("fetch_webpage", {"url": "https://html.duckduckgo.com"}, "Status: 403"))
    await rail.after_tool_call(_ctx("fetch_webpage", {"url": "https://lite.duckduckgo.com"}, "Status: 403", "tc2"))

    entries = store.entries()
    assert [key for key, _entry in entries] == [SourceHealthKey("fetch_webpage", "duckduckgo.com")]
    assert entries[0][1].fail_count == 2


@pytest.mark.asyncio
async def test_retry_storm_counts_one_logical_call() -> None:
    store = SourceHealthStore()
    rail = SourceHealthRail(store)

    for _ in range(3):
        await rail.after_tool_call(_ctx("fetch_webpage", {"url": "https://example.com"}, "Status: 403", "tc1"))

    entry = store.get(SourceHealthKey("fetch_webpage", "example.com"))
    assert entry is not None
    assert entry.fail_count == 1


@pytest.mark.asyncio
async def test_body_only_threshold_does_not_hard_block() -> None:
    store = SourceHealthStore()
    key = SourceHealthKey("fetch_webpage", "trip.com")
    store.record(key, outcome=SourceHealthOutcome.GARBAGE, provenance=SourceHealthProvenance.BODY)
    store.record(key, outcome=SourceHealthOutcome.GARBAGE, provenance=SourceHealthProvenance.BODY)
    rail = SourceHealthRail(store)
    ctx = _ctx("fetch_webpage", {"url": "https://trip.com/hotels"})

    await rail.before_tool_call(ctx)

    assert ctx.extra.get("_skip_tool") is not True


@pytest.mark.asyncio
async def test_transport_backed_threshold_hard_blocks() -> None:
    store = SourceHealthStore()
    key = SourceHealthKey("fetch_webpage", "duckduckgo.com")
    store.record(key, outcome=SourceHealthOutcome.HTTP_403, provenance=SourceHealthProvenance.TRANSPORT)
    store.record(key, outcome=SourceHealthOutcome.HTTP_403, provenance=SourceHealthProvenance.TRANSPORT)
    rail = SourceHealthRail(store)
    ctx = _ctx("fetch_webpage", {"url": "https://lite.duckduckgo.com"})

    await rail.before_tool_call(ctx)

    assert ctx.extra["_skip_tool"] is True
    assert ctx.inputs.tool_result["source_health"] == SourceHealthOutcome.HTTP_403.value
    assert "duckduckgo.com" in ctx.inputs.tool_msg.content


@pytest.mark.asyncio
async def test_recheck_window_allows_transient_source_after_turn() -> None:
    store = SourceHealthStore()
    key = SourceHealthKey("fetch_webpage", "example.com")
    store.record(
        key,
        outcome=SourceHealthOutcome.TRANSPORT,
        provenance=SourceHealthProvenance.TRANSPORT,
        turn=0,
        recheck_after_turn=5,
    )
    store.record(
        key,
        outcome=SourceHealthOutcome.TRANSPORT,
        provenance=SourceHealthProvenance.TRANSPORT,
        turn=0,
        recheck_after_turn=5,
    )
    rail = SourceHealthRail(store)
    session = SimpleNamespace(get_state=lambda key: 5 if key == "source_health_turn" else None)
    ctx = _ctx("fetch_webpage", {"url": "https://example.com"})
    ctx.session = session

    await rail.before_tool_call(ctx)

    assert ctx.extra.get("_skip_tool") is not True


@pytest.mark.asyncio
async def test_after_invoke_adds_candidate_report_for_threshold_hits() -> None:
    store = SourceHealthStore()
    key = SourceHealthKey("fetch_webpage", "duckduckgo.com")
    store.record(key, outcome=SourceHealthOutcome.HTTP_403, provenance=SourceHealthProvenance.TRANSPORT)
    store.record(key, outcome=SourceHealthOutcome.HTTP_403, provenance=SourceHealthProvenance.TRANSPORT)
    rail = SourceHealthRail(store)
    ctx = AgentCallbackContext(agent=None, inputs=SimpleNamespace(result={"output": "done"}))

    await rail.after_invoke(ctx)

    assert ctx.inputs.result["source_health_candidates"] == [
        {
            "tool_name": "fetch_webpage",
            "source": "duckduckgo.com",
            "outcome": "http_403",
            "provenance": "transport",
            "fail_count": 2,
            "first_seen_turn": 0,
            "last_seen_turn": 0,
        }
    ]
