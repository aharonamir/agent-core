# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Run-scoped source-health state shared by a DeepAgent and its subagents."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from threading import RLock


class SourceHealthOutcome(StrEnum):
    """Source-health outcome classes recorded by rails/tools."""

    OK = "ok"
    EMPTY = "empty"
    HTTP_403 = "http_403"
    HTTP_429 = "http_429"
    BLOCKED = "blocked"
    DNS_FAIL = "dns_fail"
    TRANSPORT = "transport"
    LOW_SIGNAL = "low_signal"
    GARBAGE = "garbage"


class SourceHealthProvenance(StrEnum):
    """How the health observation was derived."""

    TRANSPORT = "transport"
    BODY = "body"


@dataclass(frozen=True)
class SourceHealthKey:
    """Normalized source-health bucket key."""

    tool_name: str
    source: str


@dataclass
class SourceHealthEntry:
    """Aggregated source-health state for one bucket."""

    outcome: SourceHealthOutcome
    provenance: SourceHealthProvenance
    latency_ms: int | None = None
    first_seen_turn: int = 0
    last_seen_turn: int = 0
    fail_count: int = 0
    recheck_after_turn: int | None = None
    has_transport_observation: bool = False


class SourceHealthStore:
    """Thread-safe run-scoped source-health registry.

    A single ``DeepAgent`` owns one store and TaskTool-spawned subagents share
    that object. Parallel workers can record and consult it concurrently, so the
    implementation uses an ``RLock`` around a small mutable map instead of
    relying on session-local state.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._entries: dict[SourceHealthKey, SourceHealthEntry] = {}
        self._logical_call_keys: dict[str, SourceHealthKey] = {}

    def record(
        self,
        key: SourceHealthKey,
        *,
        outcome: SourceHealthOutcome,
        provenance: SourceHealthProvenance,
        turn: int = 0,
        latency_ms: int | None = None,
        logical_call_id: str | None = None,
        recheck_after_turn: int | None = None,
    ) -> SourceHealthEntry:
        """Record one logical health observation.

        ``logical_call_id`` deduplicates retry storms: repeated writes for the
        same logical call replace the prior key association and count as one
        observation in the final bucket.
        """
        with self._lock:
            if logical_call_id:
                prior_key = self._logical_call_keys.get(logical_call_id)
                if prior_key == key:
                    entry = self._entries.get(key)
                    if entry is not None:
                        entry.outcome = outcome
                        entry.provenance = provenance
                        entry.latency_ms = latency_ms
                        entry.last_seen_turn = turn
                        entry.recheck_after_turn = recheck_after_turn
                        entry.has_transport_observation = (
                            entry.has_transport_observation
                            or provenance == SourceHealthProvenance.TRANSPORT
                        )
                        return entry
                self._logical_call_keys[logical_call_id] = key

            entry = self._entries.get(key)
            if entry is None:
                entry = SourceHealthEntry(
                    outcome=outcome,
                    provenance=provenance,
                    latency_ms=latency_ms,
                    first_seen_turn=turn,
                    last_seen_turn=turn,
                    fail_count=0,
                    recheck_after_turn=recheck_after_turn,
                    has_transport_observation=provenance == SourceHealthProvenance.TRANSPORT,
                )
                self._entries[key] = entry
            else:
                entry.outcome = outcome
                entry.provenance = provenance
                entry.latency_ms = latency_ms
                entry.last_seen_turn = turn
                entry.recheck_after_turn = recheck_after_turn
                entry.has_transport_observation = (
                    entry.has_transport_observation
                    or provenance == SourceHealthProvenance.TRANSPORT
                )

            if outcome == SourceHealthOutcome.OK:
                entry.fail_count = 0
            else:
                entry.fail_count += 1
            return entry

    def get(self, key: SourceHealthKey) -> SourceHealthEntry | None:
        """Return a snapshot-like reference for a key, or None."""
        with self._lock:
            return self._entries.get(key)

    def threshold_hits(self) -> list[tuple[SourceHealthKey, SourceHealthEntry]]:
        """Return entries that reached their per-outcome threshold."""
        with self._lock:
            return [
                (key, entry)
                for key, entry in self._entries.items()
                if is_threshold_hit(entry)
            ]

    def entries(self) -> list[tuple[SourceHealthKey, SourceHealthEntry]]:
        """Return all current entries for reporting/tests."""
        with self._lock:
            return list(self._entries.items())

    def clear(self) -> None:
        """Clear all run-scoped source-health state."""
        with self._lock:
            self._entries.clear()
            self._logical_call_keys.clear()


_THRESHOLDS: dict[SourceHealthOutcome, int] = {
    SourceHealthOutcome.HTTP_403: 2,
    SourceHealthOutcome.BLOCKED: 2,
    SourceHealthOutcome.HTTP_429: 2,
    SourceHealthOutcome.EMPTY: 3,
    SourceHealthOutcome.LOW_SIGNAL: 2,
    SourceHealthOutcome.GARBAGE: 2,
    SourceHealthOutcome.DNS_FAIL: 2,
    SourceHealthOutcome.TRANSPORT: 2,
}

_RECHECK_WINDOWS: dict[SourceHealthOutcome, int] = {
    SourceHealthOutcome.HTTP_429: 5,
    SourceHealthOutcome.DNS_FAIL: 5,
    SourceHealthOutcome.TRANSPORT: 5,
}


def threshold_for(outcome: SourceHealthOutcome) -> int:
    """Return the configured threshold for an outcome."""
    return _THRESHOLDS.get(outcome, 1)


def recheck_window_for(outcome: SourceHealthOutcome) -> int | None:
    """Return the within-run recheck window for transient outcomes."""
    return _RECHECK_WINDOWS.get(outcome)


def is_threshold_hit(entry: SourceHealthEntry) -> bool:
    """Return whether an entry has crossed its current outcome threshold."""
    return entry.outcome != SourceHealthOutcome.OK and entry.fail_count >= threshold_for(entry.outcome)


__all__ = [
    "SourceHealthEntry",
    "SourceHealthKey",
    "SourceHealthOutcome",
    "SourceHealthProvenance",
    "SourceHealthStore",
    "is_threshold_hit",
    "recheck_window_for",
    "threshold_for",
]
