"""Stable JSON serialization for structural event audit output."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

from .models import StructuralEvent


def _canonical(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, tuple):
        return [_canonical(item) for item in value]
    if isinstance(value, dict):
        return {key: _canonical(value[key]) for key in sorted(value)}
    return value


def structural_events_jsonl(events: tuple[StructuralEvent, ...]) -> str:
    """Serialize events in their supplied order with canonical JSON fields."""

    lines: list[str] = []
    for event in events:
        row = {
            "event_id": event.event_id,
            "kind": event.kind.value,
            "instrument": event.instrument,
            "timeframe": event.timeframe,
            "available_at": _canonical(event.available_at),
            "source_candle_ids": list(event.source_candle_ids),
            "related_ids": list(event.related_ids),
            "reason": event.reason,
            "config_version": event.config_version,
            "side": event.side.value if event.side else None,
            "level_ticks": event.level_ticks,
            "details": [[key, value] for key, value in event.details],
            "provenance": [
                {
                    "source_path": item.source_path,
                    "source_sha256": item.source_sha256,
                    "source_line": item.source_line,
                    "source_row_sha256": item.source_row_sha256,
                    "record_id": item.record_id,
                    "raw_fields": [list(pair) for pair in item.raw_fields],
                }
                for item in event.provenance
            ],
            "candle_evidence": [
                {
                    "record_id": item.record_id,
                    "interval_start": _canonical(item.interval_start),
                    "interval_end": _canonical(item.interval_end),
                    "source_available_at": _canonical(item.source_available_at),
                    "provenance_record_id": item.provenance.record_id,
                    "source_path": item.provenance.source_path,
                    "source_sha256": item.provenance.source_sha256,
                    "source_line": item.provenance.source_line,
                    "source_row_sha256": item.provenance.source_row_sha256,
                }
                for item in event.candle_evidence
            ],
        }
        lines.append(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return "\n".join(lines) + ("\n" if lines else "")


def bootstrap_replay_jsonl(result) -> str:
    """Serialize bootstrap evidence with deterministic field and timestamp order."""

    groups = (
        ("event", result.events),
        ("observation_pair", result.pairs),
        ("bootstrap_manipulation", result.manipulations),
        ("bootstrap_exchange", result.exchanges),
        ("frozen_control_range", result.controls),
    )
    rows = []
    for kind, values in groups:
        for value in values:
            row = {"record_type": kind, **_canonical(asdict(value))}
            rows.append(json.dumps(row, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=False, allow_nan=False))
    return "\n".join(rows) + ("\n" if rows else "")
