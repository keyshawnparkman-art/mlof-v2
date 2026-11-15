"""Strict ISO-8601 parsing and deterministic UTC formatting."""

from __future__ import annotations

from datetime import datetime, timezone


UTC = timezone.utc


def parse_utc_timestamp(value: str) -> datetime:
    """Parse an ISO-8601 timestamp that explicitly carries a known UTC offset.

    Offset timestamps are converted to UTC. Naive timestamps and RFC3339's
    unknown-offset marker (-00:00) are rejected rather than guessed.
    """

    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("timestamp must be a non-empty ISO-8601 string without surrounding whitespace")
    if value.endswith("-00:00"):
        raise ValueError("timestamp uses -00:00, which means the source offset is unknown")
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError(f"invalid ISO-8601 timestamp: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include a timezone offset")
    return parsed.astimezone(UTC)


def format_utc_timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("cannot serialize a naive timestamp as UTC")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
