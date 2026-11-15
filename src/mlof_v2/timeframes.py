"""Supported candle durations for the data foundation."""

from __future__ import annotations

from datetime import timedelta


TIMEFRAME_SECONDS: dict[str, int] = {
    "15s": 15,
    "30s": 30,
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,

}


SESSION_TIMEFRAMES = frozenset({"4h", "1d"})

def normalize_timeframe(value: str) -> str:
    return "1d" if value.lower() == "daily" else value.lower()

def timeframe_delta(timeframe: str) -> timedelta:
    try:
        return timedelta(seconds=TIMEFRAME_SECONDS[normalize_timeframe(timeframe)])
    except KeyError as exc:
        raise ValueError(f"unsupported timeframe: {timeframe!r}") from exc
