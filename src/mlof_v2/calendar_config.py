"""Loading explicit finite session-calendar descriptions from local JSON."""

from __future__ import annotations

import json
from pathlib import Path

from .calendars import ExplicitSessionCalendar, OpenInterval, UnknownSessionCalendar
from .timeframes import normalize_timeframe
from .timestamps import parse_utc_timestamp


def load_calendar(path: str | Path | None, expected_id: str | None):
    if path is None or expected_id is None:
        return UnknownSessionCalendar()
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read calendar configuration: {exc}") from exc
    if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data.get("schema_version") != 1 or data.get("calendar_id") != expected_id:
        raise ValueError("calendar schema_version or calendar_id does not match instrument configuration")
    coverage = data.get("coverage", {})
    open_rows = data.get("open_intervals", [])
    try:
        intervals = tuple(
            OpenInterval(parse_utc_timestamp(row["start_utc"]), parse_utc_timestamp(row["end_utc"]))
            for row in open_rows
        )
        return ExplicitSessionCalendar(
            expected_id,
            parse_utc_timestamp(coverage["start_utc"]),
            parse_utc_timestamp(coverage["end_utc"]),
            intervals,
            tuple((row["instrument"].upper(), normalize_timeframe(row["timeframe"]),
                   OpenInterval(parse_utc_timestamp(row["start"]), parse_utc_timestamp(row["end"])))
                  for row in data.get("candle_intervals", [])),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid explicit calendar configuration: {exc}") from exc
