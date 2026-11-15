from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from mlof_v2.calendars import ExplicitSessionCalendar, OpenInterval
from mlof_v2.ingestion import SourceRow


UTC = timezone.utc
BASE = datetime(2024, 1, 1, tzinfo=UTC)


def stamp(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def raw_row(
    line: int,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    available: datetime | None = None,
    instrument: str = "TESTUSD",
    timeframe: str = "1m",
    open_price: str = "10.00",
    high: str = "10.20",
    low: str = "9.90",
    close: str = "10.10",
) -> SourceRow:
    start = start or BASE + timedelta(minutes=line - 2)
    end = end or start + timedelta(minutes=1)
    available = available or end
    fields = {
        "instrument": instrument,
        "timeframe": timeframe,
        "interval_start_utc": stamp(start),
        "interval_end_utc": stamp(end),
        "source_available_at_utc": stamp(available),
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
    }
    return SourceRow(line, tuple(fields.items()))


def calendar_for(start: datetime, end: datetime, *, calendar_id: str = "test-calendar") -> ExplicitSessionCalendar:
    return ExplicitSessionCalendar(calendar_id, start, end, (OpenInterval(start, end),))


def tick_spec():
    from mlof_v2.models import InstrumentSpec

    return (InstrumentSpec("TESTUSD", Decimal("0.01")),)
