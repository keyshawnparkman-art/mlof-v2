"""Session-calendar interfaces and explicit, finite calendar coverage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Protocol

from .timeframes import SESSION_TIMEFRAMES, TIMEFRAME_SECONDS


UTC = timezone.utc


class SessionCalendar(Protocol):
    """Calendar lookup. Return None when the requested period is not covered."""

    calendar_id: str

    def expected_starts(
        self,
        instrument: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> tuple[datetime, ...] | None: ...


@dataclass(frozen=True, slots=True)
class OpenInterval:
    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("calendar interval timestamps must be timezone-aware")
        if self.start.astimezone(UTC) >= self.end.astimezone(UTC):
            raise ValueError("calendar open interval must have positive duration")


@dataclass(frozen=True, slots=True)
class ExplicitSessionCalendar:
    """Finite explicit UTC open intervals; suitable for tests or verified extracts.

    This intentionally does not infer broker sessions or holidays. If requested
    coverage is outside the declared interval, it returns None (NOT_READY).
    """

    calendar_id: str
    coverage_start: datetime
    coverage_end: datetime
    open_intervals: tuple[OpenInterval, ...]
    candle_intervals: tuple[tuple[str, str, OpenInterval], ...] = ()

    def __post_init__(self) -> None:
        if self.coverage_start.tzinfo is None or self.coverage_end.tzinfo is None:
            raise ValueError("calendar coverage must be timezone-aware")
        if self.coverage_start.astimezone(UTC) >= self.coverage_end.astimezone(UTC):
            raise ValueError("calendar coverage must have positive duration")
        ordered = tuple(sorted(self.open_intervals, key=lambda item: item.start.astimezone(UTC)))
        if ordered != self.open_intervals:
            raise ValueError("open intervals must be supplied in chronological order")
        for previous, current in zip(ordered, ordered[1:]):
            if previous.end.astimezone(UTC) > current.start.astimezone(UTC):
                raise ValueError("open intervals may not overlap")

        previous_by_stream = {}
        for instrument, timeframe, interval in self.candle_intervals:
            key = instrument, timeframe
            if (not instrument or instrument != instrument.upper() or timeframe not in SESSION_TIMEFRAMES
                    or interval.start < self.coverage_start or interval.end > self.coverage_end
                    or (key in previous_by_stream and interval.start < previous_by_stream[key])):
                raise ValueError("invalid, overlapping, or unordered explicit candle intervals")
            previous_by_stream[key] = interval.end

    def session_candles(self, instrument: str, timeframe: str, start: datetime, end: datetime):
        """Return declared UTC boundary pairs whose starts fall in the inclusive window."""
        if start < self.coverage_start or end > self.coverage_end:
            return None
        declared = [(row.start.astimezone(UTC), row.end.astimezone(UTC))
                    for name, tf, row in self.candle_intervals
                    if (name, tf) == (instrument.upper(), timeframe)]
        if not declared:
            return None
        return tuple((a,b) for a,b in declared if start <= a <= end)

    def expected_starts(
        self,
        instrument: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> tuple[datetime, ...] | None:
        if timeframe in SESSION_TIMEFRAMES:
            pairs = self.session_candles(instrument, timeframe, start, end)
            return None if pairs is None else tuple(a for a, _ in pairs)
        del instrument  # Explicit interval calendars are supplied per configured use.
        seconds = TIMEFRAME_SECONDS.get(timeframe.lower())
        if seconds is None:
            return None
        start_utc, end_utc = start.astimezone(UTC), end.astimezone(UTC)
        coverage_start, coverage_end = self.coverage_start.astimezone(UTC), self.coverage_end.astimezone(UTC)
        duration = timedelta(seconds=seconds)
        if start_utc < coverage_start or end_utc + duration > coverage_end:
            return None
        epoch = datetime(1970, 1, 1, tzinfo=UTC)
        def microseconds_since_epoch(value: datetime) -> int:
            elapsed = value - epoch
            return ((elapsed.days * 86400 + elapsed.seconds) * 1_000_000) + elapsed.microseconds

        step_us = seconds * 1_000_000
        start_us = microseconds_since_epoch(start_utc)
        end_us = microseconds_since_epoch(end_utc)
        first_index = (start_us + step_us - 1) // step_us
        last_index = end_us // step_us
        expected: list[datetime] = []
        sessions = tuple((x.start.astimezone(UTC), x.end.astimezone(UTC)) for x in self.open_intervals)
        for index in range(first_index, last_index + 1):
            bar_start = epoch + timedelta(seconds=index * seconds)
            bar_end = bar_start + duration
            if any(session_start <= bar_start and bar_end <= session_end for session_start, session_end in sessions):
                expected.append(bar_start)
        return tuple(expected)


@dataclass(frozen=True, slots=True)
class UnknownSessionCalendar:
    calendar_id: str = "unknown"

    def expected_starts(
        self,
        instrument: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> None:
        del instrument, timeframe, start, end
        return None


@dataclass(frozen=True, slots=True)
class CalendarRegistry:
    """Dispatch session queries to a separately verified calendar per instrument."""

    calendars: tuple[tuple[str, SessionCalendar], ...]
    calendar_id: str = "instrument-registry"

    def session_candles(self, instrument: str, timeframe: str, start: datetime, end: datetime):
        selected = next((calendar for name, calendar in self.calendars
                         if name.upper() == instrument.upper()), None)
        return session_candles(selected, instrument, timeframe, start, end)

    def __post_init__(self) -> None:
        names = [name.upper() for name, _ in self.calendars]
        if len(names) != len(set(names)):
            raise ValueError("calendar registry has duplicate instrument identifiers")

    def expected_starts(
        self,
        instrument: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> tuple[datetime, ...] | None:
        selected = dict((name.upper(), calendar) for name, calendar in self.calendars).get(instrument.upper())
        if selected is None:
            return None
        return selected.expected_starts(instrument, timeframe, start, end)


def session_candles(calendar, instrument: str, timeframe: str, start: datetime, end: datetime):
    query = getattr(calendar, "session_candles", None)
    return None if query is None else query(instrument, timeframe, start, end)
