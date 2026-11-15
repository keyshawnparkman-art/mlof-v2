"""Immutable data records and quality results for offline candle ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Mapping


class ReadinessState(str, Enum):
    NOT_READY = "NOT_READY"
    WARMING = "WARMING"
    READY = "READY"


@dataclass(frozen=True, slots=True)
class SourceProvenance:
    source_path: str
    source_sha256: str
    source_line: int
    source_row_sha256: str
    record_id: str
    raw_fields: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class Candle:
    instrument: str
    timeframe: str
    interval_start: datetime
    interval_end: datetime
    source_available_at: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    provenance: SourceProvenance

    def __post_init__(self) -> None:
        for timestamp_name in ("interval_start", "interval_end", "source_available_at"):
            value = getattr(self, timestamp_name)
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{timestamp_name} must be timezone-aware")
        if self.interval_end <= self.interval_start:
            raise ValueError("interval_end must be after interval_start")
        if self.source_available_at < self.interval_end:
            raise ValueError("source_available_at cannot precede candle close")
        prices = (self.open, self.high, self.low, self.close)
        if any(not value.is_finite() for value in prices):
            raise ValueError("OHLC values must be finite decimals")
        if self.low > self.high or not (self.low <= self.open <= self.high) or not (self.low <= self.close <= self.high):
            raise ValueError("invalid OHLC relationship")


@dataclass(frozen=True, slots=True)
class InstrumentSpec:
    instrument: str
    tick_size: Decimal

    def price_to_ticks(self, price: Decimal) -> int:
        if not self.tick_size.is_finite() or self.tick_size <= 0:
            raise ValueError("tick size must be a finite positive number")
        ticks = price / self.tick_size
        integral = ticks.to_integral_value()
        if ticks != integral:
            raise ValueError(f"price {price} is not representable at tick size {self.tick_size}")
        return int(integral)


@dataclass(frozen=True, slots=True)
class QualityIssue:
    code: str
    message: str
    source_line: int | None = None
    instrument: str | None = None
    timeframe: str | None = None
    interval_start: datetime | None = None
    record_id: str | None = None


@dataclass(frozen=True, slots=True)
class StreamReadiness:
    instrument: str
    timeframe: str
    state: ReadinessState
    candle_count: int


@dataclass(frozen=True, slots=True)
class IngestionResult:
    candles: tuple[Candle, ...]
    issues: tuple[QualityIssue, ...]
    missing_starts: tuple[tuple[str, str, datetime], ...]
    streams: tuple[StreamReadiness, ...]
    readiness: ReadinessState
    source_sha256: str
    source_path: str
    as_of_utc: datetime

    @property
    def accepted_count(self) -> int:
        return len(self.candles)

    @property
    def rejected_count(self) -> int:
        return len({issue.source_line for issue in self.issues if issue.source_line is not None})


def decimal_fields(candle: Candle) -> Mapping[str, str]:
    """Return canonical decimal text, without exponent notation."""

    return {
        name: format(value, "f")
        for name, value in (
            ("open", candle.open),
            ("high", candle.high),
            ("low", candle.low),
            ("close", candle.close),
        )
    }
