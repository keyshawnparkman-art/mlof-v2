"""Strict local CSV parsing, validation, provenance, and readiness assessment."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Sequence

from .calendars import SessionCalendar, UnknownSessionCalendar
from .models import (
    Candle,
    IngestionResult,
    InstrumentSpec,
    QualityIssue,
    ReadinessState,
    SourceProvenance,
    StreamReadiness,
)
from .calendars import session_candles
from .timeframes import SESSION_TIMEFRAMES, TIMEFRAME_SECONDS, normalize_timeframe, timeframe_delta
from .timestamps import format_utc_timestamp, parse_utc_timestamp


UTC = timezone.utc
REQUIRED_COLUMNS = (
    "instrument",
    "timeframe",
    "interval_start_utc",
    "interval_end_utc",
    "source_available_at_utc",
    "open",
    "high",
    "low",
    "close",
)


@dataclass(frozen=True, slots=True)
class SourceRow:
    line_number: int
    fields: tuple[tuple[str, str], ...]
    malformed: bool = False


@dataclass(frozen=True, slots=True)
class ExpectedRange:
    start: datetime
    end: datetime


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_csv_source(path: str | Path) -> tuple[str, str, tuple[str, ...], tuple[SourceRow, ...], tuple[QualityIssue, ...]]:
    source_path = Path(path)
    try:
        raw_bytes = source_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read CSV {source_path}: {exc}") from exc
    source_hash = _sha256(raw_bytes)
    try:
        text = raw_bytes.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"CSV must be UTF-8 encoded: {exc}") from exc
    try:
        reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
        headers = tuple(reader.fieldnames or ())
        issues: list[QualityIssue] = []
        if not headers:
            issues.append(QualityIssue("MISSING_HEADER", "CSV has no header row"))
        if len(set(headers)) != len(headers):
            issues.append(QualityIssue("DUPLICATE_HEADER", "CSV contains duplicate column names"))
        missing = sorted(set(REQUIRED_COLUMNS) - set(headers))
        if missing:
            issues.append(QualityIssue("MISSING_REQUIRED_COLUMNS", f"Missing required columns: {', '.join(missing)}"))
        rows: list[SourceRow] = []
        for row in reader:
            line = reader.line_num
            malformed = None in row or any(value is None for value in row.values())
            fields: list[tuple[str, str]] = []
            for key, value in row.items():
                if key is None:
                    fields.append(("__extra_values__", "|".join(value or [])))
                else:
                    fields.append((key, "" if value is None else value))
            rows.append(SourceRow(line, tuple(fields), bool(malformed)))
    except csv.Error as exc:
        raise ValueError(f"malformed CSV structure: {exc}") from exc
    return source_hash, str(source_path.resolve()), headers, tuple(rows), tuple(issues)


def _canonical_row_hash(fields: tuple[tuple[str, str], ...]) -> str:
    data = json.dumps(sorted(fields), ensure_ascii=False, separators=(",", ":"))
    return _sha256(data.encode("utf-8"))


def _record_id(source_hash: str, line: int, row_hash: str) -> str:
    material = f"mlof-v2-m1|{source_hash}|{line}|{row_hash}".encode("utf-8")
    return _sha256(material)


def _issue_order(issue: QualityIssue) -> tuple[str, str, str, str, int, str]:
    return (
        issue.instrument or "",
        issue.timeframe or "",
        format_utc_timestamp(issue.interval_start) if issue.interval_start else "",
        issue.code,
        issue.source_line or 0,
        issue.record_id or "",
    )


def _parse_price(raw: str, field: str) -> Decimal:
    try:
        value = Decimal(raw)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{field} is not a valid decimal number") from exc
    if not value.is_finite():
        raise ValueError(f"{field} must be finite")
    return value


def ingest_csv(
    path: str | Path,
    *,
    instrument_specs: Sequence[InstrumentSpec],
    calendar: SessionCalendar | None,
    as_of_utc: datetime,
    min_ready_bars: int = 50,
    expected_range: ExpectedRange | None = None,
) -> IngestionResult:
    source_hash, source_path, headers, rows, header_issues = read_csv_source(path)
    return ingest_rows(
        rows,
        source_sha256=source_hash,
        source_path=source_path,
        instrument_specs=instrument_specs,
        calendar=calendar,
        as_of_utc=as_of_utc,
        min_ready_bars=min_ready_bars,
        expected_range=expected_range,
        initial_issues=header_issues,
    )


def ingest_rows(
    rows: Iterable[SourceRow],
    *,
    source_sha256: str,
    source_path: str,
    instrument_specs: Sequence[InstrumentSpec],
    calendar: SessionCalendar | None,
    as_of_utc: datetime,
    min_ready_bars: int = 50,
    expected_range: ExpectedRange | None = None,
    initial_issues: Sequence[QualityIssue] = (),
) -> IngestionResult:
    """Finalize the same deterministic result for batch or buffered ingestion.

    Rows retain their original source line. Incremental ingestion is buffered
    until finalization; it does not expose partial candles as signals.
    """

    if as_of_utc.tzinfo is None or as_of_utc.utcoffset() is None:
        raise ValueError("as_of_utc must be timezone-aware")
    if expected_range is not None:
        if expected_range.start.tzinfo is None or expected_range.end.tzinfo is None:
            raise ValueError("expected range timestamps must be timezone-aware")
        if expected_range.end.astimezone(UTC) < expected_range.start.astimezone(UTC):
            raise ValueError("expected range end must not precede its start")
    as_of = as_of_utc.astimezone(UTC)
    if min_ready_bars < 1:
        raise ValueError("min_ready_bars must be positive")
    specs = {spec.instrument.upper(): spec for spec in instrument_specs}
    issues: list[QualityIssue] = list(initial_issues)
    parsed: list[Candle] = []
    raw_intervals: list[tuple[str, str, datetime, datetime, int, str]] = []
    source_order = sorted(rows, key=lambda row: row.line_number)
    last_start_by_stream: dict[tuple[str, str], datetime] = {}

    for row in source_order:
        fields = dict(row.fields)
        row_hash = _canonical_row_hash(row.fields)
        rec_id = _record_id(source_sha256, row.line_number, row_hash)
        instrument = fields.get("instrument", "").strip().upper() or None
        timeframe = normalize_timeframe(fields.get("timeframe", "").strip()) or None

        def add(code: str, message: str, start: datetime | None = None) -> None:
            issues.append(QualityIssue(code, message, row.line_number, instrument, timeframe, start, rec_id))

        if row.malformed:
            add("MALFORMED_COLUMN_COUNT", "Row has too many or too few columns")
            continue
        missing_fields = [name for name in REQUIRED_COLUMNS if not fields.get(name, "").strip()]
        if missing_fields:
            add("MISSING_REQUIRED_VALUE", f"Missing required values: {', '.join(missing_fields)}")
            continue
        if instrument not in specs:
            add("UNKNOWN_INSTRUMENT", f"No tick-size metadata configured for {instrument}")
            continue
        if timeframe not in TIMEFRAME_SECONDS:
            add("UNSUPPORTED_TIMEFRAME", f"Unsupported timeframe {timeframe!r}")
            continue
        try:
            start = parse_utc_timestamp(fields["interval_start_utc"])
            end = parse_utc_timestamp(fields["interval_end_utc"])
            available = parse_utc_timestamp(fields["source_available_at_utc"])
        except ValueError as exc:
            add("INVALID_TIMESTAMP", str(exc))
            continue
        stream = (instrument, timeframe)
        previous = last_start_by_stream.get(stream)
        if previous is not None and start < previous:
            add("OUT_OF_ORDER_RECORD", "Source rows are not chronological within this instrument/timeframe", start)
        else:
            last_start_by_stream[stream] = start
        if end > start:
            raw_intervals.append((instrument, timeframe, start, end, row.line_number, rec_id))
        if timeframe in SESSION_TIMEFRAMES:
            pairs = session_candles(calendar, instrument, timeframe, start, start)
            if pairs is None or (start, end) not in pairs:
                add("INVALID_SESSION_CANDLE", "No exact session candle boundaries are declared", start)
                continue
        else:
            duration = timeframe_delta(timeframe)
            if end - start != duration:
                add("INVALID_INTERVAL_DURATION", f"Interval must be exactly {timeframe} ({duration})", start)
                continue
            epoch = datetime(1970, 1, 1, tzinfo=UTC)
            elapsed = start - epoch
            elapsed_microseconds = ((elapsed.days * 86400 + elapsed.seconds) * 1_000_000) + elapsed.microseconds
            timeframe_microseconds = TIMEFRAME_SECONDS[timeframe] * 1_000_000
            if elapsed_microseconds % timeframe_microseconds != 0:
                add("UNALIGNED_INTERVAL", "Interval start is not aligned to the UTC timeframe grid", start)
                continue
        if end <= start:
            add("INVALID_INTERVAL", "Interval end must be after interval start", start)
            continue
        if available < end:
            add("AVAILABLE_BEFORE_CLOSE", "Source availability precedes candle close", start)
            continue
        if end > as_of:
            add("PARTIAL_CANDLE", "Candle has not closed as of the supplied reference time", start)
            continue
        if available > as_of:
            add("NOT_YET_AVAILABLE", "Source record was not available as of the supplied reference time", start)
            continue
        try:
            prices = {name: _parse_price(fields[name], name) for name in ("open", "high", "low", "close")}
        except ValueError as exc:
            add("INVALID_PRICE", str(exc), start)
            continue
        low, high = prices["low"], prices["high"]
        if low > high or not (low <= prices["open"] <= high) or not (low <= prices["close"] <= high):
            add("INVALID_OHLC", "Require low <= open/close <= high and low <= high", start)
            continue
        spec = specs[instrument]
        try:
            for value in prices.values():
                spec.price_to_ticks(value)
        except ValueError as exc:
            add("PRICE_NOT_ALIGNED_TO_TICK", str(exc), start)
            continue
        provenance = SourceProvenance(
            source_path=source_path,
            source_sha256=source_sha256,
            source_line=row.line_number,
            source_row_sha256=row_hash,
            record_id=rec_id,
            raw_fields=tuple(sorted(row.fields)),
        )
        parsed.append(Candle(instrument, timeframe, start, end, available, prices["open"], high, low, prices["close"], provenance))

    # Inspect all parseable source intervals, including rows rejected later for
    # alignment or OHLC. Same-start cases are reported as duplicate timestamps.
    raw_by_stream: dict[tuple[str, str], list[tuple[datetime, datetime, int, str]]] = defaultdict(list)
    for instrument, timeframe, start, end, line, rec_id in raw_intervals:
        raw_by_stream[(instrument, timeframe)].append((start, end, line, rec_id))
    for (instrument, timeframe), intervals in sorted(raw_by_stream.items()):
        intervals.sort(key=lambda item: (item[0], item[2], item[3]))
        furthest_end: datetime | None = None
        furthest_owner: tuple[datetime, datetime, int, str] | None = None
        for current in intervals:
            start, end, line, rec_id = current
            if furthest_end is not None and start < furthest_end and furthest_owner is not None and start != furthest_owner[0]:
                issues.append(QualityIssue(
                    "OVERLAPPING_INTERVALS", "Interval overlaps another source interval in the same stream",
                    line, instrument, timeframe, start, rec_id,
                ))
                issues.append(QualityIssue(
                    "OVERLAPPING_INTERVALS", "Interval overlaps another source interval in the same stream",
                    furthest_owner[2], instrument, timeframe, furthest_owner[0], furthest_owner[3],
                ))
            if furthest_end is None or end > furthest_end:
                furthest_end = end
                furthest_owner = current

    # Equal starts are ambiguous even when their OHLC values happen to match.
    groups: dict[tuple[str, str, datetime], list[Candle]] = defaultdict(list)
    for candle in parsed:
        groups[(candle.instrument, candle.timeframe, candle.interval_start)].append(candle)
    duplicate_keys = {key for key, group in groups.items() if len(group) > 1}
    for instrument, timeframe, start in sorted(duplicate_keys):
        for conflicting in sorted(groups[(instrument, timeframe, start)], key=lambda c: c.provenance.source_line):
            issues.append(QualityIssue(
                "DUPLICATE_TIMESTAMP",
                "Multiple source rows have the same instrument/timeframe/interval start; all conflicting rows are excluded",
                source_line=conflicting.provenance.source_line,
                instrument=instrument,
                timeframe=timeframe,
                interval_start=start,
                record_id=conflicting.provenance.record_id,
            ))
    unique = [group[0] for key, group in groups.items() if key not in duplicate_keys]
    unique.sort(key=lambda candle: (candle.instrument, candle.timeframe, candle.interval_start, candle.provenance.record_id))

    previous_by_stream: dict[tuple[str, str], Candle] = {}
    for candle in unique:
        key = (candle.instrument, candle.timeframe)
        previous = previous_by_stream.get(key)
        if previous is not None and previous.interval_end > candle.interval_start:
            issues.append(QualityIssue(
                "OVERLAPPING_INTERVALS",
                "Candle overlaps the preceding candle in the same stream",
                candle.provenance.source_line,
                candle.instrument,
                candle.timeframe,
                candle.interval_start,
                candle.provenance.record_id,
            ))
            issues.append(QualityIssue(
                "OVERLAPPING_INTERVALS",
                "Candle overlaps the following candle in the same stream",
                previous.provenance.source_line,
                previous.instrument,
                previous.timeframe,
                previous.interval_start,
                previous.provenance.record_id,
            ))
        previous_by_stream[key] = candle

    missing_starts: list[tuple[str, str, datetime]] = []
    by_stream: dict[tuple[str, str], list[Candle]] = defaultdict(list)
    for candle in unique:
        by_stream[(candle.instrument, candle.timeframe)].append(candle)
    calendar_source = calendar or UnknownSessionCalendar()
    for (instrument, timeframe), stream_candles in sorted(by_stream.items()):
        stream_candles.sort(key=lambda candle: candle.interval_start)
        requested_start = expected_range.start.astimezone(UTC) if expected_range else stream_candles[0].interval_start
        requested_end = expected_range.end.astimezone(UTC) if expected_range else stream_candles[-1].interval_start
        if timeframe in SESSION_TIMEFRAMES:
            if expected_range is None:
                requested_end = as_of
            pairs = session_candles(calendar_source, instrument, timeframe, requested_start, requested_end)
            expected = None if pairs is None else tuple(a for a, b in pairs if b <= as_of)
        else:
            expected = calendar_source.expected_starts(instrument, timeframe, requested_start, requested_end)
        if expected is None:
            issues.append(QualityIssue(
                "CALENDAR_COVERAGE_UNKNOWN",
                f"Calendar {calendar_source.calendar_id!r} does not establish expected intervals for this range",
                instrument=instrument,
                timeframe=timeframe,
            ))
            continue
        observed = {c.interval_start for c in stream_candles}
        absent = sorted(set(expected) - observed)
        for start in absent:
            missing_starts.append((instrument, timeframe, start))
            issues.append(QualityIssue(
                "MISSING_EXPECTED_INTERVAL",
                "Expected candle interval is absent; no candle was fabricated",
                instrument=instrument,
                timeframe=timeframe,
                interval_start=start,
            ))

    # Records involved in an overlap remain in the audit result, but the entire
    # dataset is NOT_READY, so downstream consumers must not treat them as usable.
    hard_failure = bool(issues) or not unique
    streams: list[StreamReadiness] = []
    for (instrument, timeframe), stream_candles in sorted(by_stream.items()):
        stream_has_issue = any(
            issue.instrument is None
            or (issue.instrument == instrument and (issue.timeframe is None or issue.timeframe == timeframe))
            for issue in issues
        )
        if stream_has_issue:
            state = ReadinessState.NOT_READY
        elif len(stream_candles) < min_ready_bars:
            state = ReadinessState.WARMING
        else:
            state = ReadinessState.READY
        streams.append(StreamReadiness(instrument, timeframe, state, len(stream_candles)))
    if hard_failure:
        readiness = ReadinessState.NOT_READY
    elif streams and all(stream.state is ReadinessState.READY for stream in streams):
        readiness = ReadinessState.READY
    elif streams:
        readiness = ReadinessState.WARMING
    else:
        readiness = ReadinessState.NOT_READY

    ordered_issues = tuple(sorted(issues, key=_issue_order))
    return IngestionResult(
        candles=tuple(unique),
        issues=ordered_issues,
        missing_starts=tuple(sorted(missing_starts)),
        streams=tuple(streams),
        readiness=readiness,
        source_sha256=source_sha256,
        source_path=source_path,
        as_of_utc=as_of,
    )


class BufferedIngestor:
    """Offline incremental collector; output is finalized only at close/finalize."""

    def __init__(self) -> None:
        self._rows: list[SourceRow] = []

    def add(self, row: SourceRow) -> None:
        self._rows.append(row)

    def finalize(self, **kwargs: object) -> IngestionResult:
        return ingest_rows(tuple(self._rows), **kwargs)  # type: ignore[arg-type]


def issue_counts(result: IngestionResult) -> dict[str, int]:
    counts = Counter(issue.code for issue in result.issues)
    return dict(sorted(counts.items()))
