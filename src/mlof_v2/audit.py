"""Stable serialization for normalized candles and validation audit reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TextIO

from .models import Candle, IngestionResult, InstrumentSpec
from .timestamps import format_utc_timestamp


def normalized_record(candle: Candle, instrument_spec: InstrumentSpec) -> dict[str, object]:
    ticks = {
        name: instrument_spec.price_to_ticks(value)
        for name, value in (
            ("open", candle.open),
            ("high", candle.high),
            ("low", candle.low),
            ("close", candle.close),
        )
    }
    canonical_prices = {
        name: format(value.quantize(instrument_spec.tick_size), "f")
        for name, value in (
            ("open", candle.open),
            ("high", candle.high),
            ("low", candle.low),
            ("close", candle.close),
        )
    }
    return {
        "record_id": candle.provenance.record_id,
        "instrument": candle.instrument,
        "timeframe": candle.timeframe,
        "interval_start_utc": format_utc_timestamp(candle.interval_start),
        "interval_end_utc": format_utc_timestamp(candle.interval_end),
        "source_available_at_utc": format_utc_timestamp(candle.source_available_at),
        **canonical_prices,
        "tick_size": format(instrument_spec.tick_size.normalize(), "f"),
        "open_ticks": ticks["open"],
        "high_ticks": ticks["high"],
        "low_ticks": ticks["low"],
        "close_ticks": ticks["close"],
        "source_path": candle.provenance.source_path,
        "source_sha256": candle.provenance.source_sha256,
        "source_line": candle.provenance.source_line,
        "source_row_sha256": candle.provenance.source_row_sha256,
        "raw_fields": dict(candle.provenance.raw_fields),
    }


def audit_dict(result: IngestionResult) -> dict[str, object]:
    return {
        "audit_schema_version": 1,
        "source_path": result.source_path,
        "source_sha256": result.source_sha256,
        "as_of_utc": format_utc_timestamp(result.as_of_utc),
        "readiness": result.readiness.value,
        "accepted_records": result.accepted_count,
        "rejected_source_rows": len({issue.source_line for issue in result.issues if issue.source_line is not None}),
        "missing_expected_intervals": [
            {"instrument": instrument, "timeframe": timeframe, "start_utc": format_utc_timestamp(start)}
            for instrument, timeframe, start in result.missing_starts
        ],
        "streams": [
            {
                "instrument": stream.instrument,
                "timeframe": stream.timeframe,
                "readiness": stream.state.value,
                "candle_count": stream.candle_count,
            }
            for stream in result.streams
        ],
        "issues": [
            {
                "code": issue.code,
                "message": issue.message,
                "source_line": issue.source_line,
                "instrument": issue.instrument,
                "timeframe": issue.timeframe,
                "interval_start_utc": format_utc_timestamp(issue.interval_start) if issue.interval_start else None,
                "record_id": issue.record_id,
            }
            for issue in result.issues
        ],
    }


def canonical_json(result: IngestionResult) -> str:
    return json.dumps(audit_dict(result), sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"


NORMALIZED_COLUMNS = (
    "record_id", "instrument", "timeframe", "interval_start_utc", "interval_end_utc",
    "source_available_at_utc", "open", "high", "low", "close", "tick_size",
    "open_ticks", "high_ticks", "low_ticks", "close_ticks", "source_path",
    "source_sha256", "source_line", "source_row_sha256", "raw_fields_json",
)


def normalized_row(candle: Candle, instrument_spec: InstrumentSpec) -> dict[str, object]:
    row = normalized_record(candle, instrument_spec)
    raw = row.pop("raw_fields")
    row["raw_fields_json"] = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return row


def write_normalized_csv(result: IngestionResult, instrument_specs: tuple[InstrumentSpec, ...], path: str | Path) -> None:
    spec_map = {spec.instrument: spec for spec in instrument_specs}
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=NORMALIZED_COLUMNS, lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        for candle in result.candles:
            writer.writerow(normalized_row(candle, spec_map[candle.instrument]))


def write_audit_json(result: IngestionResult, path: str | Path) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(canonical_json(result), encoding="utf-8", newline="\n")
