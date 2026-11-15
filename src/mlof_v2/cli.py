"""Local-only CSV validation command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .audit import canonical_json, write_audit_json, write_normalized_csv
from .calendar_config import load_calendar
from .config import load_config
from .ingestion import ExpectedRange, ingest_csv
from .models import ReadinessState
from .timestamps import parse_utc_timestamp


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlof-v2", description="Offline MLOF v2 data validation; no strategy or trading functions.")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-csv", help="validate and normalize a local OHLC CSV")
    validate.add_argument("--csv", required=True, help="input local CSV; source is read only")
    validate.add_argument("--config", required=True, help="versioned instrument configuration JSON")
    validate.add_argument("--calendar", help="optional finite, explicit session calendar JSON")
    validate.add_argument("--as-of-utc", required=True, help="explicit ISO-8601 timestamp used for complete-candle checks")
    validate.add_argument("--min-ready-bars", type=int, help="override configured data-readiness bar count")
    validate.add_argument("--expected-start-utc", help="optional expected range start for leading-gap checks")
    validate.add_argument("--expected-end-utc", help="optional expected range end candle start for trailing-gap checks")
    validate.add_argument("--normalized-out", help="write normalized records to this new or explicitly selected CSV path")
    validate.add_argument("--audit-out", help="write deterministic audit JSON to this new or explicitly selected path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config)
        calendar = load_calendar(args.calendar, config.calendar_id)
        expected_range = None
        if bool(args.expected_start_utc) != bool(args.expected_end_utc):
            raise ValueError("expected-start and expected-end must be supplied together")
        if args.expected_start_utc:
            expected_range = ExpectedRange(
                parse_utc_timestamp(args.expected_start_utc),
                parse_utc_timestamp(args.expected_end_utc),
            )
            if expected_range.end < expected_range.start:
                raise ValueError("expected-end must not precede expected-start")
        minimum = config.min_ready_bars if args.min_ready_bars is None else args.min_ready_bars
        result = ingest_csv(
            args.csv,
            instrument_specs=config.instruments,
            calendar=calendar,
            as_of_utc=parse_utc_timestamp(args.as_of_utc),
            min_ready_bars=minimum,
            expected_range=expected_range,
        )
        project_root = Path(__file__).resolve().parents[2]
        for output in (args.normalized_out, args.audit_out):
            if output:
                destination = Path(output).resolve()
                if project_root not in destination.parents:
                    raise ValueError(f"outputs must remain inside the MLOF V2 workspace: {destination}")
                if destination.exists():
                    raise FileExistsError(f"refusing to overwrite existing output: {destination}")
        input_path = Path(args.csv).resolve()
        if any(output and Path(output).resolve() == input_path for output in (args.normalized_out, args.audit_out)):
            raise ValueError("output paths may not overwrite the input CSV")
        if args.normalized_out:
            write_normalized_csv(result, config.instruments, args.normalized_out)
        if args.audit_out:
            write_audit_json(result, args.audit_out)
        print(canonical_json(result), end="")
        return 0 if result.readiness is not ReadinessState.NOT_READY else 2
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
