from __future__ import annotations

import tempfile
import unittest
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

from mlof_v2.audit import canonical_json, normalized_record
from mlof_v2.calendars import UnknownSessionCalendar
from mlof_v2.ingestion import BufferedIngestor, ExpectedRange, REQUIRED_COLUMNS, SourceRow, ingest_csv, ingest_rows
from mlof_v2.models import InstrumentSpec, ReadinessState

from helpers import BASE, calendar_for, raw_row, tick_spec


AS_OF = BASE + timedelta(hours=2)
SOURCE_HASH = "a" * 64
SOURCE_PATH = "synthetic.csv"


class IngestionTests(unittest.TestCase):
    def ingest(self, rows, *, calendar=None, minimum=50, expected_range=None):
        return ingest_rows(
            rows, source_sha256=SOURCE_HASH, source_path=SOURCE_PATH,
            instrument_specs=tick_spec(), calendar=calendar, as_of_utc=AS_OF,
            min_ready_bars=minimum, expected_range=expected_range,
        )

    def test_valid_short_file_is_warming_and_has_provenance(self) -> None:
        result = self.ingest([raw_row(2), raw_row(3)], calendar=calendar_for(BASE, BASE + timedelta(minutes=2)))
        self.assertEqual(result.readiness, ReadinessState.WARMING)
        self.assertEqual(result.accepted_count, 2)
        record = normalized_record(result.candles[0], tick_spec()[0])
        self.assertEqual(record["source_sha256"], SOURCE_HASH)
        self.assertEqual(record["source_line"], 2)
        self.assertEqual(record["open_ticks"], 1000)

    def test_normalized_decimal_format_uses_tick_precision(self) -> None:
        source = raw_row(2, start=BASE, open_price="10.0")
        result = self.ingest([source], calendar=calendar_for(BASE, BASE + timedelta(minutes=1)))
        record = normalized_record(result.candles[0], tick_spec()[0])
        self.assertEqual(record["open"], "10.00")

    def test_missing_expected_interval_is_not_fabricated(self) -> None:
        rows = [raw_row(2, start=BASE), raw_row(3, start=BASE + timedelta(minutes=2))]
        result = self.ingest(rows, calendar=calendar_for(BASE, BASE + timedelta(minutes=3)))
        self.assertEqual(result.readiness, ReadinessState.NOT_READY)
        self.assertEqual(result.accepted_count, 2)
        self.assertEqual(result.missing_starts, (("TESTUSD", "1m", BASE + timedelta(minutes=1)),))
        self.assertNotIn(BASE + timedelta(minutes=1), [c.interval_start for c in result.candles])

    def test_stream_readiness_identifies_only_the_affected_instrument(self) -> None:
        rows = [
            raw_row(2, start=BASE, instrument="TESTUSD"),
            raw_row(3, start=BASE + timedelta(minutes=2), instrument="TESTUSD"),
            raw_row(4, start=BASE, instrument="OTHER"),
            raw_row(5, start=BASE + timedelta(minutes=1), instrument="OTHER"),
        ]
        result = ingest_rows(
            rows, source_sha256=SOURCE_HASH, source_path=SOURCE_PATH,
            instrument_specs=(InstrumentSpec("TESTUSD", Decimal("0.01")), InstrumentSpec("OTHER", Decimal("0.01"))),
            calendar=calendar_for(BASE, BASE + timedelta(minutes=3)), as_of_utc=AS_OF, min_ready_bars=2,
        )
        states = {(item.instrument, item.timeframe): item.state for item in result.streams}
        self.assertEqual(states[("TESTUSD", "1m")], ReadinessState.NOT_READY)
        self.assertEqual(states[("OTHER", "1m")], ReadinessState.READY)
        self.assertEqual(result.readiness, ReadinessState.NOT_READY)

    def test_explicit_session_closure_is_not_a_missing_candle(self) -> None:
        from mlof_v2.calendars import ExplicitSessionCalendar, OpenInterval

        rows = [
            raw_row(2, start=BASE),
            raw_row(3, start=BASE + timedelta(minutes=1)),
            raw_row(4, start=BASE + timedelta(minutes=5)),
        ]
        calendar = ExplicitSessionCalendar(
            "two-session-test", BASE, BASE + timedelta(minutes=7),
            (OpenInterval(BASE, BASE + timedelta(minutes=2)),
             OpenInterval(BASE + timedelta(minutes=5), BASE + timedelta(minutes=7))),
        )
        result = self.ingest(rows, calendar=calendar)
        self.assertNotIn("MISSING_EXPECTED_INTERVAL", [issue.code for issue in result.issues])
        self.assertEqual(result.readiness, ReadinessState.WARMING)

    def test_unknown_calendar_fails_closed(self) -> None:
        result = self.ingest([raw_row(2)], calendar=UnknownSessionCalendar())
        self.assertEqual(result.readiness, ReadinessState.NOT_READY)
        self.assertIn("CALENDAR_COVERAGE_UNKNOWN", [issue.code for issue in result.issues])

    def test_duplicate_equal_timestamp_excludes_all_conflicting_rows(self) -> None:
        result = self.ingest(
            [raw_row(2, start=BASE), raw_row(3, start=BASE)],
            calendar=calendar_for(BASE, BASE + timedelta(minutes=1)),
        )
        self.assertEqual(result.readiness, ReadinessState.NOT_READY)
        self.assertEqual(result.accepted_count, 0)
        self.assertIn("DUPLICATE_TIMESTAMP", [issue.code for issue in result.issues])

    def test_out_of_order_source_rows_are_flagged_and_output_is_canonical(self) -> None:
        rows = [raw_row(2, start=BASE + timedelta(minutes=1)), raw_row(3, start=BASE)]
        result = self.ingest(rows, calendar=calendar_for(BASE, BASE + timedelta(minutes=2)))
        self.assertEqual(result.readiness, ReadinessState.NOT_READY)
        self.assertIn("OUT_OF_ORDER_RECORD", [issue.code for issue in result.issues])
        self.assertEqual([c.interval_start for c in result.candles], [BASE, BASE + timedelta(minutes=1)])

    def test_invalid_ohlc_is_rejected(self) -> None:
        bad = raw_row(2, start=BASE, low="10.15", high="10.10")
        result = self.ingest([bad], calendar=calendar_for(BASE, BASE + timedelta(minutes=1)))
        self.assertEqual(result.readiness, ReadinessState.NOT_READY)
        self.assertEqual(result.accepted_count, 0)
        self.assertIn("INVALID_OHLC", [issue.code for issue in result.issues])

    def test_off_tick_price_is_rejected(self) -> None:
        bad = raw_row(2, start=BASE, open_price="10.001")
        result = self.ingest([bad], calendar=calendar_for(BASE, BASE + timedelta(minutes=1)))
        self.assertIn("PRICE_NOT_ALIGNED_TO_TICK", [issue.code for issue in result.issues])
        self.assertEqual(result.readiness, ReadinessState.NOT_READY)

    def test_overlapping_source_intervals_are_reported(self) -> None:
        first = raw_row(2, start=BASE, end=BASE + timedelta(minutes=1))
        overlapping = raw_row(
            3, start=BASE + timedelta(seconds=30), end=BASE + timedelta(minutes=1, seconds=30),
        )
        result = self.ingest([first, overlapping], calendar=calendar_for(BASE, BASE + timedelta(minutes=2)))
        codes = [issue.code for issue in result.issues]
        self.assertIn("OVERLAPPING_INTERVALS", codes)
        self.assertIn("UNALIGNED_INTERVAL", codes)
        self.assertEqual(result.readiness, ReadinessState.NOT_READY)

    def test_partial_candle_is_rejected_as_of_reference_time(self) -> None:
        row = raw_row(2, start=BASE, end=BASE + timedelta(minutes=1), available=BASE + timedelta(minutes=1))
        result = ingest_rows(
            [row], source_sha256=SOURCE_HASH, source_path=SOURCE_PATH,
            instrument_specs=tick_spec(), calendar=calendar_for(BASE, BASE + timedelta(minutes=1)),
            as_of_utc=BASE + timedelta(seconds=30), min_ready_bars=1,
        )
        self.assertEqual(result.readiness, ReadinessState.NOT_READY)
        self.assertIn("PARTIAL_CANDLE", [issue.code for issue in result.issues])

    def test_availability_before_close_is_rejected(self) -> None:
        row = raw_row(2, start=BASE, end=BASE + timedelta(minutes=1), available=BASE + timedelta(seconds=59))
        result = self.ingest([row], calendar=calendar_for(BASE, BASE + timedelta(minutes=1)))
        self.assertIn("AVAILABLE_BEFORE_CLOSE", [issue.code for issue in result.issues])

    def test_offset_timezone_is_normalized_to_utc(self) -> None:
        row = raw_row(2, start=BASE)
        fields = dict(row.fields)
        fields["interval_start_utc"] = "2024-01-01T01:00:00+01:00"
        fields["interval_end_utc"] = "2024-01-01T01:01:00+01:00"
        fields["source_available_at_utc"] = "2024-01-01T01:01:00+01:00"
        result = self.ingest([SourceRow(2, tuple(fields.items()))], calendar=calendar_for(BASE, BASE + timedelta(minutes=1)))
        self.assertEqual(result.candles[0].interval_start, BASE)

    def test_naive_timestamp_in_csv_is_rejected(self) -> None:
        row = raw_row(2, start=BASE)
        fields = dict(row.fields)
        fields["interval_start_utc"] = "2024-01-01T00:00:00"
        result = self.ingest([SourceRow(2, tuple(fields.items()))], calendar=calendar_for(BASE, BASE + timedelta(minutes=1)))
        self.assertIn("INVALID_TIMESTAMP", [issue.code for issue in result.issues])

    def test_batch_and_buffered_incremental_finalization_are_equivalent(self) -> None:
        rows = [raw_row(2), raw_row(3), raw_row(4)]
        kwargs = dict(
            source_sha256=SOURCE_HASH, source_path=SOURCE_PATH, instrument_specs=tick_spec(),
            calendar=calendar_for(BASE, BASE + timedelta(minutes=3)), as_of_utc=AS_OF, min_ready_bars=2,
        )
        batch = ingest_rows(rows, **kwargs)
        buffered = BufferedIngestor()
        for row in rows:
            buffered.add(row)
        incremental = buffered.finalize(**kwargs)
        self.assertEqual(canonical_json(batch), canonical_json(incremental))
        self.assertEqual(batch.candles, incremental.candles)

    def test_final_result_is_independent_of_buffer_arrival_order(self) -> None:
        rows = [raw_row(2), raw_row(3), raw_row(4)]
        kwargs = dict(
            source_sha256=SOURCE_HASH, source_path=SOURCE_PATH, instrument_specs=tick_spec(),
            calendar=calendar_for(BASE, BASE + timedelta(minutes=3)), as_of_utc=AS_OF, min_ready_bars=2,
        )
        first = BufferedIngestor()
        second = BufferedIngestor()
        for row in rows:
            first.add(row)
        for row in reversed(rows):
            second.add(row)
        self.assertEqual(canonical_json(first.finalize(**kwargs)), canonical_json(second.finalize(**kwargs)))

    def test_fifty_valid_bars_can_reach_data_ready(self) -> None:
        rows = [raw_row(index + 2, start=BASE + timedelta(minutes=index)) for index in range(50)]
        result = self.ingest(rows, calendar=calendar_for(BASE, BASE + timedelta(minutes=50)), minimum=50)
        self.assertEqual(result.readiness, ReadinessState.READY)

    def test_duplicate_csv_header_is_reported(self) -> None:
        header = ",".join(REQUIRED_COLUMNS + ("open",))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "duplicate_header.csv"
            path.write_text(header + "\n", encoding="utf-8")
            result = ingest_csv(path, instrument_specs=tick_spec(), calendar=UnknownSessionCalendar(), as_of_utc=AS_OF)
        self.assertIn("DUPLICATE_HEADER", [issue.code for issue in result.issues])
        self.assertEqual(result.readiness, ReadinessState.NOT_READY)

    def test_expected_range_detects_leading_and_trailing_gaps(self) -> None:
        rows = [raw_row(2, start=BASE + timedelta(minutes=1))]
        expected = ExpectedRange(BASE, BASE + timedelta(minutes=2))
        result = self.ingest(
            rows, calendar=calendar_for(BASE, BASE + timedelta(minutes=3)), expected_range=expected,
        )
        self.assertEqual(
            result.missing_starts,
            (("TESTUSD", "1m", BASE), ("TESTUSD", "1m", BASE + timedelta(minutes=2))),
        )

    def test_malformed_row_is_a_rejection_and_not_ready(self) -> None:
        row = raw_row(2)
        malformed = SourceRow(row.line_number, row.fields, malformed=True)
        result = self.ingest([malformed], calendar=calendar_for(BASE, BASE + timedelta(minutes=1)))
        self.assertEqual(result.readiness, ReadinessState.NOT_READY)
        self.assertIn("MALFORMED_COLUMN_COUNT", [issue.code for issue in result.issues])


if __name__ == "__main__":
    unittest.main()
