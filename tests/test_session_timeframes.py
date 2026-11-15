from dataclasses import replace
from datetime import datetime, timedelta, timezone
import unittest

from mlof_v2.calendars import ExplicitSessionCalendar, OpenInterval
from mlof_v2.models import ReadinessState
from mlof_v2.structure import StructuralAnalyzer, SwingWidths
from mlof_v2.structure.coverage import ReplayInput, SourceSeal, interval_digest, intervals
from tests.test_subsequent_exchanges import bar, make_analyzer


class SessionTimeframeTests(unittest.TestCase):
    def test_explicit_daily_and_four_hour_replay(self):
        template = make_analyzer()
        start = datetime(2024, 3, 9, 22, tzinfo=timezone.utc)
        for timeframe, hours in (("1d", (23, 24)), ("4h", (4, 3))):
            with self.subTest(timeframe=timeframe):
                middle = start + timedelta(hours=hours[0])
                end = middle + timedelta(hours=hours[1])
                cut = end + timedelta(minutes=1)
                pairs = ((start, middle), (middle, end))
                calendar = ExplicitSessionCalendar("sessions", start, cut, (),
                    tuple(("TESTUSD", timeframe, OpenInterval(a, b)) for a, b in pairs))
                candles = tuple(replace(bar(i, "100", "101", "99", "100"), timeframe=timeframe,
                                        interval_start=a, interval_end=b,
                                        source_available_at=b+timedelta(minutes=1))
                                for i, (a,b) in enumerate(pairs))
                seal = SourceSeal("local", "TESTUSD", timeframe, "UTC", "close",
                                  start, end, cut, calendar.calendar_id, interval_digest(pairs),
                                  "SEALED_COMPLETE_THROUGH_BOUNDARY", "")
                replay = ReplayInput.locally_sealed(seal, candles)
                self.assertEqual(pairs[:1], intervals(calendar, "TESTUSD", timeframe, start, end-timedelta(seconds=1)))
                self.assertIsNone(replay.input_error(calendar, cut, candles)[0])
                self.assertIsNotNone(replay.input_error(calendar, end, candles)[0])
                missing = ReplayInput.locally_sealed(seal, candles[:1])
                self.assertEqual("INPUT_GAP", missing.input_error(calendar, cut, candles[:1])[0])
                self.assertIsNone(intervals(calendar, "OTHER", timeframe, start, cut))
                config = replace(template.bootstrap.config, pivot_widths=((timeframe, SwingWidths(1,1)),))
                def engine():
                    return StructuralAnalyzer(config, tuple(template.specs.values()), calendar)
                batch, incremental = engine(), engine()
                result = batch.process_batch(candles)
                for candle in candles:
                    incremental.process_candle(candle)
                self.assertEqual(result, incremental.result())
                self.assertEqual(cut, result.readiness[0].last_available_at)
                malformed = replace(candles[0], interval_end=middle-timedelta(minutes=1))
                incremental = engine()
                incremental.process_candle(malformed)
                self.assertIs(ReadinessState.NOT_READY, incremental.readiness()[0].state)

    def test_missing_session_boundary_and_existing_hour_grid(self):
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = start + timedelta(hours=5)
        calendar = ExplicitSessionCalendar("hours", start, end, (OpenInterval(start,end),))
        self.assertIsNone(intervals(calendar, "TESTUSD", "4h", start, end))
        self.assertEqual(5, len(intervals(calendar, "TESTUSD", "1h", start, end)))
