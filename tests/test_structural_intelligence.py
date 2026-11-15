from __future__ import annotations

import json
import unittest
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from mlof_v2.calendars import ExplicitSessionCalendar, OpenInterval, UnknownSessionCalendar
from mlof_v2.models import Candle, InstrumentSpec, ReadinessState, SourceProvenance
from mlof_v2.structure import (
    PivotSide, StructuralAnalyzer, StructuralConfig, StructuralEventKind,
    SwingWidths, load_structural_config, structural_events_jsonl,
)

UTC = timezone.utc
BASE = datetime(2024, 1, 1, tzinfo=UTC)
POLICY = "extend_before_confirmation_new_cluster_after_confirmation"


def research_config(*, minimum: int = 1, timeframes: tuple[str, ...] = ("1m",)) -> StructuralConfig:
    return StructuralConfig(1, "test-structural-v1", minimum,
                            tuple((tf, SwingWidths(2, 2)) for tf in timeframes), POLICY)


def make_candles(highs: list[str], lows: list[str] | None = None, *, timeframe: str = "1m") -> list[Candle]:
    duration = timedelta(minutes=1) if timeframe == "1m" else timedelta(seconds=15)
    if lows is None:
        lows = [str(Decimal(value) - 2) for value in highs]
    result = []
    for i, (high_text, low_text) in enumerate(zip(highs, lows)):
        start = BASE + i * duration
        end = start + duration
        high, low = Decimal(high_text), Decimal(low_text)
        middle = (high + low) / 2
        provenance = SourceProvenance(
            "synthetic://structural-test", "a" * 64, i + 1, f"{i + 1:064x}",
            f"bar-{timeframe}-{i}", (("synthetic", "true"), ("index", str(i))),
        )
        result.append(Candle("TESTUSD", timeframe, start, end, end, middle, high, low, middle, provenance))
    return result


def calendar_for(candles: list[Candle]) -> ExplicitSessionCalendar:
    duration = candles[0].interval_end - candles[0].interval_start
    start = min(c.interval_start for c in candles)
    end = max(c.interval_end for c in candles) + duration
    return ExplicitSessionCalendar("synthetic-structural-calendar", start, end, (OpenInterval(start, end),))


def analyzer_for(candles: list[Candle], *, config: StructuralConfig | None = None,
                 calendar: object | None = None) -> StructuralAnalyzer:
    return StructuralAnalyzer(config or research_config(), (InstrumentSpec("TESTUSD", Decimal("0.01")),),
                              calendar or calendar_for(candles))  # type: ignore[arg-type]


def swing_high(result):
    return next(x for x in result.swings if x.side is PivotSide.HIGH)


BASE_HIGH = ["2", "3", "4", "5", "8", "5", "4", "3", "2"]


class StructuralSwingTests(unittest.TestCase):
    def test_four_left_four_right_uses_nine_bar_window_not_five(self) -> None:
        candles = make_candles(BASE_HIGH)
        analyzer = analyzer_for(candles)
        for candle in candles[:8]:
            analyzer.process_candle(candle)
        self.assertFalse(any(x.side is PivotSide.HIGH for x in analyzer.result().swings))
        analyzer.process_candle(candles[8])
        high = swing_high(analyzer.result())
        self.assertEqual(candles[4].provenance.record_id, high.anchor_candle_id)
        self.assertEqual(candles[8].interval_end, high.confirmed_at)
        self.assertEqual(9, len(high.comparison_evidence))

    def test_high_and_low_confirm_only_after_final_fourth_right_bar(self) -> None:
        lows = ["1", "0", "-1", "-2", "-4", "-2", "-1", "0", "1"]
        candles = make_candles(BASE_HIGH, lows)
        analyzer = analyzer_for(candles)
        for candle in candles[:-1]:
            analyzer.process_candle(candle)
        self.assertEqual((), analyzer.result().swings)
        analyzer.process_candle(candles[-1])
        self.assertEqual({PivotSide.HIGH, PivotSide.LOW}, {s.side for s in analyzer.result().swings})
        self.assertTrue(all(s.confirmed_at == candles[8].interval_end for s in analyzer.result().swings))

    def test_full_comparison_provenance_includes_left_anchor_right_and_availability(self) -> None:
        candles = make_candles(BASE_HIGH)
        result = analyzer_for(candles).process_batch(candles)
        high = swing_high(result)
        self.assertEqual(tuple(c.provenance.record_id for c in candles), tuple(x.record_id for x in high.comparison_evidence))
        self.assertEqual(tuple(c.provenance for c in candles), high.provenance)
        for candle, evidence in zip(candles, high.comparison_evidence):
            self.assertEqual(candle.interval_start, evidence.interval_start)
            self.assertEqual(candle.interval_end, evidence.interval_end)
            self.assertEqual(candle.source_available_at, evidence.source_available_at)
            self.assertEqual(candle.provenance, evidence.provenance)
        event = next(x for x in result.events if x.event_id == high.swing_event_id)
        self.assertEqual(high.comparison_evidence, event.candle_evidence)
        self.assertEqual(high.config_version, event.config_version)

    def test_equal_price_extends_unconfirmed_cluster_and_restarts_four_right_bars(self) -> None:
        highs = ["2", "3", "4", "5", "8", "8", "5", "4", "3", "2"]
        candles = make_candles(highs)
        analyzer = analyzer_for(candles)
        for candle in candles[:9]:
            analyzer.process_candle(candle)
        self.assertFalse(any(x.side is PivotSide.HIGH for x in analyzer.result().swings))
        analyzer.process_candle(candles[9])
        high = swing_high(analyzer.result())
        self.assertEqual(candles[4].provenance.record_id, high.anchor_candle_id)
        self.assertEqual((candles[4].provenance.record_id, candles[5].provenance.record_id), high.member_candle_ids)
        self.assertEqual(tuple(c.provenance.record_id for c in candles[6:10]), high.confirmation_evidence_ids)

    def test_strictly_more_extreme_invalidates_and_can_start_new_candidate(self) -> None:
        candles = make_candles(["2", "3", "4", "5", "8", "9", "6", "5", "4", "3"])
        result = analyzer_for(candles).process_batch(candles)
        highs = [x for x in result.swings if x.side is PivotSide.HIGH]
        self.assertEqual(1, len(highs))
        self.assertEqual(candles[5].provenance.record_id, highs[0].anchor_candle_id)
        self.assertEqual(9 * 100, highs[0].level_ticks)
        self.assertTrue(any(x.kind is StructuralEventKind.CANDIDATE_INVALIDATED for x in result.events))

    def test_confirmed_cluster_is_immutable_and_later_equal_visit_is_separate(self) -> None:
        candles = make_candles(["2", "3", "4", "5", "8", "5", "4", "3", "2",
                                "3", "4", "5", "8", "5", "4", "3", "2"])
        result = analyzer_for(candles).process_batch(candles)
        highs = [x for x in result.swings if x.side is PivotSide.HIGH]
        self.assertEqual(2, len(highs))
        self.assertNotEqual(highs[0].cluster_id, highs[1].cluster_id)
        self.assertEqual(candles[4].provenance.record_id, highs[0].anchor_candle_id)
        self.assertEqual(candles[12].provenance.record_id, highs[1].anchor_candle_id)
        with self.assertRaises(FrozenInstanceError):
            highs[0].level_ticks = 999  # type: ignore[misc]

    def test_insufficient_history_does_not_open_candidate(self) -> None:
        candles = make_candles(["2", "3", "4", "5"])
        result = analyzer_for(candles).process_batch(candles)
        self.assertFalse(result.swings)
        self.assertFalse(any(x.kind is StructuralEventKind.CANDIDATE_OPENED for x in result.events))

    def test_same_candle_high_low_candidates_are_not_intrabar_ordered(self) -> None:
        candles = make_candles(["2", "3", "4", "5", "8"], ["1", "0", "-1", "-2", "-4"])
        result = analyzer_for(candles).process_batch(candles)
        opens = [e for e in result.events if e.kind is StructuralEventKind.CANDIDATE_OPENED
                 and candles[4].provenance.record_id in e.source_candle_ids]
        self.assertEqual({PivotSide.HIGH, PivotSide.LOW}, {x.side for x in opens})
        self.assertEqual(1, len({dict(x.details)["simultaneous_candidate_group_id"] for x in opens}))

    def test_versioned_default_widths_remain_research_defaults(self) -> None:
        config = load_structural_config("config/structural_research.example.json")
        self.assertEqual(50, config.minimum_contiguous_bars)
        for tf, width in (("15s", 3), ("30s", 3), ("1m", 2), ("3m", 2),
                          ("5m", 2), ("15m", 2), ("30m", 2), ("1h", 2)):
            self.assertEqual(SwingWidths(width, width), config.widths_for(tf))

    def test_three_three_width_uses_thirteen_bar_comparison(self) -> None:
        shape = ["2", "3", "4", "5", "6", "7", "10", "7", "6", "5", "4", "3", "2"]
        candles = make_candles(shape, timeframe="15s")
        config = StructuralConfig(1, "wide-15s-test", 1, (("15s", SwingWidths(3, 3)),), POLICY)
        result = analyzer_for(candles, config=config).process_batch(candles)
        high = next(x for x in result.swings if x.side is PivotSide.HIGH)
        self.assertEqual(candles[6].provenance.record_id, high.anchor_candle_id)
        self.assertEqual(candles[12].interval_end, high.confirmed_at)
        self.assertEqual(13, len(high.comparison_evidence))


class StructuralAvailabilityTests(unittest.TestCase):
    def test_confirmation_candle_cannot_use_its_own_pool_but_next_can(self) -> None:
        candles = make_candles(BASE_HIGH + ["3"])
        analyzer = analyzer_for(candles)
        for candle in candles:
            analyzer.process_candle(candle)
        high = swing_high(analyzer.result())
        i = next(i for i, c in enumerate(candles) if c.provenance.record_id == high.confirmation_candle_id)
        self.assertNotIn(high.pool_id, {p.pool_id for p in analyzer.eligible_pools_for(candles[i])})
        self.assertIn(high.pool_id, {p.pool_id for p in analyzer.eligible_pools_for(candles[i + 1])})

    def test_delayed_source_availability_gates_pool_until_later_interval_start(self) -> None:
        base = make_candles(BASE_HIGH + ["3", "4", "5"])
        candles = [replace(c, source_available_at=c.interval_end + timedelta(seconds=90)) for c in base]
        analyzer = analyzer_for(candles)
        analyzer.process_batch(candles)
        high = swing_high(analyzer.result())
        i = next(i for i, c in enumerate(candles) if c.provenance.record_id == high.confirmation_candle_id)
        self.assertNotIn(high.pool_id, {p.pool_id for p in analyzer.eligible_pools_for(candles[i + 1])})
        self.assertNotIn(high.pool_id, {p.pool_id for p in analyzer.eligible_pools_for(candles[i + 2])})
        self.assertIn(high.pool_id, {p.pool_id for p in analyzer.eligible_pools_for(candles[i + 3])})

    def test_close_open_equal_timestamp_uses_prior_registration_order(self) -> None:
        candles = make_candles(BASE_HIGH + ["3"])
        analyzer = analyzer_for(candles)
        analyzer.process_batch(candles)
        high = swing_high(analyzer.result())
        i = next(i for i, c in enumerate(candles) if c.provenance.record_id == high.confirmation_candle_id)
        self.assertEqual(candles[i].interval_end, candles[i + 1].interval_start)
        self.assertIn(high.pool_id, {p.pool_id for p in analyzer.eligible_pools_for(candles[i + 1])})

    def test_unknown_calendar_fails_closed(self) -> None:
        candles = make_candles(BASE_HIGH)
        result = analyzer_for(candles, calendar=UnknownSessionCalendar()).process_batch(candles)
        self.assertEqual(ReadinessState.NOT_READY, result.readiness[0].state)
        self.assertFalse(result.swings)

    def test_off_tick_ohlc_fails_closed(self) -> None:
        candle = make_candles(["8"])[0]
        invalid = replace(candle, open=Decimal("7.001"))
        analyzer = analyzer_for([candle])
        analyzer.process_candle(invalid)
        self.assertEqual(ReadinessState.NOT_READY, analyzer.readiness()[0].state)
        self.assertFalse(analyzer.result().swings)

    def test_missing_expected_bar_resets_candidates_and_readiness(self) -> None:
        full = make_candles(BASE_HIGH + ["3", "4"])
        candles = [c for i, c in enumerate(full) if i != 5]
        result = analyzer_for(full).process_batch(candles)
        self.assertTrue(any(e.kind is StructuralEventKind.DATA_GAP_DETECTED for e in result.events))
        self.assertTrue(any(e.kind is StructuralEventKind.READINESS_RESET for e in result.events))
        self.assertEqual(ReadinessState.NOT_READY, result.readiness[0].state)

    def test_batch_and_incremental_processing_are_identical(self) -> None:
        candles = make_candles(BASE_HIGH + ["3", "7", "4", "2"])
        batch = analyzer_for(candles).process_batch(candles)
        incremental = analyzer_for(candles)
        for candle in candles:
            incremental.process_candle(candle)
        self.assertEqual(batch, incremental.result())
        self.assertEqual([e.event_id for e in batch.events], [e.event_id for e in incremental.result().events])

    def test_ids_and_serialized_audit_are_deterministic_with_evidence(self) -> None:
        candles = make_candles(BASE_HIGH)
        first = analyzer_for(candles).process_batch(candles)
        second = analyzer_for(candles).process_batch(candles)
        self.assertEqual([e.event_id for e in first.events], [e.event_id for e in second.events])
        serialized = structural_events_jsonl(first.events)
        self.assertEqual(serialized, structural_events_jsonl(second.events))
        row = json.loads(next(line for line in serialized.splitlines() if '"kind":"SWING_CONFIRMED"' in line))
        self.assertEqual(9, len(row["candle_evidence"]))
        self.assertTrue(all(e.provenance and e.config_version for e in first.events))

    def test_timeframe_states_are_independent(self) -> None:
        shape = BASE_HIGH
        minute, seconds = make_candles(shape, timeframe="1m"), make_candles(shape, timeframe="15s")
        config = research_config(timeframes=("1m", "15s"))
        end = BASE + timedelta(minutes=10)
        calendar = ExplicitSessionCalendar("both", BASE, end, (OpenInterval(BASE, end),))
        result = analyzer_for(minute, config=config, calendar=calendar).process_batch(minute + seconds)
        self.assertEqual({"1m", "15s"}, {s.timeframe for s in result.swings})


if __name__ == "__main__":
    unittest.main()
