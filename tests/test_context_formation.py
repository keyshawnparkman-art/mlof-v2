"""Approved context formation on explicit scheduled intervals."""
from dataclasses import replace
from datetime import timedelta
import unittest

from mlof_v2.calendars import ExplicitSessionCalendar, OpenInterval
from mlof_v2.structure import BootstrapAnalyzer, BootstrapRules, SubsequentExchangeAnalyzer, SwingWidths
from mlof_v2.structure.config import RECLAIM_PROFILE_VERSION, reclaim_deadline
from mlof_v2.structure.coverage import ReplayInput, SourceSeal, interval_digest
from tests.test_subsequent_exchanges import BASE, KEY, chronological_replay


def context_replay(timeframe, direction, *, sealed=True, gap=False, bootstrap_reclaim=None, origin=BASE, coverage_end=None, price_shift=0, trailing=None, pre_final_seal=False):
    legacy, source = chronological_replay(direction, stop_before_sweep=True)
    if bootstrap_reclaim is not None:
        self_side = direction == "bullish"  # The first controller is bearish.
        pool = source[50].close + (1 if self_side else -1)
        for index in range(50, 50 + bootstrap_reclaim):
            candle = source[index]
            source[index] = (replace(candle, high=max(candle.high, pool), close=pool)
                             if self_side else replace(candle, low=min(candle.low, pool), close=pool))
    step = {"4h": timedelta(hours=4), "1d": timedelta(days=1),
            "15m": timedelta(minutes=15)}[timeframe]
    pairs = tuple((origin+i*step, origin+(i+1)*step) for i in range(len(source)))
    bars = tuple(replace(c, timeframe=timeframe, interval_start=a, interval_end=b,
                         source_available_at=b+timedelta(minutes=(b == pairs[-1][1])),
                         open=c.open+price_shift, high=c.high+price_shift,
                         low=c.low+price_shift, close=c.close+price_shift)
                 for c,(a,b) in zip(source,pairs))
    if trailing is not None:
        trailing = replace(trailing, provenance=replace(trailing.provenance, record_id="htf-trailing"))
        bars += (trailing,)
        pairs += ((trailing.interval_start, trailing.interval_end),)
    if gap:
        bars = bars[:49]+bars[50:]
    calendar_end = coverage_end or pairs[-1][1]+timedelta(minutes=1)
    if timeframe == "15m":
        calendar = ExplicitSessionCalendar("scalp-15m", origin, calendar_end,
                                           (OpenInterval(origin, pairs[-1][1]),))
    else:
        calendar = ExplicitSessionCalendar("m42", origin, calendar_end, (),
            tuple((KEY[0], timeframe, OpenInterval(a,b)) for a,b in pairs))
    config = replace(legacy.bootstrap.config, config_version=RECLAIM_PROFILE_VERSION,
                     minimum_contiguous_bars=50, pivot_widths=((timeframe,SwingWidths(2,2)),))
    rules = BootstrapRules(RECLAIM_PROFILE_VERSION)
    specs = tuple(legacy.specs.values())

    def seal(count):
        end = pairs[count-1][1]
        statement = SourceSeal("m42-fixture", KEY[0], timeframe, "UTC", "close", origin,
                               end, end if pre_final_seal and count == len(pairs)-1 else (bars[count-1].source_available_at if not gap else end+timedelta(minutes=1)), calendar.calendar_id,
                               interval_digest(pairs[:count]), "SEALED_COMPLETE_THROUGH_BOUNDARY", "")
        return ReplayInput.locally_sealed(statement,bars)

    latest = seal(len(pairs))
    engine = SubsequentExchangeAnalyzer(BootstrapAnalyzer(config,rules,specs,calendar),rules,specs,
        spec_version="0.3.5", replay=latest if sealed else None,
        additional_seals=tuple(seal(n) for n in (*range(50,55), *((len(pairs)-1,) if trailing is not None or pre_final_seal else ()))) if sealed else ())
    return engine,bars


class ContextFormationTests(unittest.TestCase):
    def test_both_streams_and_directions_publish_only_with_proof(self):
        for tf in ("4h","1d"):
            for direction in ("bullish","bearish"):
                with self.subTest(timeframe=tf,direction=direction):
                    batch,bars = context_replay(tf,direction)
                    live,_ = context_replay(tf,direction)
                    result = batch.process_batch(bars)
                    for candle in bars:
                        live.process_candle(candle)
                    self.assertEqual(result,live.result())
                    initial, = result.bootstrap.controls
                    exchange, = result.exchanges
                    self.assertNotEqual(initial.direction,direction)
                    self.assertEqual(direction,exchange.direction)
                    self.assertEqual((initial.control_id,), result.replaced_controller_ids)
                    self.assertFalse(exchange.tradable)
                    self.assertFalse(batch.as_of(exchange.available_at-timedelta(microseconds=1)).exchanges)
                    self.assertEqual(bars[-1].source_available_at,exchange.available_at)
                    swing = next(s for s in result.bootstrap.structural.swings
                                 if s.cluster_id == initial.protective_cluster_id)
                    anchor = next(i for i,c in enumerate(bars) if c.provenance.record_id == swing.anchor_candle_id)
                    self.assertEqual(bars[anchor+4].source_available_at,swing.available_at)
                    self.assertGreaterEqual(len(swing.comparison_evidence),9)

    def test_no_seal_and_calendar_gap_block_formation(self):
        for tf in ("4h","1d"):
            for kwargs in ({"sealed":False},{"gap":True}):
                with self.subTest(timeframe=tf,options=kwargs):
                    engine,bars = context_replay(tf,"bullish",**kwargs)
                    result = engine.process_batch(bars)
                    self.assertFalse(result.bootstrap.controls)
                    self.assertFalse(result.exchanges)

    def test_deadlines_are_opt_in_and_preserve_execution(self):
        self.assertEqual([6,6,5,4,4,3],
                         [reclaim_deadline(tf,RECLAIM_PROFILE_VERSION)
                          for tf in ("15s","30s","1m","3m","5m","15m")])
        self.assertEqual(2,reclaim_deadline("4h",RECLAIM_PROFILE_VERSION))
        self.assertEqual(1,reclaim_deadline("1d",RECLAIM_PROFILE_VERSION))
        self.assertIsNone(reclaim_deadline("4h"))
        self.assertIsNone(reclaim_deadline("1d"))

    def test_bootstrap_reclaim_at_and_after_deadline(self):
        for timeframe, deadline in (("4h", 2), ("1d", 1)):
            for delay in (deadline, deadline + 1):
                with self.subTest(timeframe=timeframe, delay=delay):
                    engine, bars = context_replay(timeframe, "bullish", bootstrap_reclaim=delay)
                    result = engine.process_batch(bars[:54])
                    self.assertEqual(delay == deadline, bool(result.bootstrap.controls))
