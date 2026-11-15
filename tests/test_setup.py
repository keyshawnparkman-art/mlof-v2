from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from types import SimpleNamespace
import unittest

from mlof_v2.setup import AncestorLink, LTF, SetupDetector, _confirmation
from mlof_v2.structure.models import PivotSide
from tests.test_sealed_subsequent import sealed
from tests.test_subsequent_exchanges import bar


class SetupTests(unittest.TestCase):
    def test_ordinary_exchange_requires_explicit_link_and_preserves_history(self):
        for direction in ("bullish", "bearish"):
            with self.subTest(direction=direction):
                engine, candles = sealed(direction)
                exchange, = engine.process_batch(candles).exchanges
                detector = SetupDetector()
                at = exchange.available_at
                outcome = detector.as_of(at, exchange.exchange_id, engine, (), (), ())
                self.assertEqual("ANCESTRY_MISSING", outcome.status)
                self.assertIsNotNone(outcome.reference_open_ticks)
                self.assertFalse(outcome.tradable)
                self.assertEqual(outcome, detector.as_of(at, exchange.exchange_id, engine, (), (), ()))
                early = detector.as_of(at-timedelta(microseconds=1), exchange.exchange_id, engine, (), (), ())
                self.assertEqual("ORDINARY_EXCHANGE_UNAVAILABLE", early.status)
                bootstrap_id = engine.result().bootstrap.exchanges[0].event_id
                self.assertEqual("ORDINARY_EXCHANGE_UNAVAILABLE",
                                 detector.as_of(at, bootstrap_id, engine, (), (), ()).status)
                manipulation, = engine.result().manipulations
                link = AncestorLink(exchange.exchange_id, engine.evaluation_identity, "other",
                                    "controller", "range", manipulation.liquidity_pool_id, at,
                                    "fixture-evidence", 1)
                self.assertEqual("ANCESTRY_UNVERIFIED_OR_CONFLICTING",
                                 detector.as_of(at, exchange.exchange_id, engine, (), (link,), ()).status)

    def test_strict_reclaim_and_fresh_pool_in_both_directions(self):
        tick = Decimal("0.01")
        start = bar(0, "100", "100", "100", "100").interval_end
        for direction in ("bullish", "bearish"):
            with self.subTest(direction=direction):
                is_long = direction == "bullish"
                sweep = bar(2, "100", "100.03", "99.97", "100")
                pool = SimpleNamespace(pool_id="pool", level_ticks=10000,
                                       available_at=sweep.interval_start,
                                       side=PivotSide.LOW if is_long else PivotSide.HIGH)
                reclaim = bar(3, "100", "100.03", "99.97", "100.01" if is_long else "99.99")
                def check(rows, radius=Decimal("0.02"), entry=Decimal("100")):
                    return _confirmation(rows, start, direction, entry, radius, tick, "1m")
                self.assertIsNone(check([(sweep, (pool,))]))
                self.assertEqual(reclaim.source_available_at, check([(sweep, (pool,)), (reclaim, ())])[0])
                stale = SimpleNamespace(**{**vars(pool), "available_at": start})
                self.assertIsNone(check([(sweep, (stale,)), (reclaim, ())]))
                self.assertIsNone(check([(sweep, (pool,)), (reclaim, ())], entry=Decimal("101")))

    def test_exact_timeframe_mapping(self):
        self.assertEqual({"1m": ("15s", "30s"), "3m": ("1m",), "5m": ("1m",), "15m": ("3m",)}, LTF)

    def test_linked_context_location_and_ltf_selection(self):
        for direction in ("bullish", "bearish"):
            with self.subTest(direction=direction):
                engine, candles = sealed(direction)
                exchange, = engine.process_batch(candles).exchanges
                at = exchange.available_at
                view = engine.as_of(at)
                ctrl = next(c for c in view.controllers if c.exchange_id == exchange.exchange_id)
                manipulation, = view.manipulations
                entry = manipulation.reference_open_ticks
                execution = SimpleNamespace(evaluation_identity=engine.evaluation_identity,
                                            as_of=engine.as_of, specs=engine.specs, rules=engine.rules)
                parent = replace(ctrl, control_id="higher", timeframe="15m")
                distance = abs(entry-manipulation.manipulation_extreme_ticks)+100
                low, high = ((entry-distance, entry+2*distance) if direction == "bullish"
                             else (entry-2*distance, entry+distance))
                band = replace(view.ranges[0], range_id="higher-range", control_id=parent.control_id,
                               low_ticks=low, high_ticks=high)
                ancestor_view = SimpleNamespace(controllers=(parent,), ranges=(band,))
                ancestor = SimpleNamespace(evaluation_identity="higher-eval", specs=engine.specs,
                                           as_of=lambda _: ancestor_view)
                link = AncestorLink(exchange.exchange_id, execution.evaluation_identity, "higher-eval",
                                    parent.control_id, band.range_id, manipulation.liquidity_pool_id,
                                    at, "link", 1)
                tick = Decimal("0.01")
                opening = entry*tick
                directions = ((PivotSide.LOW, opening+tick) if direction == "bullish"
                              else (PivotSide.HIGH, opening-tick))
                side, close = directions
                def stream(tf, lag):
                    start = at+timedelta(seconds=15)
                    sweep = replace(candles[-1], timeframe=tf, interval_start=start,
                                    interval_end=start+timedelta(seconds=lag),
                                    source_available_at=start+timedelta(seconds=lag),
                                    open=opening, high=opening+3*tick, low=opening-3*tick, close=close)
                    pool = SimpleNamespace(pool_id=tf+"-pool", side=side, level_ticks=entry,
                                           available_at=start)
                    return SimpleNamespace(identity=tf, observations=lambda cut:
                                           [(sweep,(pool,))] if sweep.source_available_at <= cut else [],
                                           instrument=engine.specs[ctrl.instrument],
                                           replay=SimpleNamespace(seal=SimpleNamespace(
                                               instrument=ctrl.instrument, timeframe=tf,
                                               seal_available_at=at)))
                from unittest.mock import patch
                later = at+timedelta(minutes=1)
                with patch("mlof_v2.setup._controller_bars", return_value=candles):
                    def decide(streams, links=(link,)):
                        return SetupDetector().as_of(later, exchange.exchange_id, execution,
                                                      (ancestor,), links, streams,
                                                      verified_link_ids=frozenset({"link","extra"}))
                    tied = decide((stream("30s",30),stream("15s",30)))
                    self.assertEqual("LTF_RECLAIM_CONFIRMED", tied.status)
                    self.assertIn("15s-pool", tied.evidence_ids)
                    self.assertFalse(tied.tradable)
                    faster = decide((stream("30s",30),stream("15s",45)))
                    self.assertIn("30s-pool", faster.evidence_ids)
                    self.assertEqual("LTF_COVERAGE_INCOMPLETE", decide(()).status)
                    self.assertEqual("ANCESTRY_UNVERIFIED_OR_CONFLICTING",
                                     decide((), (link,replace(link,evidence_id="extra"))).status)
                    ancestor_view.ranges = (replace(band, low_ticks=entry-distance,
                                                    high_ticks=entry+distance),)
                    self.assertEqual("LOCATION_REJECTED", decide(()).status)
                    ancestor_view.ranges = (band,)
                    ancestor_view.controllers = (replace(parent,state="SUSPENDED"),)
                    self.assertEqual("ANCESTOR_INELIGIBLE", decide(()).status)
                    ancestor_view.controllers = (replace(parent,state="REPLACED"),)
                    self.assertEqual("ANCESTOR_INELIGIBLE", decide(()).status)
