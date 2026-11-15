from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from mlof_v2.context import ContextEvaluator, _location
from mlof_v2.setup import SetupDetector
from tests.test_sealed_subsequent import sealed


class ContextTests(TestCase):
    def test_exact_location(self):
        band = SimpleNamespace(low_ticks=90, high_ticks=111)
        self.assertIsNone(_location("bullish", (99, 100), band))
        self.assertIsNone(_location("bearish", (101, 102), band))
        for prices in ((100, 99), (99, 100)):
            self.assertEqual("MIDPOINT_EQUALITY",
                             _location("bullish", prices, replace_band(band, high_ticks=110)))
        self.assertEqual("OUTSIDE_RANGE", _location("bullish", (89, 100), band))

    def test_context_freeze_and_history(self):
        for direction in ("bullish", "bearish"):
            with self.subTest(direction=direction):
                engine, bars = sealed(direction, reset=True)
                exchange, = engine.process_batch(bars).exchanges
                at = exchange.available_at
                start = bars[65].interval_start
                old_detector = SetupDetector()
                old = old_detector.as_of(at, exchange.exchange_id, engine, (), (), ())
                specs = engine.specs
                view = engine.as_of(at)
                formed = next(c for c in view.controllers if c.exchange_id == exchange.exchange_id)
                execution = engine
                def htf(tf, *, late=False, equal=False, replaced=False, bootstrap=False, ambiguous=False,
                        state="ACTIVE", readiness="READY", identity=None):
                    time = start if equal else start + timedelta(minutes=1 if late else -1)
                    parent = replace(formed, control_id=tf, timeframe=tf, available_at=time,
                                     state=state, readiness=readiness)
                    band = replace(view.ranges[0], control_id=tf, range_id=tf+"-range",
                                   instrument=formed.instrument, timeframe=tf, available_at=time,
                                   low_ticks=0 if direction == "bullish" else 8000,
                                   high_ticks=30000 if direction == "bullish" else 10800)
                    def snapshot(cut):
                        visible = () if cut < time else (replace(parent, state="REPLACED")
                            if replaced and cut >= at else parent,)
                        return SimpleNamespace(controllers=visible* (2 if ambiguous else 1),
                                               ranges=() if bootstrap or not visible else (band,))
                    return SimpleNamespace(spec_version="0.3.5", evaluation_identity=identity or tf,
                        replay=SimpleNamespace(seal=SimpleNamespace(instrument=formed.instrument,timeframe=tf)),
                        specs=specs, as_of=snapshot,
                        bootstrap=SimpleNamespace(result=lambda: SimpleNamespace(controls=(band,) if bootstrap else ())))
                def prefix(source, cut):
                    if source is execution:
                        return tuple(bars), "execution-seal"
                    return (), source.evaluation_identity+"-seal"
                with patch("mlof_v2.context._prefix", side_effect=prefix):
                    four = htf("4h", bootstrap=True)
                    daily = htf("1d")
                    evaluator = ContextEvaluator()
                    outcome = evaluator.as_of(at, exchange.exchange_id, execution, daily, four)
                    self.assertEqual("HTF_CONTEXT_ALIGNMENT", outcome.status)
                    self.assertEqual(start, outcome.first_sweep_at)
                    self.assertEqual(("4h-range","1d-range"), (outcome.primary_range_id,outcome.daily_range_id))
                    self.assertFalse(outcome.tradable)
                    corrected = evaluator.as_of(at, exchange.exchange_id, execution,
                                               htf("1d", identity="corrected"), four)
                    self.assertNotEqual(outcome.evaluation_identity, corrected.evaluation_identity)
                    self.assertEqual(outcome, evaluator.as_of(at, exchange.exchange_id, execution, daily, four))
                    for candidate, status in (
                        (None, "CONTEXT_STREAM_UNVERIFIED"),
                        (htf("1d", late=True), "CONTEXT_NOT_AVAILABLE_AT_SWEEP"),
                        (htf("1d", equal=True), "CONTEXT_PRIOR_PROCESSING_UNPROVEN"),
                        (htf("1d", ambiguous=True), "CONTEXT_NOT_AVAILABLE_AT_SWEEP"),
                        (htf("1d", replaced=True), "CONTEXT_LIFECYCLE_BLOCKED"),
                        (htf("1d", state="SUSPENDED"), "CONTEXT_LIFECYCLE_BLOCKED"),
                        (htf("1d", readiness="NOT_READY"), "CONTEXT_LIFECYCLE_BLOCKED"),
                    ):
                        with self.subTest(status=status):
                            actual = ContextEvaluator().as_of(at, exchange.exchange_id, execution, candidate, four)
                            self.assertEqual(status, actual.status)
                self.assertEqual(old, old_detector.as_of(at, exchange.exchange_id, engine, (), (), ()))


def replace_band(band, **changes):
    return SimpleNamespace(**(vars(band) | changes))


class SealedContextIntegrationTests(TestCase):
    def replay(self, *, incremental=False, breach=None, late=None):
        from tests.test_context_formation import context_replay
        from decimal import Decimal
        from mlof_v2.calendars import ExplicitSessionCalendar, OpenInterval
        from mlof_v2.setup import LowerStream
        from mlof_v2.structure import BootstrapAnalyzer, SwingWidths
        from mlof_v2.structure.coverage import ReplayInput, SourceSeal, interval_digest, intervals
        from mlof_v2.structure.subsequent import SubsequentExchangeAnalyzer
        from tests.test_subsequent_exchanges import bar

        template, bars = sealed(reset=True)
        start = bars[65].interval_start
        formed_at = bars[-1].source_available_at
        cut = formed_at + timedelta(minutes=3)
        bars += [bar(i, "105", "106", "104", "105") for i in range(68, 71)]

        def replay_input(source, calendar, boundary):
            first = source[0].interval_start
            instrument, timeframe = source[0].instrument, source[0].timeframe
            schedule = intervals(calendar, instrument, timeframe, first, boundary)
            seal = SourceSeal("integration", instrument, timeframe, "UTC", "close", first,
                              boundary, boundary, calendar.calendar_id, interval_digest(schedule),
                              "SEALED_COMPLETE_THROUGH_BOUNDARY", "")
            return ReplayInput.locally_sealed(seal, tuple(source))

        config, specs = template.bootstrap.config, tuple(template.specs.values())
        calendar = template.bootstrap.calendar
        bootstrap = BootstrapAnalyzer(config, template.rules, specs, calendar)
        execution = SubsequentExchangeAnalyzer(
            bootstrap, template.rules, specs, spec_version="0.3.4",
            replay=replay_input(bars, calendar, cut),
            additional_seals=(replay_input(bars, calendar, formed_at),),
        )
        step = timedelta(seconds=15)
        base = formed_at-timedelta(minutes=2)
        ltf = []
        for index in range(20):
            center = Decimal("100.50")
            low = center-Decimal("0.05")
            if index == 11:
                low = center-Decimal("0.10")
            if index == 19:
                low = center-Decimal("0.11")
            a = base+index*step
            ltf.append(replace(bar(100+index, str(center), str(center+Decimal("0.05")),
                                   str(low), str(center)), timeframe="15s",
                               interval_start=a, interval_end=a+step, source_available_at=a+step))
        ltf_cal = ExplicitSessionCalendar("ltf-integration", base, cut, (OpenInterval(base,cut),))
        ltf_config = replace(config, minimum_contiguous_bars=7, pivot_widths=(("15s",SwingWidths(1,1)),))
        lower = (LowerStream(replay_input(ltf, ltf_cal, cut), ltf_cal, ltf_config, specs[0]),)
        streams = []
        for tf, hours in (("1d", 24), ("4h", 4)):
            trailing = None
            if tf == breach:
                candle = bars[-1]
                trailing = replace(candle, timeframe=tf, interval_start=start-timedelta(minutes=1),
                                   interval_end=formed_at+timedelta(minutes=1),
                                   source_available_at=formed_at+timedelta(minutes=1),
                                   low=candle.low-30)
            engine, candles = context_replay(
                tf, "bullish", origin=start-timedelta(hours=68*hours, minutes=-1 if tf == late else 3),
                coverage_end=cut, price_shift=2, trailing=trailing, pre_final_seal=tf == late)
            streams.append((engine, candles))
        for engine, candles in ((execution,bars), *streams):
            if incremental:
                for candle in candles:
                    engine.process_candle(candle)
            else:
                engine.process_batch(candles)
        return execution, streams[0][0], streams[1][0], start, cut, lower

    def test_actual_sealed_streams_align_and_match_incremental(self):
        outputs = []
        for incremental in (False, True):
            execution, daily, four, start, cut, lower = self.replay(incremental=incremental)
            for htf in (daily, four):
                controller, = [c for c in htf.as_of(start).controllers if c.state == "ACTIVE"]
                self.assertEqual("READY", controller.readiness)
                self.assertLess(controller.available_at, start)
            exchange, = execution.result().exchanges
            disposition = ContextEvaluator().as_of(cut, exchange.exchange_id, execution, daily, four)
            self.assertEqual("HTF_CONTEXT_ALIGNMENT", disposition.status)
            self.assertEqual(start, disposition.first_sweep_at)
            self.assertFalse(disposition.tradable)
            self.assertEqual(four.as_of(start).ranges[-1].range_id, disposition.primary_range_id)
            self.assertEqual(daily.as_of(start).ranges[-1].range_id, disposition.daily_range_id)
            confirmed = ContextEvaluator().as_of(cut, exchange.exchange_id, execution, daily, four, lower)
            self.assertEqual("LTF_RECLAIM_CONFIRMED", confirmed.status)
            self.assertFalse(confirmed.tradable)
            outputs.append((disposition, confirmed))
        self.assertEqual(*outputs)

    def test_actual_protective_breach_blocks_either_context_at_reclaim_cut(self):
        for tf in ("1d", "4h"):
            with self.subTest(timeframe=tf):
                execution, daily, four, start, cut, lower = self.replay(breach=tf)
                exchange, = execution.result().exchanges
                disposition = ContextEvaluator().as_of(cut, exchange.exchange_id, execution, daily, four, lower)
                self.assertEqual("CONTEXT_LIFECYCLE_BLOCKED", disposition.status)
                self.assertFalse(disposition.tradable)

    def test_later_real_controller_cannot_replace_frozen_context(self):
        for timeframe in ("1d", "4h"):
            with self.subTest(timeframe=timeframe):
                execution, daily, four, start, cut, lower = self.replay(late=timeframe)
                affected = daily if timeframe == "1d" else four
                active_at_s, = [c for c in affected.as_of(start).controllers if c.state == "ACTIVE"]
                active_at_t, = [c for c in affected.as_of(cut).controllers if c.state == "ACTIVE"]
                self.assertEqual("bearish", active_at_s.direction)
                self.assertEqual("bullish", active_at_t.direction)
                self.assertGreater(active_at_t.available_at, start)
                exchange, = execution.result().exchanges
                disposition = ContextEvaluator().as_of(cut, exchange.exchange_id, execution, daily, four, lower)
                self.assertEqual("CONTEXT_LIFECYCLE_BLOCKED", disposition.status)
                self.assertEqual(start, disposition.first_sweep_at)
