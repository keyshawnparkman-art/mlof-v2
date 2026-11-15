"""Sealed SCALP evaluation under the independently versioned v0.3.6 contract."""
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch
from unittest import TestCase

from mlof_v2.scalp import ScalpEvaluator
from mlof_v2.context import ContextEvaluator, _location
from mlof_v2.setup import SetupDetector
from tests import test_context_alignment
from tests.test_context_formation import context_replay
from tests.test_subsequent_exchanges import BASE


class ScalpTests(TestCase):
    def test_real_confirming_and_primary_replays_match_incremental(self):
        outcomes = []
        for incremental in (False, True):
            execution, daily, four, start, cut, lower = test_context_alignment.SealedContextIntegrationTests().replay(
                incremental=incremental)
            confirming, candles = context_replay(
                "15m", "bullish", origin=BASE-timedelta(minutes=68*15),
                coverage_end=cut+timedelta(minutes=15), price_shift=2,
            )
            if incremental:
                for candle in candles:
                    confirming.process_candle(candle)
            else:
                confirming.process_batch(candles)
            exchange, = execution.result().exchanges
            evaluator = ScalpEvaluator()
            active, = [c for c in confirming.as_of(start).controllers if c.state == "ACTIVE"]
            self.assertEqual("READY", active.readiness)
            self.assertLess(active.available_at, start)
            result = evaluator.as_of(cut, exchange.exchange_id, execution, four, confirming, lower)
            self.assertEqual("LTF_RECLAIM_CONFIRMED", result.status)
            self.assertEqual(start, result.first_sweep_at)
            self.assertEqual(cut, result.published_at)
            self.assertFalse(result.tradable)
            with patch("mlof_v2.scalp._prefix", return_value=None):
                self.assertEqual("EXECUTION_COVERAGE_INCOMPLETE", ScalpEvaluator().as_of(
                    cut, exchange.exchange_id, execution, four, confirming, lower).status)
            original_as_of = confirming.as_of
            for changes in ({"state": "SUSPENDED"}, {"state": "REPLACED"},
                            {"readiness": "NOT_READY"}):
                with self.subTest(lifecycle=changes):
                    def blocked(at):
                        view = original_as_of(at)
                        if at != cut:
                            return view
                        return replace(view, controllers=tuple(
                            replace(c, **changes) if c.state == "ACTIVE" else c
                            for c in view.controllers))
                    with patch.object(confirming, "as_of", side_effect=blocked):
                        self.assertEqual("CONTEXT_LIFECYCLE_BLOCKED", ScalpEvaluator().as_of(
                            cut, exchange.exchange_id, execution, four, confirming, lower).status)
            for offset, expected in ((0, "CONTEXT_PRIOR_PROCESSING_UNPROVEN"),
                                     (1, "CONTEXT_NOT_AVAILABLE_AT_SWEEP")):
                with self.subTest(context_publication_offset=offset):
                    def delayed_context(at):
                        view = original_as_of(at)
                        controllers = tuple(
                            replace(c, available_at=start+timedelta(seconds=offset))
                            if c.state == "ACTIVE" else c for c in view.controllers
                            if c.state != "ACTIVE" or at >= start+timedelta(seconds=offset))
                        return replace(view, controllers=controllers)
                    with patch.object(confirming, "as_of", side_effect=delayed_context):
                        self.assertEqual(expected, ScalpEvaluator().as_of(
                            cut, exchange.exchange_id, execution, four, confirming, lower).status)
            self.assertNotIn(daily.evaluation_identity, result.evidence_ids)
            from mlof_v2.structure.coverage import ReplayInput
            source = lower[0]
            candles = source.replay.candles
            revised = replace(candles[0], provenance=replace(
                candles[0].provenance, record_id="revised-ltf-first-row"))
            corrected = replace(source, replay=ReplayInput.locally_sealed(
                source.replay.seal, (revised, *candles[1:])))
            revised_result = evaluator.as_of(cut, exchange.exchange_id, execution, four,
                                             confirming, (corrected,))
            self.assertEqual(result.status, revised_result.status)
            self.assertNotEqual(result.evaluation_identity, revised_result.evaluation_identity)
            self.assertEqual(result, evaluator.as_of(cut, exchange.exchange_id, execution, four,
                                                     confirming, lower))

            historical = SetupDetector().as_of(cut, exchange.exchange_id, execution, (), (), ())
            self.assertNotEqual(historical.record_id, result.record_id)
            old_context = ContextEvaluator().as_of(cut, exchange.exchange_id, execution, daily, four, lower)
            self.assertEqual("LTF_RECLAIM_CONFIRMED", old_context.status)
            self.assertNotEqual(old_context.evaluation_identity, result.evaluation_identity)
            with patch("mlof_v2.scalp._confirmation", return_value=(
                    cut-timedelta(seconds=15), "pool", "sweep",
                    lower[0].replay.candles[-2].provenance.record_id)):
                delayed = ScalpEvaluator().as_of(cut, exchange.exchange_id, execution, four, confirming, lower)
            self.assertLess(delayed.reclaim_available_at, delayed.published_at)
            # Exercise the selection boundary with independently supplied eligible results.
            fifteen = lower[0]
            thirty_replay = SimpleNamespace(seal=SimpleNamespace(
                instrument=fifteen.replay.seal.instrument, timeframe="30s",
                seal_available_at=cut, declaration_id="test-30s-declaration"))
            thirty = SimpleNamespace(replay=thirty_replay, instrument=fifteen.instrument,
                identity="test-30s", observations=fifteen.observations)
            candles = fifteen.replay.candles
            def confirmation(rows, start, direction, entry, radius, tick, timeframe):
                return (cut, timeframe+"-pool", candles[-2].provenance.record_id,
                        candles[-1].provenance.record_id)
            with patch("mlof_v2.scalp._confirmation", side_effect=confirmation):
                tied = ScalpEvaluator().as_of(cut, exchange.exchange_id, execution, four,
                                               confirming, (thirty, fifteen))
            self.assertIn("15s-pool", tied.evidence_ids)
            def earlier(rows, start, direction, entry, radius, tick, timeframe):
                return (cut-timedelta(seconds=15) if timeframe == "30s" else cut,
                        timeframe+"-pool", candles[-2].provenance.record_id,
                        candles[-1].provenance.record_id)
            thirty_replay.seal.seal_available_at = cut-timedelta(seconds=15)
            with patch("mlof_v2.scalp._confirmation", side_effect=earlier):
                faster = ScalpEvaluator().as_of(cut, exchange.exchange_id, execution, four,
                                                 confirming, (fifteen, thirty))
            self.assertIn("30s-pool", faster.evidence_ids)
            initial, = execution.result().bootstrap.controls
            self.assertEqual("ORDINARY_EXCHANGE_UNAVAILABLE", evaluator.as_of(
                cut, initial.exchange_id, execution, four, confirming).status)
            self.assertEqual("EXECUTION_TIMEFRAME_UNSUPPORTED", evaluator.as_of(
                cut, exchange.exchange_id, four, four, confirming).status)
            outcomes.append(result)
            self.assertEqual("MODE_DISABLED", evaluator.as_of(
                cut, exchange.exchange_id, execution, four, confirming, mode="SWING").status)
            self.assertEqual("CONTEXT_STREAM_UNVERIFIED", evaluator.as_of(
                cut, exchange.exchange_id, execution, four, None).status)
        self.assertEqual(*outcomes)

    def test_integer_tick_midpoints_and_both_range_edges(self):
        for direction, prices in (("bullish", (90, 100)), ("bearish", (101, 111))):
            with self.subTest(direction=direction):
                band = SimpleNamespace(low_ticks=90, high_ticks=111)
                self.assertIsNone(_location(direction, prices, band))
                for candidate in ((89, prices[1]), (prices[0], 112)):
                    self.assertEqual("OUTSIDE_RANGE", _location(direction, candidate, band))
        band = SimpleNamespace(low_ticks=90, high_ticks=110)
        for direction in ("bullish", "bearish"):
            for prices in ((100, 99), (99, 100)):
                self.assertEqual("MIDPOINT_EQUALITY", _location(direction, prices, band))

    def test_real_sealed_execution_matrix(self):
        from dataclasses import replace
        from decimal import Decimal
        from mlof_v2.calendars import ExplicitSessionCalendar, OpenInterval
        from mlof_v2.setup import LowerStream
        from mlof_v2.structure import BootstrapAnalyzer, SubsequentExchangeAnalyzer, SwingWidths
        from mlof_v2.structure.coverage import ReplayInput, SourceSeal, interval_digest, intervals
        from tests.test_subsequent_exchanges import bar, chronological_replay

        def sealed_input(bars, calendar, cut):
            first = bars[0]
            expected = intervals(calendar, first.instrument, first.timeframe, first.interval_start, cut)
            declaration = SourceSeal("scalp-matrix", first.instrument, first.timeframe, "UTC", "close",
                first.interval_start, cut, cut, calendar.calendar_id, interval_digest(expected),
                "SEALED_COMPLETE_THROUGH_BOUNDARY", "")
            return ReplayInput.locally_sealed(declaration, tuple(bars))

        for minutes in (1, 3, 5):
            for direction in ("bullish", "bearish"):
                if (minutes, direction) == (1, "bullish"):
                    continue
                with self.subTest(timeframe=minutes, direction=direction):
                    template, raw = chronological_replay(direction, stop_before_sweep=True, deeper_sweep=True)
                    tf = f"{minutes}m"
                    sign = 1 if direction == "bullish" else -1
                    step = timedelta(minutes=minutes)
                    bars = [replace(c, timeframe=tf, interval_start=BASE+i*step,
                                    interval_end=BASE+(i+1)*step, source_available_at=BASE+(i+1)*step)
                            for i,c in enumerate(raw)]
                    formed = bars[-1].interval_end
                    cut = formed + 6*step
                    close = bars[-1].close
                    for i in range(68,74):
                        a = BASE+i*step
                        bars.append(replace(bar(i,str(close),str(close),str(close),str(close)),
                            timeframe=tf, interval_start=a, interval_end=a+step, source_available_at=a+step))
                    calendar = ExplicitSessionCalendar("execution", BASE, cut, (OpenInterval(BASE,cut),))
                    config = replace(template.bootstrap.config, pivot_widths=((tf,SwingWidths(2,2)),))
                    specs = tuple(template.specs.values())
                    rules = template.rules
                    execution = SubsequentExchangeAnalyzer(BootstrapAnalyzer(config,rules,specs,calendar),
                        rules,specs,spec_version="0.3.6",replay=sealed_input(bars,calendar,cut),
                        additional_seals=(sealed_input(bars,calendar,formed),))
                    exchange, = execution.process_batch(bars).exchanges
                    entry = Decimal(execution.result().manipulations[0].reference_open_ticks)/100
                    contexts = []
                    for context_tf, hours in (("4h",4),("15m",0.25)):
                        context, context_bars = context_replay(context_tf,direction,
                            origin=BASE-timedelta(hours=68*hours), coverage_end=cut+timedelta(hours=1),
                            price_shift=2 if sign == 1 else -1)
                        context.process_batch(context_bars)
                        contexts.append(context)
                    lower_tf, seconds = ("15s",15) if minutes == 1 else ("1m",60)
                    lower_step = timedelta(seconds=seconds)
                    lower_start = formed-8*lower_step
                    lower_bars = []
                    for i in range((cut-lower_start)//lower_step):
                        a = lower_start+i*lower_step
                        edge = Decimal("0.10") if i == 11 else Decimal("0.11") if i == 15 else Decimal("0.05")
                        high = entry+(Decimal("0.05") if sign == 1 else edge)
                        low = entry-(edge if sign == 1 else Decimal("0.05"))
                        lower_bars.append(replace(bar(200+i,str(entry),str(high),str(low),str(entry)),
                            timeframe=lower_tf,interval_start=a,interval_end=a+lower_step,source_available_at=a+lower_step))
                    lc = ExplicitSessionCalendar("lower",lower_start,cut,(OpenInterval(lower_start,cut),))
                    cfg = replace(config,minimum_contiguous_bars=7,pivot_widths=((lower_tf,SwingWidths(1,1)),))
                    lower = LowerStream(sealed_input(lower_bars,lc,cut),lc,cfg,specs[0])
                    outcome = ScalpEvaluator().as_of(cut,exchange.exchange_id,execution,*contexts,(lower,))
                    self.assertEqual("LTF_RECLAIM_CONFIRMED", outcome.status)
                    self.assertEqual(bars[65].interval_start,outcome.first_sweep_at)
                    self.assertLess(outcome.reclaim_available_at,outcome.published_at)
                    earlier = ScalpEvaluator().as_of(outcome.reclaim_available_at, exchange.exchange_id,
                                                     execution, *contexts, (lower,))
                    self.assertIn(earlier.status, ("EXECUTION_COVERAGE_INCOMPLETE",
                                                  "LTF_COVERAGE_INCOMPLETE"))

                    self.assertFalse(outcome.tradable)

    def test_reclaim_deadline_last_bar_and_expiration(self):
        from decimal import Decimal
        from mlof_v2.setup import RECLAIM_BARS, _confirmation
        from mlof_v2.structure.models import PivotSide
        from tests.test_subsequent_exchanges import bar

        for timeframe in ("15s", "30s", "1m"):
            deadline = RECLAIM_BARS[timeframe]
            for offset in (deadline, deadline + 1):
                with self.subTest(timeframe=timeframe, offset=offset):
                    candles = [bar(1, "100", "101", "99", "100")]
                    candles += [bar(i, "100", "101", "100", "100")
                                for i in range(2, offset + 1)]
                    candles.append(bar(offset + 1, "100", "101", "100", "100.01"))
                    pool = SimpleNamespace(pool_id="deadline-pool", level_ticks=10000,
                                           available_at=candles[0].interval_start, side=PivotSide.LOW)
                    rows = [(candles[0], (pool,))] + [(c, ()) for c in candles[1:]]
                    outcome = _confirmation(rows, BASE, "bullish", Decimal("100"),
                                            Decimal("0.10"), Decimal("0.01"), timeframe)
                    self.assertEqual(offset == deadline, outcome is not None)

    def test_sealed_primary_protective_breach_blocks_reclaim(self):
        execution, _, four, start, cut, lower = (
            test_context_alignment.SealedContextIntegrationTests().replay(breach="4h"))
        confirming, candles = context_replay(
            "15m", "bullish", origin=BASE-timedelta(minutes=68*15),
            coverage_end=cut+timedelta(minutes=15), price_shift=2)
        confirming.process_batch(candles)
        exchange, = execution.result().exchanges
        evaluator = ScalpEvaluator()
        before = evaluator.as_of(exchange.available_at, exchange.exchange_id, execution, four, confirming)
        self.assertEqual("HTF_CONTEXT_ALIGNMENT", before.status)
        after = evaluator.as_of(cut, exchange.exchange_id, execution, four, confirming, lower)
        self.assertEqual("CONTEXT_LIFECYCLE_BLOCKED", after.status)
        self.assertEqual(start, after.first_sweep_at)
        self.assertEqual(before, evaluator.as_of(
            exchange.available_at, exchange.exchange_id, execution, four, confirming))

    def test_inverted_ltf_availability_fails_closed(self):
        from mlof_v2.structure.coverage import ReplayInput

        _, _, _, _, cut, lower = test_context_alignment.SealedContextIntegrationTests().replay()
        stream = lower[0]
        first, second, *rest = stream.replay.candles
        inverted = (replace(first, source_available_at=second.source_available_at+
                            timedelta(microseconds=1)), second, *rest)
        replay = ReplayInput.locally_sealed(stream.replay.seal, inverted)
        reason, _ = replay.input_error(stream.calendar, cut, inverted)
        self.assertEqual("AVAILABILITY_INVERSION", reason)
        self.assertIsNone(replace(stream, replay=replay).observations(cut))

    def test_sealed_15m_breach_recovery_and_replacement_preserve_history(self):
        from decimal import Decimal
        from mlof_v2.calendars import ExplicitSessionCalendar, OpenInterval
        from mlof_v2.structure import BootstrapAnalyzer, SubsequentExchangeAnalyzer
        from mlof_v2.structure.coverage import ReplayInput, SourceSeal, interval_digest, intervals
        from tests.test_sealed_subsequent import sealed
        from tests.test_subsequent_exchanges import bar

        # The bullish 15M bootstrap is the context at S. Its later ordinary
        # bearish exchange provides a genuine terminal replacement.
        step = timedelta(minutes=15)
        origin = BASE - 55 * step
        cut = origin + 68 * step + timedelta(minutes=1)
        template, context_bars = context_replay(
            "15m", "bearish", origin=origin, coverage_end=cut+step, price_shift=2)
        breach = replace(context_bars[57], low=Decimal("96.99"))
        candles = (*context_bars[:57], breach, *context_bars[58:])
        calendar = template.bootstrap.calendar

        def declaration(bars, calendar, boundary, available):
            first = bars[0]
            expected = intervals(calendar, first.instrument, first.timeframe,
                                 first.interval_start, boundary)
            seal = SourceSeal("scalp-breach", first.instrument, first.timeframe, "UTC", "close",
                              first.interval_start, boundary, available, calendar.calendar_id,
                              interval_digest(expected), "SEALED_COMPLETE_THROUGH_BOUNDARY", "")
            return ReplayInput.locally_sealed(seal, tuple(bars))

        def analyzer(template, calendar, replay, prefixes):
            specs = tuple(template.specs.values())
            bootstrap = BootstrapAnalyzer(template.bootstrap.config, template.rules, specs, calendar)
            return SubsequentExchangeAnalyzer(bootstrap, template.rules, specs,
                spec_version=template.spec_version, replay=replay, additional_seals=prefixes)

        needed = (54, 55, 56, 57, 58, 59)
        confirming = analyzer(template, calendar,
            declaration(candles, calendar, candles[-1].interval_end, cut),
            tuple(declaration(candles, calendar, candles[n-1].interval_end,
                              candles[n-1].source_available_at) for n in needed))
        # Shift the real 1M exchange so its first sweep occurs after the 15M
        # bootstrap and its publication precedes the protective breach.
        template, raw = sealed(reset=True)
        base = BASE-timedelta(minutes=50)
        bars = [replace(c, interval_start=base+timedelta(minutes=i),
                        interval_end=base+timedelta(minutes=i+1),
                        source_available_at=base+timedelta(minutes=i+1))
                for i,c in enumerate(raw)]
        formed_at = bars[-1].interval_end
        breach_at = breach.source_available_at
        recovery_at = candles[58].source_available_at
        for i in range(len(bars), (cut-base)//timedelta(minutes=1)):
            a = base+timedelta(minutes=i)
            bars.append(replace(bar(500+i,"110","111","109","110"), interval_start=a,
                                interval_end=a+timedelta(minutes=1), source_available_at=a+timedelta(minutes=1)))
        calendar = ExplicitSessionCalendar("scalp-breach-execution",base,cut,(OpenInterval(base,cut),))
        execution = analyzer(template, calendar, declaration(bars,calendar,cut,cut),
                            tuple(declaration(bars,calendar,t,t) for t in (formed_at,breach_at,recovery_at)))
        exchange, = execution.process_batch(bars).exchanges
        four, four_bars = context_replay("4h","bullish",
            origin=BASE-timedelta(hours=68*4),coverage_end=cut,price_shift=2)
        four.process_batch(four_bars)
        for candle in candles[:57]:
            confirming.process_candle(candle)
        evaluator = ScalpEvaluator()
        before = evaluator.as_of(formed_at, exchange.exchange_id, execution, four, confirming)
        self.assertEqual("HTF_CONTEXT_ALIGNMENT", before.status)
        frozen, = confirming.bootstrap.result().controls
        self.assertEqual(frozen.control_id, before.frozen_controller_ids[1])
        self.assertLess(frozen.available_at, before.first_sweep_at)
        self.assertLess(breach.low, Decimal(frozen.low_ticks)/100)
        for candle in candles[57:59]:
            confirming.process_candle(candle)
        self.assertGreater(candles[58].close, Decimal(frozen.low_ticks)/100)
        for at in (breach_at, recovery_at):
            result = evaluator.as_of(at, exchange.exchange_id, execution, four, confirming)
            self.assertEqual("CONTEXT_LIFECYCLE_BLOCKED", result.status)
        for candle in candles[59:]:
            confirming.process_candle(candle)
        successor, = confirming.result().exchanges
        self.assertEqual(frozen.control_id, successor.parent_controller_id)
        self.assertIn(frozen.control_id, confirming.as_of(cut).replaced_controller_ids)
        self.assertEqual("CONTEXT_LIFECYCLE_BLOCKED", evaluator.as_of(
            cut, exchange.exchange_id, execution, four, confirming).status)
        self.assertEqual(before, evaluator.as_of(
            formed_at, exchange.exchange_id, execution, four, confirming))
