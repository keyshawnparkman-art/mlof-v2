from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

from mlof_v2.calendars import ExplicitSessionCalendar, OpenInterval
from mlof_v2.models import Candle, InstrumentSpec, SourceProvenance
from mlof_v2.structure import (PivotSide, SubsequentController, SubsequentExchangeAnalyzer,
                                SubsequentRange, SubsequentStatus)
from mlof_v2.structure import subsequent as module
from mlof_v2.structure.subsequent import _Candidate
from mlof_v2.structure.coverage import ReplayInput, SourceSeal, interval_digest, intervals
from mlof_v2.structure.bootstrap import BootstrapAnalyzer, BootstrapRules
from mlof_v2.structure.models import CandleEvidence, LiquidityPool
from tests.test_bootstrap import fixture

UTC = timezone.utc
BASE = datetime(2024, 1, 1, tzinfo=UTC)
KEY = ("TESTUSD", "1m")
SPEC_VERSION = "0.3.3"


def bar(i: int, o: str, h: str, l: str, c: str) -> Candle:
    start = BASE + timedelta(minutes=i)
    end = start + timedelta(minutes=1)
    provenance = SourceProvenance("synthetic://subsequent", "c" * 64, i + 1,
                                  f"{i + 1:064x}", f"sub-{i}", (("synthetic", "true"),))
    return Candle("TESTUSD", "1m", start, end, end, Decimal(o), Decimal(h), Decimal(l), Decimal(c), provenance)


def make_analyzer():
    _, bootstrap = fixture("bullish")
    return SubsequentExchangeAnalyzer(bootstrap, BootstrapRules("bootstrap-test-v1"),
                                     (InstrumentSpec("TESTUSD", Decimal("0.01")),),
                                     spec_version=SPEC_VERSION)


def chronological_replay(successor_direction: str, *, deeper_sweep: bool = False,
                          cross_boundary: bool = False, batch: bool = False,
                          stop_before_sweep: bool = False, delayed_fvg: bool = False,
                          multiple_deeper_sweeps: bool = False,
                          spec_version: str = SPEC_VERSION):
    """Build a complete bootstrap-plus-subsequent synthetic stream through public APIs."""
    bootstrap_direction = "bearish" if successor_direction == "bullish" else "bullish"
    initial_bars, template = fixture(bootstrap_direction)
    calendar_end = BASE + timedelta(minutes=80)
    calendar = ExplicitSessionCalendar("subsequent-e2e", BASE, calendar_end,
                                       (OpenInterval(BASE, calendar_end),))
    bootstrap = BootstrapAnalyzer(template.config, template.rules,
                                  (InstrumentSpec("TESTUSD", Decimal("0.01")),), calendar)
    bars = list(initial_bars)

    def add(i, o, h, l, c):
        current = bar(i, str(o), str(h), str(l), str(c))
        bars.append(current)
        return current

    if successor_direction == "bullish":
        # A new confirmed sell-side pool at 100.00, safely above the old 95.00 range floor.
        add(57, 102, 103, 101, 102)
        add(58, 102, 103, 101, 102)
        add(59, 102, 103, 101, 102)
        add(60, 101, 102, 100, 101)
        add(61, 102, 103, 101, 102)
        add(62, 102, 103, 101, 102)
        add(63, 101, 103, 101, 102)
        add(64, 102, 103, 101, 102)
        add(65, 101, 101.5, 99.99, 100.2)
        if multiple_deeper_sweeps:
            add(66, 100.5, 101, 99.98, 100.2)
            add(67, 100.4, 101, 99.97, 100.2)
            add(68, 102.8, 115, 102.5, 114)
        elif deeper_sweep:
            add(66, 100.5, 101, 94.98 if cross_boundary else 99.98, 100.2)
        else:
            add(66, 101, 102, 100.5, 101)
        if not multiple_deeper_sweeps:
            add(67, 102.8, 115, 101.4 if delayed_fvg else 102.5, 114)
            if delayed_fvg:
                add(68, 110, 117, 108, 116)
    else:
        # A new confirmed buy-side pool at 103.00, safely below the old 105.00 range ceiling.
        add(57, 101, 102, 100, 101)
        add(58, 101, 102, 100, 101)
        add(59, 101, 102, 100, 101)
        add(60, 102, 103, 101, 102)
        add(61, 101, 102, 100, 101)
        add(62, 101, 102, 100, 101)
        add(63, 101, 102, 100, 101)
        add(64, 101, 102, 100, 101)
        add(65, 102, 103.01, 101.5, 102.5)
        if multiple_deeper_sweeps:
            add(66, 102.5, 103.02, 101, 102)
            add(67, 102.5, 103.03, 101, 102)
            add(68, 99, 99.5, 85, 85.5)
        elif deeper_sweep:
            add(66, 102.5, 105.02 if cross_boundary else 103.02, 101, 102)
        else:
            add(66, 101.5, 102, 100, 101)
        if not multiple_deeper_sweeps:
            add(67, 99, 102 if delayed_fvg else 99.5, 85, 85.5)
            if delayed_fvg:
                add(68, 90, 95, 85, 86)

    replay = None
    if spec_version == "0.3.4":
        end = bars[-1].interval_end
        seal = SourceSeal("synthetic", *KEY, "UTC", "close", BASE, end, end, calendar.calendar_id,
                          interval_digest(intervals(calendar, *KEY, BASE, end)),
                          "SEALED_COMPLETE_THROUGH_BOUNDARY", "synthetic-seal")
        replay = ReplayInput.locally_sealed(seal, tuple(bars))
    engine = SubsequentExchangeAnalyzer(
        bootstrap, BootstrapRules("bootstrap-test-v1"),
        (InstrumentSpec("TESTUSD", Decimal("0.01")),), spec_version=spec_version, replay=replay)
    if stop_before_sweep:
        for candle in bars[:65]:
            engine.process_candle(candle)
    elif batch:
        engine.process_batch(bars)
    else:
        for candle in bars:
            engine.process_candle(candle)
    return engine, bars


class SubsequentConfirmationTests(unittest.TestCase):
    def _confirmation_state(self, direction: str):
        analyzer = make_analyzer()
        if direction == "bullish":
            bars = [bar(i, "100", "101", "99", "100") for i in range(16)]
            bars[0] = bar(0, "99", "100", "90", "99")
            protective_side, protective_level = PivotSide.LOW, 90
            break_side, break_level = PivotSide.HIGH, 120
            controller_direction = "bearish"
            low_id, high_id = "next-protective", "linked-break-ref"
            low_ticks, high_ticks = 9000, 12000
        else:
            bars = [bar(i, "100", "101", "99", "100") for i in range(16)]
            bars[0] = bar(0, "101", "110", "100", "101")
            protective_side, protective_level = PivotSide.HIGH, 110
            break_side, break_level = PivotSide.LOW, 80
            controller_direction = "bullish"
            low_id, high_id = "linked-break-ref", "next-protective"
            low_ticks, high_ticks = 8000, 11000
        evidence = tuple(CandleEvidence(c.provenance.record_id, c.interval_start, c.interval_end,
                                        c.source_available_at, c.provenance) for c in bars[:2])
        protective = SimpleNamespace(cluster_id="next-protective", swing_event_id="sweep-swing", side=protective_side,
            level_ticks=protective_level * 100, instrument="TESTUSD", timeframe="1m", anchor_candle_id=bars[0].provenance.record_id,
            member_candle_ids=(bars[0].provenance.record_id,), available_at=BASE,
            comparison_evidence=evidence)
        break_ref = SimpleNamespace(cluster_id="linked-break-ref", swing_event_id="break-swing", side=break_side,
            level_ticks=break_level * 100, instrument="TESTUSD", timeframe="1m", available_at=BASE, comparison_evidence=evidence)
        analyzer._structural.add(protective.cluster_id)
        analyzer._structural_promotion_sources[protective.cluster_id] = bars[0].provenance.record_id
        old = SubsequentController("old-control", "", "old-bootstrap-exchange", "old-manipulation",
            controller_direction, "TESTUSD", "1m", break_ref.cluster_id, break_ref.cluster_id, BASE,
            "READY", "bootstrap-test-v1", SPEC_VERSION)
        analyzer._active[KEY] = old
        analyzer._controllers[old.control_id] = old
        analyzer._controller_ranges[old.control_id] = SubsequentRange(
            "old-range", old.control_id, low_id, high_id, low_ticks, high_ticks, BASE,
            "bootstrap-test-v1", SPEC_VERSION, instrument="TESTUSD", timeframe="1m", direction=controller_direction)
        analyzer._candles[KEY] = bars
        pool = LiquidityPool("swept-pool", "sweep-swing", "TESTUSD", "1m",
            PivotSide.LOW if direction == "bullish" else PivotSide.HIGH,
            9500 if direction == "bullish" else 10500, bars[0].provenance.record_id,
            (bars[0].provenance.record_id,), BASE, 1, "bootstrap-test-v1")
        sweep_index = 2
        extreme_ticks = 9900 if direction == "bullish" else 10100
        candidate = _Candidate("parent", "parent:candidate:v1", 1, pool, direction,
            old.control_id, [bars[sweep_index].provenance.record_id], extreme_ticks, sweep_index, sweep_index,
            3, bars[3].provenance.record_id, 5, "old-range", protective.cluster_id,
            protective.level_ticks, 10000)
        analyzer._id_payloads[candidate.parent_id] = ("parent payload",)
        structure = SimpleNamespace(swings=(protective, break_ref))
        return analyzer, old, candidate, bars, protective, break_ref, structure

    def test_confirmed_bullish_exchange_creates_linked_successor_and_frozen_range(self):
        analyzer, old, candidate, bars, protective, break_ref, structure = self._confirmation_state("bullish")
        fvg = (5, tuple(b.provenance.record_id for b in bars[3:6]))
        analyzer._confirm(KEY, old, candidate, bars[5], bars, fvg, break_ref, structure)
        result = analyzer.result()
        self.assertEqual(1, len(result.exchanges))
        self.assertEqual("bullish", result.exchanges[0].direction)
        successor = next(c for c in result.controllers if c.control_id != old.control_id)
        self.assertEqual(old.control_id, successor.parent_controller_id)
        self.assertEqual("READY", successor.readiness)
        frozen = next(r for r in result.ranges if r.control_id == successor.control_id)
        self.assertEqual((9000, 12000), (frozen.low_ticks, frozen.high_ticks))
        self.assertEqual(break_ref.cluster_id, result.exchanges[0].break_reference_swing_id)
        self.assertEqual(0, len([x for x in result.events if "ORDER" in x.status.value]))

    def test_confirmed_bearish_exchange_mirrors_bullish_case(self):
        analyzer, old, candidate, bars, protective, break_ref, structure = self._confirmation_state("bearish")
        fvg = (5, tuple(b.provenance.record_id for b in bars[3:6]))
        analyzer._confirm(KEY, old, candidate, bars[5], bars, fvg, break_ref, structure)
        result = analyzer.result()
        self.assertEqual("bearish", result.exchanges[0].direction)
        self.assertTrue(all(r.low_ticks < r.high_ticks for r in result.ranges))
        self.assertIn(old.control_id, analyzer._replaced)

    def test_unrelated_internal_swing_cannot_substitute_for_linked_boundary(self):
        analyzer, old, candidate, bars, protective, break_ref, structure = self._confirmation_state("bullish")
        other = replace_swing(protective, 'other-protective')
        other.swing_event_id = 'other-origin'
        origin = replace_swing(protective, 'sweep-origin')
        analyzer._structural.add('other-protective')
        two = SimpleNamespace(swings=(other, origin, break_ref))
        fvg = (5, tuple(b.provenance.record_id for b in bars[3:6]))
        analyzer._confirm(KEY, old, candidate, bars[5], bars, fvg, break_ref, two)
        current = analyzer._active[KEY]
        self.assertEqual("NOT_READY", current.readiness)
        self.assertIsNone(current.protective_swing_id)
        self.assertEqual([], list(analyzer._ranges.values()))
        self.assertIn(old.control_id, analyzer._replaced)

    def test_unready_retained_anchor_uses_formation_deadline_and_requires_range(self):
        for direction in ("bullish", "bearish"):
            with self.subTest(direction=direction):
                analyzer, old, candidate, bars, protective, break_ref, structure = self._confirmation_state(direction)
                fvg = (5, tuple(b.provenance.record_id for b in bars[3:6]))
                analyzer._confirm(KEY, old, candidate, bars[5], bars, fvg, break_ref, structure)
                formed = analyzer._active[KEY]
                unready = replace(formed, state="NOT_READY", readiness="NOT_READY")
                analyzer._controller_ranges.pop(formed.control_id)
                new_direction = "bearish" if direction == "bullish" else "bullish"
                self.assertEqual("", analyzer._retained_anchor_error(unready, bars[8], structure))
                self.assertIn("no uniquely linked frozen range",
                              analyzer._frozen_boundary(unready, new_direction, bars[8], structure)[2])
                for invalid in (None, "another-anchor"):
                    self.assertIn("formation association", analyzer._retained_anchor_error(
                        replace(unready, protective_swing_id=invalid), bars[8], structure))
                self.assertIn("not causally available",
                              analyzer._retained_anchor_error(unready, bars[4], structure))
                analyzer._structural_promotion_sources[protective.cluster_id] = bars[4].provenance.record_id
                self.assertIn("original formation deadline",
                              analyzer._retained_anchor_error(unready, bars[8], structure))
                self.assertIn(old.control_id, analyzer._replaced)

    def test_formation_readiness_is_immutable_while_unready_successor_is_observed(self):
        for direction in ("bullish", "bearish"):
            with self.subTest(direction=direction):
                analyzer, old, candidate, bars, protective, break_ref, structure = self._confirmation_state(direction)
                trigger = bars[5]
                fvg = (5, tuple(b.provenance.record_id for b in bars[3:6]))
                # Formation evidence exists but the designated anchor missed its deadline.
                analyzer._structural_promotion_sources[protective.cluster_id] = trigger.provenance.record_id
                analyzer._confirm(KEY, old, candidate, trigger, bars, fvg, break_ref, structure)
                frozen = analyzer._active[KEY]
                self.assertEqual("NOT_READY", frozen.readiness)
                self.assertIsNone(frozen.protective_swing_id)
                self.assertFalse(frozen.tradable)
                before = analyzer.as_of(trigger.source_available_at - timedelta(microseconds=1))
                at = analyzer.as_of(trigger.source_available_at)
                self.assertIn(old.control_id, before.active_controller_ids)
                self.assertNotIn(old.control_id, at.active_controller_ids)
                self.assertIn(old.control_id, at.replaced_controller_ids)
                self.assertNotIn(frozen.control_id, at.active_controller_ids)
                self.assertFalse(at.ranges)

                # Supply later completed bars through the normal observation path.
                del bars[6:]
                for index in range(6, 24):
                    analyzer.process_candle(bar(index, "100", "101", "99", "100"))
                after = analyzer.as_of(bars[-1].source_available_at)
                self.assertEqual(frozen, analyzer._active[KEY])
                self.assertIn(frozen, after.controllers)
                self.assertEqual(1, len(after.exchanges))
                self.assertFalse(after.ranges)
                self.assertIn(old.control_id, after.replaced_controller_ids)
                self.assertEqual(before, analyzer.as_of(before.as_of_utc))
                self.assertEqual(at, analyzer.as_of(at.as_of_utc))
                self.assertGreater(len(analyzer.bootstrap.structural.result().events), 0)

    def test_late_protective_reference_is_not_used_retroactively(self):
        analyzer, old, candidate, bars, protective, break_ref, structure = self._confirmation_state("bullish")
        late = replace_swing(protective, "late", available_at=bars[6].interval_end)
        analyzer._structural.add("late")
        delayed = SimpleNamespace(swings=(late, break_ref))
        fvg = (5, tuple(b.provenance.record_id for b in bars[3:6]))
        analyzer._confirm(KEY, old, candidate, bars[5], bars, fvg, break_ref, delayed)
        self.assertEqual("NOT_READY", analyzer._active[KEY].readiness)

    def test_current_exchange_cannot_promote_its_own_protective_swing(self):
        analyzer, old, candidate, bars, protective, break_ref, structure = self._confirmation_state("bullish")
        analyzer._structural_promotion_sources[protective.cluster_id] = bars[5].provenance.record_id
        fvg = (5, tuple(b.provenance.record_id for b in bars[3:6]))
        analyzer._confirm(KEY, old, candidate, bars[5], bars, fvg, break_ref, structure)
        self.assertEqual("NOT_READY", analyzer._active[KEY].readiness)
        self.assertEqual([], list(analyzer._ranges.values()))

    def test_invalid_or_crossed_reference_sides_do_not_create_a_range(self):
        analyzer, old, candidate, bars, protective, break_ref, structure = self._confirmation_state("bullish")
        invalid_reference = SimpleNamespace(cluster_id="wrong-side-reference", swing_event_id="break-swing", side=PivotSide.LOW, level_ticks=8000, instrument="TESTUSD", timeframe="1m", available_at=BASE, comparison_evidence=())
        invalid_reference.side = PivotSide.LOW
        invalid_reference.level_ticks = 8000
        invalid = SimpleNamespace(swings=(protective, invalid_reference))
        fvg = (5, tuple(b.provenance.record_id for b in bars[3:6]))
        analyzer._confirm(KEY, old, candidate, bars[5], bars, fvg, invalid_reference, invalid)
        self.assertEqual("NOT_READY", analyzer._active[KEY].readiness)
        self.assertEqual([], list(analyzer._ranges.values()))

    def test_range_id_collision_fails_without_partial_promotion(self):
        analyzer, old, candidate, bars, protective, break_ref, structure = self._confirmation_state("bullish")
        analyzer._id_payloads["forced-range-collision"] = ("different evidence",)
        original_hash = module._canonical_hash
        def forced(prefix, payload):
            if prefix == "subsequent_range_v1:":
                return "forced-range-collision"
            return original_hash(prefix, payload)
        fvg = (5, tuple(b.provenance.record_id for b in bars[3:6]))
        with patch.object(module, "_canonical_hash", side_effect=forced):
            analyzer._confirm(KEY, old, candidate, bars[5], bars, fvg, break_ref, structure)
        self.assertIs(analyzer._active[KEY], old)
        self.assertEqual([], list(analyzer._exchanges.values()))
        self.assertEqual([], list(analyzer._ranges.values()))
        self.assertTrue(any(e.status is SubsequentStatus.REJECTED for e in analyzer._events.values()))

    def test_controller_and_range_records_are_immutable(self):
        analyzer, old, candidate, bars, protective, break_ref, structure = self._confirmation_state("bullish")
        fvg = (5, tuple(b.provenance.record_id for b in bars[3:6]))
        analyzer._confirm(KEY, old, candidate, bars[5], bars, fvg, break_ref, structure)
        with self.assertRaises(FrozenInstanceError):
            analyzer.result().ranges[0].low_ticks = 0


def replace_swing(swing, cluster_id, *, available_at=None):
    return SimpleNamespace(cluster_id=cluster_id, swing_event_id=swing.swing_event_id, side=swing.side, level_ticks=swing.level_ticks, instrument="TESTUSD", timeframe="1m",
        anchor_candle_id=swing.anchor_candle_id, member_candle_ids=swing.member_candle_ids,
        available_at=available_at or swing.available_at, comparison_evidence=swing.comparison_evidence)


class SubsequentCandidateAndReplayTests(unittest.TestCase):
    def test_delayed_fvg_confirmation_and_replacement_are_not_visible_early(self):
        for direction in ("bullish", "bearish"):
            with self.subTest(direction=direction):
                engine, bars = chronological_replay(direction, delayed_fvg=True)
                result = engine.result()
                self.assertEqual(1, len(result.exchanges))
                exchange = result.exchanges[0]
                confirmation = next(e for e in result.events
                                    if e.status is SubsequentStatus.EXCHANGE_CONFIRMED)
                replacement = next(e for e in result.events
                                   if e.status is SubsequentStatus.CONTROLLER_REPLACED)
                expected_available = bars[68].source_available_at
                self.assertEqual(expected_available, exchange.available_at)
                self.assertEqual(expected_available, confirmation.available_at)
                self.assertEqual(expected_available, replacement.available_at)
                self.assertIn(bars[68].provenance.record_id, confirmation.source_candle_ids)
                self.assertIn(bars[68].provenance.record_id, replacement.source_candle_ids)

                before_fvg = engine.as_of(bars[68].interval_start)
                self.assertEqual((), before_fvg.exchanges)
                self.assertFalse(any(e.status in (SubsequentStatus.EXCHANGE_CONFIRMED,
                                                  SubsequentStatus.CONTROLLER_REPLACED)
                                     for e in before_fvg.events))
                self.assertFalse(any(c.exchange_id == exchange.exchange_id
                                     for c in before_fvg.controllers))
                successor = next(c for c in result.controllers if c.exchange_id == exchange.exchange_id)
                self.assertFalse(any(r.control_id == successor.control_id for r in before_fvg.ranges))
                self.assertNotIn(exchange.parent_controller_id, before_fvg.replaced_controller_ids)

                after_fvg = engine.as_of(expected_available)
                self.assertIn(exchange.exchange_id, {x.exchange_id for x in after_fvg.exchanges})
                self.assertIn(confirmation.event_id, {x.event_id for x in after_fvg.events})
                self.assertIn(replacement.event_id, {x.event_id for x in after_fvg.events})
                self.assertIn(exchange.parent_controller_id, after_fvg.replaced_controller_ids)
                successor = next(c for c in after_fvg.controllers if c.exchange_id == exchange.exchange_id)
                self.assertEqual("READY", successor.readiness)

    def test_spec_version_is_explicit_on_m3b1_records_and_changes_canonical_ids(self):
        current, _ = chronological_replay("bullish")
        current_result = current.result()
        self.assertTrue(all(x.spec_version == SPEC_VERSION for x in current_result.events))
        self.assertTrue(all(x.spec_version == SPEC_VERSION for x in current_result.manipulations))
        self.assertTrue(all(x.spec_version == SPEC_VERSION for x in current_result.exchanges))
        self.assertTrue(all(x.spec_version == SPEC_VERSION for x in current_result.controllers))
        self.assertTrue(all(x.spec_version == SPEC_VERSION for x in current_result.ranges))
        self.assertTrue(all(x.event_id.startswith("subsequent_state_v2:")
                            for x in current_result.events))

        explicit_other_version, _ = chronological_replay("bullish", spec_version="0.3.4")
        other_result = explicit_other_version.result()
        self.assertTrue(all(x.spec_version == "0.3.4" for x in other_result.events))
        self.assertNotEqual(current_result.exchanges[0].exchange_id, other_result.exchanges[0].exchange_id)
        self.assertNotEqual(current_result.events[0].event_id, other_result.events[0].event_id)
        _, bootstrap = fixture("bullish")
        with self.assertRaises(TypeError):
            SubsequentExchangeAnalyzer(bootstrap, BootstrapRules("bootstrap-test-v1"),
                                       (InstrumentSpec("TESTUSD", Decimal("0.01")),))

    def test_multiple_consecutive_deeper_resets_preserve_ticks_and_lineage(self):
        for direction in ("bullish", "bearish"):
            with self.subTest(direction=direction):
                incremental, _ = chronological_replay(direction, multiple_deeper_sweeps=True)
                batch, _ = chronological_replay(direction, multiple_deeper_sweeps=True, batch=True)
                result = incremental.result()
                self.assertEqual(result, batch.result())
                resets = [e.candidate_reset for e in result.events
                          if e.status is SubsequentStatus.CANDIDATE_RESET]
                self.assertEqual(2, len(resets))
                self.assertEqual([(1, 2), (2, 3)],
                                 [(x.canceled_version, x.replacement_version) for x in resets])
                self.assertEqual(["v1", "v2"],
                                 [x.canceled_candidate_id.rsplit(":", 1)[-1] for x in resets])
                self.assertEqual(["v2", "v3"],
                                 [x.replacement_candidate_id.rsplit(":", 1)[-1] for x in resets])
                self.assertEqual([SPEC_VERSION, SPEC_VERSION], [x.spec_version for x in resets])
                self.assertTrue(all(type(x.new_extreme_ticks) is int for x in resets))
                self.assertTrue(all(x.source_evidence.record_id == x.triggering_candle_id
                                    and x.source_evidence.provenance.record_id == x.triggering_candle_id
                                    and x.source_available_at == x.source_evidence.source_available_at
                                    and x.interval_start == x.source_evidence.interval_start
                                    and x.interval_end == x.source_evidence.interval_end
                                    for x in resets))
                self.assertEqual((9998, 9997) if direction == "bullish" else (10302, 10303),
                                 tuple(x.new_extreme_ticks for x in resets))

    def test_lifecycle_event_duplicate_id_with_different_evidence_fails_without_overwrite(self):
        analyzer = make_analyzer()
        candle = bar(30, "100", "101", "99", "100")
        evidence = CandleEvidence(candle.provenance.record_id, candle.interval_start, candle.interval_end,
                                  candle.source_available_at, candle.provenance)
        analyzer._emit(SubsequentStatus.OBSERVING, candle, None, ("same-related",), "first", (evidence,))
        first = next(iter(analyzer._events.values()))
        changed_provenance = replace(candle.provenance, source_path="synthetic://different-source")
        conflicting = replace(evidence, provenance=changed_provenance)
        with self.assertRaisesRegex(ValueError, "collision"):
            analyzer._emit(SubsequentStatus.OBSERVING, candle, None, ("same-related",),
                           "conflicting evidence", (conflicting,))
        self.assertEqual(1, len(analyzer._events))
        self.assertIs(first, next(iter(analyzer._events.values())))

    def test_boundary_availability_equal_to_sweep_interval_start_is_causally_eligible(self):
        engine, bars = chronological_replay("bullish", stop_before_sweep=True)
        active = engine._active[KEY]
        frozen = engine._controller_ranges[active.control_id]
        promotion_id = engine._structural_promotion_sources[frozen.low_cluster_id]
        local_bars = engine._candles[KEY]
        index = next(i for i, b in enumerate(local_bars) if b.provenance.record_id == promotion_id)
        local_bars[index] = replace(local_bars[index], source_available_at=bars[65].interval_start)
        engine.process_candle(bars[65])
        self.assertIn(KEY, engine._candidate)
        self.assertTrue(any(e.status is SubsequentStatus.SWEEP for e in engine.result().events))

    def test_chronological_bullish_pipeline_creates_ready_successor(self):
        engine, bars = chronological_replay("bullish")
        result = engine.result()
        self.assertEqual(1, len(result.exchanges))
        exchange = result.exchanges[0]
        self.assertEqual("bullish", exchange.direction)
        old = next(c for c in result.controllers if c.control_id == exchange.parent_controller_id)
        successor = next(c for c in result.controllers if c.control_id != old.control_id)
        frozen = next(r for r in result.ranges if r.control_id == successor.control_id)
        manipulation = next(m for m in result.manipulations if m.manipulation_id == exchange.manipulation_id)
        prior_range = next(r for r in engine.bootstrap.result().controls if r.control_id == old.control_id)
        self.assertEqual("READY", successor.readiness)
        self.assertEqual(prior_range.low_cluster_id, successor.protective_swing_id)
        self.assertEqual(prior_range.high_cluster_id, exchange.break_reference_swing_id)
        self.assertEqual(prior_range.range_id, manipulation.frozen_control_range_id)
        self.assertEqual(prior_range.low_cluster_id, manipulation.successor_protective_swing_id)
        self.assertLess(frozen.low_ticks, manipulation.manipulation_extreme_ticks)
        selected = next(s for s in result.bootstrap.structural.swings
                        if s.cluster_id == prior_range.low_cluster_id)
        self.assertNotEqual(selected.level_ticks, manipulation.manipulation_extreme_ticks)
        self.assertNotIn(manipulation.sweep_candle_ids[-1], selected.member_candle_ids)
        self.assertEqual((bars[65].source_available_at, bars[67].source_available_at),
                         (manipulation.available_at, exchange.available_at))
        promotion = next(e for e in result.events if e.status is SubsequentStatus.STRUCTURAL_PROMOTION
                         and prior_range.low_cluster_id in e.related_ids)
        self.assertLessEqual(promotion.available_at, bars[65].interval_start)
        sweep_event = next(e for e in result.events if e.status is SubsequentStatus.SWEEP
                           and prior_range.low_cluster_id in e.related_ids)
        self.assertIn(prior_range.range_id, sweep_event.related_ids)
        self.assertIn(old.control_id, result.replaced_controller_ids)
        self.assertFalse(exchange.tradable or successor.tradable or frozen.tradable)

    def test_chronological_bearish_pipeline_creates_ready_successor(self):
        engine, bars = chronological_replay("bearish")
        result = engine.result()
        self.assertEqual(1, len(result.exchanges))
        exchange = result.exchanges[0]
        self.assertEqual("bearish", exchange.direction)
        old = next(c for c in result.controllers if c.control_id == exchange.parent_controller_id)
        successor = next(c for c in result.controllers if c.control_id != old.control_id)
        frozen = next(r for r in result.ranges if r.control_id == successor.control_id)
        manipulation = next(m for m in result.manipulations if m.manipulation_id == exchange.manipulation_id)
        prior_range = next(r for r in engine.bootstrap.result().controls if r.control_id == old.control_id)
        self.assertEqual("READY", successor.readiness)
        self.assertEqual(prior_range.high_cluster_id, successor.protective_swing_id)
        self.assertEqual(prior_range.low_cluster_id, exchange.break_reference_swing_id)
        self.assertEqual(prior_range.range_id, manipulation.frozen_control_range_id)
        self.assertEqual(prior_range.high_cluster_id, manipulation.successor_protective_swing_id)
        self.assertGreater(frozen.high_ticks, manipulation.manipulation_extreme_ticks)
        selected = next(s for s in result.bootstrap.structural.swings
                        if s.cluster_id == prior_range.high_cluster_id)
        self.assertNotEqual(selected.level_ticks, manipulation.manipulation_extreme_ticks)
        self.assertNotIn(manipulation.sweep_candle_ids[-1], selected.member_candle_ids)
        self.assertEqual((bars[65].source_available_at, bars[67].source_available_at),
                         (manipulation.available_at, exchange.available_at))
        promotion = next(e for e in result.events if e.status is SubsequentStatus.STRUCTURAL_PROMOTION
                         and prior_range.high_cluster_id in e.related_ids)
        self.assertLessEqual(promotion.available_at, bars[65].interval_start)
        sweep_event = next(e for e in result.events if e.status is SubsequentStatus.SWEEP
                           and prior_range.high_cluster_id in e.related_ids)
        self.assertIn(prior_range.range_id, sweep_event.related_ids)
        self.assertIn(old.control_id, result.replaced_controller_ids)

    def test_deeper_sweep_within_frozen_boundary_revises_then_reaches_ready(self):
        for direction in ("bullish", "bearish"):
            with self.subTest(direction=direction):
                engine, _ = chronological_replay(direction, deeper_sweep=True)
                result = engine.result()
                self.assertEqual(1, len(result.exchanges))
                successor = next(c for c in result.controllers
                                 if c.exchange_id == result.exchanges[0].exchange_id)
                self.assertEqual("READY", successor.readiness)
                resets = [e for e in result.events if e.status is SubsequentStatus.CANDIDATE_RESET]
                self.assertEqual(1, len(resets))
                self.assertIn(":candidate:v2", " ".join(resets[0].related_ids))
                manipulation = next(m for m in result.manipulations if m.candidate_id.endswith(":candidate:v2"))
                old = next(c for c in result.controllers if c.control_id == manipulation.parent_controller_id)
                original_range = next(r for r in engine.bootstrap.result().controls if r.control_id == old.control_id)
                expected_boundary = (original_range.low_cluster_id if direction == "bullish"
                                     else original_range.high_cluster_id)
                self.assertEqual(original_range.range_id, manipulation.frozen_control_range_id)
                self.assertEqual(expected_boundary, manipulation.successor_protective_swing_id)
                self.assertIn(original_range.range_id, resets[0].related_ids)
                self.assertIn(expected_boundary, resets[0].related_ids)

    def test_deeper_sweep_crossing_frozen_boundary_fails_closed(self):
        for direction in ("bullish", "bearish"):
            with self.subTest(direction=direction):
                engine, _ = chronological_replay(direction, deeper_sweep=True, cross_boundary=True)
                result = engine.result()
                self.assertEqual([], list(result.exchanges))
                self.assertEqual([], list(result.ranges))
                self.assertTrue(any(e.status is SubsequentStatus.NOT_READY
                                    and "containment" in e.reason for e in result.events))

    def test_missing_independent_structural_boundary_blocks_first_sweep(self):
        engine, bars = chronological_replay("bullish", stop_before_sweep=True)
        active = engine._active[KEY]
        frozen = engine._controller_ranges[active.control_id]
        engine._structural.discard(frozen.low_cluster_id)
        engine.process_candle(bars[65])
        self.assertNotIn(KEY, engine._candidate)
        self.assertFalse(engine.result().exchanges)
        self.assertTrue(any(e.status is SubsequentStatus.NOT_READY
                            and "independent STRUCTURAL" in e.reason for e in engine.result().events))

    def test_late_frozen_boundary_range_fails_before_first_sweep(self):
        engine, bars = chronological_replay("bullish", stop_before_sweep=True)
        active = engine._active[KEY]
        original = engine._controller_ranges[active.control_id]
        engine._controller_ranges[active.control_id] = replace(original, available_at=bars[65].interval_end)
        engine.process_candle(bars[65])
        self.assertNotIn(KEY, engine._candidate)
        self.assertTrue(any(e.status is SubsequentStatus.NOT_READY
                            and "range was unavailable" in e.reason for e in engine.result().events))

    def test_batch_and_incremental_full_subsequent_replays_match(self):
        for direction in ("bullish", "bearish"):
            with self.subTest(direction=direction):
                incremental, _ = chronological_replay(direction, deeper_sweep=True)
                batch, _ = chronological_replay(direction, deeper_sweep=True, batch=True)
                self.assertEqual(incremental.result(), batch.result())

    def _seed_candidate(self, direction="bullish"):
        analyzer = make_analyzer()
        bars = [bar(i, "100", "101", "99", "100") for i in range(20)]
        if direction == "bullish":
            old_direction = "bearish"
            pool = LiquidityPool("pool", "pool-swing", "TESTUSD", "1m", PivotSide.LOW, 10000,
                bars[0].provenance.record_id, (bars[0].provenance.record_id,), BASE, 1, "bootstrap-test-v1")
            ref = SimpleNamespace(cluster_id="ref", side=PivotSide.HIGH, level_ticks=12000, instrument="TESTUSD", timeframe="1m",
                available_at=BASE, comparison_evidence=())
            selected = SimpleNamespace(cluster_id="next-boundary", swing_event_id="next-swing", side=PivotSide.LOW,
                level_ticks=9000, instrument="TESTUSD", timeframe="1m", anchor_candle_id=bars[0].provenance.record_id,
                member_candle_ids=(bars[0].provenance.record_id,), available_at=BASE, comparison_evidence=())
            low_id, high_id, low_ticks, high_ticks = "next-boundary", "ref", 9000, 12000
            extreme_ticks = 9800
        else:
            old_direction = "bullish"
            pool = LiquidityPool("high-pool", "pool-swing", "TESTUSD", "1m", PivotSide.HIGH, 10000,
                bars[0].provenance.record_id, (bars[0].provenance.record_id,), BASE, 1, "bootstrap-test-v1")
            ref = SimpleNamespace(cluster_id="ref", side=PivotSide.LOW, level_ticks=9000, instrument="TESTUSD", timeframe="1m",
                available_at=BASE, comparison_evidence=())
            selected = SimpleNamespace(cluster_id="next-boundary", swing_event_id="next-swing", side=PivotSide.HIGH,
                level_ticks=11000, instrument="TESTUSD", timeframe="1m", anchor_candle_id=bars[0].provenance.record_id,
                member_candle_ids=(bars[0].provenance.record_id,), available_at=BASE, comparison_evidence=())
            low_id, high_id, low_ticks, high_ticks = "ref", "next-boundary", 9000, 11000
            extreme_ticks = 10200
        old = SubsequentController("active", "", "boot", "manip", old_direction, "TESTUSD", "1m",
            "ref", "ref", BASE, "READY", "bootstrap-test-v1", SPEC_VERSION)
        analyzer._active[KEY] = old
        analyzer._bootstrap_control_ids.add("active")
        analyzer._candles[KEY] = bars
        candidate = _Candidate("parent", "parent:candidate:v1", 1, pool, direction, "active",
            [bars[19].provenance.record_id], extreme_ticks, 19, 19, 19, bars[12].provenance.record_id, None)
        analyzer._candidate[KEY] = candidate
        analyzer._structural.add(selected.cluster_id)
        analyzer._structural_promotion_sources[selected.cluster_id] = bars[0].provenance.record_id
        analyzer._controller_ranges[old.control_id] = SubsequentRange(
            "active-range", old.control_id, low_id, high_id, low_ticks, high_ticks, BASE,
            "bootstrap-test-v1", SPEC_VERSION, instrument="TESTUSD", timeframe="1m", direction=old_direction)
        candidate.frozen_range_id = "active-range"
        candidate.protective_swing_id = selected.cluster_id
        candidate.protective_boundary_ticks = selected.level_ticks
        candidate.reference_open_ticks = 10000
        return analyzer, candidate, bars, old, SimpleNamespace(swings=(ref, selected))

    def test_strictly_deeper_post_reclaim_sweep_increments_once_and_keeps_parent(self):
        analyzer, candidate, bars, old, structure = self._seed_candidate()
        current = bar(20, "100", "101", "97", "100.5")
        bars.append(current)
        analyzer._candles[KEY] = bars
        analyzer._advance(KEY, old, current, bars, structure,
                          (candidate.pool,))
        self.assertEqual("parent", candidate.parent_id)
        self.assertEqual(2, candidate.version)
        self.assertEqual("parent:candidate:v2", candidate.candidate_id)
        self.assertIsNotNone(candidate.reclaim_index)
        self.assertTrue(any(e.status is SubsequentStatus.CANDIDATE_RESET for e in analyzer._events.values()))

    def test_equal_or_shallower_sweep_does_not_increment_version(self):
        analyzer, candidate, bars, old, structure = self._seed_candidate()
        current = bar(20, "100", "101", "98", "100.5")
        bars.append(current)
        analyzer._candles[KEY] = bars
        analyzer._advance(KEY, old, current, bars, structure, (candidate.pool,))
        self.assertEqual(1, candidate.version)
        self.assertEqual("parent:candidate:v1", candidate.candidate_id)

    def test_internal_break_does_not_replace_active_controller(self):
        analyzer, candidate, bars, old, structure = self._seed_candidate("bearish")
        # This bar closes below an unrelated 95 level but remains above linked control low 90.
        candidate.reclaim_index = 12
        current = bar(20, "100", "101", "94", "94.5")
        bars.append(current)
        analyzer._candles[KEY] = bars
        analyzer._advance(KEY, old, current, bars, structure, ())
        self.assertIs(analyzer._active[KEY], old)
        self.assertEqual([], list(analyzer._exchanges.values()))

    def test_same_candle_reclaim_and_control_break_is_rejected(self):
        analyzer, candidate, bars, old, structure = self._seed_candidate("bearish")
        candidate.reclaim_index = None
        candidate.reclaim_id = None
        ref = SimpleNamespace(cluster_id=old.protective_swing_id, swing_event_id="break-swing",
            side=PivotSide.LOW, level_ticks=9500, instrument="TESTUSD", timeframe="1m",
            available_at=BASE, comparison_evidence=())
        selected = next(s for s in structure.swings if s.cluster_id == candidate.protective_swing_id)
        structure = SimpleNamespace(swings=(ref, selected))
        analyzer._controller_ranges[old.control_id] = SubsequentRange(
            "active-range", old.control_id, ref.cluster_id, candidate.protective_swing_id, 9500, 11000,
            BASE, "bootstrap-test-v1", SPEC_VERSION, instrument="TESTUSD", timeframe="1m", direction="bullish")
        candidate.frozen_range_id = "active-range"
        candidate.protective_boundary_ticks = 11000
        current = bar(20, "100", "105", "89", "90")
        bars.append(current)
        analyzer._candles[KEY] = bars
        analyzer._advance(KEY, old, current, bars, structure, (candidate.pool,))
        self.assertIsNone(analyzer._candidate.get(KEY))
        self.assertTrue(any(e.status is SubsequentStatus.REJECTED for e in analyzer._events.values()))

    def test_fvg_is_available_only_when_its_third_candle_completes(self):
        analyzer = make_analyzer()
        bars = [bar(i, "100", "101", "99", "100") for i in range(16)]
        bars[13] = bar(13, "99.5", "100", "99", "99.5")
        bars[15] = bar(15, "102", "103", "102", "102.5")
        pool = LiquidityPool("pool", "swing", "TESTUSD", "1m", PivotSide.LOW, 9900,
            bars[0].provenance.record_id, (), BASE, 1, "bootstrap-test-v1")
        immediate = _Candidate("p1", "p1:candidate:v1", 1, pool, "bullish", "c",
            [bars[0].provenance.record_id], 9800, 0, 0, 10, bars[10].provenance.record_id, 15)
        delayed = _Candidate("p2", "p2:candidate:v1", 1, pool, "bullish", "c",
            [bars[0].provenance.record_id], 9800, 0, 0, 10, bars[10].provenance.record_id, 12)
        self.assertIsNone(analyzer._find_fvg(bars[:15], delayed))
        self.assertIsNotNone(analyzer._find_fvg(bars, delayed))  # third candle is break + 3
        too_late = _Candidate("p3", "p3:candidate:v1", 1, pool, "bullish", "c",
            [bars[0].provenance.record_id], 9800, 0, 0, 10, bars[10].provenance.record_id, 11)
        self.assertIsNone(analyzer._find_fvg(bars, too_late))  # third candle is break + 4
        self.assertIsNotNone(analyzer._find_fvg(bars, immediate))  # FVG may include the break candle

    def test_swing_confirmed_on_a_candle_cannot_be_promoted_by_that_candle(self):
        analyzer = make_analyzer()
        history = [bar(i, "100", "101", "99", "100") for i in range(16)]
        current = bar(16, "100", "120", "100", "119")
        swing = SimpleNamespace(cluster_id="late-high", side=PivotSide.HIGH, level_ticks=11000, instrument="TESTUSD", timeframe="1m",
            available_at=current.interval_end, comparison_evidence=())
        structural = SimpleNamespace(swings=(swing,))
        analyzer._candles[KEY] = history
        analyzer._promote_independent_swings(current, structural)
        self.assertNotIn("late-high", analyzer._structural)
        analyzer._candles[KEY] = history + [current]
        later = bar(17, "100", "120", "100", "119")
        analyzer._promote_independent_swings(later, structural)
        self.assertIn("late-high", analyzer._structural)

    def test_batch_and_incremental_coordinator_replay_match_and_bootstrap_is_preserved(self):
        bars1, bootstrap1 = fixture("bullish")
        incremental = SubsequentExchangeAnalyzer(bootstrap1, BootstrapRules("bootstrap-test-v1"),
            (InstrumentSpec("TESTUSD", Decimal("0.01")),), spec_version=SPEC_VERSION)
        for candle in bars1:
            incremental.process_candle(candle)
        one = incremental.result()
        bars2, bootstrap2 = fixture("bullish")
        batch = SubsequentExchangeAnalyzer(bootstrap2, BootstrapRules("bootstrap-test-v1"),
            (InstrumentSpec("TESTUSD", Decimal("0.01")),), spec_version=SPEC_VERSION)
        two = batch.process_batch(bars2)
        self.assertEqual([e.event_id for e in one.bootstrap.events], [e.event_id for e in two.bootstrap.events])
        self.assertEqual([e.event_id for e in one.events], [e.event_id for e in two.events])
        self.assertEqual([x.event_id for x in one.bootstrap.exchanges], [x.event_id for x in two.bootstrap.exchanges])
        self.assertTrue(all(not x.tradable for x in one.bootstrap.exchanges))
        self.assertEqual([], list(one.exchanges))
        self.assertEqual([], list(one.ranges))


if __name__ == "__main__":
    unittest.main()















