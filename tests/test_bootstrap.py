from __future__ import annotations

import unittest
import json
from dataclasses import FrozenInstanceError, replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from mlof_v2.calendars import ExplicitSessionCalendar, OpenInterval, UnknownSessionCalendar
from mlof_v2.models import Candle, InstrumentSpec, SourceProvenance
from mlof_v2.structure import (
    BootstrapAnalyzer, BootstrapRules, BootstrapStage, PivotSide, StructuralConfig, SwingWidths,
    bootstrap_manipulation_id, bootstrap_replay_jsonl, observation_pair_id,
    load_bootstrap_rules,
)
from mlof_v2.structure import bootstrap as bootstrap_module
from tests.test_structural_intelligence import BASE, POLICY


def fixture(direction: str = "bullish", *, unknown_calendar: bool = False):
    bars: list[Candle] = []
    for i in range(57):
        if i < 15:
            high, low = Decimal("100") + Decimal(i) / 10, Decimal("99") + Decimal(i) / 10
        elif i == 15:
            high, low = Decimal("105"), Decimal("101.5")
        elif i < 20:
            high, low = Decimal("104") - Decimal(i - 16) * Decimal("0.6"), Decimal("101.4") + Decimal(i - 16) / 10
        elif i < 30:
            high, low = Decimal("101.8") - Decimal(i - 20) / 10, Decimal("101.6") - Decimal(i - 20) / 5
        elif i == 30:
            high, low = Decimal("100.2"), Decimal("95")
        elif i < 35:
            high, low = Decimal("100.2") + Decimal(i - 30) / 10, Decimal("95") + Decimal(i - 30)
        else:
            high, low = Decimal("100.7") + Decimal(i - 35) / 10, Decimal("99.1") + Decimal(i - 35) / 10
        open_price = (high + low) / 2
        close = open_price
        if i == 50:
            if direction == "bullish":
                high, low, open_price, close = Decimal("96.2"), Decimal("94"), Decimal("95.5"), Decimal("96")
            else:
                high, low, open_price, close = Decimal("106"), Decimal("103.8"), Decimal("104.5"), Decimal("104")
        elif i == 51:
            if direction == "bullish":
                high, low, open_price, close = Decimal("97"), Decimal("96"), Decimal("96.2"), Decimal("96.8")
            else:
                high, low, open_price, close = Decimal("105"), Decimal("104"), Decimal("104.8"), Decimal("104.2")
        elif i == 52:
            if direction == "bullish":
                high, low, open_price, close = Decimal("98"), Decimal("97"), Decimal("97.2"), Decimal("97.8")
            else:
                high, low, open_price, close = Decimal("104"), Decimal("103"), Decimal("103.8"), Decimal("103.2")
        elif i == 53:
            if direction == "bullish":
                high, low, open_price, close = Decimal("108"), Decimal("102"), Decimal("102.2"), Decimal("107")
            else:
                high, low, open_price, close = Decimal("98"), Decimal("92"), Decimal("97.8"), Decimal("93")
        start = BASE + timedelta(minutes=i)
        end = start + timedelta(minutes=1)
        provenance = SourceProvenance("synthetic://bootstrap", "b" * 64, i + 1,
                                      f"{i + 1:064x}", f"boot-{direction}-{i}", (("synthetic", "true"),))
        bars.append(Candle("TESTUSD", "1m", start, end, end, open_price, high, low, close, provenance))
    calendar = (UnknownSessionCalendar() if unknown_calendar else
                ExplicitSessionCalendar("bootstrap-fixture", bars[0].interval_start,
                                        bars[-1].interval_end + timedelta(minutes=1),
                                        (OpenInterval(bars[0].interval_start,
                                                      bars[-1].interval_end + timedelta(minutes=1)),)))
    config = StructuralConfig(1, "bootstrap-test-v1", 1, (("1m", SwingWidths(2, 2)),), POLICY)
    rules = BootstrapRules("bootstrap-test-v1")
    analyzer = BootstrapAnalyzer(config, rules, (InstrumentSpec("TESTUSD", Decimal("0.01")),), calendar)
    return bars, analyzer


class BootstrapIdentityTests(unittest.TestCase):
    def test_pair_and_manipulation_ids_are_canonical_and_versioned(self):
        pair = observation_pair_id("cfg", "testusd", "1M", "high", "low")
        self.assertTrue(pair.startswith("observation_pair_v1:"))
        self.assertEqual(pair, observation_pair_id("cfg", "TESTUSD", "1m", "high", "low"))
        event = bootstrap_manipulation_id(config_version="cfg", pair_id=pair, swept_pool_id="pool",
                                          direction="bullish", sweep_candle_ids=("a", "a", "b"),
                                          reclaim_candle_id="b", sweep_reference_ticks=12345)
        self.assertTrue(event.startswith("bootstrap_manipulation_v1:"))
        self.assertEqual(event, bootstrap_manipulation_id(config_version="cfg", pair_id=pair,
                         swept_pool_id="pool", direction="bullish", sweep_candle_ids=("a", "b"),
                         reclaim_candle_id="b", sweep_reference_ticks=12345))

    def test_versioned_bootstrap_configuration_loads_and_rejects_unknown_fields(self):
        path = Path(__file__).resolve().parents[1] / "config" / "bootstrap_research.example.json"
        rules = load_bootstrap_rules(path)
        self.assertEqual("structural-research-v1", rules.config_version)
        self.assertEqual(Decimal("0.60"), rules.body_ratio_min)
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["unapproved_threshold"] = 0
        with TemporaryDirectory(dir=path.parent) as temp:
            bad = Path(temp) / "bad.json"
            bad.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_bootstrap_rules(bad)


class BootstrapReplayTests(unittest.TestCase):
    def test_bullish_bootstrap_establishes_frozen_nontradable_control(self):
        bars, analyzer = fixture("bullish")
        for candle in bars:
            analyzer.process_candle(candle)
        result = analyzer.result()
        self.assertEqual(1, len(result.controls))
        control = result.controls[0]
        self.assertEqual("bullish", control.direction)
        self.assertFalse(control.tradable)
        self.assertEqual(1, len(result.exchanges))
        self.assertFalse(result.exchanges[0].tradable)
        self.assertEqual((control.low_cluster_id, control.high_cluster_id),
                         (next(x for x in result.structural.swings if x.cluster_id == control.low_cluster_id).cluster_id,
                          next(x for x in result.structural.swings if x.cluster_id == control.high_cluster_id).cluster_id))
        self.assertTrue(all(not e.tradable for e in result.events))
        self.assertTrue(all(e.config_version == "bootstrap-test-v1" for e in result.events))
        control_event = next(e for e in result.events if e.stage is BootstrapStage.CONTROL_ESTABLISHED)
        self.assertTrue(control_event.source_evidence)

    def test_bearish_bootstrap_establishes_control(self):
        bars, analyzer = fixture("bearish")
        result = analyzer.process_batch(bars)
        self.assertEqual(1, len(result.controls))
        self.assertEqual("bearish", result.controls[0].direction)
        self.assertFalse(result.controls[0].tradable)

    def test_swept_side_close_through_arms_only_after_reclaim(self):
        cases = (
            ("bullish", "94.9", "95.5", "96.2", "94.5"),
            ("bearish", "105.1", "104.5", "105.5", "103.8"),
        )
        for direction, outside, inside, high, low in cases:
            with self.subTest(direction=direction):
                bars, live = fixture(direction)
                bars[50] = replace(bars[50], close=Decimal(outside))
                bars[51] = replace(bars[51], high=Decimal(high), low=Decimal(low),
                                   open=Decimal(inside), close=Decimal(inside))
                _, terminated = fixture(direction)
                invalidated = list(bars)
                invalidated[52] = replace(invalidated[52], high=Decimal(high), low=Decimal(low),
                                          open=Decimal(inside), close=Decimal(outside))

                self.assertEqual(1, len(live.process_batch(bars[:54]).exchanges))
                result = terminated.process_batch(invalidated[:54])
                self.assertFalse(result.exchanges)
                self.assertFalse(result.controls)
                guard = [event for event in result.events
                         if "frozen swept-side bootstrap reference" in event.reason]
                self.assertEqual(1, len(guard))
                self.assertEqual(invalidated[52].source_available_at, guard[0].available_at)
                self.assertIn(invalidated[52].provenance.record_id, guard[0].source_candle_ids)


    def test_batch_and_incremental_replay_are_identical(self):
        bars, incremental = fixture()
        _, batch = fixture()
        for candle in bars:
            incremental.process_candle(candle)
        self.assertEqual(incremental.result(), batch.process_batch(bars))

    def test_instrument_and_timeframe_states_are_independent(self):
        bars, _ = fixture()
        other_instrument = [replace(c, instrument="OTHERUSD",
                                    provenance=replace(c.provenance, record_id="other-" + c.provenance.record_id))
                            for c in bars]
        three_minute = []
        for i, candle in enumerate(bars):
            start = BASE + timedelta(minutes=3 * i)
            end = start + timedelta(minutes=3)
            three_minute.append(replace(candle, timeframe="3m", interval_start=start, interval_end=end,
                                        source_available_at=end,
                                        provenance=replace(candle.provenance, record_id=f"3m-{i}")))
        end = BASE + timedelta(minutes=3 * len(bars) + 3)
        calendar = ExplicitSessionCalendar("multi-stream", BASE, end, (OpenInterval(BASE, end),))
        config = StructuralConfig(1, "bootstrap-test-v1", 1,
                                  (("1m", SwingWidths(2, 2)), ("3m", SwingWidths(2, 2))), POLICY)
        analyzer = BootstrapAnalyzer(config, BootstrapRules("bootstrap-test-v1"),
                                     (InstrumentSpec("TESTUSD", Decimal("0.01")),
                                      InstrumentSpec("OTHERUSD", Decimal("0.01"))), calendar)
        result = analyzer.process_batch(bars + other_instrument + three_minute)
        self.assertEqual({("TESTUSD", "1m"), ("OTHERUSD", "1m"), ("TESTUSD", "3m")},
                         {(c.instrument, c.timeframe) for c in result.controls})

    def test_unknown_session_coverage_fails_closed(self):
        bars, analyzer = fixture(unknown_calendar=True)
        result = analyzer.process_batch(bars)
        self.assertEqual((), result.controls)
        self.assertTrue(any(e.stage is BootstrapStage.NOT_READY for e in result.events))

    def test_no_sweep_produces_no_bootstrap_control(self):
        bars, analyzer = fixture()
        bars[50] = replace(bars[50], low=Decimal("95.01"), close=Decimal("96"), open=Decimal("95.5"))
        result = analyzer.process_batch(bars)
        self.assertEqual((), result.controls)

    def test_sweep_without_reclaim_expires_without_control(self):
        bars, analyzer = fixture()
        bars[50] = replace(bars[50], close=Decimal("94.5"))
        for i in range(51, 57):
            bars[i] = replace(bars[i], high=Decimal("94.9"), low=Decimal("94.1"),
                              open=Decimal("94.7"), close=Decimal("94.5"))
        for candle in bars[:57]:
            analyzer.process_candle(candle)
        result = analyzer.result()
        self.assertEqual((), result.controls)
        self.assertTrue(any(e.stage is BootstrapStage.EXPIRED for e in result.events))

    def test_repeated_deeper_sweep_before_reclaim_updates_reference_and_lineage(self):
        bars, analyzer = fixture()
        bars[50] = replace(bars[50], close=Decimal("94.5"))
        bars[51] = replace(bars[51], high=Decimal("96.2"), low=Decimal("93.8"),
                           open=Decimal("94.2"), close=Decimal("95.2"))
        result = analyzer.process_batch(bars)
        self.assertEqual(1, len(result.manipulations))
        event = result.manipulations[0]
        self.assertEqual((bars[50].provenance.record_id, bars[51].provenance.record_id), event.sweep_candle_ids)
        self.assertEqual(9420, event.reference_open_ticks)
        self.assertEqual(9380, event.manipulation_extreme_ticks)

    def test_post_reclaim_deeper_sweep_preserves_parent_and_versions_candidate(self):
        bars, analyzer = fixture()
        bars[51] = replace(bars[51], high=Decimal("98"), low=Decimal("93.8"),
                           open=Decimal("94.2"), close=Decimal("96.8"))
        result = analyzer.process_batch(bars)
        resets = [event for event in result.events if event.stage is BootstrapStage.CANDIDATE_RESET]
        self.assertEqual(1, len(resets))
        parent_id, canceled_id, replacement_id = resets[0].related_ids[:3]
        self.assertEqual((parent_id + ":candidate:v1", parent_id + ":candidate:v2"),
                         (canceled_id, replacement_id))
        self.assertEqual(parent_id, result.manipulations[0].parent_id)
        self.assertEqual(replacement_id, result.manipulations[0].candidate_id)
        trigger_id = bars[51].provenance.record_id
        self.assertIn(trigger_id, resets[0].source_candle_ids)
        evidence = next(item for item in resets[0].source_evidence if item.record_id == trigger_id)
        self.assertEqual(bars[51].interval_start, evidence.interval_start)
        self.assertEqual(bars[51].source_available_at, evidence.source_available_at)
        self.assertEqual("bootstrap-test-v1", resets[0].config_version)

    def test_multiple_post_reclaim_deeper_sweeps_increment_once_each(self):
        bars, analyzer = fixture()
        bars[51] = replace(bars[51], high=Decimal("98"), low=Decimal("93.8"),
                           open=Decimal("94.2"), close=Decimal("96.8"))
        bars[52] = replace(bars[52], high=Decimal("99"), low=Decimal("93.5"),
                           open=Decimal("94.1"), close=Decimal("97.8"))
        result = analyzer.process_batch(bars)
        resets = [event for event in result.events if event.stage is BootstrapStage.CANDIDATE_RESET]
        self.assertEqual(2, len(resets))
        parent_ids = {event.related_ids[0] for event in resets}
        self.assertEqual(1, len(parent_ids))
        self.assertEqual([
            (next(iter(parent_ids)) + ":candidate:v1", next(iter(parent_ids)) + ":candidate:v2"),
            (next(iter(parent_ids)) + ":candidate:v2", next(iter(parent_ids)) + ":candidate:v3"),
        ], [event.related_ids[1:3] for event in resets])
        self.assertEqual(next(iter(parent_ids)), result.manipulations[0].parent_id)
        self.assertTrue(result.manipulations[0].candidate_id.endswith(":candidate:v3"))

    def test_post_reclaim_lineage_is_identical_for_batch_and_incremental_replay(self):
        bars, incremental = fixture()
        _, batch = fixture()
        bars[51] = replace(bars[51], high=Decimal("98"), low=Decimal("93.8"),
                           open=Decimal("94.2"), close=Decimal("96.8"))
        bars[52] = replace(bars[52], high=Decimal("99"), low=Decimal("93.5"),
                           open=Decimal("94.1"), close=Decimal("97.8"))
        for candle in bars:
            incremental.process_candle(candle)
        self.assertEqual(incremental.result(), batch.process_batch(bars))

    def test_equal_or_shallower_sweep_does_not_create_another_candidate_version(self):
        bars, analyzer = fixture()
        bars[51] = replace(bars[51], high=Decimal("98"), low=Decimal("93.8"),
                           open=Decimal("94.2"), close=Decimal("96.8"))
        bars[52] = replace(bars[52], high=Decimal("98"), low=Decimal("93.8"),
                           open=Decimal("94.2"), close=Decimal("97.8"))
        result = analyzer.process_batch(bars)
        resets = [event for event in result.events if event.stage is BootstrapStage.CANDIDATE_RESET]
        self.assertEqual(1, len(resets))
        self.assertTrue(result.manipulations[0].candidate_id.endswith(":candidate:v2"))

    def test_range_id_collision_with_different_evidence_fails_without_partial_promotion(self):
        bars, analyzer = fixture()
        for candle in bars[:53]:
            analyzer.process_candle(candle)
        collision_id = "bootstrap_range_v1:forced-collision"
        analyzer._id_payloads[collision_id] = ("existing", "different source evidence")
        real_hash = bootstrap_module._canonical_hash

        def collide_range(prefix, payload):
            if prefix == "bootstrap_range_v1:":
                return collision_id
            return real_hash(prefix, payload)

        with patch.object(bootstrap_module, "_canonical_hash", side_effect=collide_range):
            for candle in bars[53:]:
                analyzer.process_candle(candle)
        result = analyzer.result()
        self.assertEqual((), result.controls)
        self.assertEqual((), result.manipulations)
        self.assertEqual((), result.exchanges)
        self.assertTrue(any("lineage ID collision" in event.reason for event in result.events))

    def test_range_id_collision_replay_is_deterministic(self):
        collision_id = "bootstrap_range_v1:forced-collision"
        real_hash = bootstrap_module._canonical_hash

        def run():
            bars, analyzer = fixture()
            for candle in bars[:53]:
                analyzer.process_candle(candle)
            analyzer._id_payloads[collision_id] = ("existing", "different source evidence")

            def collide_range(prefix, payload):
                if prefix == "bootstrap_range_v1:":
                    return collision_id
                return real_hash(prefix, payload)

            with patch.object(bootstrap_module, "_canonical_hash", side_effect=collide_range):
                analyzer.process_batch(bars[53:])
            return analyzer.result()

        self.assertEqual(run(), run())

    def test_missing_expected_bar_resets_and_fails_closed(self):
        bars, analyzer = fixture()
        result = analyzer.process_batch(bars[:20] + bars[21:])
        self.assertEqual((), result.controls)
        self.assertTrue(any(e.stage is BootstrapStage.NOT_READY for e in result.events))

    def test_delayed_fvg_on_third_subsequent_candle_confirms_without_lookahead(self):
        bars, analyzer = fixture()
        bars[52] = replace(bars[52], high=Decimal("98"), low=Decimal("96.1"),
                           open=Decimal("96.2"), close=Decimal("97.8"))
        bars[53] = replace(bars[53], high=Decimal("108"), low=Decimal("97"),
                           open=Decimal("97.2"), close=Decimal("107"))
        bars[54] = replace(bars[54], high=Decimal("99"), low=Decimal("98"),
                           open=Decimal("98.2"), close=Decimal("98.8"))
        bars[55] = replace(bars[55], high=Decimal("99.5"), low=Decimal("98.5"),
                           open=Decimal("98.7"), close=Decimal("99.2"))
        bars[56] = replace(bars[56], high=Decimal("101"), low=Decimal("100"),
                           open=Decimal("100.2"), close=Decimal("100.8"),
                           source_available_at=bars[56].interval_end + timedelta(minutes=1))
        for candle in bars[:56]:
            analyzer.process_candle(candle)
        self.assertEqual((), analyzer.result().controls)
        analyzer.process_candle(bars[56])
        result = analyzer.result()
        self.assertEqual(1, len(result.controls))
        self.assertEqual(bars[56].source_available_at, result.exchanges[0].available_at)

    def test_delayed_pool_availability_cannot_retroactively_qualify_earlier_sweep(self):
        bars, analyzer = fixture()
        for index in range(34, len(bars)):
            bars[index] = replace(bars[index], source_available_at=bars[index].interval_end + timedelta(minutes=20))
        result = analyzer.process_batch(bars)
        self.assertEqual((), result.controls)
        self.assertFalse(any(e.stage is BootstrapStage.BOOTSTRAP_SWEEP for e in result.events))

    def test_both_pair_extremes_breached_in_one_ohlc_bar_are_ambiguous(self):
        bars, analyzer = fixture()
        bars[50] = replace(bars[50], high=Decimal("106"), low=Decimal("94"),
                           open=Decimal("100"), close=Decimal("100"))
        result = analyzer.process_batch(bars)
        self.assertEqual((), result.controls)
        self.assertTrue(any("ordering is unknowable" in e.reason for e in result.events))

    def test_confirmed_control_and_range_keep_complete_lineage_and_are_frozen(self):
        bars, analyzer = fixture()
        result = analyzer.process_batch(bars)
        manipulation, exchange, control = result.manipulations[0], result.exchanges[0], result.controls[0]
        self.assertEqual(manipulation.event_id, exchange.manipulation_id)
        self.assertEqual(exchange.event_id, control.exchange_id)
        self.assertEqual(control.pair_id, manipulation.pair_id)
        self.assertEqual(control.pair_id, exchange.pair_id)
        self.assertFalse(any(x.tradable for x in (*result.manipulations, *result.exchanges, *result.controls)))
        self.assertEqual(control.low_cluster_id, control.protective_cluster_id)
        self.assertEqual(control.high_cluster_id, control.break_reference_cluster_id)
        self.assertTrue(control.source_evidence)
        audit = bootstrap_replay_jsonl(result)
        self.assertIn('"record_type":"bootstrap_exchange"', audit)
        self.assertEqual(audit, bootstrap_replay_jsonl(analyzer.result()))
        with self.assertRaises(FrozenInstanceError):
            control.low_ticks = 0  # type: ignore[misc]

    def test_no_future_pair_visible_before_pool_registration(self):
        bars, analyzer = fixture()
        for candle in bars[:34]:
            analyzer.process_candle(candle)
        result = analyzer.result()
        self.assertEqual((), result.pairs)
        self.assertEqual((), result.controls)

    def test_break_without_displacement_or_fvg_does_not_establish_control(self):
        bars, analyzer = fixture()
        bars[53] = replace(bars[53], high=Decimal("106"), low=Decimal("104"),
                           open=Decimal("104.99"), close=Decimal("105.02"))
        result = analyzer.process_batch(bars)
        self.assertEqual((), result.controls)

    def test_reclaim_without_opposing_break_does_not_establish_control(self):
        bars, analyzer = fixture()
        for i in range(53, len(bars)):
            bars[i] = replace(bars[i], high=Decimal("104"), low=Decimal("103"),
                              open=Decimal("103.2"), close=Decimal("103.8"))
        result = analyzer.process_batch(bars)
        self.assertEqual((), result.controls)

    def test_qualifying_displacement_without_any_aligned_fvg_stays_unconfirmed(self):
        bars, analyzer = fixture()
        bars[52] = replace(bars[52], high=Decimal("98"), low=Decimal("96.1"),
                           open=Decimal("96.2"), close=Decimal("97.8"))
        bars[53] = replace(bars[53], high=Decimal("108"), low=Decimal("97"),
                           open=Decimal("97.2"), close=Decimal("107"))
        result = analyzer.process_batch(bars[:54])
        self.assertEqual((), result.controls)


if __name__ == "__main__":
    unittest.main()
