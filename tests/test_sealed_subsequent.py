"""Acceptance checks for the manifest-bound M3B.1 path."""
from dataclasses import replace
from datetime import timedelta
import json
import unittest

from mlof_v2.structure.coverage import PublicationLedger, ReplayInput, SourceSeal, interval_digest, intervals
from mlof_v2.structure.subsequent import SubsequentExchangeAnalyzer
from tests.test_subsequent_exchanges import BASE, KEY, bar, chronological_replay


def sealed(direction="bullish", *, delay=0, missing=False, reset=False, ledger=None):
    legacy, bars = chronological_replay(direction, stop_before_sweep=True, deeper_sweep=reset)
    calendar = legacy.bootstrap.calendar
    proof_time = bars[-1].interval_end + timedelta(minutes=delay)
    if delay:
        for index in range(68, 68 + delay):
            if reset and index == 68:
                bars.append(bar(index, "101", "103", "99.97", "100.2"))
            else:
                bars.append(bar(index, "105", "106", "104", "105") if direction == "bullish"
                            else bar(index, "95", "96", "94", "95"))
    expected = intervals(calendar, *KEY, BASE, proof_time)
    seal = SourceSeal("synthetic", *KEY, "UTC", "close", BASE, proof_time, proof_time,
                      calendar.calendar_id, interval_digest(expected), "SEALED_COMPLETE_THROUGH_BOUNDARY",
                      "synthetic-seal")
    if missing:
        bars.pop()
    replay = ReplayInput.locally_sealed(seal, tuple(bars))
    template = legacy.bootstrap
    bootstrap = type(template)(template.config, template.rules, tuple(legacy.specs.values()), calendar)
    engine = SubsequentExchangeAnalyzer(bootstrap, legacy.rules, tuple(legacy.specs.values()),
                                        spec_version="0.3.4", replay=replay, ledger=ledger)
    return engine, bars


class SealedReplayTests(unittest.TestCase):
    def test_independent_proof_times_keep_market_id_and_historical_state(self):
        ledger = PublicationLedger()
        for direction in ("bullish", "bearish"):
            with self.subTest(direction=direction):
                first, bars = sealed(direction, ledger=ledger)
                second, later = sealed(direction, delay=2, ledger=ledger)
                x, = first.process_batch(bars).exchanges
                y, = second.process_batch(later).exchanges
                self.assertEqual(x.exchange_id, y.exchange_id)
                self.assertLess(x.available_at, y.available_at)
                self.assertEqual(x, ledger.get(first.evaluation_identity, x.exchange_id))
                self.assertEqual(y, ledger.get(second.evaluation_identity, y.exchange_id))
                before = second.as_of(y.available_at - timedelta(microseconds=1))
                self.assertFalse(before.exchanges)
                self.assertFalse(before.manipulations)
                self.assertEqual("ACTIVE", next(c.state for c in before.controllers
                                               if c.control_id == y.parent_controller_id))
                for when in (y.available_at, y.available_at + timedelta(seconds=1)):
                    after = second.as_of(when)
                    old = next(c for c in after.controllers if c.control_id == y.parent_controller_id)
                    self.assertEqual("REPLACED", old.state)
                    self.assertEqual(y.available_at, old.replaced_at)
                self.assertEqual("ACTIVE", next(c.state for c in second.result().controllers
                                               if c.control_id == y.parent_controller_id))

    def test_missing_trailing_interval_blocks_publication(self):
        engine, bars = sealed(delay=1, missing=True)
        self.assertFalse(engine.process_batch(bars).exchanges)
        engine.evaluate(engine.replay.seal.seal_available_at)
        self.assertFalse(engine.result().exchanges)

    def test_later_reset_prevents_prior_version_publication(self):
        engine, bars = sealed(delay=1, reset=True)
        result = engine.process_batch(bars)
        self.assertFalse(result.exchanges)
        final = json.loads(result.coverage[-1].canonical_payload)
        self.assertIn("INVALIDATED", {row["state"] for row in final["outcomes"]})

    def test_raw_fvg_and_reset_replay_are_identical_incrementally(self):
        batch, bars = sealed(reset=True)
        incremental, _ = sealed(reset=True)
        expected = batch.process_batch(bars)
        self.assertEqual(1, len(expected.exchanges))
        for candle in bars:
            incremental.process_candle(candle)
        self.assertEqual(expected, incremental.result())
        self.assertIn(bars[65].provenance.record_id, expected.exchanges[0].fvg_candle_ids)

    def test_ledger_checks_full_payload_within_evaluation_atomically(self):
        ledger = PublicationLedger()
        ledger.publish("A", (("X", b"i26"),))
        ledger.publish("A", (("X", b"i26"),))
        ledger.publish("B", (("X", b"i28"),))
        with self.assertRaisesRegex(ValueError, "collision"):
            ledger.publish("A", (("Y", b"new"), ("X", b"i28")))
        self.assertIsNone(ledger.get("A", "Y"))
        self.assertEqual(b"i26", ledger.get("A", "X"))
        self.assertEqual(b"i28", ledger.get("B", "X"))

    def test_same_candle_cohort_releases_only_with_complete_proof(self):
        for missing, reused in ((False, False), (False, True), (True, False)):
            with self.subTest(missing=missing, reused=reused):
                engine, bars = sealed(delay=2, missing=missing)
                # Isolate stage C with three preliminary evidence-ready lineages.
                for candle in bars[:-1]:
                    engine.process_candle(candle)
                parent = engine._active[KEY]
                original = next(iter(engine._ready.values()))
                candidate, _, fvg, reference, structural = original
                pool = replace(candidate.pool, pool_id="other-pool")
                for lineage, candidate_pool in (("a", candidate.pool), ("b", pool)):
                    duplicate = replace(candidate, parent_id=lineage,
                                        candidate_id=lineage + ":candidate:v1", pool=candidate_pool)
                    engine._ready[duplicate.candidate_id] = (
                        duplicate, bars[67], fvg, reference, structural)
                    engine._describe(duplicate, "EVIDENCE_READY", bars[67], fvg[1])
                    engine._lineages[(parent.control_id, candidate_pool.pool_id)] = duplicate
                engine._ready.pop(candidate.candidate_id)
                seen = []

                def capture(key, old, winner, candle, prefix, evidence, ref, snapshot, **publication):
                    seen.append((winner.parent_id, publication["publish_at"]))
                    engine._replaced.add(parent.control_id)

                from unittest.mock import patch
                if missing:
                    cut = engine.replay.seal.seal_available_at
                    engine.evaluate(cut)
                    self.assertFalse(engine.result().exchanges)
                    self.assertFalse(engine._rejected_evidence)
                    continue

                # A later completion is eligible for comparison in this same proof cut.
                fresh_ids = tuple(c.provenance.record_id for c in bars[67:70] if reused)
                if not reused:
                    fresh_ids = ("fresh-fvg-1", "fresh-fvg-2", "fresh-fvg-3")
                pool = replace(candidate.pool, pool_id="fresh-pool")
                fresh = replace(candidate, parent_id="fresh", candidate_id="fresh:candidate:v1",
                                pool=pool, break_index=69)
                engine._ready[fresh.candidate_id] = (
                    fresh, bars[69], (69, fresh_ids), reference, structural)
                engine._describe(fresh, "EVIDENCE_READY", bars[69], fresh_ids)
                with (patch.object(engine, "_discover"), patch.object(engine, "_verify_trace", return_value=True),
                      patch.object(engine, "_confirm", side_effect=capture)):
                    engine.process_candle(bars[-1])
                self.assertEqual([] if reused else [("fresh", bars[-1].source_available_at)], seen)
                self.assertEqual("UNRESOLVED", engine._outcomes["a:candidate:v1"]["state"])
                self.assertEqual("UNRESOLVED", engine._outcomes["b:candidate:v1"]["state"])
                self.assertTrue(engine._rejected_evidence[parent.control_id])

    def test_multiple_pool_sweep_quarantines_every_crossed_lineage(self):
        from unittest.mock import patch

        engine, bars = sealed()
        for candle in bars[:65]:
            engine.process_candle(candle)
        eligible = engine.bootstrap.structural.eligible_pools_for

        def two_low_pools(candle):
            pools = eligible(candle)
            if candle == bars[65]:
                low = next(pool for pool in pools if pool.side.value == "LOW"
                           and pool.level_ticks == 10000)
                return (*pools, replace(low, pool_id="second-eligible-low"))
            return pools

        with (patch.object(engine.bootstrap.structural, "eligible_pools_for", side_effect=two_low_pools),
              patch.object(engine, "_verify_trace", return_value=True)):
            for candle in bars[65:]:
                engine.process_candle(candle)
        result = engine.result()
        self.assertFalse(result.exchanges)
        final = json.loads(result.coverage[-1].canonical_payload)
        conflicts = [record for record in final["conflicts"] if record["code"] == "MULTI_POOL_SWEEP"]
        self.assertEqual(1, len(conflicts))
        self.assertEqual(2, len(conflicts[0]["lineage_ids"]))
        self.assertTrue(all(row["state"] == "UNRESOLVED" for row in final["outcomes"]))

    def test_governing_path_cannot_publish_without_replay(self):
        legacy, bars = chronological_replay("bullish", stop_before_sweep=True)
        template = legacy.bootstrap
        bootstrap = type(template)(template.config, template.rules,
                                   tuple(legacy.specs.values()), template.calendar)
        engine = SubsequentExchangeAnalyzer(bootstrap, legacy.rules, tuple(legacy.specs.values()),
                                            spec_version="0.3.4")
        self.assertFalse(engine.process_batch(bars).exchanges)

    def test_missing_duplicate_and_reordered_transition_fail_closed(self):
        for fault in ("missing", "duplicate", "reorder"):
            with self.subTest(fault=fault):
                engine, bars = sealed(delay=1, reset=False)
                for candle in bars[:-1]:
                    engine.process_candle(candle)
                trace = engine._transition_trace
                if fault == "missing":
                    del trace[0]
                elif fault == "duplicate":
                    trace.insert(0, trace[0])
                else:
                    self.assertNotEqual(trace[0], trace[-1])
                    trace[0], trace[-1] = trace[-1], trace[0]
                engine.process_candle(bars[-1])
                self.assertFalse(engine.result().exchanges)
                diagnostic = json.loads(engine.result().coverage[-1].canonical_payload)
                self.assertIn("INCOMPLETE_DISCOVERY",
                              {entry["code"] for entry in diagnostic["conflicts"]})

    def test_seal_binds_dataset_and_external_evidence_requires_independent_pin(self):
        engine, _ = sealed()
        data = engine.replay
        external = replace(data, evidence=replace(data.evidence, kind="EXTERNAL"))
        self.assertFalse(external.verify_seal(frozenset()))
        self.assertTrue(external.verify_seal(frozenset({data.seal.provenance_id})))
        substituted = replace(data, seal=replace(data.seal, dataset_id="another-dataset"))
        self.assertFalse(substituted.verify_seal(frozenset()))
        bad_candle = replace(data.candles[-1], open=data.candles[-1].close)
        self.assertFalse(replace(data, candles=(*data.candles[:-1], bad_candle)).verify_seal(frozenset()))
        self.assertFalse(replace(data, evidence=None).verify_seal(frozenset()))

    def test_real_completion_cohort_and_later_independent_replacement(self):
        from decimal import Decimal

        from mlof_v2.calendars import ExplicitSessionCalendar, OpenInterval
        from mlof_v2.structure import SwingWidths

        legacy, initial = chronological_replay("bullish", stop_before_sweep=True)
        rows = [
            (66, 103, 104, 102, 103), (67, 103, 104, 102, 103),
            (68, 102, 104, 101, 103), (69, 103, 104, 102, 103),
            (70, 104, 105, 103, 104), (71, 102, 103, 100.99, 102),
            (72, 104, 105, 103, 104), (73, 108, 116, 107, 115),
            (74, 105, 106, 103, 105), (75, 106, 107, 105, 106),
            (76, 107, 108, 106, 107), (77, 104, 105, 102.5, 104),
            (78, 105, 106, 104, 105), (79, 110, 117, 109, 116),
        ]
        bullish = initial[:66] + [bar(index, *map(str, ohlc)) for index, *ohlc in rows]
        config = replace(legacy.bootstrap.config, pivot_widths=(("1m", SwingWidths(1, 1)),))
        end = bullish[-1].interval_end
        calendar = ExplicitSessionCalendar("cohort-replay", BASE, end, (OpenInterval(BASE, end),))
        for direction, release_index in (("bullish", 73), ("bullish", 74), ("bearish", 74)):
            with self.subTest(direction=direction, release_index=release_index):
                if direction == "bullish":
                    candles = tuple(bullish)
                else:
                    axis = Decimal("210")
                    candles = tuple(replace(c, open=axis-c.open, high=axis-c.low,
                                            low=axis-c.high, close=axis-c.close) for c in bullish)
                release = candles[release_index].interval_end

                def proof(boundary):
                    declaration = SourceSeal(
                        "cohort", *KEY, "UTC", "close", BASE, boundary, boundary, calendar.calendar_id,
                        interval_digest(intervals(calendar, *KEY, BASE, boundary)),
                        "SEALED_COMPLETE_THROUGH_BOUNDARY", "",
                    )
                    return ReplayInput.locally_sealed(declaration, candles)

                def engine():
                    specs = tuple(legacy.specs.values())
                    bootstrap = type(legacy.bootstrap)(config, legacy.rules, specs, calendar)
                    return SubsequentExchangeAnalyzer(bootstrap, legacy.rules, specs, spec_version="0.3.4",
                                                      replay=proof(end), additional_seals=(proof(release),))

                incremental = engine()
                for candle in candles[:release_index + 1]:
                    incremental.process_candle(candle)
                released = incremental.as_of(release)
                self.assertFalse(released.exchanges)
                self.assertEqual(1, len(released.active_controller_ids))
                previous = incremental.as_of(candles[73].interval_end - timedelta(microseconds=1))
                self.assertFalse(any(b"OVERLAPPING_COMPLETION" in p.canonical_payload
                                     for p in previous.coverage))
                final = json.loads(released.coverage[-1].canonical_payload)
                cohort, = [row for row in final["conflicts"] if row["code"] == "OVERLAPPING_COMPLETION"]
                self.assertEqual(2, len(cohort["lineage_ids"]))
                self.assertEqual("COMPLETE", final["coverage_state"])
                for candle in candles[release_index + 1:]:
                    incremental.process_candle(candle)
                actual = incremental.result()
                exchange, = actual.exchanges
                self.assertEqual(direction, exchange.direction)
                self.assertEqual(end, exchange.available_at)
                self.assertEqual(("sub-77", "sub-78", "sub-79"), exchange.fvg_candle_ids)
                self.assertEqual(released, incremental.as_of(release))
                self.assertEqual(actual, engine().process_batch(candles))
                from tests.test_independent_reference import expected_and_actual
                reference, observed = expected_and_actual(engine(), candles)
                self.assertEqual(reference[:3], observed)
                self.assertEqual([(release, 73, 2)], reference[4])

