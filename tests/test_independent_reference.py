"""Differential acceptance for sealed replay; production is used only as the system under test."""
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
import unittest

from mlof_v2.structure.coverage import PublicationLedger, ReplayInput, interval_digest, intervals
from tests.reference_m3b1 import interpret
from tests.test_sealed_subsequent import sealed
from tests.test_subsequent_exchanges import KEY, bar


def expected_and_actual(engine, candles, *, already_processed=False):
    result = engine.result() if already_processed else engine.process_batch(candles)
    initial, = result.bootstrap.controls
    tick = engine.specs[KEY[0]].tick_size
    bullish = initial.direction == "bearish"
    side = "LOW" if bullish else "HIGH"
    ids = {c.provenance.record_id: i for i, c in enumerate(candles)}
    eligible = [
        {p.pool_id: p.level_ticks*tick for p in engine.bootstrap.structural.eligible_pools_for(c)
         if p.side.value == side and p.available_at <= c.interval_start}
        for c in candles
    ]
    first = next(i for i, c in enumerate(candles) if c.interval_start >= initial.available_at)
    expected = interpret(candles, engine.bootstrap.calendar,
                         tuple(x.seal for x in engine._input_declarations), eligible,
                         bullish=bullish, boundary=(initial.low_ticks if bullish else initial.high_ticks)*tick,
                         target=(initial.high_ticks if bullish else initial.low_ticks)*tick,
                         tick=tick, first=first, rules=engine.rules)
    origins = {}
    trace = []
    for candle, pool, parent, lineage, version, action in engine._observations:
        if parent != initial.control_id:
            continue
        if lineage:
            origins.setdefault(lineage, (pool, ids[candle]))
        trace.append((ids[candle], pool, origins[lineage][1] if lineage else None, version, action))
    outcomes = {(origins[row["lineage_id"]][0], origins[row["lineage_id"]][1], row["version"]): row["state"]
                for candidate, row in engine._outcomes.items()
                if engine._outcome_parents[candidate] == initial.control_id}
    winner = None
    for exchange in result.exchanges:
        if exchange.parent_controller_id == initial.control_id:
            manipulation = next(m for m in result.manipulations if m.manipulation_id == exchange.manipulation_id)
            winner = exchange.available_at, (*origins[manipulation.parent_id],
                      int(manipulation.candidate_id.rsplit("v",1)[1])), exchange.fvg_candle_ids
    return expected, (winner, outcomes, trace)


class IndependentReferenceTests(unittest.TestCase):
    def test_ordinary_revisions_proof_and_no_exchange(self):
        for direction, delay, missing, reset in (
            ("bullish",0,False,False), ("bearish",0,False,False),
            ("bullish",0,False,True), ("bullish",1,False,True),
            ("bullish",1,True,False),
        ):
            with self.subTest(direction=direction, delay=delay, missing=missing, reset=reset):
                engine, candles = sealed(direction, delay=delay, missing=missing, reset=reset)
                expected, actual = expected_and_actual(engine, candles)
                self.assertEqual(expected[:3], actual)

    def test_frozen_boundary_close_through_and_late_reclaim(self):
        for case in ("close-through", "late-reclaim"):
            with self.subTest(case=case):
                template, candles = sealed()
                candles = candles[:66]
                if case == "close-through":
                    candles += [bar(66, "100", "101", "94.9", "94.99")]
                else:
                    candles += [bar(i, "101", "103", "100.5", "102") for i in range(66, 70)]
                    candles += [bar(70, "101", "102", "99.98", "99.99"),
                                bar(71, "101", "103", "100.5", "102")]
                end = candles[-1].interval_end
                calendar = template.bootstrap.calendar
                seal = replace(template.replay.seal, sealed_end=end, seal_available_at=end,
                               expected_intervals_digest=interval_digest(
                                   intervals(calendar, *KEY, candles[0].interval_start, end)))
                replay = ReplayInput.locally_sealed(seal, tuple(candles))
                engine = type(template)(template.bootstrap, template.rules, tuple(template.specs.values()),
                                        spec_version="0.3.4", replay=replay)
                expected, actual = expected_and_actual(engine, candles)
                self.assertIsNone(expected[0])
                self.assertEqual(expected[:3], actual)

    def test_disjoint_lineage_survives_unrelated_sweep_conflict(self):
        from unittest.mock import patch

        template, candles = sealed()
        candles = candles[:66] + [
            bar(66, "102", "103", "100.5", "102"),
            bar(67, "102", "103", "101", "102"),
            bar(68, "102", "103", "101", "102"),
            bar(69, "108", "116", "107", "115"),
        ]
        end = candles[-1].interval_end
        calendar = template.bootstrap.calendar
        seal = replace(template.replay.seal, sealed_end=end, seal_available_at=end,
                       expected_intervals_digest=interval_digest(
                           intervals(calendar, *KEY, candles[0].interval_start, end)))
        engine = type(template)(template.bootstrap, template.rules, tuple(template.specs.values()),
                                spec_version="0.3.4", replay=ReplayInput.locally_sealed(seal, tuple(candles)))
        engine.process_batch(candles[:65])
        low = next(p for p in engine.bootstrap.structural.pool_registry.all_pools()
                   if p.level_ticks == 10000 and p.side.value == "LOW")
        extras = (replace(low, pool_id="conflict-a", level_ticks=10075),
                  replace(low, pool_id="conflict-b", level_ticks=10080))

        def frozen(source, candle):
            pools = source(candle)
            return (*pools, *extras) if candle == candles[66] else pools

        # Inject the same frozen reference inputs into both production reconstructions.
        # Core transition, proof and selection functions still run unmodified.
        owner = type(engine.bootstrap.structural)
        original = owner.eligible_pools_for
        with patch.object(owner, "eligible_pools_for", lambda analyzer, c:
                          frozen(lambda x: original(analyzer, x), c)):
            for candle in candles[65:]:
                engine.process_candle(candle)
            expected, actual = expected_and_actual(engine, candles, already_processed=True)
        self.assertEqual(expected[:3], actual)
        self.assertEqual(("sub-67", "sub-68", "sub-69"), actual[0][2])
        self.assertEqual(2, list(actual[1].values()).count("UNRESOLVED"))

    def test_behavioral_mutation_is_detected_even_with_identical_shape(self):
        engine, candles = sealed()
        expected, actual = expected_and_actual(engine, candles)
        self.assertEqual(expected[:3], actual)
        winner, outcomes, trace = actual
        faulty = (winner[0]-timedelta(minutes=1), *winner[1:]), outcomes, trace
        self.assertNotEqual(expected[:3], faulty)
        self.assertEqual(actual[1:], faulty[1:])

    def test_publication_keys_and_collisions(self):
        ledger = PublicationLedger()
        expected_records = {}
        for delay in (0, 2):
            engine, candles = sealed(delay=delay, ledger=ledger)
            expected, actual = expected_and_actual(engine, candles)
            self.assertEqual(expected[:3], actual)
            exchange, = engine.result().exchanges
            key = engine.evaluation_identity, exchange.exchange_id
            expected_records[key] = expected[0][0]
            ledger.publish(key[0], ((key[1], exchange),))
            with self.assertRaises(ValueError):
                ledger.publish(key[0], ((key[1], replace(exchange, available_at=exchange.available_at+
                                                       timedelta(minutes=1))),))
        self.assertEqual(1, len({key[1] for key in expected_records}))
        self.assertEqual(expected_records, {key: ledger.get(*key).available_at for key in expected_records})
