"""Versioned non-tradable Rule A contextual evaluation."""
from dataclasses import dataclass
from typing import ClassVar
from datetime import datetime
from decimal import Decimal

from .setup import LTF, LowerStream, _confirmation, _covered, _intact
from .structure.coverage import PublicationLedger, digest
from .structure.subsequent import SubsequentExchangeAnalyzer
from .timestamps import format_utc_timestamp


@dataclass(frozen=True, slots=True)
class ContextDisposition:
    schema: ClassVar[str] = "mlof.htf_context_evaluation.v1"
    evaluation_identity: str
    record_id: str
    exchange_id: str
    as_of: datetime
    status: str
    first_sweep_at: datetime | None
    reference_open_ticks: int | None
    extreme_ticks: int | None
    evidence_ids: tuple[str, ...]
    primary_range_id: str | None
    daily_range_id: str | None
    confirmation_at: datetime | None
    tradable: bool = False


def _prefix(engine, at):
    if engine.replay is None or engine.evaluation_identity is None:
        return None
    seal = engine.replay.seal
    key = seal.instrument, seal.timeframe
    processed = engine._candles.get(key, ())
    if any(event.status.value == "NOT_READY"
           and (event.instrument, event.timeframe) == key
           and event.reason.startswith("Structural stream is not READY")
           for event in engine.as_of(at).events):
        return None
    for source in sorted(engine._input_declarations, key=lambda item: item.seal.declaration_id):
        bars = _covered(source, engine.bootstrap.calendar, at, engine._trusted_external_seals)
        if bars and tuple(processed[:len(bars)]) == bars:
            return bars, source.seal.declaration_id
    return None


def _ranges(engine, view, at):
    initial = (r for r in engine.bootstrap.result().controls if r.available_at <= at)
    return (*view.ranges, *initial)


def _one(items):
    values = tuple(items)
    return values[0] if len(values) == 1 else None


def _location(direction, prices, band):
    if not all(band.low_ticks <= p <= band.high_ticks for p in prices):
        return "OUTSIDE_RANGE"
    twice_midpoint = band.low_ticks + band.high_ticks
    if any(2*p == twice_midpoint for p in prices):
        return "MIDPOINT_EQUALITY"
    if not all((2*p < twice_midpoint) == (direction == "bullish") for p in prices):
        return "LOCATION_REJECTED"
    return None


class ContextEvaluator:
    """Evaluate independent sealed streams under v0.3.5, without ancestry inference."""

    def __init__(self, ledger=None):
        self.ledger = ledger if ledger is not None else PublicationLedger()

    def as_of(self, at: datetime, exchange_id: str, execution: SubsequentExchangeAnalyzer,
              daily: SubsequentExchangeAnalyzer | None, four_hour: SubsequentExchangeAnalyzer | None,
              lower: tuple[LowerStream, ...] = ()) -> ContextDisposition:
        stamp = format_utc_timestamp(at)
        if at.utcoffset().total_seconds() != 0:
            raise ValueError("decision cut must be explicit UTC")
        scope = digest(["0.3.5", ContextDisposition.schema, "HTF_CONTEXT_ALIGNMENT",
                        [e.evaluation_identity if e else None for e in (execution, daily, four_hour)],
                        sorted(stream.identity for stream in lower)])
        record_id = "htf_context_evaluation_v1:" + digest([scope, exchange_id, stamp])
        evidence = []
        start = opening = extreme = primary = daily_range = None

        def result(status, confirmation=None):
            record = ContextDisposition(scope, record_id, exchange_id, at, status, start, opening,
                                        extreme, tuple(evidence), primary, daily_range, confirmation)
            self.ledger.publish(scope, ((record_id, record),))
            return record

        view = execution.as_of(at)
        exchange = _one(x for x in view.exchanges if x.exchange_id == exchange_id)
        if exchange is None:
            return result("ORDINARY_EXCHANGE_UNAVAILABLE")
        manipulation = _one(m for m in view.manipulations if m.manipulation_id == exchange.manipulation_id)
        controller = _one(c for c in view.controllers if c.exchange_id == exchange_id)
        if (not manipulation or not controller or not manipulation.sweep_candle_ids
                or controller.direction != exchange.direction or manipulation.direction != exchange.direction):
            return result("FORMATION_EVIDENCE_MISSING")
        band = _one(r for r in view.ranges if r.control_id == controller.control_id)
        covered = _prefix(execution, at)
        if covered is None:
            return result("EXECUTION_COVERAGE_INCOMPLETE")
        bars, seal = covered
        tick = execution.specs[controller.instrument].tick_size
        if band is None or not _intact(controller, band, bars, tick, at):
            return result("EXECUTION_LIFECYCLE_BLOCKED")
        first_id = manipulation.sweep_candle_ids[0]
        sweep = _one(c for c in bars if c.provenance.record_id == first_id)
        if sweep is None or sweep.source_available_at > exchange.available_at:
            return result("FORMATION_EVIDENCE_MISSING")
        start = sweep.interval_start
        opening, extreme = manipulation.reference_open_ticks, manipulation.manipulation_extreme_ticks
        if type(opening) is not int or type(extreme) is not int:
            return result("FORMATION_EVIDENCE_MISSING")
        evidence.extend((seal, exchange_id, manipulation.manipulation_id, first_id,
                         controller.control_id, band.range_id))
        for tf, engine in (("4h", four_hour), ("1d", daily)):
            if (engine is None or engine.replay is None or engine.spec_version != "0.3.5"
                    or (engine.replay.seal.instrument, engine.replay.seal.timeframe)
                    != (controller.instrument, tf)
                    or engine.specs[controller.instrument].tick_size != tick):
                return result("CONTEXT_STREAM_UNVERIFIED")
            before = engine.as_of(start)
            parent = _one(c for c in before.controllers if c.state != "REPLACED"
                          and (c.instrument, c.timeframe) == (controller.instrument, tf))
            if parent is None:
                return result("CONTEXT_NOT_AVAILABLE_AT_SWEEP")
            frozen = _one(r for r in _ranges(engine, before, start) if r.control_id == parent.control_id)
            if (frozen is None or frozen.instrument != controller.instrument
                    or frozen.timeframe != tf or frozen.direction != parent.direction
                    or frozen.low_ticks >= frozen.high_ticks):
                return result("CONTEXT_RANGE_UNAVAILABLE")
            # Separate streams do not provide a trusted cross-stream processing ordinal.
            if max(parent.available_at, frozen.available_at) >= start:
                return result("CONTEXT_PRIOR_PROCESSING_UNPROVEN")
            original = _prefix(engine, start)
            current = _prefix(engine, at)
            if original is None or current is None:
                return result("CONTEXT_COVERAGE_INCOMPLETE")
            latest = engine.as_of(at)
            same = _one(c for c in latest.controllers if c.control_id == parent.control_id)
            same_range = _one(r for r in _ranges(engine, latest, at) if r.control_id == parent.control_id)
            if (same is None or same_range != frozen
                    or not _intact(parent, frozen, original[0], tick, start)
                    or not _intact(same, frozen, current[0], tick, at)):
                return result("CONTEXT_LIFECYCLE_BLOCKED")
            if parent.direction != exchange.direction:
                return result("CONTEXT_DIRECTION_CONFLICT")
            if tf == "4h":
                primary = frozen.range_id
            else:
                daily_range = frozen.range_id
            evidence.extend((parent.control_id, frozen.range_id, original[1], current[1]))
            failure = _location(exchange.direction, (opening, extreme), frozen)
            if failure:
                return result(failure)
        if not lower:
            return result("HTF_CONTEXT_ALIGNMENT")
        options = LTF.get(controller.timeframe, ())
        prior = [bar for bar in bars if bar.source_available_at <= exchange.available_at]
        period = execution.rules.atr_period
        if len(prior) <= period:
            return result("ATR_UNAVAILABLE")
        atr = sum(max(b.high-b.low, abs(b.high-a.close), abs(b.low-a.close))
                  for a,b in zip(prior[-period-1:], prior[-period:])) / period
        radius = max(2*tick, Decimal("0.10")*atr)
        confirmations = []
        ready = False
        for priority, timeframe in enumerate(options):
            matches = [item for item in lower if (item.replay.seal.instrument, item.replay.seal.timeframe)
                       == (controller.instrument, timeframe)]
            if len(matches) > 1:
                return result("LTF_COVERAGE_AMBIGUOUS")
            if not matches or matches[0].instrument.tick_size != tick:
                continue
            stream = matches[0]
            rows = stream.observations(at)
            if rows is None:
                continue
            ready = True
            found = _confirmation(rows, exchange.available_at, exchange.direction,
                                  opening*tick, radius, tick, timeframe)
            if found:
                confirmations.append((max(found[0], stream.replay.seal.seal_available_at),
                                      priority, stream.identity, found[1:]))
        if not confirmations:
            return result("WAIT_LTF_RECLAIM" if ready else "LTF_COVERAGE_INCOMPLETE")
        confirmation, _, identity, references = min(confirmations)
        evidence.extend((identity, *references))
        return result("LTF_RECLAIM_CONFIRMED", confirmation)
