"""Offline structural setup dispositions at an explicit replay cut."""
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

from .models import InstrumentSpec, ReadinessState
from .structure.coverage import PublicationLedger, ReplayInput, digest
from .structure.engine import StructuralAnalyzer
from .structure.subsequent import SubsequentExchangeAnalyzer
from .timestamps import format_utc_timestamp

LTF = {"1m": ("15s", "30s"), "3m": ("1m",), "5m": ("1m",), "15m": ("3m",)}
RECLAIM_BARS = {"15s": 6, "30s": 6, "1m": 5, "3m": 4}


@dataclass(frozen=True, slots=True)
class AncestorLink:
    exchange_id: str
    execution_evaluation: str
    ancestor_evaluation: str
    controller_id: str
    range_id: str
    pool_id: str
    available_at: datetime
    evidence_id: str
    distance: int


@dataclass(frozen=True, slots=True)
class SetupDisposition:
    record_id: str
    status: str
    as_of: datetime
    exchange_id: str
    evidence_ids: tuple[str, ...]
    reference_open_ticks: int | None = None
    extreme_ticks: int | None = None
    confirmation_at: datetime | None = None
    tradable: bool = False


@dataclass(frozen=True, slots=True)
class LowerStream:
    replay: ReplayInput
    calendar: object
    config: object
    instrument: InstrumentSpec
    trusted_external_seals: frozenset[str] = frozenset()

    @property
    def identity(self):
        return digest([self.replay.seal.declaration_id,
                       self.replay.manifest_payload(self.instrument, self.config.config_version, "0.3.4")])

    def observations(self, at):
        candles = _covered(self.replay, self.calendar, at, self.trusted_external_seals)
        if candles is None:
            return None
        analyzer = StructuralAnalyzer(self.config, (self.instrument,), self.calendar)
        rows = []
        for candle in candles:
            analyzer.process_candle(candle)
            rows.append((candle, analyzer.eligible_pools_for(candle)))
        if not analyzer.readiness() or analyzer.readiness()[0].state is not ReadinessState.READY:
            return None
        return rows


def _covered(replay, calendar, at, trusted):
    prefix = tuple(c for c in replay.candles if c.interval_end <= at)
    error, _ = replay.input_error(calendar, at, prefix, trusted)
    return None if error or not prefix else prefix


def _controller_bars(engine, at):
    if engine.replay is None or engine.evaluation_identity is None or engine.spec_version != "0.3.4":
        return None
    key = engine.replay.seal.instrument, engine.replay.seal.timeframe
    processed = engine._candles.get(key, ())
    for declaration in engine._input_declarations:
        prefix = _covered(declaration, engine.bootstrap.calendar, at, engine._trusted_external_seals)
        if prefix is not None and tuple(processed[:len(prefix)]) == prefix:
            if not any(event.status.value == "NOT_READY" and event.available_at <= at
                       and (event.instrument, event.timeframe) == key
                       and event.reason.startswith("Structural stream is not READY")
                       for event in engine.as_of(at).events):
                return prefix
    return None


def _intact(controller, frozen, bars, tick, at):
    """With no verified reactivation history, a protective breach blocks progression."""
    protective = frozen.low_ticks if controller.direction == "bullish" else frozen.high_ticks
    for candle in bars:
        if candle.interval_end <= controller.available_at or candle.source_available_at > at:
            continue
        if (controller.direction == "bullish" and candle.low <= (protective-1)*tick
                or controller.direction == "bearish" and candle.high >= (protective+1)*tick):
            return False
    return controller.state == "ACTIVE" and controller.readiness == "READY"


def _confirmation(rows, start, direction, entry, radius, tick, timeframe):
    live = {}
    bullish = direction == "bullish"
    for i, (bar, pools) in enumerate(rows):
        if bar.interval_start < start:
            continue
        live = {key: value for key, value in live.items() if i-value[0] <= RECLAIM_BARS[timeframe]}
        for pool in pools:
            if (pool.available_at <= start or pool.available_at > bar.interval_start
                    or pool.side.value != ("LOW" if bullish else "HIGH")):
                continue
            level = pool.level_ticks*tick
            swept = bar.low <= level-tick if bullish else bar.high >= level+tick
            overlap = min(bar.high, entry+radius)-max(bar.low, entry-radius)
            if swept and overlap >= tick:
                prior = live.get(pool.pool_id)
                extreme = bar.low if bullish else bar.high
                if prior is None:
                    live[pool.pool_id] = i, pool, bar, extreme
                elif (extreme < prior[3] if bullish else extreme > prior[3]):
                    live[pool.pool_id] = prior[0], pool, bar, extreme
        completed = []
        for _, pool, sweep, _ in live.values():
            level = pool.level_ticks*tick
            if (bar.close > level if bullish else bar.close < level):
                completed.append((bar.source_available_at, pool.pool_id,
                                  sweep.provenance.record_id, bar.provenance.record_id))
        if completed:
            return min(completed) if len(completed) == 1 else None
    return None


class SetupDetector:
    """Project independently supplied evidence. Links must be pinned by the caller."""

    def __init__(self, ledger=None):
        self.ledger = ledger if ledger is not None else PublicationLedger()

    def as_of(self, at: datetime, exchange_id: str, execution: SubsequentExchangeAnalyzer,
              ancestors: tuple[SubsequentExchangeAnalyzer, ...], links: tuple[AncestorLink, ...],
              lower: tuple[LowerStream, ...], *, verified_link_ids: frozenset[str] = frozenset()):
        if at.utcoffset() != timedelta(0):
            raise ValueError("setup decision requires explicit UTC")
        scope = digest([execution.evaluation_identity,
                        sorted(e.evaluation_identity or "" for e in ancestors),
                        sorted(s.identity for s in lower), sorted(verified_link_ids),
                        sorted((link.evidence_id, link.exchange_id, link.execution_evaluation,
                                link.ancestor_evaluation, link.controller_id, link.range_id,
                                link.pool_id, format_utc_timestamp(link.available_at), link.distance)
                               for link in links)])
        record_id = "setup_disposition_v1:" + digest([scope, exchange_id, format_utc_timestamp(at)])
        evidence = []
        opening = extreme = None

        def result(status, confirmation=None):
            record = SetupDisposition(record_id, status, at, exchange_id, tuple(evidence),
                                      opening, extreme, confirmation)
            self.ledger.publish(scope, ((record_id, record),))
            return record

        view = execution.as_of(at)
        exchange = next((x for x in view.exchanges if x.exchange_id == exchange_id), None)
        if exchange is None:
            return result("ORDINARY_EXCHANGE_UNAVAILABLE")
        evidence.append(exchange.exchange_id)
        bars = _controller_bars(execution, at)
        if bars is None:
            return result("EXECUTION_COVERAGE_INCOMPLETE")
        controller = next((c for c in view.controllers if c.exchange_id == exchange_id), None)
        frozen = next((r for r in view.ranges if controller and r.control_id == controller.control_id), None)
        manipulation = next((m for m in view.manipulations if m.manipulation_id == exchange.manipulation_id), None)
        if controller is None or frozen is None or manipulation is None:
            return result("CONTROLLER_NOT_READY")
        tick = execution.specs[controller.instrument].tick_size
        if not _intact(controller, frozen, bars, tick, at):
            return result("CONTROLLER_INELIGIBLE")
        evidence.extend((manipulation.manipulation_id, controller.control_id, frozen.range_id))
        opening, extreme = manipulation.reference_open_ticks, manipulation.manipulation_extreme_ticks
        if opening is None:
            return result("MANIPULATION_OPEN_UNAVAILABLE")
        applicable = [link for link in links if link.exchange_id == exchange_id
                      and link.execution_evaluation == execution.evaluation_identity
                      and link.available_at <= at]
        if not applicable:
            return result("ANCESTRY_MISSING")
        distances = []
        seen = set()
        for link in applicable:
            if (link.evidence_id not in verified_link_ids or link.distance < 1
                    or link.pool_id != manipulation.liquidity_pool_id or link.controller_id in seen):
                return result("ANCESTRY_UNVERIFIED_OR_CONFLICTING")
            matches = [engine for engine in ancestors if engine.evaluation_identity == link.ancestor_evaluation]
            if len(matches) != 1:
                return result("ANCESTRY_UNVERIFIED_OR_CONFLICTING")
            ancestor = matches[0]
            history = _controller_bars(ancestor, at)
            if history is None:
                return result("ANCESTOR_COVERAGE_INCOMPLETE")
            snapshot = ancestor.as_of(at)
            parent = next((c for c in snapshot.controllers if c.control_id == link.controller_id), None)
            band = next((r for r in snapshot.ranges if r.range_id == link.range_id
                         and r.control_id == link.controller_id), None)
            if (parent is None or band is None or parent.instrument != controller.instrument
                    or parent.timeframe == controller.timeframe
                    or ancestor.specs[parent.instrument].tick_size != tick):
                return result("ANCESTOR_RANGE_UNAVAILABLE")
            if not _intact(parent, band, history, tick, at):
                return result("ANCESTOR_INELIGIBLE")
            if not all(band.low_ticks <= price <= band.high_ticks for price in (opening, extreme)):
                return result("ANCESTOR_RANGE_EXCLUDES_MANIPULATION")
            midpoint_twice = band.low_ticks + band.high_ticks
            bullish = exchange.direction == "bullish"
            if not all(2*price < midpoint_twice if bullish else 2*price > midpoint_twice
                       for price in (opening, extreme)):
                return result("LOCATION_REJECTED")
            seen.add(link.controller_id)
            distances.append(link.distance)
            evidence.extend((link.evidence_id, parent.control_id, band.range_id))
        if distances.count(min(distances)) != 1:
            return result("ANCESTRY_UNVERIFIED_OR_CONFLICTING")
        options = LTF.get(controller.timeframe)
        if options is None:
            return result("EXECUTION_TIMEFRAME_UNSUPPORTED")
        # Freeze ATR from completed evidence visible at exchange publication.
        past = [c for c in bars if c.source_available_at <= exchange.available_at]
        period = execution.rules.atr_period
        if len(past) <= period:
            return result("ATR_UNAVAILABLE")
        ranges = [max(b.high-b.low, abs(b.high-a.close), abs(b.low-a.close))
                  for a, b in zip(past[-period-1:], past[-period:])]
        radius = max(2*tick, Decimal("0.10")*sum(ranges)/period)
        available = []
        confirmations = []
        for timeframe in options:
            matches = [stream for stream in lower if
                       (stream.replay.seal.instrument, stream.replay.seal.timeframe) ==
                       (controller.instrument, timeframe)]
            if len(matches) != 1 or matches[0].instrument.tick_size != tick:
                continue
            stream = matches[0]
            rows = stream.observations(at)
            if rows is None:
                continue
            available.append(timeframe)
            confirmation = _confirmation(rows, exchange.available_at, exchange.direction,
                                         opening*tick, radius, tick, timeframe)
            if confirmation:
                confirmations.append((max(confirmation[0], stream.replay.seal.seal_available_at),
                                      options.index(timeframe), stream.identity, confirmation))
        if not available:
            return result("LTF_COVERAGE_INCOMPLETE")
        if any(sum((stream.replay.seal.instrument, stream.replay.seal.timeframe) ==
                   (controller.instrument, tf) for stream in lower) > 1 for tf in options):
            return result("LTF_COVERAGE_AMBIGUOUS")
        if not confirmations:
            return result("WAIT_LTF_RECLAIM")
        visible_at, _, identity, chosen = min(confirmations)
        evidence.extend((identity, chosen[1], chosen[2], chosen[3]))
        return result("LTF_RECLAIM_CONFIRMED", visible_at)
