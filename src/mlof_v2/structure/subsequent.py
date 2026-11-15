"""Offline, non-tradable ordinary exchanges after bootstrap control."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Iterable, Sequence

from ..calendars import SessionCalendar
from ..models import Candle, InstrumentSpec, ReadinessState
from ..timestamps import format_utc_timestamp
from .bootstrap import (BootstrapAnalyzer, BootstrapExchange, BootstrapReplayResult,
                        BootstrapRules, FrozenControlRange, _canonical_hash)
from .coverage import (CoverageRecord, PublicationLedger, ReplayInput, canonical_bytes, digest, interval_digest)
from .config import RECLAIM_PROFILE_VERSION, SwingWidths, reclaim_deadline
from .models import CandleEvidence, LiquidityPool, PivotSide, StructuralReplayResult, SwingCluster


class SubsequentStatus(str, Enum):
    OBSERVING = "OBSERVING"
    SWEEP = "SWEEP"
    RECLAIM = "RECLAIM"
    DISPLACEMENT = "DISPLACEMENT"
    EXCHANGE_CONFIRMED = "EXCHANGE_CONFIRMED"
    CANDIDATE_RESET = "CANDIDATE_RESET"
    REJECTED = "REJECTED"
    NOT_READY = "NOT_READY"
    CONTROLLER_REPLACED = "CONTROLLER_REPLACED"
    STRUCTURAL_PROMOTION = "STRUCTURAL_PROMOTION"


@dataclass(frozen=True, slots=True)
class SubsequentEvent:
    event_id: str
    status: SubsequentStatus
    instrument: str
    timeframe: str
    direction: str | None
    available_at: datetime
    source_candle_ids: tuple[str, ...]
    related_ids: tuple[str, ...]
    reason: str
    config_version: str
    spec_version: str
    source_evidence: tuple[CandleEvidence, ...] = ()
    candidate_reset: CandidateResetEvidence | None = None
    tradable: bool = False


@dataclass(frozen=True, slots=True)
class CandidateResetEvidence:
    parent_id: str
    canceled_candidate_id: str
    replacement_candidate_id: str
    parent_controller_id: str
    liquidity_pool_id: str
    direction: str
    triggering_candle_id: str
    interval_start: datetime
    interval_end: datetime
    source_available_at: datetime
    new_extreme_ticks: int
    canceled_version: int
    replacement_version: int
    config_version: str
    spec_version: str
    source_evidence: CandleEvidence

    def __post_init__(self) -> None:
        if type(self.new_extreme_ticks) is not int:
            raise ValueError("candidate reset extreme must be an integer tick value")
        if self.replacement_version != self.canceled_version + 1:
            raise ValueError("candidate reset versions must advance exactly once")
        if self.source_evidence.record_id != self.triggering_candle_id:
            raise ValueError("candidate reset provenance must identify its triggering candle")
        if (self.interval_start != self.source_evidence.interval_start
                or self.interval_end != self.source_evidence.interval_end
                or self.source_available_at != self.source_evidence.source_available_at):
            raise ValueError("candidate reset timing must match its triggering candle provenance")


@dataclass(frozen=True, slots=True)
class SubsequentExchange:
    exchange_id: str
    parent_controller_id: str
    manipulation_id: str
    break_reference_swing_id: str
    displacement_candle_id: str
    fvg_candle_ids: tuple[str, str, str]
    direction: str
    available_at: datetime
    config_version: str
    spec_version: str
    source_evidence: tuple[CandleEvidence, ...]
    tradable: bool = False


@dataclass(frozen=True, slots=True)
class SubsequentManipulation:
    manipulation_id: str
    parent_id: str
    candidate_id: str
    parent_controller_id: str
    liquidity_pool_id: str
    origin_swing_cluster_id: str
    direction: str
    sweep_candle_ids: tuple[str, ...]
    reclaim_candle_id: str
    manipulation_extreme_ticks: int
    available_at: datetime
    config_version: str
    spec_version: str
    source_evidence: tuple[CandleEvidence, ...]
    tradable: bool = False
    frozen_control_range_id: str = ""
    successor_protective_swing_id: str = ""
    reference_open_ticks: int | None = None


@dataclass(frozen=True, slots=True)
class SubsequentController:
    control_id: str
    parent_controller_id: str
    exchange_id: str
    manipulation_id: str
    direction: str
    instrument: str
    timeframe: str
    protective_swing_id: str | None
    break_reference_swing_id: str
    available_at: datetime
    readiness: str
    config_version: str
    spec_version: str
    replaced_at: datetime | None = None
    state: str = "ACTIVE"
    tradable: bool = False


@dataclass(frozen=True, slots=True)
class SubsequentRange:
    range_id: str
    control_id: str
    low_swing_id: str
    high_swing_id: str
    low_ticks: int
    high_ticks: int
    available_at: datetime
    config_version: str
    spec_version: str
    tradable: bool = False
    instrument: str = ""
    timeframe: str = ""
    direction: str = ""


@dataclass(frozen=True, slots=True)
class SubsequentReplayResult:
    bootstrap: BootstrapReplayResult
    events: tuple[SubsequentEvent, ...]
    manipulations: tuple[SubsequentManipulation, ...]
    exchanges: tuple[SubsequentExchange, ...]
    controllers: tuple[SubsequentController, ...]
    ranges: tuple[SubsequentRange, ...]
    structural_swing_ids: tuple[str, ...]
    active_controller_ids: tuple[str, ...]
    replaced_controller_ids: tuple[str, ...]
    coverage: tuple[CoverageRecord, ...] = ()


@dataclass(frozen=True, slots=True)
class SubsequentAsOfView:
    """Ordinary-exchange records visible at one explicit UTC information time."""

    as_of_utc: datetime
    events: tuple[SubsequentEvent, ...]
    manipulations: tuple[SubsequentManipulation, ...]
    exchanges: tuple[SubsequentExchange, ...]
    controllers: tuple[SubsequentController, ...]
    ranges: tuple[SubsequentRange, ...]
    active_controller_ids: tuple[str, ...]
    replaced_controller_ids: tuple[str, ...]
    coverage: tuple[CoverageRecord, ...] = ()


@dataclass(slots=True)
class _Candidate:
    parent_id: str
    candidate_id: str
    version: int
    pool: LiquidityPool
    direction: str
    parent_controller_id: str
    sweep_ids: list[str]
    extreme_ticks: int
    first_sweep_index: int
    last_sweep_index: int
    reclaim_index: int | None = None
    reclaim_id: str | None = None
    break_index: int | None = None
    frozen_range_id: str = ""
    protective_swing_id: str = ""
    protective_boundary_ticks: int | None = None
    reference_open_ticks: int | None = None


class SubsequentExchangeAnalyzer:
    """Compose the existing bootstrap and structural analyzers with ordinary exchanges.

    This emits immutable structural evidence and non-tradable controller records only.
    It has no signal, setup, order, position, valuation, fill, or P&L interface.
    """

    def __init__(self, bootstrap: BootstrapAnalyzer, rules: BootstrapRules,
                 instrument_specs: Sequence[InstrumentSpec], *, spec_version: str,
                 replay: ReplayInput | None = None, ledger: PublicationLedger | None = None,
                 trusted_external_seals: frozenset[str] = frozenset(),
                 additional_seals: tuple[ReplayInput, ...] = ()) -> None:
        if bootstrap.rules.config_version != rules.config_version:
            raise ValueError("bootstrap and subsequent-exchange configuration versions must match")
        if not isinstance(spec_version, str) or not spec_version.strip():
            raise ValueError("the controlled specification version must be supplied explicitly")
        if rules.config_version == RECLAIM_PROFILE_VERSION:
            if (spec_version != "0.3.5" or rules != BootstrapRules(RECLAIM_PROFILE_VERSION)
                    or bootstrap.rules != rules
                    or bootstrap.config.config_version != RECLAIM_PROFILE_VERSION
                    or bootstrap.config.minimum_contiguous_bars != 50
                    or any(widths != SwingWidths(2, 2) for timeframe, widths in bootstrap.config.pivot_widths
                           if timeframe in ("4h", "1d"))):
                raise ValueError("M4.2 requires the approved v0.3.5 formation profile")
        self.bootstrap = bootstrap
        self.rules = rules
        self.spec_version = spec_version
        self.specs = {x.instrument.upper(): x for x in instrument_specs}
        self.replay = replay
        self._trusted_external_seals = frozenset(trusted_external_seals)
        self._input_declarations = ((replay,) + tuple(additional_seals)) if replay else ()
        for declaration in additional_seals:
            if (replay is None or declaration.candles != replay.candles
                    or any(getattr(declaration.seal, name) != getattr(replay.seal, name)
                           for name in ("dataset_id", "instrument", "timeframe", "declared_start",
                                        "calendar_version", "source_timezone", "timestamp_convention"))):
                raise ValueError("a rolling seal must bind the same replay dataset and calendar")
        self._cut_input = replay
        self.ledger = ledger if ledger is not None else PublicationLedger()
        self.evaluation_identity = None
        self._manifest = None
        self._coverage: list[CoverageRecord] = []
        self._lineages: dict[tuple[str, str], _Candidate] = {}
        self._ready: dict[str, tuple] = {}
        self._outcomes: dict[str, dict] = {}
        self._outcome_parents: dict[str, str] = {}
        self._conflicts: list[tuple[str, str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = []
        self._structural_rows: list[list] = []
        self._observations: list[list] = []
        self._transition_trace: list[tuple] = []
        self._verifying = False
        self._resolution = None
        self._expected_resolution = None
        self._expected_sweeps: set[tuple[str, str, str]] = set()
        self._rejected_evidence: dict[str, set[str]] = {}
        self._input_invalid = False
        self._publication_failed = False
        self._cut = None
        if replay is not None:
            seal = replay.seal
            self._manifest = replay.manifest_payload(self.specs[seal.instrument], rules.config_version, spec_version)
            self.manifest_id = "replay_manifest_v1:" + digest(self._manifest)
            self.evaluation_identity = "exchange_evaluation_v1:" + digest(
                [self.manifest_id, sorted(item.seal.declaration_id for item in self._input_declarations),
                 "m3b1-0.3.4-v2"])
        self._candles: dict[tuple[str, str], list[Candle]] = {}
        self._active: dict[tuple[str, str], SubsequentController] = {}
        self._candidate: dict[tuple[str, str], _Candidate] = {}
        self._events: dict[str, SubsequentEvent] = {}
        self._manipulations: dict[str, SubsequentManipulation] = {}
        self._exchanges: dict[str, SubsequentExchange] = {}
        self._controllers: dict[str, SubsequentController] = {}
        self._ranges: dict[str, SubsequentRange] = {}
        self._controller_ranges: dict[str, FrozenControlRange | SubsequentRange] = {}
        self._id_payloads: dict[str, object] = {}
        self._structural: set[str] = set()
        self._structural_promotion_sources: dict[str, str] = {}
        self._last_bootstrap_ids: set[str] = set()
        self._bootstrap_control_ids: set[str] = set()
        self._replaced: set[str] = set()
        self._bootstrap_control_ids: set[str] = set()

    def process_candle(self, candle: Candle) -> tuple[SubsequentEvent, ...]:
        if self.replay is None:
            return self._process_candle(candle)
        seal = self.replay.seal
        key = (seal.instrument, seal.timeframe)
        prior = self._candles.get(key, ())
        index = len(prior)
        if (index >= len(self.replay.candles) or self.replay.candles[index] != candle
                or (prior and candle.source_available_at < prior[-1].source_available_at)):
            self._input_invalid = True
            raise ValueError("incremental input differs from immutable market-ordered replay")
        before = set(self._events)
        at = candle.source_available_at
        if prior:
            self._between_cuts(prior[-1].source_available_at, at)
        self._process_candle(candle)
        if index + 1 == len(self.replay.candles) or self.replay.candles[index + 1].source_available_at > at:
            self.evaluate(at)
        return self._new(before)

    def _between_cuts(self, previous: datetime, current: datetime) -> None:
        times = {item.seal.seal_available_at for item in self._input_declarations}
        for at in sorted(time for time in times if previous < time < current):
            self.evaluate(at)

    def _process_candle(self, candle: Candle) -> tuple[SubsequentEvent, ...]:
        before = set(self._events)
        key = (candle.instrument.upper(), candle.timeframe.lower())
        bars = self._candles.setdefault(key, [])
        # Bootstrap owns the one initial-control path and advances structural state once.
        context = key[1] in ("4h", "1d")
        if context:
            verified = key not in self._active and self.rules.config_version == RECLAIM_PROFILE_VERSION and any(
                declaration.input_error(
                    self.bootstrap.calendar, candle.source_available_at,
                    tuple(bars) + (candle,),
                    self._trusted_external_seals,
                )[0] is None
                for declaration in self._input_declarations
            )
            self.bootstrap._process_candle(candle, verified_context=verified)
        else:
            self.bootstrap.process_candle(candle)
        result = self.bootstrap.result()
        stream_readiness = next((r for r in result.structural.readiness
                                 if (r.instrument, r.timeframe) == key), None)
        if stream_readiness is not None and stream_readiness.state is ReadinessState.READY:
            self._promote_independent_swings(candle, result.structural)
        current_boot = next((x for x in result.controls if (x.instrument, x.timeframe) == key), None)
        active = self._active.get(key)
        if active is None and current_boot is not None:
            initial_exchange = next((x for x in result.exchanges if x.event_id == current_boot.exchange_id), None)
            if initial_exchange is None:
                self._emit(SubsequentStatus.NOT_READY, candle, None, (current_boot.control_id,),
                           "Bootstrap exchange lineage is missing; no ordinary controller can be initialized")
                bars.append(candle)
                return self._new(before)
            active = SubsequentController(
                current_boot.control_id, "", current_boot.exchange_id,
                initial_exchange.manipulation_id,
                current_boot.direction, key[0], key[1], current_boot.protective_cluster_id,
                current_boot.break_reference_cluster_id, current_boot.available_at,
                "READY", current_boot.config_version, self.spec_version)
            self._active[key] = active
            self._controllers.setdefault(active.control_id, active)
            self._controller_ranges[active.control_id] = current_boot
            self._last_bootstrap_ids.add(active.control_id)
            self._bootstrap_control_ids.add(active.control_id)
            self._bootstrap_control_ids.add(active.control_id)
            # The bootstrap completion candle cannot also begin a dependent ordinary event.
            bars.append(candle)
            return self._new(before)
        if active is None:
            bars.append(candle)
            return self._new(before)
        # Range readiness is fixed at formation. The current controller still
        # participates in discovery; _frozen_boundary gates qualification.
        if active.available_at > candle.interval_start:
            self._emit(SubsequentStatus.NOT_READY, candle, None, (active.control_id,),
                       "Current controller is not causally available")
            bars.append(candle)
            return self._new(before)
        readiness = next((r for r in result.structural.readiness
                          if (r.instrument, r.timeframe) == key), None)
        if readiness is None or readiness.state is not ReadinessState.READY:
            if self.replay is not None:
                self._input_invalid = True
            self._candidate.pop(key, None)
            self._emit(SubsequentStatus.NOT_READY, candle, None, (active.control_id,),
                       "Structural stream is not READY; ordinary exchange evidence is blocked")
            bars.append(candle)
            return self._new(before)
        bars.append(candle)
        if len(bars) < self.rules.atr_period + 2:
            self._emit(SubsequentStatus.OBSERVING, candle, None, (active.control_id,),
                       "Insufficient contiguous ATR history for qualified displacement")
            return self._new(before)
        pools = self.bootstrap.structural.eligible_pools_for(candle)
        if self.replay is None:
            if self.spec_version == "0.3.4":
                self._emit(SubsequentStatus.NOT_READY, candle, None, (active.control_id,),
                           "Governing v0.3.4 requires a source seal and immutable replay")
            else:
                self._advance(key, active, candle, bars, result.structural, pools)
        else:
            self._discover(key, active, candle, bars, result.structural, pools)
        return self._new(before)

    def process_batch(self, candles: Iterable[Candle]) -> SubsequentReplayResult:
        ordered = sorted(candles, key=lambda c: (c.instrument.upper(), c.timeframe.lower(),
                                                  c.interval_start, c.source_available_at,
                                                  c.provenance.record_id))
        for candle in ordered:
            self.process_candle(candle)
        if self.replay is not None and ordered:
            seal = self.replay.seal
            processed = self._candles.get((seal.instrument, seal.timeframe), ())
            if len(processed) < len(self.replay.candles):
                return self.result()
            for at in sorted({item.seal.seal_available_at for item in self._input_declarations}):
                if at > ordered[-1].source_available_at:
                    self.evaluate(at)
        return self.result()

    def result(self) -> SubsequentReplayResult:
        return SubsequentReplayResult(
            self.bootstrap.result(),
            tuple(sorted(self._events.values(), key=lambda e: (e.available_at, e.instrument, e.timeframe, e.event_id))),
            tuple(sorted(self._manipulations.values(), key=lambda x: (x.available_at, x.manipulation_id))),
            tuple(sorted(self._exchanges.values(), key=lambda x: (x.available_at, x.exchange_id))),
            tuple(sorted(self._controllers.values(), key=lambda x: (x.available_at, x.control_id))),
            tuple(sorted(self._ranges.values(), key=lambda x: (x.available_at, x.range_id))),
            tuple(sorted(self._structural)),
            tuple(sorted(x.control_id for x in self._active.values()
                         if x.state == "ACTIVE" and x.readiness == "READY")),
            tuple(sorted(self._replaced)),
            tuple(self._coverage),
        )

    def as_of(self, as_of_utc: datetime) -> SubsequentAsOfView:
        """Return ordinary-exchange records whose recorded information time has arrived."""
        if (as_of_utc.tzinfo is None or as_of_utc.utcoffset() is None
                or as_of_utc.utcoffset() != timedelta(0)):
            raise ValueError("as_of_utc must be an explicit UTC timestamp")
        visible_events = tuple(sorted(
            (e for e in self._events.values() if e.available_at <= as_of_utc),
            key=lambda e: (e.available_at, e.instrument, e.timeframe, e.event_id)))
        visible_manipulations = tuple(sorted(
            (x for x in self._manipulations.values() if x.available_at <= as_of_utc),
            key=lambda x: (x.available_at, x.manipulation_id)))
        visible_exchanges = tuple(sorted(
            (x for x in self._exchanges.values() if x.available_at <= as_of_utc),
            key=lambda x: (x.available_at, x.exchange_id)))
        visible_controllers = tuple(sorted(
            (x for x in self._controllers.values() if x.available_at <= as_of_utc),
            key=lambda x: (x.available_at, x.control_id)))
        visible_ranges = tuple(sorted(
            (x for x in self._ranges.values() if x.available_at <= as_of_utc),
            key=lambda x: (x.available_at, x.range_id)))
        replacement_times = {x.parent_controller_id: x.available_at for x in visible_exchanges}
        replaced = set(replacement_times)
        visible_controllers = tuple(
            replace(x, state="REPLACED", replaced_at=replacement_times[x.control_id])
            if x.control_id in replaced else x for x in visible_controllers)
        active = tuple(sorted(
            x.control_id for x in visible_controllers
            if x.state == "ACTIVE" and x.readiness == "READY" and x.control_id not in replaced))
        return SubsequentAsOfView(
            as_of_utc, visible_events, visible_manipulations, visible_exchanges,
            visible_controllers, visible_ranges, active, tuple(sorted(replaced)),
            tuple(record for record in self._coverage if record.information_time <= as_of_utc))

    def _promote_independent_swings(self, candle: Candle, structural: StructuralReplayResult) -> None:
        """Section 3 candidate->STRUCTURAL promotion on a later quality close-break."""
        key = (candle.instrument.upper(), candle.timeframe.lower())
        bars = self._candles.setdefault(key, [])
        evidence_bars = bars if bars and bars[-1].provenance.record_id == candle.provenance.record_id else bars + [candle]
        spec = self.specs.get(key[0])
        if spec is None or len(evidence_bars) <= self.rules.atr_period:
            return
        close = spec.price_to_ticks(candle.close)
        for swing in structural.swings:
            if (swing.instrument, swing.timeframe) != key or swing.cluster_id in self._structural:
                continue
            if swing.available_at > candle.interval_start:
                continue
            if swing.side is PivotSide.HIGH:
                direction, broken = "bullish", close >= swing.level_ticks + 1
            else:
                direction, broken = "bearish", close <= swing.level_ticks - 1
            if broken and self._qualifies_displacement(evidence_bars, len(evidence_bars) - 1, direction, swing.level_ticks):
                self._structural.add(swing.cluster_id)
                self._structural_promotion_sources[swing.cluster_id] = candle.provenance.record_id
                proof = list(swing.comparison_evidence)
                proof.append(CandleEvidence(candle.provenance.record_id, candle.interval_start,
                                            candle.interval_end, candle.source_available_at, candle.provenance))
                proof = tuple(sorted({x.record_id: x for x in proof}.values(),
                                     key=lambda x: (x.interval_start, x.interval_end, x.record_id)))
                self._emit(SubsequentStatus.STRUCTURAL_PROMOTION, candle, direction,
                           (swing.cluster_id,), "Pre-registered swing independently promoted by a later qualified close-break", proof)

    def _retained_anchor_error(self, controller: SubsequentController, sweep: Candle,
                               structural: StructuralReplayResult) -> str:
        manipulation = self._manipulations.get(controller.manipulation_id)
        exchange = self._exchanges.get(controller.exchange_id)
        anchor = controller.protective_swing_id
        if (not anchor or manipulation is None or exchange is None
                or manipulation.successor_protective_swing_id != anchor
                or exchange.manipulation_id != manipulation.manipulation_id
                or exchange.parent_controller_id != controller.parent_controller_id
                or manipulation.parent_controller_id != controller.parent_controller_id
                or exchange.direction != controller.direction
                or manipulation.direction != controller.direction
                or exchange.break_reference_swing_id != controller.break_reference_swing_id
                or exchange.available_at != controller.available_at):
            return "Missing or contradictory retained controller formation association"
        key = controller.instrument, controller.timeframe
        if ((sweep.instrument.upper(), sweep.timeframe.lower()) != key
                or controller.available_at > sweep.interval_start
                or manipulation.available_at > sweep.interval_start):
            return "Retained association was not causally available for this stream"
        bars = self._candles.get(key, ())
        first_ids = manipulation.sweep_candle_ids[:1]
        formation = [bar for bar in bars if bar.provenance.record_id in first_ids]
        promotions = [bar for bar in bars if bar.provenance.record_id ==
                      self._structural_promotion_sources.get(anchor)]
        swings = [swing for swing in structural.swings if swing.cluster_id == anchor]
        side = PivotSide.LOW if controller.direction == "bullish" else PivotSide.HIGH
        if (len(formation) != 1 or len(promotions) != 1 or len(swings) != 1
                or anchor not in self._structural or swings[0].side is not side
                or (swings[0].instrument, swings[0].timeframe) != key):
            return "Retained anchor lacks unique same-stream STRUCTURAL evidence"
        deadline = formation[0].interval_start
        if max(swings[0].available_at, promotions[0].interval_end,
               promotions[0].source_available_at) > deadline:
            return "Retained anchor missed its original formation deadline"
        return ""

    def _frozen_boundary(self, active: SubsequentController, direction: str, candle: Candle,
                         structural: StructuralReplayResult):
        """Resolve only the direction-appropriate boundary already frozen to the active controller."""
        frozen = self._controller_ranges.get(active.control_id)
        if active.readiness != "READY":
            error = self._retained_anchor_error(active, candle, structural)
            if error:
                return frozen, None, error
        if frozen is None or frozen.control_id != active.control_id:
            return None, None, "Active controller has no uniquely linked frozen range"
        if frozen.available_at > candle.interval_start:
            return frozen, None, "Active controller range was unavailable before the first sweep interval"
        if (getattr(frozen, "instrument", active.instrument).upper() != active.instrument.upper()
                or getattr(frozen, "timeframe", active.timeframe).lower() != active.timeframe.lower()):
            return frozen, None, "Frozen range belongs to a different instrument or timeframe"
        if getattr(frozen, "direction", active.direction) not in ("", active.direction):
            return frozen, None, "Frozen range direction conflicts with its active controller"
        low_swing_id = getattr(frozen, "low_cluster_id", getattr(frozen, "low_swing_id", ""))
        high_swing_id = getattr(frozen, "high_cluster_id", getattr(frozen, "high_swing_id", ""))

        if direction == "bullish":
            side, swing_id, boundary_ticks = PivotSide.LOW, low_swing_id, frozen.low_ticks
            opposite_id = high_swing_id
        else:
            side, swing_id, boundary_ticks = PivotSide.HIGH, high_swing_id, frozen.high_ticks
            opposite_id = low_swing_id
        if active.direction != ("bearish" if direction == "bullish" else "bullish"):
            return frozen, None, "Successor direction does not oppose the active controller"
        if active.protective_swing_id != opposite_id:
            return frozen, None, "Frozen range's opposing boundary is not the active controller's linked protective swing"

        matches = [s for s in structural.swings if s.cluster_id == swing_id]
        if len(matches) != 1:
            return frozen, None, "Exact frozen protective boundary has missing or ambiguous swing provenance"
        swing = matches[0]
        key = (active.instrument.upper(), active.timeframe.lower())
        if ((swing.instrument.upper(), swing.timeframe.lower()) != key or swing.side is not side
                or swing.level_ticks != boundary_ticks):
            return frozen, None, "Frozen protective boundary does not match its same-stream swing identity and price"
        if swing.available_at > candle.interval_start:
            return frozen, None, "Frozen protective swing was unavailable before the first sweep interval"
        if swing.cluster_id not in self._structural:
            return frozen, None, "Frozen protective boundary lacks independent STRUCTURAL status"
        promotion_source_id = self._structural_promotion_sources.get(swing.cluster_id)
        source_candle = next((b for b in self._candles.get(key, ())
                              if b.provenance.record_id == promotion_source_id), None)
        if (source_candle is None or source_candle.interval_end > candle.interval_start
                or source_candle.source_available_at > candle.interval_start):
            return frozen, None, "Independent structural promotion was not available before the first sweep interval"
        return frozen, swing, ""

    @staticmethod
    def _inside_frozen_range(direction: str, extreme_ticks: int, reference_open_ticks: int,
                             frozen: FrozenControlRange | SubsequentRange) -> bool:
        if direction == "bullish":
            return (frozen.low_ticks < extreme_ticks <= frozen.high_ticks
                    and frozen.low_ticks <= reference_open_ticks <= frozen.high_ticks)
        return (frozen.low_ticks <= extreme_ticks < frozen.high_ticks
                and frozen.low_ticks <= reference_open_ticks <= frozen.high_ticks)

    def _observe(self, candle, pool, parent, candidate, action):
        self._observations.append([
            candle.provenance.record_id, pool.pool_id, parent.control_id,
            candidate.parent_id if candidate else None, candidate.version if candidate else None, action,
        ])

    def _describe(self, candidate, state, completion=None, extra=()):
        self._transition_trace.append((candidate.candidate_id, state,
                                       completion.provenance.record_id if completion else None,
                                       candidate.reclaim_id, candidate.break_index, tuple(extra),
                                       self._candles[(candidate.pool.instrument, candidate.pool.timeframe)][-1]
                                       .provenance.record_id))
        self._outcome_parents[candidate.candidate_id] = candidate.parent_controller_id
        evidence = set(candidate.sweep_ids).union(extra)
        if candidate.reclaim_id:
            evidence.add(candidate.reclaim_id)
        self._outcomes[candidate.candidate_id] = {
            "lineage_id": candidate.parent_id, "version": candidate.version, "state": state,
            "completion_interval": ([format_utc_timestamp(completion.interval_start),
                                     format_utc_timestamp(completion.interval_end)] if completion else None),
            "evidence_ready_at": None, "event_id": None, "evidence_ids": sorted(evidence),
        }

    def _conflict_if_completion(self, parent, candidate, candle, bars):
        current = candidate
        if candidate.break_index is None:
            reference = next((s for s in self.bootstrap.structural.result().swings
                              if s.cluster_id == parent.protective_swing_id
                              and s.available_at <= candle.interval_start), None)
            index = len(bars) - 1
            if reference is None or not self._qualifies_displacement(
                    bars, index, candidate.direction, reference.level_ticks):
                return
            current = replace(candidate, break_index=index)
        fvg = self._find_fvg(bars, current)
        if fvg is not None and max(current.break_index, fvg[0]) == len(bars) - 1:
            self._describe(candidate, "UNRESOLVED", candle, fvg[1])
            conflict = (parent.control_id, "SAME_CANDLE_LEG_END",
                        (candle.provenance.record_id,), (candidate.pool.pool_id,), (candidate.parent_id,))
            if conflict not in self._conflicts:
                self._conflicts.append(conflict)

    def _discover(self, key, parent, candle, bars, structural, pools):
        spec = self.specs[key[0]]
        bullish = parent.direction == "bearish"
        side = PivotSide.LOW if bullish else PivotSide.HIGH
        edge = spec.price_to_ticks(candle.low if bullish else candle.high)
        # Independently reconstruct sweep observations from the full eligibility snapshot.
        self._structural_rows.append([
            candle.provenance.record_id, "READY", sorted(p.pool_id for p in pools),
            sorted(s.cluster_id for s in structural.swings if s.available_at <= candle.interval_start),
        ])
        for pool in pools:
            if pool.side is side and (edge < pool.level_ticks if bullish else edge > pool.level_ticks):
                self._expected_sweeps.add((candle.provenance.record_id, pool.pool_id, parent.control_id))
        crossed = {
            pool.pool_id: pool for pool in pools if pool.side is side
            and (pool.level_ticks - edge >= 1 if bullish else edge - pool.level_ticks >= 1)
        }
        owned = {pool_id: candidate for (controller_id, pool_id), candidate in self._lineages.items()
                 if controller_id == parent.control_id}
        index = len(bars) - 1
        deadline = reclaim_deadline(key[1], self.rules.config_version)
        if deadline is None:
            self._input_invalid = True
            return
        for pool_id, candidate in tuple(owned.items()):
            reclaim = candidate.reclaim_index
            expired = (index > candidate.first_sweep_index + deadline if reclaim is None
                       else index > reclaim + self.rules.leg_max_bars)
            if expired:
                self._ready.pop(candidate.candidate_id, None)
                self._describe(candidate, "EXPIRED")
                del self._lineages[parent.control_id, pool_id]
                del owned[pool_id]
        if len(crossed) > 1:
            cohort = []
            for pool_id, pool in sorted(crossed.items()):
                candidate = owned.pop(pool_id, None)
                if candidate is None:
                    lineage = self._parent_id(key, parent, pool, candle)
                    candidate = _Candidate(lineage, lineage + ":candidate:v1", 1, pool,
                                           "bullish" if bullish else "bearish", parent.control_id,
                                           [candle.provenance.record_id], edge, index, index)
                self._lineages.pop((parent.control_id, pool_id), None)
                self._ready.pop(candidate.candidate_id, None)
                self._describe(candidate, "UNRESOLVED", extra=(candle.provenance.record_id,))
                self._observe(candle, pool, parent, candidate, "CONFLICT")
                cohort.append(candidate.parent_id)
            self._conflicts.append((parent.control_id, "MULTI_POOL_SWEEP",
                                    (candle.provenance.record_id,), tuple(sorted(crossed)), tuple(sorted(cohort))))
            crossed.clear()
        for pool_id in sorted(set(owned) | set(crossed)):
            candidate = owned.get(pool_id)
            if candidate is not None:
                self._candidate[key] = candidate
            before = len(self._observations)
            self._advance(key, parent, candle, bars, structural,
                          (crossed[pool_id],) if pool_id in crossed else ())
            survivor = self._candidate.pop(key, None)
            lineage_key = parent.control_id, pool_id
            if survivor is None:
                self._lineages.pop(lineage_key, None)
                if candidate is not None:
                    self._ready.pop(candidate.candidate_id, None)
                    if self._outcomes.get(candidate.candidate_id, {}).get("state") != "UNRESOLVED":
                        self._describe(candidate, "INVALIDATED")
                if pool_id in crossed and before == len(self._observations):
                    self._observe(candle, crossed[pool_id], parent, candidate, "REJECT")
            else:
                self._lineages[lineage_key] = survivor
                if survivor.candidate_id not in self._ready:
                    self._describe(survivor, "PENDING")

    def _parent_id(self, key, controller, pool, candle):
        return _canonical_hash("subsequent_manipulation_parent_v1:", {
            "schema": "mlof.subsequent_manipulation_parent.v1", "spec_version": self.spec_version,
            "config_version": self.rules.config_version, "instrument": key[0], "timeframe": key[1],
            "liquidity_pool_id": pool.pool_id, "first_sweep_candle_id": candle.provenance.record_id,
            "direction": "bearish" if controller.direction == "bullish" else "bullish",
            "active_controller_id": controller.control_id,
        })

    def _verify_trace(self, at: datetime, bars) -> bool:
        """Rebuild from the original manifest with fresh analyzers and isolated storage."""
        fresh = BootstrapAnalyzer(self.bootstrap.config, self.bootstrap.rules,
                                  tuple(self.specs.values()), self.bootstrap.calendar)
        reference = SubsequentExchangeAnalyzer(
            fresh, self.rules, tuple(self.specs.values()), spec_version=self.spec_version,
            replay=self.replay, trusted_external_seals=self._trusted_external_seals,
            additional_seals=self._input_declarations[1:],
        )
        reference._verifying = True
        # Compare at the pre-resolution boundary, after replay of all earlier cuts.
        count = len(bars)
        if not count:
            return False
        for candle in self.replay.candles[:count - 1]:
            reference.process_candle(candle)
        last = self.replay.candles[count - 1]
        if count > 1:
            reference._between_cuts(bars[-2].source_available_at, last.source_available_at)
        reference._process_candle(last)
        if last.source_available_at < at:
            reference.evaluate(last.source_available_at)
            reference._between_cuts(last.source_available_at, at)
        if self._trace_snapshot() != reference._trace_snapshot():
            return False
        reference.evaluate(at)
        self._expected_resolution = reference._resolution
        return True

    def _trace_snapshot(self):
        return (
            self._observations, self._transition_trace, self._structural_rows,
            self._outcomes, self._lineages, self._ready, self._conflicts,
            self._rejected_evidence, self._active,
            tuple(self._exchanges.values()), tuple(self._controllers.values()),
            tuple(self._events.values()), tuple(self._coverage),
        )

    def evaluate(self, at: datetime) -> None:
        """Evaluate a manifest-bound information cut, including a seal-only cut."""
        if self.replay is None:
            raise ValueError("a source declaration and immutable manifest are required")
        if at.utcoffset() != timedelta(0):
            raise ValueError("the decision cut must be explicit UTC")
        if self._cut is not None and at <= self._cut:
            return
        seal = self.replay.seal
        key = seal.instrument, seal.timeframe
        bars = self._candles.get(key, ())
        if bars and bars[-1].source_available_at > at:
            raise ValueError("decision cut precedes processed evidence")
        parent = self._active.get(key)
        self._cut_input = self.replay
        error, expected = self.replay.input_error(self.bootstrap.calendar, at, tuple(bars),
                                                  self._trusted_external_seals)
        if error is not None:
            for declaration in sorted(self._input_declarations[1:],
                                      key=lambda item: item.seal.declaration_id):
                alternative_error, intervals = declaration.input_error(
                    self.bootstrap.calendar, at, tuple(bars), self._trusted_external_seals)
                if alternative_error is None:
                    self._cut_input = declaration
                    error, expected = None, intervals
                    break
        if parent is None:
            return
        if self._publication_failed:
            error = "ID_COLLISION"
        elif self._input_invalid:
            error = "INVALID_INPUT"
        actual = {tuple(row[:3]) for row in self._observations}
        accounted_for = all(
            (row[3] is None and row[5] == "REJECT")
            or f"{row[3]}:candidate:v{row[4]}" in self._outcomes
            for row in self._observations
        )
        if actual != self._expected_sweeps or not accounted_for:
            error = "INCOMPLETE_DISCOVERY"
        self._expected_resolution = None
        if error is None and expected and not self._verifying and not self._verify_trace(at, bars):
            error = "INCOMPLETE_DISCOVERY"
        if error is not None:
            diagnostic = self._coverage_record(parent, bars, at, expected, error=error)
            self.ledger.publish(self.evaluation_identity, ((diagnostic.coverage_id, diagnostic),))
            if diagnostic not in self._coverage:
                self._coverage.append(diagnostic)
            return
        if not expected:
            return
        ready = [record for record in self._ready.values()
                 if record[0].parent_controller_id == parent.control_id]
        completion = lambda record: max(record[0].break_index, record[2][0])
        cut_text = format_utc_timestamp(at)
        for candidate, _, _, _, _ in ready:
            outcome = self._outcomes[candidate.candidate_id]
            if outcome["evidence_ready_at"] is None:
                outcome["evidence_ready_at"] = cut_text
        readiness_key = lambda record: (
            self._outcomes[record[0].candidate_id]["evidence_ready_at"], completion(record))
        ready.sort(key=readiness_key)
        preliminary = self._coverage_record(parent, bars, at, expected)
        prior_outcomes = {cid: outcome.copy() for cid, outcome in self._outcomes.items()}
        prior_conflicts = len(self._conflicts)
        prior_forbidden = self._rejected_evidence.get(parent.control_id, set()).copy()
        rejected_ids = set()
        rejected_evidence = set()
        forbidden = self._rejected_evidence.get(parent.control_id, set())
        by_id = {c.provenance.record_id: c.interval_end for c in bars}
        conflicts = [item for item in self._conflicts if item[0] == parent.control_id]
        winner = None
        for rank in sorted({readiness_key(record) for record in ready}):
            cohort = [record for record in ready if readiness_key(record) == rank]
            index = rank[1]
            completed = bars[index].interval_end
            if any(code == "SAME_CANDLE_LEG_END" and any(by_id[i] <= completed for i in candles)
                   for _, code, candles, _, _ in conflicts):
                break
            if len(cohort) > 1:
                candle_id = bars[index].provenance.record_id
                ids = tuple(sorted(record[0].parent_id for record in cohort))
                pools = tuple(sorted(record[0].pool.pool_id for record in cohort))
                self._conflicts.append((parent.control_id, "OVERLAPPING_COMPLETION",
                                        (candle_id,), pools, ids))
                for candidate, _, fvg, _, _ in cohort:
                    rejected_ids.add(candidate.candidate_id)
                    rejected_evidence.update(fvg[1])
                    rejected_evidence.add(bars[candidate.break_index].provenance.record_id)
                continue
            record = cohort[0]
            candidate, _, fvg, _, _ = record
            confirm_ids = set(fvg[1]) | {bars[candidate.break_index].provenance.record_id}
            required_ids = confirm_ids | set(candidate.sweep_ids) | {candidate.reclaim_id}
            if (forbidden | rejected_evidence) & confirm_ids:
                rejected_ids.add(candidate.candidate_id)
                continue
            if any(code == "MULTI_POOL_SWEEP" and
                   (required_ids.intersection(candles) or any(by_id[i] >= completed for i in candles))
                   for _, code, candles, _, _ in conflicts):
                break
            winner = record
            break
        self._rejected_evidence.setdefault(parent.control_id, set()).update(rejected_evidence)
        for candidate_id in rejected_ids:
            self._outcomes[candidate_id]["state"] = "UNRESOLVED"
        self._resolution = (at, winner[0].candidate_id if winner else None,
                            tuple(sorted(rejected_ids)), tuple(sorted(rejected_evidence)))
        if self._expected_resolution is not None and self._resolution != self._expected_resolution:
            self._outcomes = prior_outcomes
            del self._conflicts[prior_conflicts:]
            self._rejected_evidence[parent.control_id] = prior_forbidden
            diagnostic = self._coverage_record(parent, bars, at, expected, error="INCOMPLETE_DISCOVERY")
            self.ledger.publish(self.evaluation_identity, ((diagnostic.coverage_id, diagnostic),))
            self._coverage.append(diagnostic)
            self._cut = at
            return
        self._cut = at
        try:
            if winner is None:
                final = self._coverage_record(parent, bars, at, expected, preliminary)
                self.ledger.publish(self.evaluation_identity,
                                    ((preliminary.coverage_id, preliminary), (final.coverage_id, final)))
                self._coverage.extend((preliminary, final))
            else:
                candidate, candle, fvg, reference, structural = winner
                self._confirm(key, parent, candidate, candle, bars, fvg, reference, structural,
                              publish_at=at, coverage=(preliminary, expected))
                if parent.control_id not in self._replaced:
                    raise ValueError("exchange publication did not commit")
        except ValueError:
            self._outcomes = prior_outcomes
            del self._conflicts[prior_conflicts:]
            self._rejected_evidence[parent.control_id] = prior_forbidden
            self._publication_failed = True
            return
        for candidate_id in rejected_ids:
            rejected = self._ready.pop(candidate_id)[0]
            self._lineages.pop((parent.control_id, rejected.pool.pool_id), None)
        if parent.control_id in self._replaced:
            for lineage_key in tuple(self._lineages):
                if lineage_key[0] == parent.control_id:
                    prior = self._lineages.pop(lineage_key)
                    self._ready.pop(prior.candidate_id, None)

    def _coverage_record(self, parent, bars, at, expected, preliminary=None, *, error=None):
        time = format_utc_timestamp(at)
        order = {c.provenance.record_id: i for i, c in enumerate(bars)}
        ancestry = []
        node = parent
        while node is not None:
            frozen = self._controller_ranges.get(node.control_id)
            ancestry.append([node.control_id, node.exchange_id, frozen.range_id if frozen else None,
                             format_utc_timestamp(node.available_at)])
            node = self._controllers.get(node.parent_controller_id)
        conflicts = [{"code": code, "candle_ids": list(candles), "pool_ids": list(pools),
                      "lineage_ids": list(lineages)}
                     for pid, code, candles, pools, lineages in self._conflicts if pid == parent.control_id]
        if error is not None:
            conflicts.append({"code": error, "candle_ids": [], "pool_ids": [], "lineage_ids": []})
        payload = {
            "schema": "mlof.exchange_coverage.v1", "stage": "FINAL" if preliminary else "PRELIMINARY",
            "producer_version": "m3b1-0.3.4-v2", "manifest_id": self.manifest_id,
            "declaration_id": self._cut_input.seal.declaration_id, "parent_controller_id": parent.control_id,
            "ancestry_digest": digest(list(reversed(ancestry))),
            "structural_digest": digest(self._structural_rows),
            "input_prefix_digest": digest(self._manifest["records"][:len(bars)]),
            "expected_intervals_digest": interval_digest(expected),
            "observations_digest": digest(sorted(self._observations, key=lambda
                row: (order[row[0]], row[1], row[3] or "", row[4] or 0, row[5]))),
            "information_time": time, "sealed_prefix_end": format_utc_timestamp(expected[-1][1]) if expected else None,
            "preliminary_id": preliminary.coverage_id if preliminary else None,
            "outcomes": sorted((outcome for cid, outcome in self._outcomes.items()
                                if self._outcome_parents[cid] == parent.control_id),
                               key=lambda o: (o["lineage_id"], o["version"])),
            "conflicts": sorted(conflicts, key=canonical_bytes),
            "coverage_state": "UNRESOLVED" if error else "COMPLETE",
        }
        return CoverageRecord("exchange_coverage_v1:" + digest(payload), self.evaluation_identity,
                              at, canonical_bytes(payload))

    def _advance(self, key, active, candle, bars, structural, pools) -> None:
        spec = self.specs[key[0]]
        index = len(bars) - 1
        expected = "bearish" if active.direction == "bullish" else "bullish"
        sweep_side = PivotSide.HIGH if expected == "bearish" else PivotSide.LOW
        crossed = []
        for pool in pools:
            if pool.side is not sweep_side:
                continue
            level = pool.level_ticks
            edge = spec.price_to_ticks(candle.high if expected == "bearish" else candle.low)
            if (expected == "bearish" and edge >= level + 1) or (expected == "bullish" and edge <= level - 1):
                crossed.append(pool)
        candidate = self._candidate.get(key)
        armed_id = None
        close_through = False
        if candidate is not None and candidate.reclaim_index is not None and self.replay is not None:
            armed_id = candidate.candidate_id
            close = spec.price_to_ticks(candle.close)
            boundary = candidate.protective_boundary_ticks
            close_through = close <= boundary - 1 if expected == "bullish" else close >= boundary + 1
            if close_through:
                self._conflict_if_completion(active, candidate, candle, bars)
        if candidate is None:
            if len(crossed) > 1:
                self._emit(SubsequentStatus.REJECTED, candle, expected, (active.control_id,) + tuple(p.pool_id for p in crossed),
                           "Multiple eligible pools swept in one OHLC candle; ordering is ambiguous")
                return
            if not crossed:
                self._emit(SubsequentStatus.OBSERVING, candle, None, (active.control_id,), "No eligible opposing liquidity sweep")
                return
            pool = crossed[0]
            frozen, protective, boundary_error = self._frozen_boundary(active, expected, candle, structural)
            extreme_ticks = spec.price_to_ticks(candle.low if expected == "bullish" else candle.high)
            reference_open_ticks = spec.price_to_ticks(candle.open)
            if boundary_error:
                self._emit(SubsequentStatus.NOT_READY, candle, expected,
                           (active.control_id, pool.pool_id, getattr(frozen, "range_id", ""),
                            getattr(protective, "cluster_id", "")), boundary_error)
                return
            if not self._inside_frozen_range(expected, extreme_ticks, reference_open_ticks, frozen):
                self._emit(SubsequentStatus.NOT_READY, candle, expected,
                           (active.control_id, pool.pool_id, frozen.range_id, protective.cluster_id),
                           "Initial manipulation extreme or reference open is outside the linked frozen range")
                return
            parent = _canonical_hash("subsequent_manipulation_parent_v1:", {
                "schema": "mlof.subsequent_manipulation_parent.v1", "spec_version": self.spec_version,
                "config_version": self.rules.config_version, "instrument": key[0], "timeframe": key[1],
                "liquidity_pool_id": pool.pool_id, "first_sweep_candle_id": candle.provenance.record_id,
                "direction": expected, "active_controller_id": active.control_id})
            candidate = _Candidate(parent, parent + ":candidate:v1", 1, pool, expected, active.control_id,
                                   [candle.provenance.record_id],
                                   extreme_ticks, index, index, frozen_range_id=frozen.range_id,
                                   protective_swing_id=protective.cluster_id,
                                   protective_boundary_ticks=protective.level_ticks,
                                   reference_open_ticks=reference_open_ticks)
            parent_payload = (key, pool.pool_id, candle.provenance.record_id, expected, active.control_id)
            try:
                self._check_unique(parent, parent_payload)
                self._check_unique(candidate.candidate_id, (parent, 1))
            except ValueError as exc:
                self._emit(SubsequentStatus.REJECTED, candle, expected, (active.control_id, pool.pool_id),
                           f"Candidate identity collision; no candidate registered: {exc}")
                return
            self._store_unique(parent, parent_payload)
            self._store_unique(candidate.candidate_id, (parent, 1))
            self._candidate[key] = candidate
            if self.replay is not None:
                self._observe(candle, pool, active, candidate, "START")
            self._emit(SubsequentStatus.SWEEP, candle, expected, (parent, candidate.candidate_id, pool.pool_id,
                       pool.swing_event_id, active.control_id, frozen.range_id, protective.cluster_id),
                       "Pre-registered eligible pool swept; exact independently STRUCTURAL frozen boundary association captured")
        else:
            same = any(p.pool_id == candidate.pool.pool_id for p in crossed)
            if same:
                now_extreme = spec.price_to_ticks(candle.low if expected == "bullish" else candle.high)
                deeper = now_extreme < candidate.extreme_ticks if expected == "bullish" else now_extreme > candidate.extreme_ticks
                candidate.sweep_ids.append(candle.provenance.record_id)
                candidate.last_sweep_index = index
                if self.replay is not None:
                    self._observe(candle, candidate.pool, active, candidate, "REVISIT")
                if deeper:
                    candidate.extreme_ticks = now_extreme
                    candidate.reference_open_ticks = spec.price_to_ticks(candle.open)
                    if candidate.reclaim_index is not None:
                        old = candidate.candidate_id
                        old_version = candidate.version
                        next_version = candidate.version + 1
                        next_id = f"{candidate.parent_id}:candidate:v{next_version}"
                        try:
                            self._check_unique(next_id, (candidate.parent_id, next_version))
                        except ValueError as exc:
                            self._candidate.pop(key, None)
                            self._emit(SubsequentStatus.REJECTED, candle, expected,
                                       (candidate.parent_id, old), f"Candidate revision collision; state failed closed: {exc}")
                            return
                        if self.replay is not None:
                            self._ready.pop(old, None)
                            self._describe(candidate, "INVALIDATED")
                            self._conflict_if_completion(active, candidate, candle, bars)
                        candidate.version = next_version
                        candidate.candidate_id = next_id
                        self._store_unique(next_id, (candidate.parent_id, next_version))
                        candidate.reclaim_index = None
                        candidate.reclaim_id = None
                        candidate.break_index = None
                        if self.replay is not None:
                            self._observe(candle, candidate.pool, active, candidate, "RESET")
                        reset_evidence = CandidateResetEvidence(
                            parent_id=candidate.parent_id,
                            canceled_candidate_id=old,
                            replacement_candidate_id=candidate.candidate_id,
                            parent_controller_id=candidate.parent_controller_id,
                            liquidity_pool_id=candidate.pool.pool_id,
                            direction=candidate.direction,
                            triggering_candle_id=candle.provenance.record_id,
                            interval_start=candle.interval_start,
                            interval_end=candle.interval_end,
                            source_available_at=candle.source_available_at,
                            new_extreme_ticks=candidate.extreme_ticks,
                            canceled_version=old_version,
                            replacement_version=candidate.version,
                            config_version=self.rules.config_version,
                            spec_version=self.spec_version,
                            source_evidence=CandleEvidence(
                                candle.provenance.record_id, candle.interval_start, candle.interval_end,
                                candle.source_available_at, candle.provenance))
                        self._emit(SubsequentStatus.CANDIDATE_RESET, candle, expected,
                                   (candidate.parent_id, old, candidate.candidate_id,
                                    candidate.frozen_range_id, candidate.protective_swing_id),
                                   "Strictly deeper post-reclaim sweep canceled the prior candidate and requires fresh reclaim",
                                   candidate_reset=reset_evidence)
                    frozen = self._controller_ranges.get(active.control_id)
                    frozen_low_id = (getattr(frozen, "low_cluster_id", getattr(frozen, "low_swing_id", ""))
                                     if frozen is not None else "")
                    frozen_high_id = (getattr(frozen, "high_cluster_id", getattr(frozen, "high_swing_id", ""))
                                      if frozen is not None else "")
                    if (frozen is None or frozen.range_id != candidate.frozen_range_id
                            or candidate.protective_swing_id != (frozen_low_id if expected == "bullish"
                                                                 else frozen_high_id)
                            or not self._inside_frozen_range(expected, candidate.extreme_ticks,
                                                            candidate.reference_open_ticks, frozen)):
                        self._candidate.pop(key, None)
                        self._emit(SubsequentStatus.NOT_READY, candle, expected,
                                   (candidate.parent_id, candidate.candidate_id, candidate.frozen_range_id,
                                    candidate.protective_swing_id),
                                   "Deeper sweep invalidated strict containment; frozen boundary cannot be replaced")
                        return
            unrelated = [p for p in crossed if p.pool_id != candidate.pool.pool_id]
            if unrelated:
                self._candidate.pop(key, None)
                self._emit(SubsequentStatus.REJECTED, candle, expected,
                           (candidate.candidate_id,) + tuple(p.pool_id for p in unrelated),
                           "A competing pool sweep conflicts with the active manipulation candidate")
                return
        if self.replay is not None and close_through:
            self._ready.pop(armed_id, None)
            if candidate.candidate_id == armed_id:
                self._candidate.pop(key, None)
                if self._outcomes.get(candidate.candidate_id, {}).get("state") != "UNRESOLVED":
                    self._describe(candidate, "INVALIDATED")
                self._emit(SubsequentStatus.REJECTED, candle, candidate.direction,
                           (candidate.candidate_id, candidate.protective_swing_id),
                           "Completed close crossed the frozen successor boundary")
                return
        reference = next((s for s in structural.swings if s.cluster_id == active.protective_swing_id), None)
        if reference is None or reference.available_at > candle.interval_start or (active.control_id not in self._bootstrap_control_ids
                                 and reference.cluster_id not in self._structural):
            self._emit(SubsequentStatus.NOT_READY, candle, expected,
                       (active.control_id, active.protective_swing_id or ""),
                       "Linked protective reference is missing or not independently STRUCTURAL")
            return
        close = spec.price_to_ticks(candle.close)
        broke = close <= reference.level_ticks - 1 if expected == "bearish" else close >= reference.level_ticks + 1
        if candidate.reclaim_index is None:
            deadline = reclaim_deadline(key[1], self.rules.config_version)
            if deadline is None or index - candidate.first_sweep_index > deadline:
                self._candidate.pop(key, None)
                self._emit(SubsequentStatus.REJECTED, candle, expected, (candidate.candidate_id,), "Reclaim window expired")
                return
            reclaimed = close < candidate.pool.level_ticks if expected == "bearish" else close > candidate.pool.level_ticks
            if reclaimed:
                candidate.reclaim_index, candidate.reclaim_id = index, candle.provenance.record_id
                self._emit(SubsequentStatus.RECLAIM, candle, expected, (candidate.candidate_id, candidate.pool.pool_id),
                           "Completed candle closed strictly back across the swept pool")
                if broke:
                    self._emit(SubsequentStatus.REJECTED, candle, expected, (candidate.candidate_id, reference.cluster_id),
                               "Reclaim and control break occur on one candle; lifecycle order is ambiguous")
                    self._candidate.pop(key, None)
                return
            return
        if index - candidate.reclaim_index > self.rules.leg_max_bars:
            self._candidate.pop(key, None)
            self._emit(SubsequentStatus.REJECTED, candle, expected, (candidate.candidate_id,), "Exchange leg expired")
            return
        if candidate.break_index is None and broke:
            if index == candidate.reclaim_index or not self._qualifies_displacement(bars, index, expected, reference.level_ticks):
                self._emit(SubsequentStatus.REJECTED, candle, expected,
                           (candidate.candidate_id, reference.cluster_id), "Break failed displacement or separate-candle ordering")
                return
            candidate.break_index = index
            self._emit(SubsequentStatus.DISPLACEMENT, candle, expected,
                       (candidate.candidate_id, reference.cluster_id), "Quality-qualified close broke the linked protective swing")
        if candidate.break_index is None:
            return
        fvg = self._find_fvg(bars, candidate)
        if fvg is None:
            if index - candidate.break_index > self.rules.fvg_latest_third_offset:
                self._candidate.pop(key, None)
                self._emit(SubsequentStatus.REJECTED, candle, expected, (candidate.candidate_id,), "No aligned same-leg FVG within configured window")
            return
        if self.replay is None:
            self._confirm(key, active, candidate, candle, bars, fvg, reference, structural)
        else:
            self._ready[candidate.candidate_id] = (candidate, candle, fvg, reference, structural)
            prior_ready = self._outcomes.get(candidate.candidate_id, {}).get("evidence_ready_at")
            self._describe(candidate, "EVIDENCE_READY", bars[max(candidate.break_index, fvg[0])],
                           (*fvg[1], bars[candidate.break_index].provenance.record_id))
            self._outcomes[candidate.candidate_id]["evidence_ready_at"] = prior_ready

    def _confirm(self, key, old, candidate, candle, bars, fvg, break_reference, structural,
                 *, publish_at=None, coverage=None) -> None:
        evaluation_indices = [i for i, item in enumerate(bars)
                              if item.provenance.record_id == candle.provenance.record_id]
        if len(evaluation_indices) != 1:
            self._candidate.pop(key, None)
            self._emit(SubsequentStatus.NOT_READY, candle, candidate.direction,
                       (candidate.candidate_id,),
                       "Exchange evaluation candle is missing or ambiguous in the contiguous replay prefix")
            return
        evaluation_index = evaluation_indices[0]
        # Readiness time is derived from the complete source-available prefix used by
        # this evaluation, not from wall time or whichever candle happened to be processed last.
        ready_prefix_available = max(x.source_available_at for x in bars[:evaluation_index + 1])
        first_sweep = bars[candidate.first_sweep_index]
        frozen, protective, boundary_error = self._frozen_boundary(
            old, candidate.direction, first_sweep, structural)
        association_valid = (
            not boundary_error
            and frozen is not None
            and protective is not None
            and frozen.range_id == candidate.frozen_range_id
            and protective.cluster_id == candidate.protective_swing_id
            and protective.level_ticks == candidate.protective_boundary_ticks
            and candidate.reference_open_ticks is not None
            and self._inside_frozen_range(candidate.direction, candidate.extreme_ticks,
                                          candidate.reference_open_ticks, frozen)
            and break_reference.cluster_id == old.protective_swing_id
        )
        if not association_valid:
            protective = None
        origin_swings = [s for s in structural.swings if s.swing_event_id == candidate.pool.swing_event_id]
        if len(origin_swings) != 1:
            self._emit(SubsequentStatus.NOT_READY, candle, candidate.direction,
                       (candidate.candidate_id, candidate.pool.pool_id),
                       "Swept pool has missing or ambiguous origin-swing provenance; successor promotion blocked")
            self._candidate.pop(key, None)
            return
        origin_swing = origin_swings[0]
        manipulation_id = _canonical_hash("subsequent_manipulation_v1:", {
            "schema": "mlof.subsequent_manipulation.v1", "parent_id": candidate.parent_id,
            "candidate_id": candidate.candidate_id, "sweep_candle_ids": candidate.sweep_ids,
            "reclaim_candle_id": candidate.reclaim_id, "pool_id": candidate.pool.pool_id,
            "origin_swing_cluster_id": origin_swing.cluster_id,
            "controller_id": old.control_id, "direction": candidate.direction,
            "manipulation_extreme_ticks": candidate.extreme_ticks,
            "spec_version": self.spec_version, "config_version": self.rules.config_version})
        break_id = bars[candidate.break_index].provenance.record_id
        fvg_ids = fvg[1]
        exchange_id = _canonical_hash("subsequent_exchange_v1:", {
            "schema": "mlof.subsequent_exchange.v1", "parent_controller_id": old.control_id,
            "manipulation_id": manipulation_id, "break_reference_swing_id": break_reference.cluster_id,
            "displacement_candle_id": break_id, "fvg_candle_ids": list(fvg_ids), "direction": candidate.direction,
            "spec_version": self.spec_version, "config_version": self.rules.config_version})
        successor_id = _canonical_hash("subsequent_control_v1:", {
            "schema": "mlof.subsequent_control.v1", "parent_controller_id": old.control_id,
            "exchange_id": exchange_id, "direction": candidate.direction,
            "spec_version": self.spec_version, "config_version": self.rules.config_version})
        evidence_ids = tuple(dict.fromkeys(candidate.sweep_ids + [candidate.reclaim_id or "", break_id] + list(fvg_ids)))
        by_id = {c.provenance.record_id: c for c in bars}
        evidence_list = [CandleEvidence(by_id[i].provenance.record_id, by_id[i].interval_start, by_id[i].interval_end,
                                        by_id[i].source_available_at, by_id[i].provenance)
                         for i in evidence_ids if i in by_id]
        evidence_list.extend(break_reference.comparison_evidence)
        evidence_list.extend(origin_swing.comparison_evidence)
        if protective is not None:
            evidence_list.extend(protective.comparison_evidence)
        if frozen is not None:
            evidence_list.extend(getattr(frozen, "source_evidence", ()))
        evidence_map = {x.record_id: x for x in evidence_list}
        evidence = tuple(sorted(evidence_map.values(), key=lambda x: (x.interval_start, x.interval_end, x.record_id)))
        available = max(max(x.source_available_at for x in evidence), ready_prefix_available,
                        break_reference.available_at,
                        protective.available_at if protective is not None else break_reference.available_at,
                        frozen.available_at if frozen is not None else break_reference.available_at)
        if publish_at is not None:
            available = publish_at
        manipulation_ids = tuple(dict.fromkeys(candidate.sweep_ids + [candidate.reclaim_id or ""]))
        manipulation_evidence = tuple(evidence_map[i] for i in manipulation_ids if i in evidence_map)
        manipulation_available = publish_at or max(x.source_available_at for x in manipulation_evidence)
        manipulation = SubsequentManipulation(
            manipulation_id, candidate.parent_id, candidate.candidate_id, old.control_id,
            candidate.pool.pool_id, origin_swing.cluster_id, candidate.direction, tuple(candidate.sweep_ids),
            candidate.reclaim_id or "", candidate.extreme_ticks, manipulation_available,
            self.rules.config_version, self.spec_version, manipulation_evidence,
            frozen_control_range_id=candidate.frozen_range_id,
            successor_protective_swing_id=candidate.protective_swing_id,
            reference_open_ticks=candidate.reference_open_ticks)
        exch = SubsequentExchange(exchange_id, old.control_id, manipulation_id, break_reference.cluster_id,
                                  break_id, fvg_ids, candidate.direction, available,
                                  self.rules.config_version, self.spec_version, evidence)
        if protective is None:
            new_controller = SubsequentController(successor_id, old.control_id, exchange_id, manipulation_id,
                candidate.direction, key[0], key[1], None, break_reference.cluster_id, available,
                "NOT_READY", self.rules.config_version, self.spec_version, state="NOT_READY")
            new_range = None
        else:
            low_swing = protective if protective.side is PivotSide.LOW else break_reference
            high_swing = protective if protective.side is PivotSide.HIGH else break_reference
            low, high = sorted((low_swing.level_ticks, high_swing.level_ticks))
            correct_sides = ((candidate.direction == "bullish" and protective.side is PivotSide.LOW
                              and break_reference.side is PivotSide.HIGH)
                             or (candidate.direction == "bearish" and protective.side is PivotSide.HIGH
                                 and break_reference.side is PivotSide.LOW))
            contained = (candidate.extreme_ticks > low and candidate.extreme_ticks <= high
                         and low <= candidate.reference_open_ticks <= high) if candidate.direction == "bullish" else (
                         candidate.extreme_ticks >= low and candidate.extreme_ticks < high
                         and low <= candidate.reference_open_ticks <= high)
            if (low >= high or not correct_sides or not contained
                    or break_reference.cluster_id != old.protective_swing_id):
                new_controller = SubsequentController(successor_id, old.control_id, exchange_id, manipulation_id,
                    candidate.direction, key[0], key[1], protective.cluster_id, break_reference.cluster_id,
                    available, "NOT_READY", self.rules.config_version, self.spec_version,
                    state="NOT_READY")
                new_range = None
            else:
                range_id = _canonical_hash("subsequent_range_v1:", {
                    "schema": "mlof.subsequent_range.v1", "control_id": successor_id,
                    "protective_swing_id": protective.cluster_id, "break_reference_swing_id": break_reference.cluster_id,
                    "low_ticks": low, "high_ticks": high, "instrument": key[0], "timeframe": key[1],
                    "spec_version": self.spec_version, "config_version": self.rules.config_version})
                new_controller = SubsequentController(successor_id, old.control_id, exchange_id, manipulation_id,
                    candidate.direction, key[0], key[1], protective.cluster_id, break_reference.cluster_id,
                    available, "READY", self.rules.config_version, self.spec_version, state="ACTIVE")
                new_range = SubsequentRange(range_id, successor_id, low_swing.cluster_id, high_swing.cluster_id,
                                            low, high, available, self.rules.config_version, self.spec_version,
                                            instrument=key[0], timeframe=key[1], direction=candidate.direction)
        payloads = [(manipulation_id, manipulation), (exchange_id, exch), (successor_id, new_controller)]
        if new_range is not None:
            payloads.append((new_range.range_id, new_range))
        confirmation_event = self._make_event(
            SubsequentStatus.EXCHANGE_CONFIRMED, bars[candidate.break_index], candidate.direction,
            (exchange_id, manipulation_id, old.control_id, successor_id),
            "Complete ordinary exchange is immutable; prior controller replaced; successor readiness is explicit",
            evidence, available_at_floor=available)
        replacement_event = self._make_event(
            SubsequentStatus.CONTROLLER_REPLACED, bars[candidate.break_index], candidate.direction,
            (old.control_id, successor_id, exchange_id),
            "Prior controller eligibility ended at opposing exchange availability", evidence,
            available_at_floor=available)
        payloads.extend(((confirmation_event.event_id, confirmation_event),
                         (replacement_event.event_id, replacement_event)))
        try:
            for event_id, payload in payloads:
                self._check_unique(event_id, payload)
            if coverage is not None:
                preliminary, expected = coverage
                previous_outcomes = {key: value.copy() for key, value in self._outcomes.items()}
                outcome = self._outcomes[candidate.candidate_id]
                outcome.update(state="CONFIRMED", event_id=exchange_id)
                for other in self._lineages.values():
                    state = self._outcomes[other.candidate_id]
                    if (other.parent_controller_id == old.control_id and other is not candidate
                            and state["state"] in ("PENDING", "EVIDENCE_READY")):
                        state["state"] = "STALE_PARENT"
                final = self._coverage_record(old, bars, publish_at, expected, preliminary)
                records = tuple(payloads) + ((preliminary.coverage_id, preliminary), (final.coverage_id, final))
                try:
                    self.ledger.publish(self.evaluation_identity, records)
                except ValueError:
                    self._outcomes = previous_outcomes
                    raise
        except ValueError as exc:
            self._emit(SubsequentStatus.REJECTED, bars[candidate.break_index], candidate.direction,
                       (old.control_id, candidate.candidate_id), f"Identity collision; no promotion committed: {exc}")
            self._candidate.pop(key, None)
            return
        self._store_unique(manipulation_id, manipulation)
        self._store_unique(exchange_id, exch)
        self._store_unique(successor_id, new_controller)
        if new_range is not None:
            self._store_unique(new_range.range_id, new_range)
            self._ranges[new_range.range_id] = new_range
            self._controller_ranges[successor_id] = new_range
        self._manipulations[manipulation_id] = manipulation
        self._exchanges[exchange_id] = exch
        # Lifecycle evidence is preflighted with all entity IDs and committed only
        # after every collision check succeeds.
        self._store_event(confirmation_event)
        self._store_event(replacement_event)
        if coverage is not None:
            self._coverage.extend((preliminary, final))
        self._replaced.add(old.control_id)
        self._controllers[successor_id] = new_controller
        self._active[key] = new_controller
        self._candidate.pop(key, None)

    def _qualifies_displacement(self, bars: Sequence[Candle], index: int, direction: str, level: int) -> bool:
        if index < 1 or index < self.rules.atr_period + 1:
            return False
        c, prev = bars[index], bars[index - 1]
        spread = c.high - c.low
        body = abs(c.close - c.open)
        if spread <= 0:
            return False
        tr = max(spread, abs(c.high - prev.close), abs(c.low - prev.close))
        atr_values = [max(b.high-b.low, abs(b.high-bars[i-1].close), abs(b.low-bars[i-1].close))
                      for i, b in ((j, bars[j]) for j in range(index-self.rules.atr_period, index))]
        atr = sum(atr_values, Decimal(0)) / Decimal(len(atr_values)) if atr_values else Decimal(0)
        if atr <= 0 or body / spread < self.rules.body_ratio_min or tr < self.rules.range_atr_multiple_min * atr:
            return False
        ticks = self.specs[c.instrument.upper()].price_to_ticks(c.close)
        if direction == "bullish":
            return c.close > c.open and (c.close-c.low)/spread >= self.rules.close_location_min and ticks >= level+1
        return c.close < c.open and (c.high-c.close)/spread >= self.rules.close_location_min and ticks <= level-1

    def _find_fvg(self, bars: Sequence[Candle], candidate: _Candidate):
        assert candidate.break_index is not None and candidate.reclaim_index is not None
        spec = self.specs[bars[-1].instrument.upper()]
        bi = candidate.break_index
        earliest = max(2, bi + self.rules.fvg_earliest_third_offset)
        latest = min(len(bars)-1, bi + self.rules.fvg_latest_third_offset,
                     candidate.reclaim_index + self.rules.leg_max_bars)
        end = min(len(bars)-1, candidate.reclaim_index + self.rules.leg_max_bars)
        forbidden = self._rejected_evidence.get(candidate.parent_controller_id, set())
        for third in range(earliest, latest+1):
            first, middle, last = bars[third-2:third+1]
            if any(c.provenance.record_id in forbidden for c in (first, middle, last)):
                continue
            if third < candidate.reclaim_index or third-2 > end:
                continue
            gap = last.low-first.high if candidate.direction == "bullish" else first.low-last.high
            if gap <= 0:
                continue
            prior = [max(bars[i].high-bars[i].low, abs(bars[i].high-bars[i-1].close), abs(bars[i].low-bars[i-1].close))
                     for i in range(max(1,third-self.rules.atr_period),third)]
            if len(prior) < self.rules.atr_period:
                continue
            atr = sum(prior, Decimal(0))/Decimal(len(prior))
            minimum = max(Decimal(self.rules.fvg_min_ticks)*spec.tick_size, self.rules.fvg_atr_ratio_min*atr)
            if gap >= minimum:
                return third, (first.provenance.record_id,middle.provenance.record_id,last.provenance.record_id)
        return None

    def _make_event(self, status, candle, direction, related, reason, evidence=(), *,
                    candidate_reset: CandidateResetEvidence | None = None,
                    available_at_floor: datetime | None = None) -> SubsequentEvent:
        if not evidence:
            evidence = (CandleEvidence(candle.provenance.record_id, candle.interval_start,
                                       candle.interval_end, candle.source_available_at, candle.provenance),)
        evidence_ids = tuple(x.record_id for x in evidence)
        source_ids = tuple(dict.fromkeys((candle.provenance.record_id,) + evidence_ids))
        payload = {"schema":"mlof.subsequent_state.v2", "spec_version":self.spec_version,
                   "config_version":self.rules.config_version,
                   "instrument":candle.instrument.upper(), "timeframe":candle.timeframe.lower(),
                   "status":status.value, "direction":direction, "candle_id":candle.provenance.record_id,
                   "related_ids":list(related)}
        if candidate_reset is not None:
            payload["candidate_reset"] = {
                "parent_id": candidate_reset.parent_id,
                "canceled_candidate_id": candidate_reset.canceled_candidate_id,
                "replacement_candidate_id": candidate_reset.replacement_candidate_id,
                "parent_controller_id": candidate_reset.parent_controller_id,
                "liquidity_pool_id": candidate_reset.liquidity_pool_id,
                "direction": candidate_reset.direction,
                "triggering_candle_id": candidate_reset.triggering_candle_id,
                "interval_start": format_utc_timestamp(candidate_reset.interval_start),
                "interval_end": format_utc_timestamp(candidate_reset.interval_end),
                "source_available_at": format_utc_timestamp(candidate_reset.source_available_at),
                "new_extreme_ticks": candidate_reset.new_extreme_ticks,
                "canceled_version": candidate_reset.canceled_version,
                "replacement_version": candidate_reset.replacement_version,
            }
        event_id = _canonical_hash("subsequent_state_v2:", payload)
        required_availability = max(
            candle.source_available_at,
            max(item.source_available_at for item in evidence),
            available_at_floor or candle.source_available_at)
        return SubsequentEvent(
            event_id=event_id,
            status=status,
            instrument=candle.instrument.upper(),
            timeframe=candle.timeframe.lower(),
            direction=direction,
            available_at=required_availability,
            source_candle_ids=source_ids,
            related_ids=tuple(related),
            reason=reason,
            config_version=self.rules.config_version,
            spec_version=self.spec_version,
            source_evidence=tuple(evidence),
            candidate_reset=candidate_reset,
            tradable=False)

    def _store_event(self, event: SubsequentEvent) -> None:
        self._store_unique(event.event_id, event)
        if event.event_id not in self._events:
            self._events[event.event_id] = event

    def _emit(self, status, candle, direction, related, reason, evidence=(), *,
              candidate_reset: CandidateResetEvidence | None = None) -> None:
        event = self._make_event(status, candle, direction, related, reason, evidence,
                                 candidate_reset=candidate_reset)
        self._store_event(event)

    def _check_unique(self, event_id: str, payload: object) -> None:
        prior = self._id_payloads.get(event_id)
        if prior is not None and prior != payload:
            raise ValueError("stable subsequent-event ID collision")

    def _store_unique(self, event_id: str, payload: object) -> None:
        self._check_unique(event_id,payload)
        self._id_payloads[event_id]=payload

    def _new(self,before):
        return tuple(sorted((v for k,v in self._events.items() if k not in before), key=lambda x:(x.available_at,x.event_id)))
