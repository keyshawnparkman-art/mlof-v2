"""Causal confirmed-pivot detection and time-gated liquidity-pool registry.

This module intentionally stops at structural references. It has no sweep,
break, manipulation, exchange, range, signal, risk, or execution logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable, Sequence

from ..calendars import SessionCalendar, session_candles
from ..models import Candle, InstrumentSpec, ReadinessState, SourceProvenance
from ..timeframes import SESSION_TIMEFRAMES, TIMEFRAME_SECONDS, timeframe_delta
from ..timestamps import format_utc_timestamp
from .config import StructuralConfig, SwingWidths
from .ids import stable_structural_id
from .models import (
    CandleEvidence,
    LiquidityPool,
    PivotSide,
    ProvisionalStructuralReference,
    ReadinessSnapshot,
    StructuralEvent,
    StructuralEventKind,
    StructuralReplayResult,
    SwingCluster,
)


UTC = timezone.utc


@dataclass(slots=True)
class _Candidate:
    cluster_id: str
    candidate_id: str
    side: PivotSide
    level_ticks: int
    anchor: Candle
    members: list[Candle]
    comparison_candles: list[Candle]
    right_bars: list[Candle] = field(default_factory=list)
    last_equal_order: int = 0


@dataclass(slots=True)
class _Stream:
    instrument: str
    timeframe: str
    widths: SwingWidths
    history: list[Candle] = field(default_factory=list)
    candidates: dict[str, _Candidate] = field(default_factory=dict)
    last_candle: Candle | None = None
    observation_order: int = 0
    contiguous_count: int = 0
    sticky_data_issue: str | None = None
    candle_orders: dict[str, int] = field(default_factory=dict)
    candles_by_id: dict[str, Candle] = field(default_factory=dict)
    readiness_at_candle: dict[str, ReadinessState] = field(default_factory=dict)


class LiquidityPoolRegistry:
    """Stores only confirmed pivot pools and checks point-in-time visibility."""

    def __init__(self) -> None:
        self._pools: dict[str, LiquidityPool] = {}

    def register(self, pool: LiquidityPool) -> None:
        existing = self._pools.get(pool.pool_id)
        if existing is not None and existing != pool:
            raise ValueError("immutable pool ID collision")
        self._pools[pool.pool_id] = pool

    def all_pools(self) -> tuple[LiquidityPool, ...]:
        return tuple(sorted(self._pools.values(), key=lambda item: (item.available_at, item.pool_id)))

    @staticmethod
    def is_known_before(
        pool: LiquidityPool,
        candle: Candle,
        observation_order: int,
    ) -> bool:
        """A pool is visible only after registration and by interval start.

        The order check encodes that the registration event was processed before
        the candle's first observation, including an equal timestamp at adjacent
        close/open boundaries.
        """

        return (
            pool.instrument == candle.instrument
            and pool.timeframe == candle.timeframe
            and pool.available_at <= candle.interval_start
            and pool.registered_order < observation_order
        )

    def known_before_for(
        self,
        candle: Candle,
        observation_order: int,
    ) -> tuple[LiquidityPool, ...]:
        return tuple(
            pool for pool in self.all_pools()
            if self.is_known_before(pool, candle, observation_order)
        )


class StructuralAnalyzer:
    """Incremental/batch confirmed-swing analyzer for independent timeframes."""

    _EVENT_PRIORITY = {
        StructuralEventKind.DATA_NOT_READY: 0,
        StructuralEventKind.DATA_GAP_DETECTED: 1,
        StructuralEventKind.READINESS_RESET: 2,
        StructuralEventKind.CANDIDATE_OPENED: 3,
        StructuralEventKind.CLUSTER_EXTENDED: 4,
        StructuralEventKind.CANDIDATE_INVALIDATED: 5,
        StructuralEventKind.SWING_CONFIRMED: 6,
        StructuralEventKind.PROVISIONAL_REFERENCE_REGISTERED: 7,
        StructuralEventKind.LIQUIDITY_POOL_REGISTERED: 8,
    }

    def __init__(
        self,
        config: StructuralConfig,
        instrument_specs: Sequence[InstrumentSpec],
        calendar: SessionCalendar,
    ) -> None:
        self.config = config
        self.instrument_specs = {spec.instrument.upper(): spec for spec in instrument_specs}
        self.calendar = calendar
        self.pool_registry = LiquidityPoolRegistry()
        self._streams: dict[tuple[str, str], _Stream] = {}
        self._events: dict[str, StructuralEvent] = {}
        self._swings: dict[str, SwingCluster] = {}
        self._references: dict[str, ProvisionalStructuralReference] = {}

    def process_candle(self, candle: Candle) -> tuple[StructuralEvent, ...]:
        """Consume one completed bar; per-stream event time must be monotonic."""

        before = set(self._events)
        instrument = candle.instrument.upper()
        timeframe = candle.timeframe.lower()
        key = (instrument, timeframe)
        try:
            widths = self.config.widths_for(timeframe)
        except KeyError:
            self._emit_data_issue(candle, "TIMEFRAME_NOT_CONFIGURED", "No versioned pivot widths configured")
            return self._new_events(before)
        stream = self._streams.setdefault(key, _Stream(instrument, timeframe, widths))

        spec = self.instrument_specs.get(instrument)
        if spec is None:
            self._mark_not_ready(stream, candle, "TICK_SIZE_NOT_CONFIGURED", "Instrument has no configured tick size")
            return self._new_events(before)
        if not self._is_utc(candle.interval_start) or not self._is_utc(candle.interval_end) or not self._is_utc(candle.source_available_at):
            self._mark_not_ready(stream, candle, "TIMESTAMP_NOT_UTC", "Structural input timestamps must already be normalized to UTC")
            return self._new_events(before)
        try:
            if timeframe in SESSION_TIMEFRAMES:
                pairs = session_candles(self.calendar, instrument, timeframe,
                                        candle.interval_start, candle.interval_start)
                if pairs is None or (candle.interval_start, candle.interval_end) not in pairs:
                    raise ValueError("no exact session candle boundaries are declared")
            elif candle.interval_end - candle.interval_start != timeframe_delta(timeframe):
                raise ValueError("interval duration does not match timeframe")
            if (timeframe not in SESSION_TIMEFRAMES and self._microseconds_from_epoch(candle.interval_start)
                    % (TIMEFRAME_SECONDS[timeframe] * 1_000_000) != 0):
                raise ValueError("interval start is not UTC timeframe-aligned")
            # Fail closed unless every OHLC value is representable in ticks.
            for price in (candle.open, candle.high, candle.low, candle.close):
                spec.price_to_ticks(price)
            high_ticks = spec.price_to_ticks(candle.high)
            low_ticks = spec.price_to_ticks(candle.low)
        except (ValueError, KeyError) as exc:
            self._mark_not_ready(stream, candle, "INVALID_STRUCTURAL_CANDLE", str(exc))
            return self._new_events(before)

        previous = stream.last_candle
        if previous is not None:
            if candle.interval_start <= previous.interval_start:
                self._mark_not_ready(stream, candle, "OUT_OF_ORDER_OR_DUPLICATE_CANDLE", "Stream intervals must advance strictly")
                return self._new_events(before)
            if candle.interval_start < previous.interval_end:
                self._mark_not_ready(stream, candle, "OVERLAPPING_CANDLE", "Structural input intervals overlap")
                return self._new_events(before)
            if candle.source_available_at < previous.source_available_at:
                self._mark_not_ready(stream, candle, "OUT_OF_EVENT_TIME_ORDER", "A later interval arrived with an earlier availability timestamp")
                return self._new_events(before)

        expected_current = self.calendar.expected_starts(instrument, timeframe, candle.interval_start, candle.interval_start)
        if expected_current is None:
            self._mark_not_ready(stream, candle, "CALENDAR_COVERAGE_UNKNOWN", "Calendar cannot establish this interval's session status")
            return self._new_events(before)
        if candle.interval_start not in expected_current:
            self._mark_not_ready(stream, candle, "CANDLE_OUTSIDE_SESSION", "Calendar does not mark this interval as expected")
            return self._new_events(before)

        if previous is not None:
            expected_between = self.calendar.expected_starts(
                instrument, timeframe, previous.interval_start, candle.interval_start,
            )
            if expected_between is None:
                self._mark_not_ready(stream, candle, "CALENDAR_COVERAGE_UNKNOWN", "Calendar coverage is incomplete between consecutive bars")
                return self._new_events(before)
            missing = tuple(
                stamp for stamp in expected_between
                if previous.interval_start < stamp < candle.interval_start
            )
            if missing:
                stream.sticky_data_issue = "MISSING_EXPECTED_INTERVAL"
                stream.history.clear()
                stream.candidates.clear()
                stream.contiguous_count = 0
                gap_id = stable_structural_id("data_gap", {
                    "config_version": self.config.config_version,
                    "instrument": instrument,
                    "timeframe": timeframe,
                    "previous_candle_id": previous.provenance.record_id,
                    "current_candle_id": candle.provenance.record_id,
                    "missing_starts": [format_utc_timestamp(item) for item in missing],
                })
                self._record(StructuralEvent(
                    gap_id,
                    StructuralEventKind.DATA_GAP_DETECTED,
                    instrument,
                    timeframe,
                    candle.source_available_at,
                    (previous.provenance.record_id, candle.provenance.record_id),
                    (),
                    "Expected completed intervals are missing; provisional pivot state was reset and this stream remains NOT_READY",
                    self.config.config_version,
                    details=(("missing_starts_utc", ",".join(format_utc_timestamp(item) for item in missing)),),
                    provenance=(previous.provenance, candle.provenance),
                ))
                reset_id = stable_structural_id("readiness_reset", {
                    "config_version": self.config.config_version,
                    "instrument": instrument,
                    "timeframe": timeframe,
                    "trigger_event_id": gap_id,
                })
                self._record(StructuralEvent(
                    reset_id,
                    StructuralEventKind.READINESS_RESET,
                    instrument,
                    timeframe,
                    candle.source_available_at,
                    (candle.provenance.record_id,),
                    (gap_id,),
                    "Contiguous history restarted after a confirmed data gap; missing bars were not fabricated",
                    self.config.config_version,
                    provenance=(candle.provenance,),
                ))

        stream.observation_order += 1
        observation_order = stream.observation_order
        stream.candle_orders[candle.provenance.record_id] = observation_order
        stream.candles_by_id[candle.provenance.record_id] = candle

        events_before_candle = set(self._events)
        self._advance_candidates(stream, candle, high_ticks, low_ticks, observation_order)
        self._start_candidates(stream, candle, high_ticks, low_ticks, observation_order)

        stream.history.append(candle)
        keep = max(stream.widths.left + stream.widths.right + 2, stream.widths.left + 1)
        if len(stream.history) > keep:
            stream.history = stream.history[-keep:]
        stream.contiguous_count += 1
        stream.last_candle = candle
        state = self._readiness_for_stream(stream)
        stream.readiness_at_candle[candle.provenance.record_id] = state
        return self._new_events(before)

    def process_batch(self, candles: Iterable[Candle]) -> StructuralReplayResult:
        """Replay bars deterministically in instrument/timeframe/interval order."""

        # Preserve market interval order within each stream. A source that
        # reports a later bar available before an earlier one is rejected by
        # process_candle instead of being allowed to reorder market history.
        ordered = sorted(
            candles,
            key=lambda candle: (
                candle.instrument.upper(),
                candle.timeframe.lower(),
                candle.interval_start,
                candle.source_available_at,
                candle.provenance.record_id,
            ),
        )
        for candle in ordered:
            self.process_candle(candle)
        return self.result()

    def eligible_pools_for(self, candle: Candle) -> tuple[LiquidityPool, ...]:
        """Return pools processed before this candle and available at its open.

        This is a timing/visibility query only. It does not declare a pool
        swept, significant, controlling, or trade-eligible.
        """

        key = (candle.instrument.upper(), candle.timeframe.lower())
        stream = self._streams.get(key)
        if stream is None:
            return ()
        order = stream.candle_orders.get(candle.provenance.record_id)
        if (
            order is None
            or stream.candles_by_id.get(candle.provenance.record_id) != candle
            or stream.readiness_at_candle.get(candle.provenance.record_id) is ReadinessState.NOT_READY
        ):
            return ()
        return self.pool_registry.known_before_for(candle, order)

    def readiness(self) -> tuple[ReadinessSnapshot, ...]:
        return tuple(
            ReadinessSnapshot(
                state.instrument,
                state.timeframe,
                self._readiness_for_stream(state),
                state.contiguous_count,
                state.sticky_data_issue,
                state.last_candle.source_available_at if state.last_candle else None,
            )
            for _, state in sorted(self._streams.items())
        )

    def result(self) -> StructuralReplayResult:
        return StructuralReplayResult(
            events=tuple(sorted(self._events.values(), key=self._event_sort_key)),
            swings=tuple(sorted(self._swings.values(), key=lambda item: (item.available_at, item.instrument, item.timeframe, item.swing_event_id))),
            provisional_references=tuple(sorted(self._references.values(), key=lambda item: (item.available_at, item.reference_id))),
            liquidity_pools=self.pool_registry.all_pools(),
            readiness=self.readiness(),
        )

    def _advance_candidates(
        self,
        stream: _Stream,
        candle: Candle,
        high_ticks: int,
        low_ticks: int,
        observation_order: int,
    ) -> None:
        price_by_side = {PivotSide.HIGH: high_ticks, PivotSide.LOW: low_ticks}
        for candidate_id, candidate in sorted(tuple(stream.candidates.items())):
            current = price_by_side[candidate.side]
            invalid = (
                candidate.side is PivotSide.HIGH and current > candidate.level_ticks
            ) or (
                candidate.side is PivotSide.LOW and current < candidate.level_ticks
            )
            if invalid:
                event_id = stable_structural_id("pivot_invalidated", {
                    "candidate_id": candidate.candidate_id,
                    "invalidating_candle_id": candle.provenance.record_id,
                    "config_version": self.config.config_version,
                })
                self._record(StructuralEvent(
                    event_id,
                    StructuralEventKind.CANDIDATE_INVALIDATED,
                    stream.instrument,
                    stream.timeframe,
                    candle.source_available_at,
                    tuple(item.provenance.record_id for item in self._ordered_evidence(candidate.comparison_candles + [candle])),
                    (candidate.cluster_id,),
                    "A more extreme completed candle formed before right-side confirmation",
                    self.config.config_version,
                    candidate.side,
                    candidate.level_ticks,
                    (("invalidating_candle_id", candle.provenance.record_id),),
                    tuple(item.provenance for item in self._ordered_evidence(candidate.comparison_candles + [candle])),
                    tuple(self._candle_evidence(item) for item in self._ordered_evidence(candidate.comparison_candles + [candle])),
                ))
                stream.candidates.pop(candidate_id, None)
                continue
            if current == candidate.level_ticks:
                comparison_left = stream.history[-(2 * stream.widths.left):]
                candidate.members.append(candle)
                candidate.comparison_candles = self._ordered_evidence(
                    candidate.comparison_candles + comparison_left + [candle]
                )
                candidate.right_bars.clear()
                candidate.last_equal_order = observation_order
                event_id = stable_structural_id("pivot_cluster_extended", {
                    "cluster_id": candidate.cluster_id,
                    "equal_candle_id": candle.provenance.record_id,
                    "config_version": self.config.config_version,
                })
                self._record(StructuralEvent(
                    event_id,
                    StructuralEventKind.CLUSTER_EXTENDED,
                    stream.instrument,
                    stream.timeframe,
                    candle.source_available_at,
                    (candle.provenance.record_id,),
                    (candidate.cluster_id,),
                    "Equal-price observation extended the unconfirmed cluster and restarted its right-side count",
                    self.config.config_version,
                    candidate.side,
                    candidate.level_ticks,
                    (("anchor_candle_id", candidate.anchor.provenance.record_id),
                     ("equal_member_count", str(len(candidate.members)))),
                    (candle.provenance,),
                    tuple(self._candle_evidence(item) for item in self._ordered_evidence(comparison_left + [candle])),
                ))
                continue

            candidate.right_bars.append(candle)
            candidate.comparison_candles = self._ordered_evidence(candidate.comparison_candles + [candle])
            if len(candidate.right_bars) >= 2 * stream.widths.right:
                self._confirm_candidate(stream, candidate, candle)
                stream.candidates.pop(candidate_id, None)

    def _start_candidates(
        self,
        stream: _Stream,
        candle: Candle,
        high_ticks: int,
        low_ticks: int,
        observation_order: int,
    ) -> None:
        left_count = 2 * stream.widths.left
        right_count = 2 * stream.widths.right
        if len(stream.history) < left_count:
            return
        left = stream.history[-left_count:]
        spec = self.instrument_specs[candle.instrument.upper()]
        previous_highs = [spec.price_to_ticks(item.high) for item in left]
        previous_lows = [spec.price_to_ticks(item.low) for item in left]
        side_values = (
            (PivotSide.HIGH, high_ticks, max(previous_highs)),
            (PivotSide.LOW, low_ticks, min(previous_lows)),
        )
        startable: list[tuple[PivotSide, int]] = []
        for side, current, extreme in side_values:
            if (side is PivotSide.HIGH and current < extreme) or (side is PivotSide.LOW and current > extreme):
                continue
            if any(item.side is side and item.level_ticks == current for item in stream.candidates.values()):
                continue
            startable.append((side, current))
        same_bar_group = None
        if len(startable) == 2:
            same_bar_group = stable_structural_id("simultaneous_pivot_candidates", {
                "config_version": self.config.config_version,
                "candle_id": candle.provenance.record_id,
            })
        for side, level in startable:
            cluster_id = stable_structural_id("pivot_cluster", {
                "config_version": self.config.config_version,
                "instrument": stream.instrument,
                "timeframe": stream.timeframe,
                "side": side.value,
                "anchor_candle_id": candle.provenance.record_id,
                "level_ticks": level,
            })
            candidate_id = stable_structural_id("pivot_candidate", {
                "cluster_id": cluster_id,
                "config_version": self.config.config_version,
            })
            candidate = _Candidate(
                cluster_id, candidate_id, side, level, candle, [candle], list(left) + [candle], [], observation_order,
            )
            stream.candidates[candidate_id] = candidate
            event_id = stable_structural_id("pivot_candidate_opened", {
                "candidate_id": candidate_id,
                "anchor_candle_id": candle.provenance.record_id,
                "config_version": self.config.config_version,
            })
            details = [
                ("left_width", str(stream.widths.left)),
                ("right_width", str(stream.widths.right)),
                ("left_comparison_count", str(left_count)),
                ("right_confirmation_count", str(right_count)),
                ("comparison_window_count", str(left_count + 1 + right_count)),
            ]
            if same_bar_group:
                details.append(("simultaneous_candidate_group_id", same_bar_group))
            self._record(StructuralEvent(
                event_id,
                StructuralEventKind.CANDIDATE_OPENED,
                stream.instrument,
                stream.timeframe,
                candle.source_available_at,
                tuple(item.provenance.record_id for item in self._ordered_evidence(candidate.comparison_candles)),
                (cluster_id,),
                "Candidate is a completed-bar extreme against all configured left comparison candles; right-side confirmation is still required",
                self.config.config_version,
                side,
                level,
                tuple(details),
                tuple(item.provenance for item in self._ordered_evidence(candidate.comparison_candles)),
                tuple(self._candle_evidence(item) for item in self._ordered_evidence(candidate.comparison_candles)),
            ))

    def _confirm_candidate(self, stream: _Stream, candidate: _Candidate, confirmation_candle: Candle) -> None:
        member_ids = tuple(item.provenance.record_id for item in candidate.members)
        right_ids = tuple(item.provenance.record_id for item in candidate.right_bars)
        evidence_candles = self._ordered_evidence(candidate.comparison_candles)
        evidence = tuple(item.provenance for item in evidence_candles)
        candle_evidence = tuple(self._candle_evidence(item) for item in evidence_candles)
        swing_event_id = stable_structural_id("swing_confirmed", {
            "cluster_id": candidate.cluster_id,
            "confirmation_candle_id": confirmation_candle.provenance.record_id,
            "config_version": self.config.config_version,
        })
        pool_id = stable_structural_id("liquidity_pool", {
            "cluster_id": candidate.cluster_id,
            "config_version": self.config.config_version,
        })
        swing = SwingCluster(
            cluster_id=candidate.cluster_id,
            swing_event_id=swing_event_id,
            pool_id=pool_id,
            instrument=stream.instrument,
            timeframe=stream.timeframe,
            side=candidate.side,
            level_ticks=candidate.level_ticks,
            anchor_candle_id=candidate.anchor.provenance.record_id,
            member_candle_ids=member_ids,
            confirmation_candle_id=confirmation_candle.provenance.record_id,
            confirmation_evidence_ids=right_ids,
            confirmed_at=confirmation_candle.interval_end,
            available_at=confirmation_candle.source_available_at,
            config_version=self.config.config_version,
            provenance=evidence,
            comparison_evidence=candle_evidence,
        )
        self._swings[swing_event_id] = swing
        pool = LiquidityPool(
            pool_id=pool_id,
            swing_event_id=swing_event_id,
            instrument=stream.instrument,
            timeframe=stream.timeframe,
            side=candidate.side,
            level_ticks=candidate.level_ticks,
            anchor_candle_id=candidate.anchor.provenance.record_id,
            member_candle_ids=member_ids,
            available_at=confirmation_candle.source_available_at,
            registered_order=stream.observation_order,
            config_version=self.config.config_version,
        )
        self.pool_registry.register(pool)
        reference_id = stable_structural_id("provisional_structural_reference", {
            "swing_event_id": swing_event_id,
            "config_version": self.config.config_version,
        })
        reference = ProvisionalStructuralReference(
            reference_id,
            swing_event_id,
            pool_id,
            stream.instrument,
            stream.timeframe,
            candidate.side,
            candidate.level_ticks,
            confirmation_candle.source_available_at,
        )
        self._references[reference_id] = reference
        common = {
            "instrument": stream.instrument,
            "timeframe": stream.timeframe,
            "available_at": confirmation_candle.source_available_at,
            "source_candle_ids": tuple(item.provenance.record_id for item in evidence_candles),
            "config_version": self.config.config_version,
            "side": candidate.side,
            "level_ticks": candidate.level_ticks,
            "provenance": evidence,
            "candle_evidence": candle_evidence,
        }
        self._record(StructuralEvent(
            swing_event_id,
            StructuralEventKind.SWING_CONFIRMED,
            related_ids=(candidate.cluster_id, pool_id),
            reason="Right-side completed candles confirmed the pivot after the final equal-price member",
            details=(("anchor_candle_id", candidate.anchor.provenance.record_id),
                     ("member_count", str(len(member_ids))),
                     ("confirmation_candle_id", confirmation_candle.provenance.record_id),
                     ("confirmed_at_utc", format_utc_timestamp(confirmation_candle.interval_end))),
            **common,
        ))
        self._record(StructuralEvent(
            reference_id,
            StructuralEventKind.PROVISIONAL_REFERENCE_REGISTERED,
            related_ids=(swing_event_id, pool_id),
            reason="Confirmed pivot is a provisional structural reference only; no directional control is assigned",
            details=(("directional_control", "false"), ("control_direction", "none")),
            **common,
        ))
        pool_event_id = stable_structural_id("liquidity_pool_registered", {
            "pool_id": pool_id,
            "confirmation_candle_id": confirmation_candle.provenance.record_id,
            "config_version": self.config.config_version,
        })
        self._record(StructuralEvent(
            pool_event_id,
            StructuralEventKind.LIQUIDITY_POOL_REGISTERED,
            related_ids=(pool_id, swing_event_id),
            reason="Confirmed pivot registered as a timestamped liquidity reference; visibility is interval-start gated",
            details=(("registered_order", str(stream.observation_order)),),
            **common,
        ))

    def _mark_not_ready(self, stream: _Stream, candle: Candle, code: str, reason: str) -> None:
        stream.sticky_data_issue = code
        stream.history.clear()
        stream.candidates.clear()
        stream.contiguous_count = 0
        event_id = stable_structural_id("structural_data_not_ready", {
            "config_version": self.config.config_version,
            "instrument": stream.instrument,
            "timeframe": stream.timeframe,
            "candle_id": candle.provenance.record_id,
            "reason_code": code,
        })
        self._record(StructuralEvent(
            event_id,
            StructuralEventKind.DATA_NOT_READY,
            stream.instrument,
            stream.timeframe,
            candle.source_available_at,
            (candle.provenance.record_id,),
            (),
            reason,
            self.config.config_version,
            details=(("reason_code", code),),
            provenance=(candle.provenance,),
        ))

    def _emit_data_issue(self, candle: Candle, code: str, reason: str) -> None:
        event_id = stable_structural_id("structural_data_not_ready", {
            "config_version": self.config.config_version,
            "instrument": candle.instrument.upper(),
            "timeframe": candle.timeframe.lower(),
            "candle_id": candle.provenance.record_id,
            "reason_code": code,
        })
        self._record(StructuralEvent(
            event_id,
            StructuralEventKind.DATA_NOT_READY,
            candle.instrument.upper(),
            candle.timeframe.lower(),
            candle.source_available_at,
            (candle.provenance.record_id,),
            (),
            reason,
            self.config.config_version,
            details=(("reason_code", code),),
            provenance=(candle.provenance,),
        ))

    def _readiness_for_stream(self, stream: _Stream) -> ReadinessState:
        if stream.sticky_data_issue is not None:
            return ReadinessState.NOT_READY
        if stream.contiguous_count >= self.config.minimum_contiguous_bars:
            return ReadinessState.READY
        return ReadinessState.WARMING

    def _record(self, event: StructuralEvent) -> None:
        existing = self._events.get(event.event_id)
        if existing is not None and existing != event:
            raise ValueError("stable structural event ID collision")
        self._events[event.event_id] = event

    def _new_events(self, before: set[str]) -> tuple[StructuralEvent, ...]:
        events = [event for event_id, event in self._events.items() if event_id not in before]
        return tuple(sorted(events, key=self._event_sort_key))

    @classmethod
    def _event_sort_key(cls, event: StructuralEvent) -> tuple[datetime, str, str, int, str]:
        return (
            event.available_at,
            event.instrument,
            event.timeframe,
            cls._EVENT_PRIORITY[event.kind],
            event.event_id,
        )

    @staticmethod
    def _is_utc(value: datetime) -> bool:
        return value.tzinfo == UTC and value.utcoffset() == timedelta(0)

    @staticmethod
    def _microseconds_from_epoch(value: datetime) -> int:
        epoch = datetime(1970, 1, 1, tzinfo=UTC)
        elapsed = value - epoch
        return ((elapsed.days * 86400 + elapsed.seconds) * 1_000_000) + elapsed.microseconds

    @staticmethod
    def _ordered_evidence(candles: list[Candle]) -> list[Candle]:
        unique = {item.provenance.record_id: item for item in candles}
        return sorted(unique.values(), key=lambda item: (item.interval_start, item.provenance.record_id))

    @staticmethod
    def _candle_evidence(candle: Candle) -> CandleEvidence:
        return CandleEvidence(
            candle.provenance.record_id,
            candle.interval_start,
            candle.interval_end,
            candle.source_available_at,
            candle.provenance,
        )
