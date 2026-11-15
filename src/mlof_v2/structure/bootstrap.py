"""Offline, non-tradable first-control bootstrap from confirmed structure.

This module consumes completed candles and the existing causal pivot/pool
registry. It emits structural evidence only; it has no signal, order, risk,
valuation, fill, or P&L functionality.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Iterable, Sequence

from ..calendars import SessionCalendar
from ..models import Candle, InstrumentSpec, ReadinessState
from .config import StructuralConfig
from .engine import StructuralAnalyzer
from .config import reclaim_deadline
from .models import CandleEvidence, LiquidityPool, PivotSide, StructuralReplayResult, SwingCluster


class BootstrapStage(str, Enum):
    OBSERVING_NO_CONTROL = "OBSERVING_NO_CONTROL"
    PAIR_READY = "PAIR_READY"
    BOOTSTRAP_SWEEP = "BOOTSTRAP_SWEEP"
    RECLAIM_CONFIRMED = "RECLAIM_CONFIRMED"
    DISPLACEMENT_CONFIRMED = "DISPLACEMENT_CONFIRMED"
    BREAK_AND_FVG_CONFIRMED = "BREAK_AND_FVG_CONFIRMED"
    CONTROL_ESTABLISHED = "CONTROL_ESTABLISHED"
    EXPIRED = "EXPIRED"
    NOT_READY = "NOT_READY"
    CANDIDATE_RESET = "CANDIDATE_RESET"


@dataclass(frozen=True, slots=True)
class BootstrapRules:
    """Versioned Gate 0 research defaults used by the bootstrap path."""

    config_version: str
    minimum_contiguous_bars: int = 50
    atr_period: int = 14
    body_ratio_min: Decimal = Decimal("0.60")
    range_atr_multiple_min: Decimal = Decimal("1.20")
    close_location_min: Decimal = Decimal("0.75")
    fvg_min_ticks: int = 2
    fvg_atr_ratio_min: Decimal = Decimal("0.10")
    fvg_earliest_third_offset: int = -2
    fvg_latest_third_offset: int = 3
    leg_max_bars: int = 12

    def __post_init__(self) -> None:
        if not self.config_version.strip() or self.minimum_contiguous_bars < 1 or self.atr_period < 1:
            raise ValueError("bootstrap version and history settings must be valid")
        if min(self.body_ratio_min, self.range_atr_multiple_min, self.close_location_min,
               self.fvg_atr_ratio_min) <= 0 or self.fvg_min_ticks < 1 or self.leg_max_bars < 1:
            raise ValueError("bootstrap quality thresholds must be positive")
        if self.body_ratio_min > 1 or self.close_location_min > 1:
            raise ValueError("body and close-location ratios cannot exceed one")
        if self.fvg_earliest_third_offset > 0 or self.fvg_latest_third_offset < 0:
            raise ValueError("FVG window must include the break candle")


def load_bootstrap_rules(path: str | Path) -> BootstrapRules:
    """Load explicit versioned offline bootstrap thresholds from local JSON."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read bootstrap configuration {path}: {exc}") from exc
    required = {"schema_version", "config_version", "minimum_contiguous_bars", "atr_period",
                "body_ratio_min", "range_atr_multiple_min", "close_location_min", "fvg_min_ticks",
                "fvg_atr_ratio_min", "fvg_earliest_third_offset", "fvg_latest_third_offset", "leg_max_bars"}
    if not isinstance(raw, dict) or set(raw) != required or raw.get("schema_version") != 1:
        raise ValueError("bootstrap configuration fields do not match schema version 1")
    if any(not isinstance(raw[name], str) for name in
           ("body_ratio_min", "range_atr_multiple_min", "close_location_min", "fvg_atr_ratio_min")):
        raise ValueError("bootstrap decimal thresholds must be strings")
    integer_fields = ("minimum_contiguous_bars", "atr_period", "fvg_min_ticks",
                      "fvg_earliest_third_offset", "fvg_latest_third_offset", "leg_max_bars")
    if any(type(raw[name]) is not int for name in integer_fields) or not isinstance(raw["config_version"], str):
        raise ValueError("bootstrap version or integer settings have invalid types")
    try:
        decimals = {name: Decimal(raw[name]) for name in (
            "body_ratio_min", "range_atr_multiple_min", "close_location_min", "fvg_atr_ratio_min")}
    except (TypeError, ValueError) as exc:
        raise ValueError("bootstrap decimal thresholds must be strings containing finite decimals") from exc
    if any(not value.is_finite() for value in decimals.values()):
        raise ValueError("bootstrap thresholds must be finite")
    return BootstrapRules(
        raw["config_version"], raw["minimum_contiguous_bars"], raw["atr_period"],
        decimals["body_ratio_min"], decimals["range_atr_multiple_min"],
        decimals["close_location_min"], raw["fvg_min_ticks"], decimals["fvg_atr_ratio_min"],
        raw["fvg_earliest_third_offset"], raw["fvg_latest_third_offset"], raw["leg_max_bars"],
    )


@dataclass(frozen=True, slots=True)
class ObservationPair:
    pair_id: str
    instrument: str
    timeframe: str
    high_cluster_id: str
    low_cluster_id: str
    high_pool_id: str
    low_pool_id: str
    high_ticks: int
    low_ticks: int
    high_available_at: datetime
    low_available_at: datetime
    pair_available_at: datetime
    pair_anchor_end: datetime
    config_version: str
    provenance_cluster_ids: tuple[str, str]
    high_pool_registered_order: int
    low_pool_registered_order: int
    high_pool_available_at: datetime
    low_pool_available_at: datetime
    source_evidence: tuple[CandleEvidence, ...] = ()


@dataclass(frozen=True, slots=True)
class BootstrapManipulation:
    event_id: str
    pair_id: str
    swept_pool_id: str
    pivot_cluster_id: str
    direction: str
    sweep_candle_ids: tuple[str, ...]
    reclaim_candle_id: str
    sweep_reference_ticks: int
    manipulation_extreme_ticks: int
    reference_open_ticks: int
    confirmation_available_at: datetime
    config_version: str
    parent_id: str = ""
    candidate_id: str = ""
    source_evidence: tuple[CandleEvidence, ...] = ()
    tradable: bool = False


@dataclass(frozen=True, slots=True)
class BootstrapExchange:
    event_id: str
    manipulation_id: str
    pair_id: str
    displacement_candle_id: str
    fvg_candle_ids: tuple[str, str, str]
    direction: str
    available_at: datetime
    config_version: str
    tradable: bool = False
    source_evidence: tuple[CandleEvidence, ...] = ()


@dataclass(frozen=True, slots=True)
class FrozenControlRange:
    range_id: str
    control_id: str
    instrument: str
    timeframe: str
    pair_id: str
    exchange_id: str
    low_cluster_id: str
    high_cluster_id: str
    low_ticks: int
    high_ticks: int
    direction: str
    protective_cluster_id: str
    break_reference_cluster_id: str
    available_at: datetime
    config_version: str
    tradable: bool = False
    source_evidence: tuple[CandleEvidence, ...] = ()


@dataclass(frozen=True, slots=True)
class BootstrapEvent:
    event_id: str
    stage: BootstrapStage
    instrument: str
    timeframe: str
    available_at: datetime
    reason: str
    related_ids: tuple[str, ...]
    source_candle_ids: tuple[str, ...]
    config_version: str
    tradable: bool = False
    source_evidence: tuple[CandleEvidence, ...] = ()


@dataclass(frozen=True, slots=True)
class BootstrapReplayResult:
    structural: StructuralReplayResult
    events: tuple[BootstrapEvent, ...]
    pairs: tuple[ObservationPair, ...]
    manipulations: tuple[BootstrapManipulation, ...]
    exchanges: tuple[BootstrapExchange, ...]
    controls: tuple[FrozenControlRange, ...]


@dataclass(slots=True)
class _Candidate:
    pair: ObservationPair
    pool_id: str
    cluster_id: str
    direction: str
    parent_id: str
    candidate_id: str
    candidate_version: int
    sweep_ids: list[str]
    sweep_extreme_ticks: int
    reference_open_ticks: int
    first_sweep_index: int
    last_sweep_index: int
    reclaim_index: int | None = None
    reclaim_candle_id: str | None = None
    break_index: int | None = None


def _canonical_hash(prefix: str, payload: dict[str, object]) -> str:
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return prefix + hashlib.sha256(body.encode("utf-8")).hexdigest()


def observation_pair_id(config_version: str, instrument: str, timeframe: str,
                        high_cluster_id: str, low_cluster_id: str) -> str:
    return _canonical_hash("observation_pair_v1:", {
        "schema": "mlof.observation_pair.v1",
        "config_version": config_version,
        "instrument": instrument.upper(),
        "timeframe": timeframe.lower(),
        "high_cluster_id": high_cluster_id,
        "low_cluster_id": low_cluster_id,
    })


def bootstrap_manipulation_id(*, config_version: str, pair_id: str, swept_pool_id: str,
                              direction: str, sweep_candle_ids: Sequence[str],
                              reclaim_candle_id: str, sweep_reference_ticks: int) -> str:
    if direction not in {"bullish", "bearish"}:
        raise ValueError("direction must be bullish or bearish")
    return _canonical_hash("bootstrap_manipulation_v1:", {
        "schema": "mlof.bootstrap_manipulation.v1",
        "config_version": config_version,
        "pair_id": pair_id,
        "swept_pool_id": swept_pool_id,
        "direction": direction,
        "sweep_candle_ids": list(dict.fromkeys(sweep_candle_ids)),
        "reclaim_candle_id": reclaim_candle_id,
        "sweep_reference_ticks": sweep_reference_ticks,
    })


def bootstrap_parent_id(*, config_version: str, instrument: str, pool_id: str,
                        first_sweep_bar_id: str, direction: str,
                        control_ancestor_id: str) -> str:
    """Stable Section 5 parent identity; bootstrap pair substitutes for absent ancestor."""
    if direction not in {"bullish", "bearish"}:
        raise ValueError("direction must be bullish or bearish")
    return _canonical_hash("bootstrap_parent_v1:", {
        "instrument": instrument.upper(),
        "pool_id": pool_id,
        "first_sweep_bar_id": first_sweep_bar_id,
        "direction": direction,
        "control_ancestor_id": control_ancestor_id,
        "spec_version": config_version,
    })


def _candidate_id(parent_id: str, version: int) -> str:
    if version < 1:
        raise ValueError("candidate version must be positive")
    return f"{parent_id}:candidate:v{version}"


class BootstrapAnalyzer:
    """Chronological, one-time initial-control analyzer for independent streams."""

    _STAGE_ORDER = {
        BootstrapStage.NOT_READY: 0,
        BootstrapStage.OBSERVING_NO_CONTROL: 1,
        BootstrapStage.PAIR_READY: 2,
        BootstrapStage.BOOTSTRAP_SWEEP: 3,
        BootstrapStage.CANDIDATE_RESET: 4,
        BootstrapStage.RECLAIM_CONFIRMED: 5,
        BootstrapStage.DISPLACEMENT_CONFIRMED: 6,
        BootstrapStage.BREAK_AND_FVG_CONFIRMED: 7,
        BootstrapStage.CONTROL_ESTABLISHED: 8,
        BootstrapStage.EXPIRED: 9,
    }

    def __init__(self, structural_config: StructuralConfig, rules: BootstrapRules,
                 instrument_specs: Sequence[InstrumentSpec], calendar: SessionCalendar) -> None:
        if structural_config.config_version != rules.config_version:
            raise ValueError("bootstrap and structural configuration versions must match")
        self.config = structural_config
        self.rules = rules
        self.specs = {x.instrument.upper(): x for x in instrument_specs}
        self.calendar = calendar
        self.structural = StructuralAnalyzer(structural_config, instrument_specs, calendar)
        self._candles: dict[tuple[str, str], list[Candle]] = {}
        self._candidate: dict[tuple[str, str], _Candidate] = {}
        self._controls: dict[tuple[str, str], FrozenControlRange] = {}
        self._events: dict[str, BootstrapEvent] = {}
        self._pairs: dict[str, ObservationPair] = {}
        self._manipulations: dict[str, BootstrapManipulation] = {}
        self._exchanges: dict[str, BootstrapExchange] = {}
        self._id_payloads: dict[str, object] = {}

    def process_candle(self, candle: Candle) -> tuple[BootstrapEvent, ...]:
        return self._process_candle(candle)

    def _process_candle(self, candle: Candle, *, verified_context: bool = False) -> tuple[BootstrapEvent, ...]:
        before = set(self._events)
        key = (candle.instrument.upper(), candle.timeframe.lower())
        stream = self._candles.setdefault(key, [])
        if key in self._controls:
            self.structural.process_candle(candle)
            return ()
        self.structural.process_candle(candle)
        if reclaim_deadline(key[1], self.rules.config_version) is None:
            self._candidate.pop(key, None)
            self._emit(BootstrapStage.NOT_READY, candle,
                       "No approved bootstrap reclaim window exists for this timeframe", (), ())
            return self._new_events(before)
        readiness = next((r for r in self.structural.readiness()
                          if (r.instrument, r.timeframe) == key), None)
        if readiness is None or readiness.state is ReadinessState.NOT_READY:
            self._candidate.pop(key, None)
            self._emit(BootstrapStage.NOT_READY, candle, "Structural stream is not ready; bootstrap is suspended",
                       (), (candle.provenance.record_id,))
            return self._new_events(before)
        stream.append(candle)
        if key[1] in ("4h", "1d") and not verified_context:
            self._candidate.pop(key, None)
            self._emit(BootstrapStage.NOT_READY, candle,
                       "Context formation requires an approved profile and sealed coverage", (), ())
            return self._new_events(before)
        widths = self.config.widths_for(candle.timeframe)
        required = max(self.rules.minimum_contiguous_bars,
                       2 * widths.left + 2 * widths.right + 1 + self.rules.atr_period)
        if len(stream) < required:
            self._emit(BootstrapStage.OBSERVING_NO_CONTROL, candle,
                       f"Continuous history is warming ({len(stream)}/{required} bars)", (), ())
            return self._new_events(before)

        structural_result = self.structural.result()
        eligible_pools = self.structural.eligible_pools_for(candle)
        active = self._candidate.get(key)
        if active is not None:
            self._advance_candidate(key, active, candle, stream, eligible_pools)
            return self._new_events(before)
        try:
            selected = self._select_pair(
                key,
                [swing for swing in structural_result.swings if swing.available_at < candle.interval_start],
                [pool for pool in eligible_pools if pool.available_at < candle.interval_start],
            )
        except (KeyError, ValueError) as exc:
            self._emit(BootstrapStage.OBSERVING_NO_CONTROL, candle,
                       f"Required pair evidence is missing, inconsistent, or colliding: {exc}", (), ())
            return self._new_events(before)
        if selected is None:
            self._emit(BootstrapStage.OBSERVING_NO_CONTROL, candle,
                       "No eligible adjacent confirmed high/low pair is available before this interval",
                       (), ())
            return self._new_events(before)
        try:
            self._register_pair(selected)
        except ValueError as exc:
            self._emit(BootstrapStage.OBSERVING_NO_CONTROL, candle,
                       f"Pair identity/evidence collision: {exc}", (selected.pair_id,), ())
            return self._new_events(before)
        self._emit(BootstrapStage.PAIR_READY, candle, "Latest eligible pair frozen before sweep evaluation",
                   (selected.pair_id,), tuple(e.record_id for e in selected.source_evidence),
                   selected.source_evidence)
        sweeps = self._pair_sweeps(candle, selected, eligible_pools)
        if len(sweeps) > 1:
            self._emit(BootstrapStage.OBSERVING_NO_CONTROL, candle,
                       "OHLC candle breaches both pair pools; intrabar ordering is unknowable",
                       (selected.pair_id,), (candle.provenance.record_id,))
        elif sweeps:
            pool, cluster, direction, extreme = sweeps[0]
            first_sweep_id = candle.provenance.record_id
            parent_id = bootstrap_parent_id(
                config_version=self.rules.config_version, instrument=candle.instrument,
                pool_id=pool.pool_id, first_sweep_bar_id=first_sweep_id,
                direction=direction, control_ancestor_id=selected.pair_id)
            candidate_id = _candidate_id(parent_id, 1)
            parent_payload = (candle.instrument.upper(), pool.pool_id, first_sweep_id,
                              direction, selected.pair_id, self.rules.config_version)
            self._store_unique(parent_id, parent_payload)
            self._store_unique(candidate_id, (parent_id, 1))
            candidate = _Candidate(selected, pool.pool_id, cluster.cluster_id, direction,
                                   parent_id, candidate_id, 1,
                                   [first_sweep_id], extreme,
                                   self.specs[key[0]].price_to_ticks(candle.open),
                                   len(stream) - 1, len(stream) - 1)
            self._candidate[key] = candidate
            self._emit(BootstrapStage.BOOTSTRAP_SWEEP, candle, "Registered selected-pair pool swept by at least one tick",
                       (selected.pair_id, pool.pool_id, cluster.cluster_id,
                        candidate.parent_id, candidate.candidate_id),
                       tuple(e.record_id for e in self._pair_candle_evidence(selected, candle)),
                       self._pair_candle_evidence(selected, candle))
            self._try_reclaim(candidate, candle, len(stream) - 1)
        return self._new_events(before)

    def process_batch(self, candles: Iterable[Candle]) -> BootstrapReplayResult:
        ordered = sorted(candles, key=lambda c: (c.instrument.upper(), c.timeframe.lower(),
                                                 c.interval_start, c.source_available_at,
                                                 c.provenance.record_id))
        for candle in ordered:
            self.process_candle(candle)
        return self.result()

    def result(self) -> BootstrapReplayResult:
        return BootstrapReplayResult(
            self.structural.result(),
            tuple(sorted(self._events.values(), key=lambda e: (e.available_at, e.instrument, e.timeframe,
                                                                self._STAGE_ORDER[e.stage], e.event_id))),
            tuple(sorted(self._pairs.values(), key=lambda p: (p.pair_available_at, p.pair_id))),
            tuple(sorted(self._manipulations.values(), key=lambda m: (m.confirmation_available_at, m.event_id))),
            tuple(sorted(self._exchanges.values(), key=lambda e: (e.available_at, e.event_id))),
            tuple(sorted(self._controls.values(), key=lambda c: (c.available_at, c.control_id))),
        )

    def _select_pair(self, key: tuple[str, str], swings: Sequence[SwingCluster],
                     pools: Sequence[LiquidityPool]) -> ObservationPair | None:
        pool_map = {p.pool_id: p for p in pools}
        known = [s for s in swings if (s.instrument, s.timeframe) == key and s.pool_id in pool_map]
        def order(swing: SwingCluster) -> tuple[datetime, datetime, str]:
            anchor = next((x.interval_start for x in swing.comparison_evidence
                           if x.record_id == swing.anchor_candle_id), None)
            if anchor is None:
                raise ValueError("confirmed swing lacks anchor provenance")
            return anchor, swing.available_at, swing.swing_event_id
        known.sort(key=order)
        candidate_pairs: list[ObservationPair] = []
        instrument, timeframe = key
        for left, right in zip(known, known[1:]):
            if left.side is right.side or left.anchor_candle_id == right.anchor_candle_id:
                continue
            high = left if left.side is PivotSide.HIGH else right
            low = left if left.side is PivotSide.LOW else right
            if abs(high.level_ticks - low.level_ticks) < 2:
                continue
            hp, lp = pool_map.get(high.pool_id), pool_map.get(low.pool_id)
            if hp is None or lp is None or hp.swing_event_id != high.swing_event_id or lp.swing_event_id != low.swing_event_id:
                continue
            pair_id = observation_pair_id(self.rules.config_version, instrument, timeframe,
                                          high.cluster_id, low.cluster_id)
            high_anchor, low_anchor = order(high)[0], order(low)[0]
            candidate_pairs.append(ObservationPair(
                pair_id, instrument, timeframe, high.cluster_id, low.cluster_id,
                hp.pool_id, lp.pool_id, high.level_ticks, low.level_ticks,
                high.available_at, low.available_at, max(high.available_at, low.available_at),
                max(high_anchor, low_anchor), self.rules.config_version,
                (high.cluster_id, low.cluster_id),
                hp.registered_order, lp.registered_order, hp.available_at, lp.available_at,
                tuple(sorted({e.record_id: e for e in high.comparison_evidence + low.comparison_evidence}.values(),
                             key=lambda e: (e.interval_start, e.interval_end, e.record_id))),
            ))
        if not candidate_pairs:
            return None
        pair_ids: dict[str, ObservationPair] = {}
        for pair in candidate_pairs:
            prior = pair_ids.get(pair.pair_id)
            if prior is not None and prior != pair:
                raise ValueError("distinct eligible pairs share one canonical ID")
            pair_ids[pair.pair_id] = pair
        latest_available = max(p.pair_available_at for p in candidate_pairs)
        finalists = [p for p in candidate_pairs if p.pair_available_at == latest_available]
        latest_anchor = max(p.pair_anchor_end for p in finalists)
        finalists = [p for p in finalists if p.pair_anchor_end == latest_anchor]
        return min(finalists, key=lambda p: p.pair_id)

    def _register_pair(self, pair: ObservationPair) -> None:
        prior = self._pairs.get(pair.pair_id)
        if prior is not None and prior != pair:
            raise ValueError("observation-pair ID collision")
        for other in self._pairs.values():
            if other.pair_id == pair.pair_id and other != pair:
                raise ValueError("distinct observation pairs share an ID")
        self._pairs[pair.pair_id] = pair

    def _pair_sweeps(self, candle: Candle, pair: ObservationPair, pools: Sequence[LiquidityPool]):
        spec = self.specs.get(candle.instrument.upper())
        if spec is None:
            return []
        by_id = {p.pool_id: p for p in pools}
        result = self.structural.result()
        swings = {s.pool_id: s for s in result.swings}
        matches = []
        for pool_id, side, level in ((pair.low_pool_id, PivotSide.LOW, pair.low_ticks),
                                     (pair.high_pool_id, PivotSide.HIGH, pair.high_ticks)):
            pool = by_id.get(pool_id)
            swing = swings.get(pool_id)
            if pool is None or swing is None or pool.swing_event_id != swing.swing_event_id:
                continue
            extreme = spec.price_to_ticks(candle.low if side is PivotSide.LOW else candle.high)
            if side is PivotSide.LOW and extreme <= level - 1:
                matches.append((pool, swing, "bullish", extreme))
            if side is PivotSide.HIGH and extreme >= level + 1:
                matches.append((pool, swing, "bearish", extreme))
        return matches

    def _advance_candidate(self, key: tuple[str, str], candidate: _Candidate, candle: Candle,
                           stream: list[Candle], eligible_pools: Sequence[LiquidityPool]) -> None:
        index = len(stream) - 1
        spec = self.specs[key[0]]
        observed_sweeps = self._pair_sweeps(candle, candidate.pair, eligible_pools)
        close_ticks = spec.price_to_ticks(candle.close)
        if candidate.reclaim_index is not None and index - candidate.reclaim_index > self.rules.leg_max_bars:
            self._candidate.pop(key, None)
            self._emit(BootstrapStage.EXPIRED, candle, "Bootstrap exchange window ended without confirmation",
                       (candidate.pair.pair_id, candidate.pool_id), (candle.provenance.record_id,))
            return
        close_through = candidate.reclaim_index is not None and (
            close_ticks <= candidate.pair.low_ticks - 1 if candidate.direction == "bullish"
            else close_ticks >= candidate.pair.high_ticks + 1
        )
        conflicting_fvg = self._find_fvg(stream, candidate) if close_through and candidate.break_index is not None else None
        expected_opposing_break = (
            candidate.direction == "bullish" and close_ticks >= candidate.pair.high_ticks + 1
        ) or (
            candidate.direction == "bearish" and close_ticks <= candidate.pair.low_ticks - 1
        )
        expected_break_pool = (candidate.pair.high_pool_id if candidate.direction == "bullish"
                               else candidate.pair.low_pool_id)
        conflicts = [item for item in observed_sweeps
                     if item[0].pool_id != candidate.pool_id
                     and not (candidate.reclaim_index is not None and expected_opposing_break
                              and item[0].pool_id == expected_break_pool)]
        if conflicts or len(observed_sweeps) > 1:
            self._candidate.pop(key, None)
            self._emit(BootstrapStage.EXPIRED, candle,
                       "Conflicting selected-pair sweeps in one provisional event; OHLC cannot establish ordering",
                       (candidate.pair.pair_id, candidate.pool_id) + tuple(item[0].pool_id for item in conflicts),
                       (candle.provenance.record_id,))
            return
        matching_sweep = next((item for item in observed_sweeps if item[0].pool_id == candidate.pool_id), None)
        current_extreme = spec.price_to_ticks(candle.low if candidate.direction == "bullish" else candle.high)
        deeper = current_extreme < candidate.sweep_extreme_ticks if candidate.direction == "bullish" else current_extreme > candidate.sweep_extreme_ticks
        qualifies_sweep = matching_sweep is not None
        opposite_level = candidate.pair.high_ticks if candidate.direction == "bullish" else candidate.pair.low_ticks
        opp_pool_id = candidate.pair.high_pool_id if candidate.direction == "bullish" else candidate.pair.low_pool_id
        pool_ready = any(p.pool_id == opp_pool_id for p in eligible_pools)
        breaks_level = close_ticks >= opposite_level + 1 if candidate.direction == "bullish" else close_ticks <= opposite_level - 1
        if qualifies_sweep:
            candidate.sweep_ids.append(candle.provenance.record_id)
            candidate.last_sweep_index = index
            if deeper:
                candidate.sweep_extreme_ticks = current_extreme
                candidate.reference_open_ticks = spec.price_to_ticks(candle.open)
                if candidate.reclaim_index is not None:
                    canceled_candidate_id = candidate.candidate_id
                    next_version = candidate.candidate_version + 1
                    replacement_candidate_id = _candidate_id(candidate.parent_id, next_version)
                    self._store_unique(replacement_candidate_id, (candidate.parent_id, next_version))
                    candidate.reclaim_index = None
                    candidate.reclaim_candle_id = None
                    candidate.break_index = None
                    candidate.candidate_version = next_version
                    candidate.candidate_id = replacement_candidate_id
                    self._emit(BootstrapStage.CANDIDATE_RESET, candle,
                               "A deeper post-reclaim sweep canceled the stale reclaim and requires a fresh reclaim",
                               (candidate.parent_id, canceled_candidate_id, replacement_candidate_id,
                                candidate.pair.pair_id, candidate.pool_id),
                               (candle.provenance.record_id,),
                               self._pair_candle_evidence(candidate.pair, candle))
        if close_through:
            # A simultaneous reset already ended the old version and cleared its qualification.
            revised = candidate.reclaim_index is None
            reason = "Completed close crossed the frozen swept-side bootstrap reference"
            stage = BootstrapStage.EXPIRED
            if conflicting_fvg is not None and conflicting_fvg[0] == index:
                stage = BootstrapStage.NOT_READY
                reason = "UNRESOLVED_SAME_CANDLE_ORDER: bootstrap close-through and FVG completion"
            self._emit(stage, candle, reason,
                       (candidate.parent_id, candidate.pair.pair_id, candidate.cluster_id, candidate.pool_id),
                       conflicting_fvg[1] if stage is BootstrapStage.NOT_READY else (),
                       self._pair_candle_evidence(candidate.pair, candle))
            if not revised:
                self._candidate.pop(key, None)
                return
        if candidate.reclaim_index is None:
            deadline = reclaim_deadline(key[1], self.rules.config_version)
            if deadline is None:
                self._candidate.pop(key, None)
                self._emit(BootstrapStage.NOT_READY, candle, "No versioned reclaim window exists for timeframe",
                           (candidate.pair.pair_id,), ())
                return
            if index - candidate.first_sweep_index > deadline:
                self._candidate.pop(key, None)
                self._emit(BootstrapStage.EXPIRED, candle, "Reclaim window expired",
                           (candidate.pair.pair_id, candidate.pool_id), (candle.provenance.record_id,))
                return
            if self._try_reclaim(candidate, candle, index):
                return
        if candidate.reclaim_index is None:
            return

        if candidate.break_index is None and breaks_level and pool_ready:
            if not self._qualifies_displacement(stream, index, candidate.direction, opposite_level, spec.tick_size):
                self._emit(BootstrapStage.OBSERVING_NO_CONTROL, candle,
                           "Opposing level was crossed but the completed candle failed displacement quality",
                           (candidate.pair.pair_id,), (candle.provenance.record_id,))
            else:
                candidate.break_index = index
                self._emit(BootstrapStage.DISPLACEMENT_CONFIRMED, candle,
                           "Pre-registered opposing pair pivot broken by a qualifying displacement candle",
                           (candidate.pair.pair_id, candidate.pool_id),
                           tuple(e.record_id for e in self._pair_candle_evidence(candidate.pair, candle)),
                           self._pair_candle_evidence(candidate.pair, candle))
        if candidate.break_index is not None:
            fvg = self._find_fvg(stream, candidate)
            if fvg is not None:
                self._confirm_exchange(key, candidate, stream, fvg)
                self._candidate.pop(key, None)
            elif index - candidate.break_index > self.rules.fvg_latest_third_offset:
                self._candidate.pop(key, None)
                self._emit(BootstrapStage.EXPIRED, candle, "No qualifying same-leg FVG formed within the approved window",
                           (candidate.pair.pair_id, candidate.pool_id), (candle.provenance.record_id,))

    def _try_reclaim(self, candidate: _Candidate, candle: Candle, index: int) -> bool:
        spec = self.specs[candle.instrument.upper()]
        close = spec.price_to_ticks(candle.close)
        level = candidate.pair.low_ticks if candidate.direction == "bullish" else candidate.pair.high_ticks
        reclaimed = close > level if candidate.direction == "bullish" else close < level
        if reclaimed:
            candidate.reclaim_index = index
            candidate.reclaim_candle_id = candle.provenance.record_id
            self._emit(BootstrapStage.RECLAIM_CONFIRMED, candle,
                       "Completed candle closed strictly back across the swept pool",
                       (candidate.pair.pair_id, candidate.pool_id),
                       tuple(e.record_id for e in self._pair_candle_evidence(candidate.pair, candle)),
                       self._pair_candle_evidence(candidate.pair, candle))
            return True
        return False

    def _atr_before(self, candles: Sequence[Candle], index: int) -> Decimal | None:
        period = self.rules.atr_period
        if index < period + 1:
            return None
        trs: list[Decimal] = []
        for i in range(index - period, index):
            bar, prev = candles[i], candles[i - 1]
            trs.append(max(bar.high - bar.low, abs(bar.high - prev.close), abs(bar.low - prev.close)))
        atr = sum(trs, Decimal(0)) / Decimal(period)
        return atr if atr > 0 else None

    def _qualifies_displacement(self, candles: Sequence[Candle], index: int, direction: str,
                                break_level_ticks: int, tick: Decimal) -> bool:
        candle = candles[index]
        atr = self._atr_before(candles, index)
        spread = candle.high - candle.low
        body = abs(candle.close - candle.open)
        if atr is None or spread <= 0:
            return False
        tr = max(spread, abs(candle.high - candles[index - 1].close), abs(candle.low - candles[index - 1].close))
        if body / spread < self.rules.body_ratio_min or tr < self.rules.range_atr_multiple_min * atr:
            return False
        if direction == "bullish":
            close_location = (candle.close - candle.low) / spread
            return (candle.close > candle.open and close_location >= self.rules.close_location_min
                    and self.specs[candle.instrument.upper()].price_to_ticks(candle.close) >= break_level_ticks + 1)
        close_location = (candle.high - candle.close) / spread
        return (candle.close < candle.open and close_location >= self.rules.close_location_min
                and self.specs[candle.instrument.upper()].price_to_ticks(candle.close) <= break_level_ticks - 1)

    def _find_fvg(self, candles: Sequence[Candle], candidate: _Candidate):
        assert candidate.break_index is not None and candidate.reclaim_index is not None
        spec = self.specs[candles[-1].instrument.upper()]
        bi = candidate.break_index
        earliest = max(2, bi + self.rules.fvg_earliest_third_offset)
        latest = min(len(candles) - 1, bi + self.rules.fvg_latest_third_offset,
                     candidate.reclaim_index + self.rules.leg_max_bars)
        for third in range(earliest, latest + 1):
            first, middle, last = candles[third - 2:third + 1]
            if third < candidate.reclaim_index:
                continue
            if candidate.direction == "bullish":
                gap = last.low - first.high
            else:
                gap = first.low - last.high
            atr = self._atr_before(candles, third)
            if atr is None:
                continue
            minimum = max(Decimal(self.rules.fvg_min_ticks) * spec.tick_size,
                          self.rules.fvg_atr_ratio_min * atr)
            if gap >= minimum:
                return (third, (first.provenance.record_id, middle.provenance.record_id,
                                last.provenance.record_id), gap)
        return None

    def _confirm_exchange(self, key: tuple[str, str], candidate: _Candidate,
                          candles: Sequence[Candle], fvg) -> None:
        assert candidate.reclaim_candle_id is not None and candidate.reclaim_index is not None
        spec = self.specs[key[0]]
        ordered_sweeps = sorted(set(candidate.sweep_ids), key=lambda cid: next(
            (c.interval_start, c.interval_end, cid) for c in candles if c.provenance.record_id == cid))
        manipulation_id = bootstrap_manipulation_id(
            config_version=self.rules.config_version, pair_id=candidate.pair.pair_id,
            swept_pool_id=candidate.pool_id, direction=candidate.direction,
            sweep_candle_ids=ordered_sweeps, reclaim_candle_id=candidate.reclaim_candle_id,
            sweep_reference_ticks=(candidate.pair.low_ticks if candidate.direction == "bullish" else candidate.pair.high_ticks))
        break_candle = candles[candidate.break_index]
        available = max(break_candle.source_available_at, candles[fvg[0]].source_available_at)
        exchange_id = _canonical_hash("bootstrap_exchange_v1:", {
            "schema": "mlof.bootstrap_exchange.v1", "config_version": self.rules.config_version,
            "manipulation_id": manipulation_id, "break_candle_id": break_candle.provenance.record_id,
            "fvg_candle_ids": list(fvg[1]), "direction": candidate.direction,
        })
        control_id = _canonical_hash("bootstrap_control_v1:", {
            "schema": "mlof.bootstrap_control.v1", "config_version": self.rules.config_version,
            "pair_id": candidate.pair.pair_id, "exchange_id": exchange_id,
            "protective_cluster_id": candidate.cluster_id,
            "break_reference_cluster_id": (candidate.pair.high_cluster_id if candidate.direction == "bullish"
                                             else candidate.pair.low_cluster_id),
        })
        range_id = _canonical_hash("bootstrap_range_v1:", {
            "schema": "mlof.bootstrap_range.v1", "config_version": self.rules.config_version,
            "control_id": control_id, "pair_id": candidate.pair.pair_id,
            "low_cluster_id": candidate.pair.low_cluster_id,
            "high_cluster_id": candidate.pair.high_cluster_id,
            "low_ticks": candidate.pair.low_ticks, "high_ticks": candidate.pair.high_ticks,
        })
        evidence = self._exchange_evidence(candidate, candles, fvg)
        manipulation = BootstrapManipulation(
            manipulation_id, candidate.pair.pair_id, candidate.pool_id, candidate.cluster_id,
            candidate.direction, tuple(ordered_sweeps), candidate.reclaim_candle_id,
            candidate.pair.low_ticks if candidate.direction == "bullish" else candidate.pair.high_ticks,
            candidate.sweep_extreme_ticks, candidate.reference_open_ticks,
            max(candles[candidate.reclaim_index].source_available_at,
                candles[candidate.last_sweep_index].source_available_at), self.rules.config_version,
            candidate.parent_id, candidate.candidate_id, evidence)
        exchange = BootstrapExchange(exchange_id, manipulation_id, candidate.pair.pair_id,
                                     break_candle.provenance.record_id, fvg[1], candidate.direction,
                                     available, self.rules.config_version, False, evidence)
        control = FrozenControlRange(range_id, control_id, key[0], key[1], candidate.pair.pair_id, exchange_id,
                                    candidate.pair.low_cluster_id, candidate.pair.high_cluster_id,
                                    candidate.pair.low_ticks, candidate.pair.high_ticks,
                                    candidate.direction, candidate.cluster_id,
                                    (candidate.pair.high_cluster_id if candidate.direction == "bullish"
                                     else candidate.pair.low_cluster_id),
                                    available, self.rules.config_version, False, evidence)
        try:
            self._check_unique(manipulation_id, manipulation)
            self._check_unique(exchange_id, exchange)
            self._check_unique(control_id, control)
            self._check_unique(range_id, control)
        except ValueError as exc:
            self._emit(BootstrapStage.OBSERVING_NO_CONTROL, candles[-1],
                       f"Bootstrap lineage ID collision: {exc}", (candidate.pair.pair_id,), ())
            return
        self._store_unique(manipulation_id, manipulation)
        self._store_unique(exchange_id, exchange)
        self._store_unique(control_id, control)
        self._store_unique(range_id, control)
        self._manipulations[manipulation_id] = manipulation
        self._exchanges[exchange_id] = exchange
        self._controls[key] = control
        self._emit(BootstrapStage.BREAK_AND_FVG_CONFIRMED, candles[-1],
                   "Qualifying displacement and same-leg FVG complete the first non-tradable exchange",
                   (candidate.parent_id, candidate.candidate_id, candidate.pair.pair_id,
                    manipulation_id, exchange_id),
                   tuple(ordered_sweeps) + (candidate.reclaim_candle_id, break_candle.provenance.record_id) + fvg[1])
        self._emit(BootstrapStage.CONTROL_ESTABLISHED, candles[-1],
                   "First control and frozen range established; bootstrap is permanently non-tradable",
                   (exchange_id, control_id, range_id, candidate.pair.pair_id),
                   (candidate.pair.high_cluster_id, candidate.pair.low_cluster_id,
                    manipulation_id, exchange_id, control_id, range_id), evidence)

    def _exchange_evidence(self, candidate: _Candidate, candles: Sequence[Candle], fvg):
        swing_map = {s.cluster_id: s for s in self.structural.result().swings}
        pair_swings = (swing_map[candidate.pair.high_cluster_id], swing_map[candidate.pair.low_cluster_id])
        by_id = {c.provenance.record_id: c for c in candles}
        ids = set(candidate.sweep_ids)
        ids.add(candidate.reclaim_candle_id or "")
        ids.add(candles[candidate.break_index].provenance.record_id)
        ids.update(fvg[1])
        candles_used = [by_id[cid] for cid in ids if cid in by_id]
        evidence = [item for swing in pair_swings for item in swing.comparison_evidence]
        evidence.extend(CandleEvidence(c.provenance.record_id, c.interval_start, c.interval_end,
                                       c.source_available_at, c.provenance) for c in candles_used)
        unique = {item.record_id: item for item in evidence}
        return tuple(sorted(unique.values(), key=lambda item: (item.interval_start, item.interval_end, item.record_id)))

    @staticmethod
    def _pair_candle_evidence(pair: ObservationPair, candle: Candle) -> tuple[CandleEvidence, ...]:
        current = CandleEvidence(candle.provenance.record_id, candle.interval_start, candle.interval_end,
                                 candle.source_available_at, candle.provenance)
        unique = {item.record_id: item for item in pair.source_evidence}
        unique[current.record_id] = current
        return tuple(sorted(unique.values(), key=lambda item: (item.interval_start, item.interval_end, item.record_id)))

    def _store_unique(self, event_id: str, value: object) -> None:
        self._check_unique(event_id, value)
        self._id_payloads[event_id] = value

    def _check_unique(self, event_id: str, value: object) -> None:
        prior = self._id_payloads.get(event_id)
        if prior is not None and prior != value:
            raise ValueError("stable bootstrap ID collision")

    def _emit(self, stage: BootstrapStage, candle: Candle, reason: str,
              related: tuple[str, ...], source: tuple[str, ...],
              source_evidence: tuple[CandleEvidence, ...] = ()) -> None:
        source = tuple(dict.fromkeys(source + tuple(item.record_id for item in source_evidence)))
        payload = {"schema": "mlof.bootstrap_state_event.v1", "config_version": self.rules.config_version,
                   "instrument": candle.instrument.upper(), "timeframe": candle.timeframe.lower(),
                   "stage": stage.value, "candle_id": candle.provenance.record_id,
                   "related_ids": list(related), "reason": reason}
        event_id = _canonical_hash("bootstrap_state_v1:", payload)
        item = BootstrapEvent(event_id, stage, candle.instrument.upper(), candle.timeframe.lower(),
                              candle.source_available_at, reason, related, source,
                              self.rules.config_version, False, source_evidence)
        self._store_unique(event_id, item)
        self._events[event_id] = item

    def _new_events(self, before: set[str]) -> tuple[BootstrapEvent, ...]:
        return tuple(sorted((event for key, event in self._events.items() if key not in before),
                            key=lambda e: (e.available_at, self._STAGE_ORDER[e.stage], e.event_id)))

