"""Immutable structural event and evidence records for Phase 1."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from ..models import ReadinessState, SourceProvenance


class PivotSide(str, Enum):
    HIGH = "HIGH"
    LOW = "LOW"


@dataclass(frozen=True, slots=True)
class CandleEvidence:
    """A source candle plus its exact interval and availability timestamps."""

    record_id: str
    interval_start: datetime
    interval_end: datetime
    source_available_at: datetime
    provenance: SourceProvenance


class StructuralEventKind(str, Enum):
    CANDIDATE_OPENED = "CANDIDATE_OPENED"
    CLUSTER_EXTENDED = "CLUSTER_EXTENDED"
    CANDIDATE_INVALIDATED = "CANDIDATE_INVALIDATED"
    SWING_CONFIRMED = "SWING_CONFIRMED"
    PROVISIONAL_REFERENCE_REGISTERED = "PROVISIONAL_REFERENCE_REGISTERED"
    LIQUIDITY_POOL_REGISTERED = "LIQUIDITY_POOL_REGISTERED"
    DATA_GAP_DETECTED = "DATA_GAP_DETECTED"
    READINESS_RESET = "READINESS_RESET"
    DATA_NOT_READY = "DATA_NOT_READY"


@dataclass(frozen=True, slots=True)
class SwingCluster:
    cluster_id: str
    swing_event_id: str
    pool_id: str
    instrument: str
    timeframe: str
    side: PivotSide
    level_ticks: int
    anchor_candle_id: str
    member_candle_ids: tuple[str, ...]
    confirmation_candle_id: str
    confirmation_evidence_ids: tuple[str, ...]
    confirmed_at: datetime
    available_at: datetime
    config_version: str
    provenance: tuple[SourceProvenance, ...]
    comparison_evidence: tuple[CandleEvidence, ...]


@dataclass(frozen=True, slots=True)
class ProvisionalStructuralReference:
    reference_id: str
    swing_event_id: str
    pool_id: str
    instrument: str
    timeframe: str
    side: PivotSide
    level_ticks: int
    available_at: datetime
    control_direction: None = None
    directional_control: bool = False


@dataclass(frozen=True, slots=True)
class LiquidityPool:
    pool_id: str
    swing_event_id: str
    instrument: str
    timeframe: str
    side: PivotSide
    level_ticks: int
    anchor_candle_id: str
    member_candle_ids: tuple[str, ...]
    available_at: datetime
    registered_order: int
    config_version: str


@dataclass(frozen=True, slots=True)
class StructuralEvent:
    event_id: str
    kind: StructuralEventKind
    instrument: str
    timeframe: str
    available_at: datetime
    source_candle_ids: tuple[str, ...]
    related_ids: tuple[str, ...]
    reason: str
    config_version: str
    side: PivotSide | None = None
    level_ticks: int | None = None
    details: tuple[tuple[str, str], ...] = ()
    provenance: tuple[SourceProvenance, ...] = ()
    candle_evidence: tuple[CandleEvidence, ...] = ()


@dataclass(frozen=True, slots=True)
class ReadinessSnapshot:
    instrument: str
    timeframe: str
    state: ReadinessState
    contiguous_bar_count: int
    sticky_data_issue: str | None
    last_available_at: datetime | None


@dataclass(frozen=True, slots=True)
class StructuralReplayResult:
    events: tuple[StructuralEvent, ...]
    swings: tuple[SwingCluster, ...]
    provisional_references: tuple[ProvisionalStructuralReference, ...]
    liquidity_pools: tuple[LiquidityPool, ...]
    readiness: tuple[ReadinessSnapshot, ...]

