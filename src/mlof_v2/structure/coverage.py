"""Immutable source declarations and calendar-bound input validation for M3B.1."""
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import hashlib
import json

from ..calendars import SessionCalendar, session_candles
from ..models import Candle, InstrumentSpec, SourceProvenance
from ..timeframes import SESSION_TIMEFRAMES, TIMEFRAME_SECONDS
from ..timestamps import format_utc_timestamp


def canonical_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def digest(payload: object) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def intervals(calendar: SessionCalendar, instrument: str, timeframe: str,
              start: datetime, end: datetime):
    if timeframe in SESSION_TIMEFRAMES:
        pairs = session_candles(calendar, instrument, timeframe, start, end)
        return None if pairs is None else tuple((a,b) for a,b in pairs if b <= end)
    duration = timedelta(seconds=TIMEFRAME_SECONDS[timeframe])
    starts = calendar.expected_starts(instrument, timeframe, start, max(start, end - duration))
    if starts is None:
        return None
    return tuple((item, item + duration) for item in starts if item + duration <= end)


def interval_digest(expected) -> str:
    return digest([[format_utc_timestamp(start), format_utc_timestamp(end)]
                   for start, end in expected])


@dataclass(frozen=True, slots=True)
class SourceSeal:
    dataset_id: str
    instrument: str
    timeframe: str
    source_timezone: str
    timestamp_convention: str
    declared_start: datetime
    sealed_end: datetime
    seal_available_at: datetime
    calendar_version: str
    expected_intervals_digest: str
    assertion: str
    provenance_id: str

    def __post_init__(self):
        if (self.instrument != self.instrument.upper() or self.timeframe not in TIMEFRAME_SECONDS
                or any(item.utcoffset() != timedelta(0) for item in
                       (self.declared_start, self.sealed_end, self.seal_available_at))
                or self.declared_start >= self.sealed_end
                or self.seal_available_at < self.sealed_end):
            raise ValueError("invalid source seal stream or interval")
        try:
            ZoneInfo(self.source_timezone)
        except (ValueError, ZoneInfoNotFoundError) as exc:
            # UTC is intrinsic even on systems without a timezone database.
            if self.source_timezone != "UTC":
                raise ValueError("unknown source timezone") from exc

    @property
    def declaration_id(self) -> str:
        payload = asdict(self)
        for key in ("declared_start", "sealed_end", "seal_available_at"):
            payload[key] = format_utc_timestamp(payload[key])
        return "source_coverage_contract_v1:" + digest({
            "schema": "mlof.source_coverage_contract.v1", **payload})


@dataclass(frozen=True, slots=True)
class SealEvidence:
    """A locally reconstructed statement or an independently pinned external artifact."""

    kind: str
    statement: bytes

    @property
    def provenance_id(self) -> str:
        return "source_seal_evidence_v1:" + hashlib.sha256(self.statement).hexdigest()


@dataclass(frozen=True, slots=True)
class ReplayInput:
    seal: SourceSeal
    candles: tuple[Candle, ...]
    evidence: SealEvidence | None = None

    def seal_statement(self) -> bytes:
        declaration = asdict(self.seal)
        del declaration["provenance_id"]
        for key in ("declared_start", "sealed_end", "seal_available_at"):
            declaration[key] = format_utc_timestamp(declaration[key])
        records = [
            {
                "stream": [c.instrument.upper(), c.timeframe.lower()],
                "interval": [format_utc_timestamp(c.interval_start), format_utc_timestamp(c.interval_end)],
                "available_at": format_utc_timestamp(c.source_available_at),
                "ohlc": [format(getattr(c, name), "f") for name in ("open", "high", "low", "close")],
                "source": asdict(c.provenance),
            }
            for c in self.candles if c.interval_end <= self.seal.sealed_end
        ]
        return canonical_bytes({"schema": "mlof.source_seal_evidence.v1",
                                "declaration": declaration, "sealed_records_digest": digest(records)})

    @classmethod
    def locally_sealed(cls, seal: SourceSeal, candles: tuple[Candle, ...]) -> "ReplayInput":
        from dataclasses import replace

        draft = cls(seal, candles)
        evidence = SealEvidence("LOCAL", draft.seal_statement())
        return cls(replace(seal, provenance_id=evidence.provenance_id), candles, evidence)

    def verify_seal(self, trusted_external: frozenset[str]) -> bool:
        evidence = self.evidence
        if (evidence is None or evidence.provenance_id != self.seal.provenance_id
                or evidence.statement != self.seal_statement()):
            return False
        return evidence.kind == "LOCAL" or (
            evidence.kind == "EXTERNAL" and evidence.provenance_id in trusted_external
        )

    @staticmethod
    def _valid_provenance(provenance: SourceProvenance) -> bool:
        return bool(provenance.record_id and provenance.source_path and provenance.source_line >= 1
                    and all(len(value) == 64 and all(c in "0123456789abcdef" for c in value)
                            for value in (provenance.source_sha256, provenance.source_row_sha256)))

    def manifest_payload(self, spec: InstrumentSpec, config_version: str, spec_version: str):
        def record(candle: Candle):
            return {
                "candle_id": candle.provenance.record_id,
                "interval_start": format_utc_timestamp(candle.interval_start),
                "interval_end": format_utc_timestamp(candle.interval_end),
                "source_available_at": format_utc_timestamp(candle.source_available_at),
                "ohlc_ticks": [spec.price_to_ticks(getattr(candle, name))
                               for name in ("open", "high", "low", "close")],
                "provenance_id": digest(asdict(candle.provenance)),
            }

        return {
            "schema": "mlof.replay_manifest.v1", "spec_version": spec_version,
            "config_version": config_version, "calendar_version": self.seal.calendar_version,
            "instrument": self.seal.instrument, "timeframe": self.seal.timeframe,
            "tick_size": format(spec.tick_size.normalize(), "f"),
            "records": [record(candle) for candle in self.candles],
        }

    def input_error(self, calendar: SessionCalendar, at: datetime, processed: tuple[Candle, ...],
                    trusted_external: frozenset[str] = frozenset()):
        seal = self.seal
        expected = intervals(calendar, seal.instrument, seal.timeframe, seal.declared_start, at)
        if expected is None:
            return "UNKNOWN_CALENDAR", ()
        declared = intervals(calendar, seal.instrument, seal.timeframe,
                             seal.declared_start, seal.sealed_end)
        if (seal.assertion != "SEALED_COMPLETE_THROUGH_BOUNDARY"
                or seal.seal_available_at > at or not self.verify_seal(trusted_external)
                or not seal.dataset_id or not seal.source_timezone or not seal.timestamp_convention
                or calendar.calendar_id != seal.calendar_version or declared is None
                or interval_digest(declared) != seal.expected_intervals_digest
                or (expected and expected[-1][1] > seal.sealed_end)):
            return "UNVERIFIED_SEAL", expected
        prefix = tuple(c for c in self.candles if c.interval_end <= at)
        if tuple((c.interval_start, c.interval_end) for c in prefix) != expected:
            return "INPUT_GAP", expected
        if prefix != processed or any(c.source_available_at > at for c in prefix):
            return "INPUT_GAP", expected
        if any(b.source_available_at < a.source_available_at for a, b in zip(prefix, prefix[1:])):
            return "AVAILABILITY_INVERSION", expected
        if (len({c.provenance.record_id for c in prefix}) != len(prefix)
                or any((c.instrument.upper(), c.timeframe.lower()) != (seal.instrument, seal.timeframe)
                       or not self._valid_provenance(c.provenance) for c in prefix)):
            return "INVALID_INPUT", expected
        return None, expected


@dataclass(frozen=True, slots=True)
class CoverageRecord:
    coverage_id: str
    evaluation_identity: str
    information_time: datetime
    canonical_payload: bytes


class PublicationLedger:
    """Evaluation-scoped immutable publication records with atomic collision preflight."""

    def __init__(self):
        self._records: dict[tuple[str, str], object] = {}

    def get(self, evaluation_identity: str, entity_id: str):
        return self._records.get((evaluation_identity, entity_id))

    def publish(self, evaluation_identity: str, records: tuple[tuple[str, object], ...]) -> None:
        staged: dict[tuple[str, str], object] = {}
        for entity_id, payload in records:
            key = evaluation_identity, entity_id
            prior = staged.get(key, self._records.get(key))
            if prior is not None and prior != payload:
                raise ValueError("replay publication ID collision")
            staged[key] = payload
        self._records.update(staged)
