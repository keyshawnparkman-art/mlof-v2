"""Versioned JSON configuration loading using only the Python standard library."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .models import InstrumentSpec


@dataclass(frozen=True, slots=True)
class AppConfig:
    config_version: str
    schema_version: int
    min_ready_bars: int
    instruments: tuple[InstrumentSpec, ...]
    calendar_id: str | None

    def instrument(self, identifier: str) -> InstrumentSpec:
        normalized = identifier.strip().upper()
        for spec in self.instruments:
            if spec.instrument == normalized:
                return spec
        raise KeyError(f"instrument {normalized!r} has no configured tick size")


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read configuration {config_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("configuration root must be a JSON object")
    allowed_keys = {"schema_version", "config_version", "min_ready_bars", "instruments", "calendar_id"}
    if set(data) - allowed_keys:
        raise ValueError(f"unknown configuration fields: {', '.join(sorted(set(data) - allowed_keys))}")
    if type(data.get("schema_version")) is not int or data.get("schema_version") != 1:
        raise ValueError("unsupported or missing schema_version; expected 1")
    version = data.get("config_version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("config_version must be a non-empty string")
    min_ready_bars = data.get("min_ready_bars", 50)
    if not isinstance(min_ready_bars, int) or isinstance(min_ready_bars, bool) or min_ready_bars < 1:
        raise ValueError("min_ready_bars must be a positive integer")
    raw_instruments = data.get("instruments")
    if not isinstance(raw_instruments, list) or not raw_instruments:
        raise ValueError("instruments must be a non-empty list")
    instruments: list[InstrumentSpec] = []
    seen: set[str] = set()
    for item in raw_instruments:
        if not isinstance(item, dict):
            raise ValueError("each instrument entry must be an object")
        if set(item) - {"instrument", "tick_size"}:
            raise ValueError("instrument entries may contain only instrument and tick_size")
        name = item.get("instrument")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("instrument must be a non-empty string")
        name = name.strip().upper()
        if name in seen:
            raise ValueError(f"duplicate instrument configuration: {name}")
        seen.add(name)
        raw_tick = str(item.get("tick_size", ""))
        if re.fullmatch(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?", raw_tick) is None:
            raise ValueError(f"tick_size for {name} must be a plain decimal string")
        try:
            tick = Decimal(raw_tick)
        except InvalidOperation as exc:
            raise ValueError(f"invalid tick_size for {name}") from exc
        if not tick.is_finite() or tick <= 0:
            raise ValueError(f"tick_size for {name} must be finite and positive")
        instruments.append(InstrumentSpec(name, tick))
    calendar_id = data.get("calendar_id")
    if calendar_id is not None and (not isinstance(calendar_id, str) or not calendar_id.strip()):
        raise ValueError("calendar_id must be null or a non-empty string")
    return AppConfig(version, 1, min_ready_bars, tuple(instruments), calendar_id)
