"""Versioned structural research configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..timeframes import TIMEFRAME_SECONDS


@dataclass(frozen=True, slots=True)
class SwingWidths:
    left: int
    right: int

    def __post_init__(self) -> None:
        if self.left < 1 or self.right < 1:
            raise ValueError("pivot left/right widths must be positive integers")


@dataclass(frozen=True, slots=True)
class StructuralConfig:
    schema_version: int
    config_version: str
    minimum_contiguous_bars: int
    pivot_widths: tuple[tuple[str, SwingWidths], ...]
    equal_cluster_policy: str

    def __post_init__(self) -> None:
        names = [timeframe for timeframe, _ in self.pivot_widths]
        if len(names) != len(set(names)):
            raise ValueError("pivot configuration contains duplicate timeframes")
        if self.minimum_contiguous_bars < 1:
            raise ValueError("minimum_contiguous_bars must be positive")
        if self.schema_version != 1:
            raise ValueError("unsupported structural config schema")
        if not self.config_version.strip():
            raise ValueError("config_version must be non-empty")
        if self.equal_cluster_policy != "extend_before_confirmation_new_cluster_after_confirmation":
            raise ValueError("unsupported equal-price cluster policy")

    def widths_for(self, timeframe: str) -> SwingWidths:
        normalized = timeframe.lower()
        for name, widths in self.pivot_widths:
            if name == normalized:
                return widths
        raise KeyError(f"no pivot widths configured for timeframe {timeframe!r}")


def load_structural_config(path: str | Path) -> StructuralConfig:
    config_path = Path(path)
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read structural configuration {config_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("structural configuration root must be a JSON object")
    required = {
        "schema_version", "config_version", "minimum_contiguous_bars",
        "pivot_widths", "equal_cluster_policy",
    }
    if set(data) != required:
        raise ValueError("structural configuration fields do not match schema version 1")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("unsupported structural configuration schema_version")
    if not isinstance(data["config_version"], str):
        raise ValueError("config_version must be a string")
    minimum = data["minimum_contiguous_bars"]
    if type(minimum) is not int or minimum < 1:
        raise ValueError("minimum_contiguous_bars must be a positive integer")
    raw_widths = data["pivot_widths"]
    if not isinstance(raw_widths, dict) or not raw_widths:
        raise ValueError("pivot_widths must be a non-empty object")
    widths: list[tuple[str, SwingWidths]] = []
    for timeframe, raw in sorted(raw_widths.items()):
        normalized = timeframe.lower()
        if normalized not in TIMEFRAME_SECONDS:
            raise ValueError(f"unsupported timeframe in pivot_widths: {timeframe}")
        if not isinstance(raw, dict) or set(raw) != {"left", "right"}:
            raise ValueError(f"pivot widths for {timeframe} must contain left and right only")
        if type(raw["left"]) is not int or type(raw["right"]) is not int:
            raise ValueError(f"pivot widths for {timeframe} must be integers")
        widths.append((normalized, SwingWidths(raw["left"], raw["right"])))
    policy = data["equal_cluster_policy"]
    if not isinstance(policy, str):
        raise ValueError("equal_cluster_policy must be a string")
    return StructuralConfig(1, data["config_version"], minimum, tuple(widths), policy)


# Context deadlines are enabled only for the approved research profile.
RECLAIM_PROFILE_VERSION = "m4.2-research-v1"
_EXECUTION_RECLAIM_BARS = (("15s", 6), ("30s", 6), ("1m", 5),
                           ("3m", 4), ("5m", 4), ("15m", 3))
_CONTEXT_RECLAIM_BARS = (("4h", 2), ("1d", 1))


def reclaim_deadline(timeframe: str, profile: str | None = None) -> int | None:
    deadlines = _EXECUTION_RECLAIM_BARS
    if profile == RECLAIM_PROFILE_VERSION:
        deadlines += _CONTEXT_RECLAIM_BARS
    return dict(deadlines).get(timeframe)
