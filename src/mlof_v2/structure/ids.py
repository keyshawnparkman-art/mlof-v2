"""Stable identifiers for structural evidence; no randomness or clock input."""

from __future__ import annotations

import hashlib
import json
from typing import Mapping, Any


def stable_structural_id(namespace: str, fields: Mapping[str, Any]) -> str:
    body = json.dumps(
        {"namespace": namespace, **fields},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return f"{namespace.lower()}_{digest}"
