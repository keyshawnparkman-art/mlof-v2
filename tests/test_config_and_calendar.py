from __future__ import annotations

import json
import tomllib
import tempfile
import unittest
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

from mlof_v2.calendars import CalendarRegistry
from mlof_v2.calendar_config import load_calendar
from mlof_v2.config import load_config

from helpers import BASE


class ConfigAndCalendarTests(unittest.TestCase):
    def test_example_config_has_versioned_tick_metadata(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "config" / "instruments.example.json")
        self.assertEqual(config.schema_version, 1)
        self.assertEqual(config.config_version, "mlof-v2-data-example-1")
        self.assertEqual(config.instrument("testusd").tick_size, Decimal("0.01"))

    def test_unknown_calendar_configuration_is_not_ready(self) -> None:
        calendar = load_calendar(None, "not-configured")
        result = calendar.expected_starts("TESTUSD", "1m", BASE, BASE + timedelta(minutes=1))
        self.assertIsNone(result)

    def test_calendar_outside_coverage_is_unknown(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "config" / "instruments.example.json")
        calendar = load_calendar(root / "config" / "calendar.synthetic.example.json", config.calendar_id)
        result = calendar.expected_starts("TESTUSD", "1m", BASE, BASE + timedelta(minutes=10))
        self.assertIsNone(result)

    def test_invalid_configuration_fails_closed(self) -> None:
        content = {
            "schema_version": 1,
            "config_version": "bad",
            "min_ready_bars": 50,
            "calendar_id": None,
            "instruments": [{"instrument": "TESTUSD", "tick_size": "0"}],
        }
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "data") as temp:
            path = Path(temp) / "bad_config.json"
            path.write_text(json.dumps(content), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "finite and positive"):
                load_config(path)

    def test_schema_files_and_project_metadata_parse(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for name in ("candle.schema.json", "config.schema.json"):
            schema = json.loads((root / "schemas" / name).read_text(encoding="utf-8"))
            self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(metadata["project"]["dependencies"], [])

    def test_calendar_registry_fails_closed_for_unmapped_instrument(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "config" / "instruments.example.json")
        calendar = load_calendar(root / "config" / "calendar.synthetic.example.json", config.calendar_id)
        registry = CalendarRegistry((("TESTUSD", calendar),))
        result = registry.expected_starts("OTHER", "1m", BASE, BASE + timedelta(minutes=1))
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
