from __future__ import annotations

import unittest

from mlof_v2.timestamps import format_utc_timestamp, parse_utc_timestamp


class TimestampTests(unittest.TestCase):
    def test_z_timestamp_is_utc(self) -> None:
        parsed = parse_utc_timestamp("2024-01-01T00:00:00Z")
        self.assertEqual(parsed.utcoffset().total_seconds(), 0)
        self.assertEqual(format_utc_timestamp(parsed), "2024-01-01T00:00:00Z")

    def test_offset_timestamp_converts_to_utc(self) -> None:
        parsed = parse_utc_timestamp("2024-01-01T01:00:00+01:00")
        self.assertEqual(parsed.isoformat(), "2024-01-01T00:00:00+00:00")

    def test_naive_timestamp_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "timezone offset"):
            parse_utc_timestamp("2024-01-01T00:00:00")

    def test_unknown_minus_zero_offset_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "offset is unknown"):
            parse_utc_timestamp("2024-01-01T00:00:00-00:00")

    def test_invalid_timestamp_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid ISO-8601"):
            parse_utc_timestamp("yesterday")

    def test_surrounding_whitespace_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_utc_timestamp(" 2024-01-01T00:00:00Z")


if __name__ == "__main__":
    unittest.main()
