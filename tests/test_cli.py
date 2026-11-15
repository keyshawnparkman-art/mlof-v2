from __future__ import annotations

import contextlib
import io
import unittest
from pathlib import Path

from mlof_v2.cli import main
from mlof_v2.models import ReadinessState


class CliTests(unittest.TestCase):
    def test_synthetic_example_command_validates_without_network(self) -> None:
        root = Path(__file__).resolve().parents[1]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main([
                "validate-csv",
                "--csv", str(root / "data" / "sample" / "synthetic_ohlc.csv"),
                "--config", str(root / "config" / "instruments.example.json"),
                "--calendar", str(root / "config" / "calendar.synthetic.example.json"),
                "--as-of-utc", "2024-01-01T00:06:00Z",
            ])
        self.assertEqual(code, 0)
        self.assertIn('"readiness":"WARMING"', output.getvalue())

    def test_invalid_output_outside_workspace_is_refused(self) -> None:
        root = Path(__file__).resolve().parents[1]
        output = io.StringIO()
        errors = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            code = main([
                "validate-csv",
                "--csv", str(root / "data" / "sample" / "synthetic_ohlc.csv"),
                "--config", str(root / "config" / "instruments.example.json"),
                "--calendar", str(root / "config" / "calendar.synthetic.example.json"),
                "--as-of-utc", "2024-01-01T00:06:00Z",
                "--audit-out", str(root.parent / "forbidden.json"),
            ])
        self.assertEqual(code, 2)
        self.assertIn("outputs must remain inside", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
