from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from who_fits_where.contrasts import category_level_contrasts, compute_contrast  # noqa: E402


class ContrastTests(unittest.TestCase):
    def setUp(self):
        self.values = {"MB": 100.0, "MH": 95.0, "MT": 110.0, "FB": 100.0, "FH": 108.0, "FT": 104.0}

    def test_requested_directions(self):
        self.assertEqual(compute_contrast(self.values, "MB-FB"), 0.0)
        self.assertEqual(compute_contrast(self.values, "MH-MB"), -5.0)
        self.assertEqual(compute_contrast(self.values, "MT-MB"), 10.0)
        self.assertEqual(compute_contrast(self.values, "FH-FB"), 8.0)
        self.assertEqual(compute_contrast(self.values, "FT-FB"), 4.0)

    def test_gender_interaction_directions(self):
        self.assertEqual(compute_contrast(self.values, "hardworking_gender_interaction"), 13.0)
        self.assertEqual(compute_contrast(self.values, "talent_gender_interaction"), -6.0)

    def test_category_level_before_aggregation(self):
        frame = pd.DataFrame(
            [
                {
                    "source_category": category,
                    "profile": profile,
                    "wage": value + offset,
                }
                for category, offset in (("001", 0), ("002", 5))
                for profile, value in self.values.items()
            ]
        )
        result = category_level_contrasts(frame, value_col="wage")
        h = result[result["contrast_id"].eq("hardworking_gender_interaction")]
        self.assertEqual(h["difference"].tolist(), [13.0, 13.0])


if __name__ == "__main__":
    unittest.main()
