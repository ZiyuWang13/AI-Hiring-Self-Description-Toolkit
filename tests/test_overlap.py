from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from who_fits_where.overlap import (  # noqa: E402
    all_profile_pair_overlaps,
    top10_job_overlap,
    top6_metrics,
)
from who_fits_where.validation import PROFILE_ORDER  # noqa: E402


class OverlapTests(unittest.TestCase):
    def test_stage1_metrics(self):
        left = ["O001", "O002", "O003", "O004", "O005", "O006"]
        right = ["O001", "O002", "O007", "O004", "O008", "O009"]
        metrics = top6_metrics(left, right)
        self.assertEqual(metrics["top6_overlap_count"], 3)
        self.assertEqual(metrics["replacement_count"], 3)
        self.assertEqual(metrics["rank1_agreement"], 1)
        self.assertEqual(metrics["top1_3_overlap_count"], 2)
        self.assertEqual(metrics["top4_6_overlap_count"], 1)

    def test_top10_uses_original_identity(self):
        left = [f"ORIGINAL_{i:02d}" for i in range(1, 11)]
        right = [f"ORIGINAL_{i:02d}" for i in range(6, 16)]
        self.assertEqual(top10_job_overlap(left, right), 5)

    def test_all_fifteen_profile_pairs(self):
        recommendations = {
            profile: [f"{profile}_ORIGINAL_{i:02d}" for i in range(1, 11)]
            for profile in PROFILE_ORDER
        }
        result = all_profile_pair_overlaps(recommendations)
        self.assertEqual(len(result), 15)
        self.assertTrue(result["overlap_count"].eq(0).all())


if __name__ == "__main__":
    unittest.main()
