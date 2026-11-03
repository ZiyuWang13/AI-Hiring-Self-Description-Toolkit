from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from who_fits_where.decomposition import (  # noqa: E402
    assert_reconstruction,
    decompose_contrast,
    decompose_job_wages,
    decompose_wage,
    gender_interaction_decomposition,
)


class DecompositionTests(unittest.TestCase):
    def test_profile_identity(self):
        result = decompose_wage(100.0, 112.0)
        self.assertEqual(result.selection_premium, 12.0)
        assert_reconstruction(result)

    def test_job_wage_means(self):
        result = decompose_job_wages([100.0] * 60, [110.0] * 10)
        self.assertEqual(result.pool_value, 100.0)
        self.assertEqual(result.selection_premium, 10.0)
        self.assertEqual(result.final_wage, 110.0)

    def test_contrast_identity(self):
        pool = {"MB": 100.0, "MH": 100.0, "MT": 100.0, "FB": 100.0, "FH": 100.0, "FT": 100.0}
        final = {"MB": 100.0, "MH": 95.0, "MT": 110.0, "FB": 100.0, "FH": 108.0, "FT": 104.0}
        result = decompose_contrast(pool, final, "hardworking_gender_interaction")
        self.assertEqual(result.pool_value_component, 0.0)
        self.assertEqual(result.selection_premium_component, 13.0)
        self.assertEqual(result.final_wage_effect, 13.0)
        assert_reconstruction(result)

    def test_gender_interaction_decomposition(self):
        pool = {"MB": 100.0, "MH": 99.0, "MT": 103.0, "FB": 98.0, "FH": 101.0, "FT": 102.0}
        final = {"MB": 105.0, "MH": 102.0, "MT": 111.0, "FB": 100.0, "FH": 107.0, "FT": 104.0}
        for signal in ("Hardworking", "Talent"):
            result = gender_interaction_decomposition(pool, final, signal=signal)
            self.assertTrue(math.isclose(result.reconstruction_error, 0.0, abs_tol=1e-8))


if __name__ == "__main__":
    unittest.main()
