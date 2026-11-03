from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from who_fits_where.multiplicity import (  # noqa: E402
    MultiplicityError,
    add_bh_fdr,
    benjamini_hochberg,
)


class MultiplicityTests(unittest.TestCase):
    def test_bh_matches_independently_calculated_example(self):
        observed = benjamini_hochberg([0.01, 0.04, 0.03, 0.002, math.nan])
        expected = np.array([0.02, 0.04, 0.04, 0.008, math.nan])
        np.testing.assert_allclose(observed, expected, equal_nan=True)

    def test_bh_preserves_original_order(self):
        observed = benjamini_hochberg([0.5, 0.001, 0.02])
        np.testing.assert_allclose(observed, [0.5, 0.003, 0.03])

    def test_families_are_adjusted_separately(self):
        frame = pd.DataFrame(
            {
                "family": ["wage", "wage", "overlap", "overlap"],
                "p": [0.01, 0.04, 0.01, 0.20],
            }
        )
        result = add_bh_fdr(frame, p_col="p", family_cols=("family",))
        np.testing.assert_allclose(result["p_fdr_bh"], [0.02, 0.04, 0.02, 0.20])

    def test_family_definition_is_mandatory(self):
        frame = pd.DataFrame({"p": [0.01]})
        with self.assertRaises(MultiplicityError):
            add_bh_fdr(frame, p_col="p", family_cols=())

    def test_invalid_probability_fails_closed(self):
        with self.assertRaises(MultiplicityError):
            benjamini_hochberg([0.2, 1.1])


if __name__ == "__main__":
    unittest.main()
