from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from who_fits_where.contrasts import category_level_contrasts  # noqa: E402
from who_fits_where.inference import (  # noqa: E402
    BootstrapConfig,
    InferenceError,
    bootstrap_category_mean,
    category_values,
    cluster_rows_by_category,
    paired_category_inference,
    stable_label_seed,
)


class InferenceTests(unittest.TestCase):
    def test_stable_label_seed_is_deterministic_and_label_specific(self):
        first = stable_label_seed(20260928, "wage|00|MH-MB")
        self.assertEqual(first, stable_label_seed(20260928, "wage|00|MH-MB"))
        self.assertNotEqual(first, stable_label_seed(20260928, "wage|00|MT-MB"))

    def test_bootstrap_matches_independent_small_empirical_example(self):
        values = np.array([-1.0, 1.0, 3.0])
        config = BootstrapConfig(200, 17, "empirical", "direct")
        observed = bootstrap_category_mean(values, label="small", config=config)

        rng = np.random.default_rng(17)
        indices = rng.integers(0, 3, size=(200, 3))
        means = values[indices].mean(axis=1)
        expected_low, expected_high = np.quantile(means, [0.025, 0.975])
        expected_p = min(1.0, 2 * min(np.mean(means <= 0), np.mean(means >= 0)))
        self.assertEqual(observed["bootstrap_CI95_low"], expected_low)
        self.assertEqual(observed["bootstrap_CI95_high"], expected_high)
        self.assertEqual(observed["bootstrap_p"], expected_p)

    def test_plus_one_pvalue_rule_differs_from_empirical_rule(self):
        values = [5.0, 6.0, 7.0]
        empirical = bootstrap_category_mean(
            values,
            label="positive",
            config=BootstrapConfig(100, 9, "empirical"),
        )
        corrected = bootstrap_category_mean(
            values,
            label="positive",
            config=BootstrapConfig(100, 9, "plus_one"),
        )
        self.assertEqual(empirical["bootstrap_p"], 0.0)
        self.assertAlmostEqual(corrected["bootstrap_p"], 2 / 101)

    def test_paired_inference_uses_category_differences(self):
        values = [1.0, 2.0, 3.0, 4.0]
        result = paired_category_inference(
            values,
            label="synthetic",
            expected_n=4,
            bootstrap_config=BootstrapConfig(250, 11, "plus_one"),
        )
        self.assertEqual(result["N"], 4)
        self.assertEqual(result["estimate"], 2.5)
        self.assertAlmostEqual(result["cohens_dz"], 2.5 / np.std(values, ddof=1))
        self.assertLess(result["paired_t_p"], 0.05)

    def test_formal_constant_effect_size_variants_are_explicit(self):
        haiku = paired_category_inference(
            [2.0, 2.0, 2.0],
            label="constant-haiku",
            expected_n=3,
            bootstrap_config=None,
        )
        cross_model = paired_category_inference(
            [2.0, 2.0, 2.0],
            label="constant-cross-model",
            expected_n=3,
            bootstrap_config=None,
            constant_effect_size_rule="signed_inf",
        )
        self.assertTrue(math.isnan(haiku["cohens_dz"]))
        self.assertEqual(cross_model["cohens_dz"], math.inf)

    def test_contrasts_are_formed_before_inference(self):
        base = {"MB": 100, "MH": 90, "MT": 110, "FB": 100, "FH": 105, "FT": 103}
        frame = pd.DataFrame(
            [
                {"source_category": category, "profile": profile, "wage": value + shift}
                for category, shift in (("001", 0), ("002", 50), ("003", -20))
                for profile, value in base.items()
            ]
        )
        contrasts = category_level_contrasts(
            frame,
            value_col="wage",
            contrast_ids=("hardworking_gender_interaction",),
        )
        result = paired_category_inference(
            contrasts["difference"],
            label="paired",
            expected_n=3,
            bootstrap_config=None,
        )
        self.assertEqual(result["N"], 3)
        self.assertEqual(result["estimate"], 15.0)

    def test_category_extraction_drops_missing_without_imputation(self):
        frame = pd.DataFrame(
            {"source_category": ["001", "002", "003"], "effect": [1.0, math.nan, 3.0]}
        )
        values = category_values(
            frame,
            category_col="source_category",
            value_col="effect",
            expected_n=None,
        )
        np.testing.assert_array_equal(values, [1.0, 3.0])

    def test_duplicate_category_fails_closed(self):
        frame = pd.DataFrame(
            {"source_category": ["001", "001"], "effect": [1.0, 2.0]}
        )
        with self.assertRaises(InferenceError):
            category_values(
                frame,
                category_col="source_category",
                value_col="effect",
                expected_n=None,
            )

    def test_cluster_table_retains_dependent_columns_per_category(self):
        frame = pd.DataFrame(
            {
                "source_category": ["001", "002"],
                "original": [1.0, 2.0],
                "repeat01": [1.5, 1.0],
            }
        )
        result = cluster_rows_by_category(
            frame,
            category_col="source_category",
            dependent_cols=("original", "repeat01"),
            expected_n=2,
        )
        self.assertEqual(result.to_dict("records"), frame.to_dict("records"))


if __name__ == "__main__":
    unittest.main()
