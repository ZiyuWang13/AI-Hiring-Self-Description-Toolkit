from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from who_fits_where.inference import BootstrapConfig, paired_category_inference  # noqa: E402
from who_fits_where.result_tables import (  # noqa: E402
    INFERENCE_COLUMNS,
    ResultTableError,
    result_row,
    result_table,
)


SOURCE_HASH = "a" * 64


def metadata() -> dict:
    return {
        "model": "Synthetic",
        "run": "test",
        "condition": "00",
        "analysis_family": "wage",
        "test_family": "wage_primary",
        "outcome_id": "top10_mean_annual_wage",
        "contrast_id": "MH-MB",
        "contrast": "MH − MB",
        "units": "RMB/year",
        "inference_unit": "source_category",
        "methodology_source": "synthetic-formal-source.py",
        "methodology_sha256": SOURCE_HASH,
        "validation_status": "SYNTHETIC_TEST_PASS",
    }


class ResultTableTests(unittest.TestCase):
    def test_schema_and_order_are_stable(self):
        inference = paired_category_inference(
            [1.0, 2.0, 3.0],
            label="table",
            expected_n=3,
            bootstrap_config=BootstrapConfig(100, 5, "plus_one"),
        )
        row = result_row(metadata=metadata(), inference=inference, adjusted_p=0.04)
        frame = result_table([row])
        self.assertEqual(tuple(frame.columns), INFERENCE_COLUMNS)
        self.assertEqual(frame.loc[0, "N"], 3)
        self.assertEqual(frame.loc[0, "p_fdr_bh"], 0.04)

    def test_missing_adjusted_p_remains_missing(self):
        inference = paired_category_inference(
            [0.0, 0.0, 0.0],
            label="table-null",
            expected_n=3,
            bootstrap_config=None,
        )
        frame = result_table([result_row(metadata=metadata(), inference=inference)])
        self.assertTrue(math.isnan(frame.loc[0, "p_fdr_bh"]))

    def test_unknown_statistic_is_not_silently_added(self):
        with self.assertRaises(ResultTableError):
            result_row(metadata=metadata(), inference={"invented_statistic": 1.0})

    def test_invalid_condition_fails_closed(self):
        bad = metadata()
        bad["condition"] = "12"
        inference = paired_category_inference(
            [1.0, 2.0],
            label="bad-condition",
            expected_n=2,
            bootstrap_config=None,
        )
        with self.assertRaises(ResultTableError):
            result_table([result_row(metadata=bad, inference=inference)])


if __name__ == "__main__":
    unittest.main()
