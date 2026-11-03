from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from who_fits_where.salary import (  # noqa: E402
    annual_wage,
    annualize_dataframe,
    parse_salary_months,
    pool_mean_wage,
    salary_coverage,
    top10_mean_wage,
)


class SalaryTests(unittest.TestCase):
    def test_annualization_matches_formal_rule(self):
        self.assertEqual(annual_wage(10_000, 14_000, ""), 144_000)
        self.assertEqual(annual_wage("10000", "14000", "13薪"), 156_000)
        self.assertEqual(parse_salary_months("14薪"), 14.0)

    def test_missing_and_unparsable_values_are_not_imputed(self):
        self.assertTrue(math.isnan(annual_wage(None, 14_000, "13薪")))
        self.assertTrue(math.isnan(annual_wage("not numeric", 14_000, "13薪")))
        self.assertTrue(math.isnan(parse_salary_months("bonus unspecified")))

    def test_dataframe_columns_match_formal_construction(self):
        frame = pd.DataFrame({"salary_min": [10_000, ""], "salary_max": [14_000, 20_000], "bonus": ["13薪", ""]})
        result = annualize_dataframe(frame)
        self.assertEqual(result.loc[0, "mid_salary"], 12_000)
        self.assertEqual(result.loc[0, "annual_wage"], 156_000)
        self.assertTrue(pd.isna(result.loc[1, "annual_wage"]))

    def test_coverage_and_observed_means(self):
        top10 = [100.0] * 9 + [math.nan]
        pool = [100.0] * 58 + [math.nan, math.nan]
        self.assertEqual(salary_coverage(top10), 0.9)
        self.assertEqual(top10_mean_wage(top10), 100.0)
        self.assertEqual(pool_mean_wage(pool), 100.0)


if __name__ == "__main__":
    unittest.main()
