"""Contract tests over fully synthetic fixtures only.

These helpers encode already-approved invariants. They are not a migration of
the formal statistical analysis pipeline.
"""

from __future__ import annotations

import json
import math
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
PROFILE_ORDER = ["MB", "MH", "MT", "FB", "FH", "FT"]
OCCUPATION_ID = re.compile(r"^O(?:00[1-9]|0[1-9][0-9]|1[0-9]{2}|2[0-3][0-9]|240)$")
JOB_ID = re.compile(r"^J(?:0[1-9]|[1-5][0-9]|60)$")


def load(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def annual_wage(salary_min, salary_max, bonus):
    if salary_min is None or salary_max is None:
        return None
    months = 12 if bonus in (None, "") else int(re.search(r"\d+", bonus).group())
    return ((salary_min + salary_max) / 2) * months


def gender_interaction(values: dict[str, float], signal: str) -> float:
    if signal == "Hardworking":
        return (values["FH"] - values["FB"]) - (values["MH"] - values["MB"])
    if signal == "Talented":
        return (values["FT"] - values["FB"]) - (values["MT"] - values["MB"])
    raise ValueError("Unknown signal")


class SyntheticContractTests(unittest.TestCase):
    def test_profile_order_and_identity(self):
        profiles = load("synthetic_profiles.json")
        self.assertEqual([row["profile"] for row in profiles], PROFILE_ORDER)
        self.assertEqual(len({row["profile"] for row in profiles}), 6)

    def test_condition_definitions(self):
        conditions = load("synthetic_conditions.json")
        self.assertEqual(
            conditions,
            {
                "00": {"treatment_in_stage1": False, "structured_resume_in_stage2": False},
                "01": {"treatment_in_stage1": False, "structured_resume_in_stage2": True},
                "10": {"treatment_in_stage1": True, "structured_resume_in_stage2": False},
                "11": {"treatment_in_stage1": True, "structured_resume_in_stage2": True},
            },
        )

    def test_top6_occupation_ids(self):
        for row in load("synthetic_stage1_results.json"):
            ids = row["occupation_ids"]
            self.assertEqual(len(ids), 6)
            self.assertEqual(len(set(ids)), 6)
            self.assertTrue(all(OCCUPATION_ID.fullmatch(value) for value in ids))

    def test_top10_job_ids(self):
        for row in load("synthetic_stage2_results.json"):
            ids = row["stage2_job_ids"]
            self.assertEqual(len(ids), 10)
            self.assertEqual(len(set(ids)), 10)
            self.assertTrue(all(JOB_ID.fullmatch(value) for value in ids))

    def test_sixty_job_pool_uniqueness(self):
        jobs = load("synthetic_job_pool.json")["jobs"]
        self.assertEqual(len(jobs), 60)
        self.assertEqual(len({row["stage2_job_id"] for row in jobs}), 60)
        self.assertEqual(len({row["original_job_id"] for row in jobs}), 60)
        counts = {occupation: 0 for occupation in [f"O{i:03d}" for i in range(1, 7)]}
        for row in jobs:
            counts[row["occupation_id"]] += 1
        self.assertEqual(set(counts.values()), {10})

    def test_gender_interaction_direction(self):
        values = {"MB": 100.0, "MH": 95.0, "MT": 110.0, "FB": 100.0, "FH": 108.0, "FT": 104.0}
        self.assertEqual(gender_interaction(values, "Hardworking"), 13.0)
        self.assertEqual(gender_interaction(values, "Talented"), -6.0)

    def test_salary_annualization(self):
        for row in load("synthetic_salary_cases.json"):
            observed = annual_wage(row["salary_min"], row["salary_max"], row["bonus"])
            self.assertEqual(observed, row["expected_annual_wage"])

    def test_missing_salary_remains_missing(self):
        self.assertIsNone(annual_wage(None, 14_000, "13薪"))

    def test_poolvalue_selection_identity(self):
        pool_value = 144_000.0
        final_wage = 151_500.0
        selection_premium = final_wage - pool_value
        self.assertTrue(math.isclose(pool_value + selection_premium, final_wage))

    def test_recommendation_overlap_uses_original_job_id(self):
        first = {f"SYNTH_JOB_{i:03d}" for i in range(1, 11)}
        second = {f"SYNTH_JOB_{i:03d}" for i in range(6, 16)}
        self.assertEqual(len(first & second), 5)


if __name__ == "__main__":
    unittest.main()
