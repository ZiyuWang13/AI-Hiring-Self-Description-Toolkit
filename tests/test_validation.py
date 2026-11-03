from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from who_fits_where.validation import (  # noqa: E402
    ValidationError,
    detect_missing_and_duplicates,
    map_original_job_ids,
    validate_condition,
    validate_job_pool,
    validate_profiles,
    validate_source_category_completeness,
    validate_stage1_top6,
    validate_stage2_top10,
)


def fixture(name: str):
    return json.loads((ROOT / "tests" / "fixtures" / name).read_text(encoding="utf-8"))


class ValidationTests(unittest.TestCase):
    def test_profiles_and_conditions(self):
        rows = fixture("synthetic_profiles.json")
        self.assertEqual(
            validate_profiles(
                [row["profile"] for row in rows], require_fixed_order=True
            ),
            ("MB", "MH", "MT", "FB", "FH", "FT"),
        )
        self.assertEqual(
            tuple(validate_condition(value) for value in (0, 1, 10, 11)),
            ("00", "01", "10", "11"),
        )

    def test_ranked_results(self):
        stage1 = fixture("synthetic_stage1_results.json")[0]["occupation_ids"]
        stage2 = fixture("synthetic_stage2_results.json")[0]["stage2_job_ids"]
        self.assertEqual(len(validate_stage1_top6(stage1)), 6)
        self.assertEqual(len(validate_stage2_top10(stage2)), 10)
        with self.assertRaises(ValidationError):
            validate_stage2_top10(["J01"] * 10)

    def test_pool_and_original_mapping(self):
        pool = fixture("synthetic_job_pool.json")["jobs"]
        self.assertEqual(len(validate_job_pool(pool)), 60)
        originals = map_original_job_ids([f"J{i:02d}" for i in range(1, 11)], pool)
        self.assertEqual(originals[0], "SYNTH_JOB_001")
        self.assertEqual(len(set(originals)), 10)

    def test_completeness_and_duplicate_audit(self):
        rows = fixture("synthetic_profiles.json")
        records = [{"source_category": "001", "profile": row["profile"], "value": 1} for row in rows]
        validate_source_category_completeness(records, expected_categories=("001",))
        audit = detect_missing_and_duplicates(
            records,
            key_fields=("source_category", "profile"),
            required_fields=("value",),
        )
        self.assertTrue(audit.passed)
        duplicate = detect_missing_and_duplicates(
            records + [records[0]], key_fields=("source_category", "profile")
        )
        self.assertEqual(duplicate.duplicate_keys, 1)


if __name__ == "__main__":
    unittest.main()
