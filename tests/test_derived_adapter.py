from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from who_fits_where.derived_adapter import (  # noqa: E402
    DerivedDataAccessError,
    DerivedDataAdapter,
    sha256_file,
    validate_replay_output_root,
)


PROFILES = ["MB", "MH", "MT", "FB", "FH", "FT"]


def schema(columns: list[str], dtypes: dict[str, str], **extra) -> dict:
    return {
        "columns": columns,
        "dtypes": dtypes,
        "unique_key": extra.get("unique_key", []),
        "required_nonmissing": extra.get("required_nonmissing", []),
        "categorical_domains": extra.get("categorical_domains", {}),
        **({"pairing": extra["pairing"]} if "pairing" in extra else {}),
    }


class DerivedAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "tests")
        self.folder = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write_manifest(self, entries: list[dict], enabled: bool = True) -> Path:
        path = self.folder / "manifest.json"
        path.write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "real_data_access_enabled": enabled,
                    "approved_inputs": entries,
                    "notes": "synthetic test only",
                }
            ),
            encoding="utf-8",
        )
        return path

    def csv_entry(self, path: Path, schema_definition: dict, input_id: str = "synthetic") -> dict:
        return {
            "input_id": input_id,
            "path": str(path),
            "sha256": sha256_file(path),
            "file_size": path.stat().st_size,
            "format": "csv",
            "schema": schema_definition,
            "provenance_asset_id": "SYNTHETIC",
            "approval_id": "SYNTHETIC_TEST",
        }

    def test_repository_template_empty_allowlist_denies_access(self):
        adapter = DerivedDataAdapter(
            ROOT / "config/derived_input_manifest.example.json",
            repository_root=ROOT,
        )
        self.assertEqual(adapter.approved_input_ids(), ())
        with self.assertRaises(DerivedDataAccessError):
            adapter.load_csv("anything")

    def test_unauthorized_id_is_rejected(self):
        data = self.folder / "derived.csv"
        pd.DataFrame({"source_category": ["001"], "effect": [1.0]}).to_csv(data, index=False)
        spec = schema(
            ["source_category", "effect"],
            {"source_category": "string", "effect": "float"},
            unique_key=["source_category"],
            required_nonmissing=["source_category"],
        )
        adapter = DerivedDataAdapter(
            self.write_manifest([self.csv_entry(data, spec)]), repository_root=ROOT
        )
        with self.assertRaises(DerivedDataAccessError):
            adapter.load_csv("not-approved")

    def test_path_traversal_is_rejected(self):
        data = self.folder / "derived.csv"
        pd.DataFrame({"source_category": ["001"], "effect": [1.0]}).to_csv(data, index=False)
        spec = schema(
            ["source_category", "effect"],
            {"source_category": "string", "effect": "float"},
        )
        entry = self.csv_entry(data, spec)
        entry["path"] = str(self.folder / "sub" / ".." / "derived.csv")
        adapter = DerivedDataAdapter(self.write_manifest([entry]), repository_root=ROOT)
        with self.assertRaises(DerivedDataAccessError):
            adapter.load_csv("synthetic")

    def test_symlink_input_is_rejected(self):
        data = self.folder / "derived.csv"
        pd.DataFrame({"source_category": ["001"], "effect": [1.0]}).to_csv(data, index=False)
        link = self.folder / "derived-link.csv"
        link.symlink_to(data)
        spec = schema(
            ["source_category", "effect"],
            {"source_category": "string", "effect": "float"},
        )
        entry = self.csv_entry(data, spec)
        entry["path"] = str(link)
        adapter = DerivedDataAdapter(self.write_manifest([entry]), repository_root=ROOT)
        with self.assertRaises(DerivedDataAccessError):
            adapter.load_csv("synthetic")

    def test_read_only_load_preserves_source_hash_and_missing_wage(self):
        data = self.folder / "derived.csv"
        pd.DataFrame(
            {"source_category": ["001", "002"], "annual_wage": [120000.0, None]}
        ).to_csv(data, index=False)
        spec = schema(
            ["source_category", "annual_wage"],
            {"source_category": "string", "annual_wage": "float"},
            unique_key=["source_category"],
            required_nonmissing=["source_category"],
        )
        before = sha256_file(data)
        adapter = DerivedDataAdapter(
            self.write_manifest([self.csv_entry(data, spec)]), repository_root=ROOT
        )
        loaded = adapter.load_csv("synthetic")
        self.assertEqual(loaded.loc[0, "source_category"], "001")
        self.assertTrue(pd.isna(loaded.loc[1, "annual_wage"]))
        self.assertEqual(before, sha256_file(data))

    def test_unexpected_columns_and_restricted_schema_fail(self):
        data = self.folder / "derived.csv"
        pd.DataFrame({"source_category": ["001"], "effect": [1.0], "extra": [2]}).to_csv(
            data, index=False
        )
        spec = schema(
            ["source_category", "effect"],
            {"source_category": "string", "effect": "float"},
        )
        adapter = DerivedDataAdapter(
            self.write_manifest([self.csv_entry(data, spec)]), repository_root=ROOT
        )
        with self.assertRaises(DerivedDataAccessError):
            adapter.load_csv("synthetic")

        restricted = schema(["phone"], {"phone": "string"})
        with self.assertRaises(DerivedDataAccessError):
            DerivedDataAdapter(
                self.write_manifest([self.csv_entry(data, restricted)]), repository_root=ROOT
            )

    def test_sameprompt_method_variable_is_not_raw_prompt_text(self):
        data = self.folder / "method.csv"
        column = "sameprompt_three_pair_benchmark_top6_distance"
        pd.DataFrame({"source_category": ["001"], column: [1.0]}).to_csv(data, index=False)
        spec = schema(
            ["source_category", column],
            {"source_category": "string", column: "float"},
            unique_key=["source_category"],
            required_nonmissing=["source_category"],
        )
        adapter = DerivedDataAdapter(
            self.write_manifest([self.csv_entry(data, spec)]), repository_root=ROOT
        )
        self.assertEqual(adapter.load_csv("synthetic").loc[0, column], 1.0)

    def test_duplicate_key_and_incomplete_pairing_fail(self):
        data = self.folder / "profiles.csv"
        rows = [
            {"source_category": "001", "profile": profile, "effect": index}
            for index, profile in enumerate(PROFILES[:-1])
        ]
        pd.DataFrame(rows).to_csv(data, index=False)
        spec = schema(
            ["source_category", "profile", "effect"],
            {"source_category": "string", "profile": "string", "effect": "float"},
            unique_key=["source_category", "profile"],
            required_nonmissing=["source_category", "profile"],
            categorical_domains={"profile": PROFILES},
            pairing={
                "group_by": ["source_category"],
                "member_column": "profile",
                "expected_members": PROFILES,
            },
        )
        adapter = DerivedDataAdapter(
            self.write_manifest([self.csv_entry(data, spec)]), repository_root=ROOT
        )
        with self.assertRaises(DerivedDataAccessError):
            adapter.load_csv("synthetic")

    def test_three_run_identifiers_and_pairing_pass(self):
        data = self.folder / "runs.csv"
        rows = [
            {"source_category": "001", "run": run, "effect": index}
            for index, run in enumerate(["Original", "Repeat01", "Repeat02"])
        ]
        pd.DataFrame(rows).to_csv(data, index=False)
        runs = ["Original", "Repeat01", "Repeat02"]
        spec = schema(
            ["source_category", "run", "effect"],
            {"source_category": "string", "run": "string", "effect": "float"},
            unique_key=["source_category", "run"],
            required_nonmissing=["source_category", "run"],
            categorical_domains={"run": runs},
            pairing={
                "group_by": ["source_category"],
                "member_column": "run",
                "expected_members": runs,
            },
        )
        adapter = DerivedDataAdapter(
            self.write_manifest([self.csv_entry(data, spec)]), repository_root=ROOT
        )
        loaded = adapter.load_csv("synthetic")
        self.assertEqual(set(loaded["run"]), set(runs))

    def test_output_root_inside_repository_is_rejected(self):
        with self.assertRaises(DerivedDataAccessError):
            validate_replay_output_root(ROOT / "results/replay", repository_root=ROOT)
        outside = validate_replay_output_root(
            ROOT.parent / ".synthetic-replay-outside", repository_root=ROOT
        )
        self.assertNotIn(ROOT, outside.parents)


if __name__ == "__main__":
    unittest.main()
