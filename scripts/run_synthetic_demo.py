#!/usr/bin/env python3
"""Run an end-to-end analysis over deterministic, entirely synthetic data."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from who_fits_where.contrasts import (  # noqa: E402
    CONTRAST_LABELS,
    PRIMARY_REQUESTED_CONTRASTS,
    category_level_contrasts,
)
from who_fits_where.inference import (  # noqa: E402
    BootstrapConfig,
    paired_category_inference,
)
from who_fits_where.multiplicity import add_bh_fdr  # noqa: E402
from who_fits_where.result_tables import result_row, result_table  # noqa: E402


def synthetic_profile_wages(n_categories: int, seed: int) -> pd.DataFrame:
    """Create visibly artificial matched profile outcomes."""

    rng = np.random.default_rng(seed)
    profile_effects = {
        "MB": 0.0,
        "MH": -2_500.0,
        "MT": 3_200.0,
        "FB": -800.0,
        "FH": 2_400.0,
        "FT": 600.0,
    }
    rows: list[dict[str, object]] = []
    for number in range(1, n_categories + 1):
        category = f"{number:03d}"
        baseline = 145_000.0 + rng.normal(0.0, 9_000.0)
        common_shock = rng.normal(0.0, 1_500.0)
        for profile, effect in profile_effects.items():
            rows.append(
                {
                    "source_category": category,
                    "profile": profile,
                    "top10_mean_annual_wage": (
                        baseline + common_shock + effect + rng.normal(0.0, 2_500.0)
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_demo(*, n_categories: int, seed: int) -> pd.DataFrame:
    """Create a schema-controlled inferential table from synthetic outcomes."""

    outcomes = synthetic_profile_wages(n_categories, seed)
    effects = category_level_contrasts(
        outcomes,
        value_col="top10_mean_annual_wage",
        contrast_ids=PRIMARY_REQUESTED_CONTRASTS,
    )
    intermediate: list[tuple[str, dict[str, object]]] = []
    for contrast_id, group in effects.groupby("contrast_id", sort=False):
        inference = paired_category_inference(
            group["difference"],
            label=f"synthetic-demo|{contrast_id}",
            expected_n=n_categories,
            bootstrap_config=BootstrapConfig(
                repetitions=2_000,
                seed=seed,
                pvalue_rule="plus_one",
                seed_rule="stable_label",
            ),
        )
        intermediate.append((contrast_id, inference))

    p_values = pd.DataFrame(
        {
            "family": "synthetic_wage_contrasts",
            "contrast_id": [item[0] for item in intermediate],
            "paired_t_p": [item[1]["paired_t_p"] for item in intermediate],
        }
    )
    adjusted = add_bh_fdr(
        p_values,
        p_col="paired_t_p",
        family_cols=("family",),
    ).set_index("contrast_id")["p_fdr_bh"]

    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    rows = []
    for contrast_id, inference in intermediate:
        metadata = {
            "model": "Synthetic",
            "run": "demo",
            "condition": "00",
            "analysis_family": "wage",
            "test_family": "synthetic_wage_contrasts",
            "outcome_id": "top10_mean_annual_wage",
            "contrast_id": contrast_id,
            "contrast": CONTRAST_LABELS[contrast_id],
            "units": "synthetic RMB/year",
            "inference_unit": "source_category",
            "methodology_source": "scripts/run_synthetic_demo.py",
            "methodology_sha256": script_hash,
            "validation_status": "SYNTHETIC_DEMO",
        }
        rows.append(
            result_row(
                metadata=metadata,
                inference=inference,
                adjusted_p=float(adjusted.loc[contrast_id]),
            )
        )
    return result_table(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "examples" / "output" / "synthetic_inference.csv",
    )
    args = parser.parse_args()
    if args.categories < 2 or args.categories > 105:
        parser.error("--categories must be between 2 and 105")
    table = run_demo(n_categories=args.categories, seed=args.seed)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(f"Synthetic demo PASS: {len(table)} contrasts, N={args.categories}")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
