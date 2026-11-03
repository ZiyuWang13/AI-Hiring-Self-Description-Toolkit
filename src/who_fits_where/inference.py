"""Matched source-category inference used by the formal analyses.

The functions in this module operate on one already-constructed effect per
source category.  They never treat job rows or profile pairs as independent
inferential observations.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats


class InferenceError(ValueError):
    """Raised when an inferential input violates the matched design."""


BootstrapPValueRule = Literal["plus_one", "empirical"]
SeedRule = Literal["direct", "stable_label"]
ConstantEffectSizeRule = Literal["nan", "signed_inf"]


@dataclass(frozen=True)
class BootstrapConfig:
    """A documented formal bootstrap configuration."""

    repetitions: int
    seed: int
    pvalue_rule: BootstrapPValueRule
    seed_rule: SeedRule = "direct"

    def __post_init__(self) -> None:
        if self.repetitions < 1:
            raise InferenceError("Bootstrap repetitions must be positive")
        if self.pvalue_rule not in {"plus_one", "empirical"}:
            raise InferenceError(f"Unsupported bootstrap p-value rule: {self.pvalue_rule}")
        if self.seed_rule not in {"direct", "stable_label"}:
            raise InferenceError(f"Unsupported bootstrap seed rule: {self.seed_rule}")


# Formal configurations found in the validated source scripts.
HAIKU_SINGLE_MODEL_BOOTSTRAP = BootstrapConfig(
    repetitions=10_000,
    seed=20260928,
    pvalue_rule="plus_one",
    seed_rule="stable_label",
)
CROSS_MODEL_BOOTSTRAP = BootstrapConfig(
    repetitions=10_000,
    seed=20260928,
    pvalue_rule="empirical",
    seed_rule="direct",
)
HAIKU_THREE_RUN_BOOTSTRAP = BootstrapConfig(
    repetitions=20_000,
    seed=20261022,
    pvalue_rule="plus_one",
    seed_rule="direct",
)


def stable_label_seed(base_seed: int, label: str) -> int:
    """Reproduce the Haiku single-model SHA-256 label-derived seed."""

    digest = hashlib.sha256(f"{base_seed}|{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _numeric_observations(values: Iterable[float]) -> np.ndarray:
    """Coerce values to float and drop missing observations without imputation."""

    return pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().to_numpy(float)


def category_values(
    frame: pd.DataFrame,
    *,
    category_col: str,
    value_col: str,
    expected_n: int | None = 105,
) -> np.ndarray:
    """Extract one observed effect per source category.

    Missing values are removed as complete cases. Duplicate categories fail
    closed because resampling them as separate observations would break the
    paired design.
    """

    missing = {category_col, value_col} - set(frame.columns)
    if missing:
        raise InferenceError(f"Missing category-inference columns: {sorted(missing)}")
    data = frame[[category_col, value_col]].copy()
    if data[category_col].isna().any():
        raise InferenceError("Source-category identifiers may not be missing")
    if data[category_col].duplicated().any():
        raise InferenceError("Expected one inferential observation per source category")
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    observed = data.dropna(subset=[value_col])
    if expected_n is not None and len(observed) != expected_n:
        raise InferenceError(f"Expected N={expected_n}, observed {len(observed)} complete cases")
    return observed[value_col].to_numpy(float)


def _resolved_seed(config: BootstrapConfig, label: str, seed_offset: int) -> int:
    if config.seed_rule == "stable_label":
        if seed_offset:
            raise InferenceError("Seed offsets are not used with stable-label seeds")
        return stable_label_seed(config.seed, label)
    return config.seed + seed_offset


def bootstrap_category_mean(
    values: Iterable[float],
    *,
    label: str,
    config: BootstrapConfig,
    seed_offset: int = 0,
    chunk_size: int = 1_000,
) -> dict[str, float | int | str]:
    """Percentile bootstrap for a mean of matched category-level effects.

    Each bootstrap draw resamples the complete category-effect row. Therefore
    any within-category profile/run dependence must be incorporated before this
    function is called, as in the formal analyses.
    """

    x = _numeric_observations(values)
    if len(x) == 0:
        return {
            "bootstrap_CI95_low": math.nan,
            "bootstrap_CI95_high": math.nan,
            "bootstrap_p": math.nan,
            "bootstrap_repetitions": config.repetitions,
            "bootstrap_seed": _resolved_seed(config, label, seed_offset),
            "bootstrap_pvalue_rule": config.pvalue_rule,
        }
    if chunk_size < 1:
        raise InferenceError("Bootstrap chunk size must be positive")
    seed = _resolved_seed(config, label, seed_offset)
    rng = np.random.default_rng(seed)
    means = np.empty(config.repetitions, dtype=float)
    for start in range(0, config.repetitions, chunk_size):
        size = min(chunk_size, config.repetitions - start)
        indices = rng.integers(0, len(x), size=(size, len(x)))
        means[start : start + size] = x[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    if config.pvalue_rule == "plus_one":
        lower = (np.count_nonzero(means <= 0) + 1) / (config.repetitions + 1)
        upper = (np.count_nonzero(means >= 0) + 1) / (config.repetitions + 1)
    else:
        lower = np.mean(means <= 0)
        upper = np.mean(means >= 0)
    return {
        "bootstrap_CI95_low": float(low),
        "bootstrap_CI95_high": float(high),
        "bootstrap_p": min(1.0, 2.0 * min(float(lower), float(upper))),
        "bootstrap_repetitions": config.repetitions,
        "bootstrap_seed": seed,
        "bootstrap_pvalue_rule": config.pvalue_rule,
    }


def paired_category_inference(
    values: Iterable[float],
    *,
    label: str,
    expected_n: int | None = 105,
    bootstrap_config: BootstrapConfig | None = HAIKU_SINGLE_MODEL_BOOTSTRAP,
    bootstrap_seed_offset: int = 0,
    constant_effect_size_rule: ConstantEffectSizeRule = "nan",
) -> dict[str, float | int | str]:
    """Formal one-sample inference on within-category paired differences."""

    x = _numeric_observations(values)
    n = len(x)
    if expected_n is not None and n != expected_n:
        raise InferenceError(f"{label}: expected N={expected_n}, observed {n}")
    if n < 2:
        raise InferenceError(f"{label}: fewer than two paired category observations")
    estimate = float(x.mean())
    sd = float(x.std(ddof=1))
    se = sd / math.sqrt(n)
    critical = float(stats.t.ppf(0.975, n - 1))
    if constant_effect_size_rule not in {"nan", "signed_inf"}:
        raise InferenceError(
            f"Unsupported constant-effect-size rule: {constant_effect_size_rule}"
        )
    if np.isclose(sd, 0):
        t_stat = 0.0 if np.isclose(estimate, 0) else math.copysign(math.inf, estimate)
        paired_t_p = 1.0 if np.isclose(estimate, 0) else 0.0
        if constant_effect_size_rule == "signed_inf":
            cohens_dz = 0.0 if np.isclose(estimate, 0) else math.copysign(math.inf, estimate)
        else:
            cohens_dz = math.nan
    else:
        test = stats.ttest_1samp(x, 0)
        t_stat = float(test.statistic)
        paired_t_p = float(test.pvalue)
        cohens_dz = estimate / sd
    if np.allclose(x, 0):
        wilcoxon_stat, wilcoxon_p = 0.0, 1.0
    else:
        try:
            test_w = stats.wilcoxon(
                x,
                zero_method="wilcox",
                correction=False,
                alternative="two-sided",
            )
            wilcoxon_stat = float(test_w.statistic)
            wilcoxon_p = float(test_w.pvalue)
        except ValueError:
            wilcoxon_stat, wilcoxon_p = math.nan, math.nan
    result: dict[str, float | int | str] = {
        "N": n,
        "estimate": estimate,
        "SD": sd,
        "SE": se,
        "CI95_low": estimate - critical * se,
        "CI95_high": estimate + critical * se,
        "t_stat": t_stat,
        "paired_t_p": paired_t_p,
        "wilcoxon_stat": wilcoxon_stat,
        "wilcoxon_p": wilcoxon_p,
        "cohens_dz": cohens_dz,
    }
    if bootstrap_config is None:
        result.update(
            {
                "bootstrap_CI95_low": math.nan,
                "bootstrap_CI95_high": math.nan,
                "bootstrap_p": math.nan,
                "bootstrap_repetitions": 0,
                "bootstrap_seed": math.nan,
                "bootstrap_pvalue_rule": "not_run",
            }
        )
    else:
        result.update(
            bootstrap_category_mean(
                x,
                label=label,
                config=bootstrap_config,
                seed_offset=bootstrap_seed_offset,
            )
        )
    return result


def cluster_rows_by_category(
    frame: pd.DataFrame,
    *,
    category_col: str,
    dependent_cols: Sequence[str],
    expected_n: int | None = 105,
) -> pd.DataFrame:
    """Validate a wide category-cluster table before joint resampling.

    This is the representation used by the three-run formal analysis: one row
    per source category, with all dependent run/profile quantities retained in
    that row. It intentionally performs no inference by itself.
    """

    required = {category_col, *dependent_cols}
    missing = required - set(frame.columns)
    if missing:
        raise InferenceError(f"Missing cluster columns: {sorted(missing)}")
    data = frame[[category_col, *dependent_cols]].copy()
    if data[category_col].isna().any() or data[category_col].duplicated().any():
        raise InferenceError("Cluster table must contain one unique row per source category")
    if expected_n is not None and len(data) != expected_n:
        raise InferenceError(f"Expected {expected_n} category clusters, observed {len(data)}")
    return data.reset_index(drop=True)
