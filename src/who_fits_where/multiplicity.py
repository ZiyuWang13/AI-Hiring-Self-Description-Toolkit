"""Documented multiple-comparison adjustment utilities."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd


class MultiplicityError(ValueError):
    """Raised when a hypothesis family is undefined or invalid."""


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    """Benjamini–Hochberg FDR adjustment, preserving input order and NaNs."""

    p = pd.to_numeric(pd.Series(list(p_values)), errors="coerce").to_numpy(float)
    result = np.full(len(p), np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(p))
    if len(valid) == 0:
        return result
    if np.any((p[valid] < 0) | (p[valid] > 1)):
        raise MultiplicityError("P-values must lie between zero and one")
    order = valid[np.argsort(p[valid], kind="stable")]
    ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def add_bh_fdr(
    frame: pd.DataFrame,
    *,
    p_col: str,
    family_cols: Sequence[str],
    output_col: str = "p_fdr_bh",
) -> pd.DataFrame:
    """Adjust p-values inside explicitly supplied, predefined families.

    Requiring family columns prevents accidental redefinition of a hypothesis
    family. For a single family, callers should first add a constant family ID.
    """

    if not family_cols:
        raise MultiplicityError("At least one explicit hypothesis-family column is required")
    missing = {p_col, *family_cols} - set(frame.columns)
    if missing:
        raise MultiplicityError(f"Missing FDR columns: {sorted(missing)}")
    result = frame.copy()
    result[output_col] = np.nan
    grouped = result.groupby(list(family_cols), sort=False, dropna=False).groups
    for index in grouped.values():
        result.loc[index, output_col] = benjamini_hochberg(result.loc[index, p_col])
    return result
