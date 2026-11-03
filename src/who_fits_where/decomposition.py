"""Descriptive PoolValue–selection accounting.

This module implements an arithmetic decomposition, not causal mediation:
Final Top10 wage = 60-job PoolValue + selection premium.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .contrasts import compute_contrast
from .salary import pool_mean_wage, top10_mean_wage
from .validation import PROFILE_ORDER, ValidationError, profile_code


@dataclass(frozen=True)
class WageDecomposition:
    """One profile's descriptive wage accounting components."""

    pool_value: float
    selection_premium: float
    final_wage: float

    @property
    def reconstruction_error(self) -> float:
        return self.final_wage - self.pool_value - self.selection_premium


@dataclass(frozen=True)
class ContrastDecomposition:
    """One profile contrast decomposed into pool and selection components."""

    contrast_id: str
    pool_value_component: float
    selection_premium_component: float
    final_wage_effect: float

    @property
    def reconstruction_error(self) -> float:
        return self.final_wage_effect - self.pool_value_component - self.selection_premium_component


def decompose_wage(pool_value: Any, final_wage: Any) -> WageDecomposition:
    """Construct profile-level accounting components while preserving missingness."""

    if pd.isna(pool_value) or pd.isna(final_wage):
        return WageDecomposition(math.nan, math.nan, math.nan)
    pool = float(pool_value)
    final = float(final_wage)
    return WageDecomposition(pool, final - pool, final)


def decompose_job_wages(pool_wages: list[Any], top10_wages: list[Any]) -> WageDecomposition:
    """Build PoolValue and final wage from 60 and 10 observed/missing job wages."""

    return decompose_wage(pool_mean_wage(pool_wages), top10_mean_wage(top10_wages))


def decompose_contrast(
    pool_values: Mapping[str, Any],
    final_values: Mapping[str, Any],
    contrast_id: str,
) -> ContrastDecomposition:
    """Decompose a formally defined within-category profile contrast."""

    normalized_pool = {profile_code(key): value for key, value in pool_values.items()}
    normalized_final = {profile_code(key): value for key, value in final_values.items()}
    if set(normalized_pool) != set(PROFILE_ORDER) or set(normalized_final) != set(PROFILE_ORDER):
        raise ValidationError("Decomposition requires all six profile values")
    selection = {
        profile: (
            math.nan
            if pd.isna(normalized_pool[profile]) or pd.isna(normalized_final[profile])
            else float(normalized_final[profile]) - float(normalized_pool[profile])
        )
        for profile in PROFILE_ORDER
    }
    pool_component = compute_contrast(normalized_pool, contrast_id)
    selection_component = compute_contrast(selection, contrast_id)
    final_effect = compute_contrast(normalized_final, contrast_id)
    return ContrastDecomposition(
        contrast_id,
        pool_component,
        selection_component,
        final_effect,
    )


def assert_reconstruction(value: WageDecomposition | ContrastDecomposition, *, atol: float = 1e-8) -> None:
    """Fail when the descriptive accounting identity does not reconstruct."""

    error = value.reconstruction_error
    if math.isnan(error):
        return
    if not math.isclose(error, 0.0, abs_tol=atol, rel_tol=0.0):
        raise ValidationError(f"Decomposition reconstruction error {error} exceeds {atol}")


def gender_interaction_decomposition(
    pool_values: Mapping[str, Any],
    final_values: Mapping[str, Any],
    *,
    signal: str,
) -> ContrastDecomposition:
    """Decompose the fixed Hardworking or Talent × Gender interaction."""

    normalized = signal.strip().lower()
    if normalized == "hardworking":
        contrast_id = "hardworking_gender_interaction"
    elif normalized in {"talent", "talented"}:
        contrast_id = "talent_gender_interaction"
    else:
        raise ValueError("signal must be Hardworking or Talent")
    result = decompose_contrast(pool_values, final_values, contrast_id)
    assert_reconstruction(result)
    return result
