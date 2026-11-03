"""Profile contrasts computed within matched source categories."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from .validation import PROFILE_ORDER, ValidationError, profile_code, source_category_id


CONTRASTS: dict[str, dict[str, float]] = {
    "MB-FB": {"MB": 1.0, "FB": -1.0},
    "MH-FH": {"MH": 1.0, "FH": -1.0},
    "MT-FT": {"MT": 1.0, "FT": -1.0},
    "MH-MB": {"MH": 1.0, "MB": -1.0},
    "MT-MB": {"MT": 1.0, "MB": -1.0},
    "FH-FB": {"FH": 1.0, "FB": -1.0},
    "FT-FB": {"FT": 1.0, "FB": -1.0},
    "hardworking_gender_interaction": {"FH": 1.0, "FB": -1.0, "MH": -1.0, "MB": 1.0},
    "talent_gender_interaction": {"FT": 1.0, "FB": -1.0, "MT": -1.0, "MB": 1.0},
}

CONTRAST_LABELS = {
    "MB-FB": "MB − FB",
    "MH-FH": "MH − FH",
    "MT-FT": "MT − FT",
    "MH-MB": "MH − MB",
    "MT-MB": "MT − MB",
    "FH-FB": "FH − FB",
    "FT-FB": "FT − FB",
    "hardworking_gender_interaction": "Hardworking × Gender",
    "talent_gender_interaction": "Talent × Gender",
}

PRIMARY_REQUESTED_CONTRASTS = (
    "MB-FB",
    "MH-MB",
    "MT-MB",
    "FH-FB",
    "FT-FB",
    "hardworking_gender_interaction",
    "talent_gender_interaction",
)


def linear_contrast(values: Mapping[str, Any], coefficients: Mapping[str, float]) -> float:
    """Evaluate a profile contrast; missing inputs propagate as NaN."""

    normalized = {profile_code(key): value for key, value in values.items()}
    missing_profiles = set(coefficients) - set(normalized)
    if missing_profiles:
        raise ValidationError(f"Contrast missing profiles: {sorted(missing_profiles)}")
    total = 0.0
    for profile, coefficient in coefficients.items():
        value = normalized[profile]
        if pd.isna(value):
            return math.nan
        total += float(value) * coefficient
    return total


def compute_contrast(values: Mapping[str, Any], contrast_id: str) -> float:
    """Compute one formally defined profile contrast."""

    if contrast_id not in CONTRASTS:
        raise KeyError(f"Unknown contrast: {contrast_id}")
    return linear_contrast(values, CONTRASTS[contrast_id])


def category_level_contrasts(
    frame: pd.DataFrame,
    *,
    value_col: str,
    category_col: str = "source_category",
    profile_col: str = "profile",
    contrast_ids: Sequence[str] = PRIMARY_REQUESTED_CONTRASTS,
) -> pd.DataFrame:
    """Compute contrasts inside each source category before any aggregation."""

    required = {category_col, profile_col, value_col}
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise KeyError(f"Missing contrast columns: {sorted(missing_columns)}")
    data = frame[[category_col, profile_col, value_col]].copy()
    data[category_col] = data[category_col].map(source_category_id)
    data[profile_col] = data[profile_col].map(profile_code)
    if data.duplicated([category_col, profile_col]).any():
        raise ValidationError("Duplicate source-category × profile outcomes")
    wide = data.pivot(index=category_col, columns=profile_col, values=value_col)
    if set(wide.columns) != set(PROFILE_ORDER):
        raise ValidationError("Each source category must contain all six profiles")
    rows: list[dict[str, Any]] = []
    for category, values in wide.iterrows():
        mapping = values.to_dict()
        for contrast_id in contrast_ids:
            rows.append(
                {
                    "source_category": category,
                    "contrast_id": contrast_id,
                    "contrast": CONTRAST_LABELS[contrast_id],
                    "difference": compute_contrast(mapping, contrast_id),
                }
            )
    return pd.DataFrame(rows)


def paired_complete_cases(
    left: pd.Series,
    right: pd.Series,
) -> pd.DataFrame:
    """Align two category-indexed outcomes and retain only observed pairs."""

    aligned = pd.concat([left.rename("left"), right.rename("right")], axis=1)
    return aligned.dropna(subset=["left", "right"]).assign(
        difference=lambda data: data["left"] - data["right"]
    )
