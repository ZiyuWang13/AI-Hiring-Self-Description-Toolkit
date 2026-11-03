"""Schema-controlled construction of analysis-ready inferential tables."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

from .validation import CONDITIONS


class ResultTableError(ValueError):
    """Raised when an inferential result table violates its schema."""


INFERENCE_COLUMNS = (
    "model",
    "run",
    "condition",
    "analysis_family",
    "test_family",
    "outcome_id",
    "contrast_id",
    "contrast",
    "units",
    "N",
    "estimate",
    "SD",
    "SE",
    "CI95_low",
    "CI95_high",
    "paired_t_p",
    "p_fdr_bh",
    "wilcoxon_p",
    "cohens_dz",
    "bootstrap_CI95_low",
    "bootstrap_CI95_high",
    "bootstrap_p",
    "inference_unit",
    "bootstrap_repetitions",
    "bootstrap_seed",
    "bootstrap_pvalue_rule",
    "methodology_source",
    "methodology_sha256",
    "validation_status",
)

NUMERIC_COLUMNS = {
    "N",
    "estimate",
    "SD",
    "SE",
    "CI95_low",
    "CI95_high",
    "paired_t_p",
    "p_fdr_bh",
    "wilcoxon_p",
    "cohens_dz",
    "bootstrap_CI95_low",
    "bootstrap_CI95_high",
    "bootstrap_p",
    "bootstrap_repetitions",
    "bootstrap_seed",
}


def result_row(
    *,
    metadata: Mapping[str, Any],
    inference: Mapping[str, Any],
    adjusted_p: float = math.nan,
) -> dict[str, Any]:
    """Create one ordered result row without manufacturing missing statistics."""

    row = {column: math.nan if column in NUMERIC_COLUMNS else "" for column in INFERENCE_COLUMNS}
    unknown = (set(metadata) | set(inference)) - set(INFERENCE_COLUMNS) - {"t_stat", "wilcoxon_stat"}
    if unknown:
        raise ResultTableError(f"Unsupported result fields: {sorted(unknown)}")
    for source in (metadata, inference):
        for key, value in source.items():
            if key in row:
                row[key] = value
    row["p_fdr_bh"] = adjusted_p
    return row


def result_table(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Build and validate a stable-column inferential table."""

    frame = pd.DataFrame(list(rows), columns=INFERENCE_COLUMNS)
    validate_result_table(frame)
    return frame


def validate_result_table(frame: pd.DataFrame) -> None:
    """Fail closed on malformed methods, p-values, conditions, or provenance."""

    if tuple(frame.columns) != INFERENCE_COLUMNS:
        raise ResultTableError("Inferential result columns or order differ from the schema")
    if frame.empty:
        return
    required_text = (
        "model",
        "analysis_family",
        "test_family",
        "outcome_id",
        "units",
        "inference_unit",
        "methodology_source",
        "methodology_sha256",
        "validation_status",
    )
    for column in required_text:
        if frame[column].astype(str).str.strip().eq("").any():
            raise ResultTableError(f"Required result metadata is blank: {column}")
    conditions = frame["condition"].astype(str)
    invalid_conditions = ~conditions.isin((*CONDITIONS, "", "NA"))
    if invalid_conditions.any():
        raise ResultTableError("Invalid experimental condition in result table")
    if not frame["methodology_sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
        raise ResultTableError("Methodology SHA-256 is missing or invalid")
    n_values = pd.to_numeric(frame["N"], errors="coerce")
    if n_values.isna().any() or (n_values < 2).any():
        raise ResultTableError("N must be observed and at least two")
    for column in ("paired_t_p", "p_fdr_bh", "wilcoxon_p", "bootstrap_p"):
        values = pd.to_numeric(frame[column], errors="coerce")
        observed = values.dropna()
        if ((observed < 0) | (observed > 1)).any():
            raise ResultTableError(f"Invalid probability in {column}")
    if frame["inference_unit"].ne("source_category").any():
        raise ResultTableError("Formal inferential unit must be source_category")
