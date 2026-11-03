"""Occupation and recommendation overlap using stable identities."""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from .validation import PROFILE_ORDER, ValidationError, profile_code, validate_stage1_top6


def set_overlap(left: Sequence[Any], right: Sequence[Any]) -> int:
    """Count unique identities shared by two duplicate-free ranked lists."""

    if len(left) != len(set(left)) or len(right) != len(set(right)):
        raise ValidationError("Overlap inputs must not contain duplicate identities")
    return len(set(left) & set(right))


def top6_metrics(structured_top6: Sequence[str], treatment_top6: Sequence[str]) -> dict[str, float | int]:
    """Compute the approved Stage 1 set and rank-stability metrics."""

    structured = validate_stage1_top6(structured_top6)
    treatment = validate_stage1_top6(treatment_top6)
    overlap = set_overlap(structured, treatment)
    top3 = set_overlap(structured[:3], treatment[:3])
    bottom3 = set_overlap(structured[3:], treatment[3:])
    return {
        "top6_overlap_count": overlap,
        "top6_overlap_share": overlap / 6.0,
        "replacement_count": 6 - overlap,
        "any_occupation_changed": int(overlap < 6),
        "exact_top6_set": int(overlap == 6),
        "rank1_agreement": int(structured[0] == treatment[0]),
        "top1_3_overlap_count": top3,
        "top1_3_retention_share": top3 / 3.0,
        "top4_6_overlap_count": bottom3,
        "top4_6_retention_share": bottom3 / 3.0,
    }


def top10_job_overlap(left_original_job_ids: Sequence[str], right_original_job_ids: Sequence[str]) -> int:
    """Count shared Top-10 jobs by original_job_id, never by cross-pool J-ID."""

    if len(left_original_job_ids) != 10 or len(right_original_job_ids) != 10:
        raise ValidationError("Top10 overlap requires two lists of length 10")
    return set_overlap(left_original_job_ids, right_original_job_ids)


def all_profile_pair_overlaps(
    recommendations: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Return all 15 unordered profile-pair overlaps for one source category."""

    normalized = {profile_code(key): tuple(value) for key, value in recommendations.items()}
    if set(normalized) != set(PROFILE_ORDER):
        raise ValidationError("Recommendation mapping must contain all six profiles")
    rows: list[dict[str, Any]] = []
    for profile_a, profile_b in itertools.combinations(PROFILE_ORDER, 2):
        overlap = top10_job_overlap(normalized[profile_a], normalized[profile_b])
        rows.append(
            {
                "profile_a": profile_a,
                "profile_b": profile_b,
                "profile_pair": f"{profile_a} vs {profile_b}",
                "overlap_count": overlap,
                "overlap_share": overlap / 10.0,
                "set_identical": int(set(normalized[profile_a]) == set(normalized[profile_b])),
                "ranked_identical": int(normalized[profile_a] == normalized[profile_b]),
            }
        )
    return pd.DataFrame(rows)
