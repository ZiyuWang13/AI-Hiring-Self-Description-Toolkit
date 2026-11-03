"""Pure validation helpers for the completed two-stage experiments.

The functions fail closed and have no file-system, API, or output side effects.
They encode the validated 240-occupation design without loading research data.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


PROFILE_ORDER = ("MB", "MH", "MT", "FB", "FH", "FT")
CONDITIONS = ("00", "01", "10", "11")
EXPECTED_SOURCE_CATEGORIES = tuple(f"{number:03d}" for number in range(1, 106))
VALID_OCCUPATION_IDS = frozenset(f"O{number:03d}" for number in range(1, 241))
VALID_JOB_IDS = frozenset(f"J{number:02d}" for number in range(1, 61))

PROFILE_NAME_TO_CODE = {
    "Male_Baseline": "MB",
    "Male_Hardworking": "MH",
    "Male_Talented": "MT",
    "Female_Baseline": "FB",
    "Female_Hardworking": "FH",
    "Female_Talented": "FT",
}

CONDITION_DEFINITIONS = {
    "00": {"treatment_in_stage1": False, "structured_resume_in_stage2": False},
    "01": {"treatment_in_stage1": False, "structured_resume_in_stage2": True},
    "10": {"treatment_in_stage1": True, "structured_resume_in_stage2": False},
    "11": {"treatment_in_stage1": True, "structured_resume_in_stage2": True},
}


class ValidationError(ValueError):
    """Raised when a fixed experimental invariant is violated."""


@dataclass(frozen=True)
class RecordAudit:
    """Missing and duplicate key counts for an in-memory record collection."""

    n_records: int
    missing_required_values: int
    duplicate_keys: int

    @property
    def passed(self) -> bool:
        return self.missing_required_values == 0 and self.duplicate_keys == 0


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def source_category_id(value: Any) -> str:
    """Return the fixed three-digit source-category ID used by formal analyses."""

    if isinstance(value, int) and 1 <= value <= 999:
        return f"{value:03d}"
    match = re.match(r"^(\d{3})(?:_|$)", _clean(value))
    if not match:
        raise ValidationError(f"Invalid source category: {value!r}")
    return match.group(1)


def profile_code(value: Any) -> str:
    """Normalize a short or full experimental profile label."""

    text = _clean(value)
    code = PROFILE_NAME_TO_CODE.get(text, text)
    if code not in PROFILE_ORDER:
        raise ValidationError(f"Invalid profile: {value!r}")
    return code


def validate_profiles(values: Iterable[Any], *, require_fixed_order: bool = False) -> tuple[str, ...]:
    """Validate exactly one occurrence of each of the six profiles."""

    codes = tuple(profile_code(value) for value in values)
    if len(codes) != 6 or set(codes) != set(PROFILE_ORDER):
        raise ValidationError("Expected exactly the six profiles MB/MH/MT/FB/FH/FT")
    if require_fixed_order and codes != PROFILE_ORDER:
        raise ValidationError(f"Profiles are not in fixed order: {codes}")
    return codes


def validate_condition(value: Any) -> str:
    """Validate and normalize a two-character architecture condition."""

    text = _clean(value)
    if text.endswith(".0"):
        text = text[:-2]
    text = text.zfill(2)
    if text not in CONDITIONS:
        raise ValidationError(f"Invalid condition: {value!r}")
    return text


def _validate_ranked_ids(
    values: Sequence[Any],
    *,
    expected_n: int,
    universe: frozenset[str] | set[str],
    label: str,
) -> tuple[str, ...]:
    cleaned = tuple(_clean(value) for value in values)
    if len(cleaned) != expected_n:
        raise ValidationError(f"{label} must contain exactly {expected_n} IDs")
    if len(set(cleaned)) != expected_n:
        raise ValidationError(f"{label} contains duplicate IDs")
    invalid = sorted(set(cleaned) - set(universe))
    if invalid:
        raise ValidationError(f"{label} contains invalid IDs: {invalid}")
    return cleaned


def validate_stage1_top6(
    values: Sequence[Any],
    *,
    occupation_universe: Iterable[str] = VALID_OCCUPATION_IDS,
) -> tuple[str, ...]:
    """Validate a ranked Top-6 against the frozen occupation universe."""

    return _validate_ranked_ids(
        values,
        expected_n=6,
        universe=set(occupation_universe),
        label="Stage 1 Top6",
    )


def validate_stage2_top10(
    values: Sequence[Any],
    *,
    job_universe: Iterable[str] = VALID_JOB_IDS,
) -> tuple[str, ...]:
    """Validate a ranked Top-10 against the relevant frozen J-ID universe."""

    return _validate_ranked_ids(
        values,
        expected_n=10,
        universe=set(job_universe),
        label="Stage 2 Top10",
    )


def validate_job_pool(
    records: Iterable[Mapping[str, Any]],
    *,
    stage2_job_key: str = "stage2_job_id",
    original_job_key: str = "original_job_id",
    occupation_key: str = "occupation_id",
) -> tuple[Mapping[str, Any], ...]:
    """Validate a 6-occupation × 10-job pool with 60 unique job identities."""

    rows = tuple(records)
    if len(rows) != 60:
        raise ValidationError(f"Job pool must have exactly 60 rows, found {len(rows)}")
    j_ids = tuple(_clean(row.get(stage2_job_key)) for row in rows)
    originals = tuple(_clean(row.get(original_job_key)) for row in rows)
    occupations = tuple(_clean(row.get(occupation_key)) for row in rows)
    if set(j_ids) != set(VALID_JOB_IDS) or len(set(j_ids)) != 60:
        raise ValidationError("Job pool must contain J01-J60 exactly once")
    if "" in originals or len(set(originals)) != 60:
        raise ValidationError("Job pool must contain 60 nonblank unique original_job_id values")
    if "" in occupations or len(set(occupations)) != 6:
        raise ValidationError("Job pool must contain exactly six occupations")
    invalid_occupations = sorted(set(occupations) - set(VALID_OCCUPATION_IDS))
    if invalid_occupations:
        raise ValidationError(f"Job pool contains invalid occupation IDs: {invalid_occupations}")
    counts = Counter(occupations)
    if set(counts.values()) != {10}:
        raise ValidationError("Each pool occupation must contribute exactly 10 jobs")
    return rows


def map_original_job_ids(
    selected_stage2_job_ids: Sequence[Any],
    pool_records: Iterable[Mapping[str, Any]],
    *,
    stage2_job_key: str = "stage2_job_id",
    original_job_key: str = "original_job_id",
) -> tuple[str, ...]:
    """Recover ordered original job identities for a valid Stage 2 Top-10."""

    rows = validate_job_pool(pool_records)
    selected = validate_stage2_top10(selected_stage2_job_ids)
    lookup = {_clean(row[stage2_job_key]): _clean(row[original_job_key]) for row in rows}
    originals = tuple(lookup[job_id] for job_id in selected)
    if len(set(originals)) != 10:
        raise ValidationError("Selected jobs do not map to 10 unique original_job_id values")
    return originals


def detect_missing_and_duplicates(
    records: Iterable[Mapping[str, Any]],
    *,
    key_fields: Sequence[str],
    required_fields: Sequence[str] = (),
) -> RecordAudit:
    """Count missing required values and duplicate composite keys."""

    rows = tuple(records)
    keys: list[tuple[str, ...]] = []
    missing = 0
    for row in rows:
        missing += sum(not _clean(row.get(field)) for field in required_fields)
        keys.append(tuple(_clean(row.get(field)) for field in key_fields))
    duplicates = len(keys) - len(set(keys))
    return RecordAudit(len(rows), missing, duplicates)


def validate_source_category_completeness(
    records: Iterable[Mapping[str, Any]],
    *,
    category_key: str = "source_category",
    profile_key: str = "profile",
    expected_categories: Sequence[str] = EXPECTED_SOURCE_CATEGORIES,
) -> None:
    """Require one record for every expected source-category × profile key."""

    rows = tuple(records)
    actual = [(source_category_id(row.get(category_key)), profile_code(row.get(profile_key))) for row in rows]
    expected = {(category, profile) for category in expected_categories for profile in PROFILE_ORDER}
    if len(actual) != len(expected):
        raise ValidationError(f"Expected {len(expected)} category-profile records, found {len(actual)}")
    if len(set(actual)) != len(actual):
        raise ValidationError("Duplicate source-category × profile records detected")
    missing = expected - set(actual)
    unexpected = set(actual) - expected
    if missing or unexpected:
        raise ValidationError(
            f"Category-profile coverage mismatch: missing={len(missing)}, unexpected={len(unexpected)}"
        )
