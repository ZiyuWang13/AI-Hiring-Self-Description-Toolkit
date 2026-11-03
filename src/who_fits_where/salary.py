"""Approved salary construction for the formal 240-occupation analyses.

Inputs `salary_min` and `salary_max` are monthly RMB amounts already standardized
by the research data pipeline. Annual wage is their midpoint multiplied by the
salary-month factor: blank bonus means 12; otherwise use the first numeric value.
Missing or unparsable bounds remain missing and are never imputed.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


def is_missing(value: Any) -> bool:
    """Return True for the missing values used by the approved analysis."""

    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if not isinstance(missing, (np.ndarray, pd.Series)) else False


def clean_text(value: Any) -> str:
    """Match the formal scripts' blank/NaN string normalization."""

    if is_missing(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def parse_salary_number(value: Any) -> float:
    """Parse a numeric monthly salary bound; return NaN when parsing fails."""

    text = clean_text(value)
    if not text:
        return math.nan
    try:
        return float(text)
    except (TypeError, ValueError):
        return math.nan


def parse_salary_months(value: Any) -> float:
    """Return 12 for blank bonus, otherwise the first numeric salary-month value."""

    text = clean_text(value)
    if not text:
        return 12.0
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    return float(match.group(1)) if match else math.nan


def annual_wage(salary_min: Any, salary_max: Any, bonus: Any = None) -> float:
    """Compute annual RMB wage as monthly midpoint × salary months."""

    lower = parse_salary_number(salary_min)
    upper = parse_salary_number(salary_max)
    months = parse_salary_months(bonus)
    if any(math.isnan(value) for value in (lower, upper, months)):
        return math.nan
    return ((lower + upper) / 2.0) * months


def annualize_dataframe(
    frame: pd.DataFrame,
    *,
    salary_min_col: str = "salary_min",
    salary_max_col: str = "salary_max",
    bonus_col: str = "bonus",
) -> pd.DataFrame:
    """Add the exact formal salary-construction columns to a dataframe copy."""

    required = {salary_min_col, salary_max_col, bonus_col}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Missing salary columns: {sorted(missing)}")
    result = frame.copy()
    result["salary_min_numeric"] = pd.to_numeric(result[salary_min_col], errors="coerce")
    result["salary_max_numeric"] = pd.to_numeric(result[salary_max_col], errors="coerce")
    result["salary_months"] = result[bonus_col].map(parse_salary_months)
    observed_bounds = result["salary_min_numeric"].notna() & result["salary_max_numeric"].notna()
    result["mid_salary"] = np.where(
        observed_bounds,
        (result["salary_min_numeric"] + result["salary_max_numeric"]) / 2.0,
        np.nan,
    )
    result["annual_wage"] = np.where(
        pd.Series(result["mid_salary"], index=result.index).notna()
        & result["salary_months"].notna(),
        result["mid_salary"] * result["salary_months"],
        np.nan,
    )
    return result


def observed_mean(values: Iterable[Any]) -> float:
    """Mean over observed numeric wages, preserving all-missing as NaN."""

    series = pd.to_numeric(pd.Series(list(values), dtype="object"), errors="coerce")
    return float(series.mean()) if series.notna().any() else math.nan


def salary_coverage(values: Iterable[Any]) -> float:
    """Fraction of jobs with observed numeric annual wage."""

    series = pd.to_numeric(pd.Series(list(values), dtype="object"), errors="coerce")
    return float(series.notna().mean()) if len(series) else math.nan


def top10_mean_wage(values: Iterable[Any]) -> float:
    """Compute the formal Top-10 mean over observed wages only."""

    wages = list(values)
    if len(wages) != 10:
        raise ValueError(f"Top10 outcome requires 10 jobs, found {len(wages)}")
    return observed_mean(wages)


def pool_mean_wage(values: Iterable[Any]) -> float:
    """Compute the formal 60-job pool mean over observed wages only."""

    wages = list(values)
    if len(wages) != 60:
        raise ValueError(f"PoolValue requires 60 jobs, found {len(wages)}")
    return observed_mean(wages)
