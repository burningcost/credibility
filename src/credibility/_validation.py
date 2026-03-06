"""Input validation utilities for credibility models."""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd


def validate_panel_data(
    data: pd.DataFrame,
    group_col: str,
    period_col: str,
    loss_col: str,
    weight_col: str,
) -> None:
    """
    Check that a DataFrame is suitable for Bühlmann-Straub estimation.

    Raises ValueError for unrecoverable problems, warns for things that will
    be handled automatically (e.g. negative a_hat).

    Parameters
    ----------
    data:
        Panel data — one row per (group, period).
    group_col, period_col, loss_col, weight_col:
        Column names in `data`.
    """
    required = {group_col, period_col, loss_col, weight_col}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Columns not found in data: {missing}")

    if data.empty:
        raise ValueError("data is empty.")

    # No NaNs in key columns
    for col in [group_col, period_col, loss_col, weight_col]:
        n_null = data[col].isna().sum()
        if n_null > 0:
            raise ValueError(
                f"Column '{col}' contains {n_null} null value(s). "
                "Remove or impute before fitting."
            )

    # Weights must be strictly positive
    if (data[weight_col] <= 0).any():
        raise ValueError(
            f"Column '{weight_col}' contains non-positive values. "
            "Weights must be strictly positive exposure measures."
        )

    # Loss values must be finite
    if not np.isfinite(data[loss_col]).all():
        raise ValueError(
            f"Column '{loss_col}' contains non-finite values (inf or nan)."
        )

    # Check minimum period count per group
    periods_per_group = data.groupby(group_col)[period_col].nunique()
    single_period_groups = periods_per_group[periods_per_group == 1].index.tolist()
    if len(single_period_groups) == len(periods_per_group):
        raise ValueError(
            "Every group has exactly one period. "
            "The within-group variance v cannot be estimated without at least "
            "two periods for at least one group. "
            "If you only have one period of data, consider supplying v directly "
            "or using a Bayesian approach."
        )

    if single_period_groups:
        warnings.warn(
            f"{len(single_period_groups)} group(s) have only one period and "
            "contribute nothing to the v_hat estimate: "
            f"{single_period_groups[:5]}{'...' if len(single_period_groups) > 5 else ''}. "
            "These groups still receive credibility premiums, but their single-period "
            "observations carry no weight in estimating within-group variance.",
            stacklevel=4,
        )

    # Need at least 2 groups for a_hat estimation
    n_groups = data[group_col].nunique()
    if n_groups < 2:
        raise ValueError(
            "At least 2 groups are required to estimate between-group variance (a). "
            f"Found {n_groups} group."
        )


def check_duplicate_periods(
    data: pd.DataFrame, group_col: str, period_col: str
) -> None:
    """Warn if any (group, period) pair appears more than once."""
    dupes = data.duplicated(subset=[group_col, period_col]).sum()
    if dupes > 0:
        warnings.warn(
            f"{dupes} duplicate (group, period) row(s) found. "
            "Each (group, period) combination should be a single row. "
            "Consider aggregating before fitting.",
            stacklevel=4,
        )
