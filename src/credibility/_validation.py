"""Input validation utilities for credibility models."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Union

import numpy as np
import polars as pl

if TYPE_CHECKING:
    pass


def _to_polars(data: Union[pl.DataFrame, "pd.DataFrame"]) -> pl.DataFrame:  # type: ignore[name-defined]
    """
    Normalise input to a Polars DataFrame.

    Accepts a Polars DataFrame directly (zero-copy path) or a pandas DataFrame
    (converted via Polars' from_pandas). This is the single bridge point between
    the old pandas-based API and the Polars-native internals.
    """
    if isinstance(data, pl.DataFrame):
        return data
    # Attempt pandas conversion. If pandas is not installed and the caller
    # somehow passes a non-Polars object, the AttributeError from from_pandas
    # will surface a clear message.
    try:
        return pl.from_pandas(data)
    except Exception as exc:
        raise TypeError(
            f"data must be a polars.DataFrame or pandas.DataFrame. "
            f"Got {type(data).__name__}."
        ) from exc


def validate_panel_data(
    data: pl.DataFrame,
    group_col: str,
    period_col: str,
    loss_col: str,
    weight_col: str,
) -> None:
    """
    Check that a Polars DataFrame is suitable for Bühlmann-Straub estimation.

    Raises ValueError for unrecoverable problems, warns for things that will
    be handled automatically (e.g. negative a_hat).

    Parameters
    ----------
    data:
        Panel data — one row per (group, period). Must be a pl.DataFrame.
    group_col, period_col, loss_col, weight_col:
        Column names in `data`.
    """
    required = {group_col, period_col, loss_col, weight_col}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Columns not found in data: {missing}")

    if data.is_empty():
        raise ValueError("data is empty.")

    # No nulls in key columns
    for col in [group_col, period_col, loss_col, weight_col]:
        n_null = data[col].null_count()
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
    loss_vals = data[loss_col].to_numpy()
    if not np.isfinite(loss_vals).all():
        raise ValueError(
            f"Column '{loss_col}' contains non-finite values (inf or nan)."
        )

    # Check minimum period count per group
    periods_per_group = (
        data.group_by(group_col)
        .agg(pl.col(period_col).n_unique().alias("n_periods"))
    )
    single_period_groups = (
        periods_per_group
        .filter(pl.col("n_periods") == 1)[group_col]
        .to_list()
    )
    n_groups = periods_per_group.height

    if len(single_period_groups) == n_groups:
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
    if n_groups < 2:
        raise ValueError(
            "At least 2 groups are required to estimate between-group variance (a). "
            f"Found {n_groups} group."
        )


def check_duplicate_periods(
    data: pl.DataFrame, group_col: str, period_col: str
) -> None:
    """Warn if any (group, period) pair appears more than once."""
    n_dupes = data.height - data.unique(subset=[group_col, period_col]).height
    if n_dupes > 0:
        warnings.warn(
            f"{n_dupes} duplicate (group, period) row(s) found. "
            "Each (group, period) combination should be a single row. "
            "Consider aggregating before fitting.",
            stacklevel=4,
        )
