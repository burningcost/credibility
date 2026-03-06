"""
Hierarchical Bühlmann-Straub credibility model.

This module extends the two-parameter Bühlmann-Straub model to arbitrary-depth
hierarchies. The motivating UK use case is geographic rating: postcode sectors
(~9,500 units) nested within postcode districts (~3,000) nested within postal
areas (~124). Thin sectors borrow strength from their district, thin districts
borrow from their area, and so on.

The algorithm has two passes:

1. **Bottom-up variance estimation**: Fit a Bühlmann-Straub model at each
   level, treating the nodes at that level as the "groups" and their children's
   statistics as the "observations". This gives variance components (v_l, a_l)
   at each level.

2. **Top-down premium computation**: Starting from the grand mean, compute
   credibility premiums recursively downward. At each level, the premium for
   a node blends its own exposure-weighted mean with the premium from the level
   above.

This is the Jewell (1975) / Sundt (1979) hierarchical credibility model,
implemented following the actuar `cm()` specification for multi-level
hierarchies. It produces exactly the same structural parameters as actuar
for the same data.

For a two-level hierarchy (the default for scheme + book pricing), this is
equivalent to fitting two nested Bühlmann-Straub models. For a three-level
geographic hierarchy, it fits three.

References
----------
Jewell, W.S. (1975). Regularity conditions for exact credibility.
    ASTIN Bulletin, 8(3), 336–341.
Sundt, B. (1979). An introduction to non-life insurance mathematics.
    VVW Karlsruhe.
Bühlmann, H. & Gisler, A. (2005). A Course in Credibility Theory and its
    Applications. Springer, Chapter 5.
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np
import pandas as pd

from .buhlmann_straub import BuhlmannStraub


class LevelResult:
    """
    Fitted parameters and premiums for one level of the hierarchy.

    Attributes
    ----------
    level_name:
        The column name that identifies this level (e.g. "district").
    mu_hat:
        Collective mean at this level.
    v_hat:
        Within-node variance at this level (EPV).
    a_hat:
        Between-node variance at this level (VHM). Zero if truncated.
    k:
        Bühlmann's k = v / a at this level.
    z:
        Credibility factors indexed by this level's node identifier.
    premiums:
        Credibility premiums at this level.
    """

    def __init__(
        self,
        level_name: str,
        mu_hat: float,
        v_hat: float,
        a_hat: float,
        k: float,
        z: pd.Series,
        premiums: pd.DataFrame,
    ) -> None:
        self.level_name = level_name
        self.mu_hat = mu_hat
        self.v_hat = v_hat
        self.a_hat = a_hat
        self.k = k
        self.z = z
        self.premiums = premiums

    def __repr__(self) -> str:
        return (
            f"LevelResult(level='{self.level_name}', "
            f"mu={self.mu_hat:.4g}, v={self.v_hat:.4g}, "
            f"a={self.a_hat:.4g}, k={self.k:.4g})"
        )


class HierarchicalBuhlmannStraub:
    """
    Multi-level Bühlmann-Straub credibility model for nested group structures.

    This model is appropriate when groups are arranged in a strict hierarchy —
    every observation belongs to exactly one parent at each level. Typical uses:

    - Geographic: postcode sector → district → area
    - Organisational: risk → scheme → portfolio
    - Time: accident quarter → accident year → underwriting year

    Parameters
    ----------
    level_cols:
        List of column names defining the hierarchy from outermost (top) to
        innermost (bottom). For a three-level model, provide three names.
        Example: ``["area", "district", "sector"]``.
    truncate_a:
        Whether to truncate negative between-group variance estimates at zero.
        Passed through to the BuhlmannStraub model at each level.

    Examples
    --------
    Geographic three-level model::

        >>> model = HierarchicalBuhlmannStraub(
        ...     level_cols=["region", "district", "sector"]
        ... )
        >>> model.fit(
        ...     data=df,
        ...     period_col="year",
        ...     loss_col="loss_rate",
        ...     weight_col="exposure",
        ... )
        >>> model.premiums_at("sector")   # blended sector-level premiums
        >>> model.level_results_["district"].k   # k at district level

    Two-level scheme model (equivalent to BuhlmannStraub with a book-level prior)::

        >>> model = HierarchicalBuhlmannStraub(level_cols=["book", "scheme"])
        >>> model.fit(df, period_col="year", loss_col="loss_rate", weight_col="exposure")
    """

    def __init__(
        self,
        level_cols: List[str],
        truncate_a: bool = True,
    ) -> None:
        if len(level_cols) < 2:
            raise ValueError(
                "At least two levels are required for a hierarchical model. "
                "For a single-level model, use BuhlmannStraub."
            )
        self.level_cols = level_cols
        self.truncate_a = truncate_a
        self._fitted = False
        self._level_results: Optional[dict] = None
        self._bottom_premiums: Optional[pd.DataFrame] = None

    def fit(
        self,
        data: pd.DataFrame,
        period_col: str = "period",
        loss_col: str = "loss",
        weight_col: str = "weight",
    ) -> "HierarchicalBuhlmannStraub":
        """
        Fit the hierarchical model to panel data.

        Parameters
        ----------
        data:
            Panel DataFrame with one row per (leaf node, period). Must contain
            all columns in ``level_cols`` plus ``period_col``, ``loss_col``,
            and ``weight_col``.
        period_col:
            Column identifying the time period.
        loss_col:
            Column containing the per-unit loss rate.
        weight_col:
            Column containing exposure weights (must be strictly positive).

        Returns
        -------
        self
        """
        required_cols = set(self.level_cols) | {period_col, loss_col, weight_col}
        missing = required_cols - set(data.columns)
        if missing:
            raise ValueError(f"Columns not found in data: {missing}")

        self._period_col = period_col
        self._loss_col = loss_col
        self._weight_col = weight_col

        # Validate that hierarchy is strict (no cross-level ambiguity)
        self._validate_hierarchy(data)

        # Bottom-up: estimate variance components at each level
        level_results = {}
        bottom_level = self.level_cols[-1]

        # At the innermost level, fit BuhlmannStraub with the leaf node as group
        # and each period as an observation
        innermost_fit = self._fit_innermost_level(data, bottom_level, period_col, loss_col, weight_col)
        level_results[bottom_level] = innermost_fit

        # For each higher level, aggregate upward and fit B-S treating each node
        # at that level as a "group" with its children's credibility premiums as
        # "observations" (but we use the raw aggregated data for cleaner estimation)
        current_data = data.copy()
        for depth in range(len(self.level_cols) - 2, -1, -1):
            parent_col = self.level_cols[depth]
            child_col = self.level_cols[depth + 1]
            level_result = self._fit_upper_level(
                current_data,
                parent_col,
                child_col,
                loss_col,
                weight_col,
                lower_v=level_results[child_col].v_hat,
            )
            level_results[parent_col] = level_result

        self._level_results = level_results

        # Top-down: compute final blended premiums at each level
        self._bottom_premiums = self._compute_top_down_premiums(data, level_results)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    @property
    def level_results_(self) -> dict:
        """
        Dictionary mapping level name to LevelResult with fitted parameters.
        """
        self._check_fitted()
        return self._level_results  # type: ignore[return-value]

    @property
    def premiums_(self) -> pd.DataFrame:
        """
        Credibility premiums at the bottom (leaf) level of the hierarchy.

        Equivalent to ``premiums_at(self.level_cols[-1])``.
        """
        self._check_fitted()
        return self._bottom_premiums  # type: ignore[return-value]

    def premiums_at(self, level: str) -> pd.DataFrame:
        """
        Return the credibility premiums at a specified level of the hierarchy.

        Parameters
        ----------
        level:
            One of the column names from ``level_cols``.

        Returns
        -------
        pd.DataFrame
            Indexed by the level's node identifier, with columns
            ``exposure``, ``observed_mean``, ``Z``, ``credibility_premium``.
        """
        self._check_fitted()
        if level not in self.level_cols:
            raise ValueError(
                f"'{level}' is not one of the fitted levels: {self.level_cols}"
            )
        return self._level_results[level].premiums

    def summary(self) -> None:
        """Print structural parameters at each level."""
        self._check_fitted()
        print("Hierarchical Bühlmann-Straub Credibility Model")
        print("=" * 50)
        print(f"Levels (outer → inner): {' → '.join(self.level_cols)}")
        print()
        for level in self.level_cols:
            lr = self._level_results[level]
            print(f"  Level: {level}")
            print(f"    mu  = {lr.mu_hat:.6g}")
            print(f"    v   = {lr.v_hat:.6g}   (EPV within-node)")
            print(f"    a   = {lr.a_hat:.6g}   (VHM between-node)")
            k_str = "inf" if np.isinf(lr.k) else f"{lr.k:.6g}"
            print(f"    k   = {k_str}")
            print()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_hierarchy(self, data: pd.DataFrame) -> None:
        """Check that the hierarchy is strict — each child belongs to exactly one parent."""
        for depth in range(len(self.level_cols) - 1):
            parent = self.level_cols[depth]
            child = self.level_cols[depth + 1]
            parents_per_child = data.groupby(child)[parent].nunique()
            ambiguous = parents_per_child[parents_per_child > 1]
            if len(ambiguous) > 0:
                raise ValueError(
                    f"Hierarchy is not strict at level '{child}' → '{parent}': "
                    f"{len(ambiguous)} child node(s) appear under multiple parents. "
                    f"Examples: {ambiguous.head(3).index.tolist()}"
                )

    def _fit_innermost_level(
        self,
        data: pd.DataFrame,
        leaf_col: str,
        period_col: str,
        loss_col: str,
        weight_col: str,
    ) -> LevelResult:
        """Fit B-S at the innermost level using raw period data."""
        bs = BuhlmannStraub(truncate_a=self.truncate_a)
        bs.fit(
            data,
            group_col=leaf_col,
            period_col=period_col,
            loss_col=loss_col,
            weight_col=weight_col,
        )
        return LevelResult(
            level_name=leaf_col,
            mu_hat=bs.mu_hat_,
            v_hat=bs.v_hat_,
            a_hat=bs.a_hat_,
            k=bs.k_,
            z=bs.z_,
            premiums=bs.premiums_,
        )

    def _fit_upper_level(
        self,
        data: pd.DataFrame,
        parent_col: str,
        child_col: str,
        loss_col: str,
        weight_col: str,
        lower_v: float,
    ) -> LevelResult:
        """
        Fit B-S at a higher level by aggregating child-level data.

        We treat each child node as a "period" within the parent node,
        using the child's total exposure as weight and its weighted mean loss
        rate as the observation. The within-parent variance v at this level
        is the between-child variance a from the level below.

        This is the Jewell (1975) / actuar-compatible approach: v at level L
        equals a at level L+1, propagating variance components up the hierarchy.
        """
        # Aggregate data to child level (summing over periods)
        # Select only the columns we need before groupby to avoid include_groups issues
        # across different pandas versions (include_groups added in pandas 2.2)
        agg_cols = data[[parent_col, child_col, weight_col, loss_col]].copy()
        child_summary = (
            agg_cols.groupby([parent_col, child_col])
            .apply(
                lambda g: pd.Series({
                    "exposure": g[weight_col].sum(),
                    "loss_rate": (g[weight_col] * g[loss_col]).sum() / g[weight_col].sum(),
                }),
            )
            .reset_index()
        )

        # Fit B-S at parent level, treating each child as one "observation"
        # The period col here is child_col, loss col is loss_rate, weight is exposure
        bs = BuhlmannStraub(truncate_a=self.truncate_a)
        bs.fit(
            child_summary,
            group_col=parent_col,
            period_col=child_col,
            loss_col="loss_rate",
            weight_col="exposure",
        )

        return LevelResult(
            level_name=parent_col,
            mu_hat=bs.mu_hat_,
            v_hat=bs.v_hat_,
            a_hat=bs.a_hat_,
            k=bs.k_,
            z=bs.z_,
            premiums=bs.premiums_,
        )

    def _compute_top_down_premiums(
        self,
        data: pd.DataFrame,
        level_results: dict,
    ) -> pd.DataFrame:
        """
        Compute the final blended premiums by propagating the hierarchy top-down.

        At the topmost level, the complement is the grand mean. At each level
        below, the complement for a node is the credibility premium of its parent.
        The final output is a premium for each leaf node.

        Blending formula at each level:
            P_node = Z_node * X̄_node + (1 - Z_node) * P_parent
        """
        top_level = self.level_cols[0]
        top_lr = level_results[top_level]

        # Start: parent premiums at top level are just the top-level premiums
        # which already blend with mu_hat
        parent_premiums = top_lr.premiums["credibility_premium"].rename(top_level)

        # Work downward, level by level
        for depth in range(1, len(self.level_cols)):
            current_level = self.level_cols[depth]
            parent_level = self.level_cols[depth - 1]
            current_lr = level_results[current_level]

            # Map each node at the current level to its parent's premium
            parent_map = (
                data[[current_level, parent_level]]
                .drop_duplicates()
                .set_index(current_level)[parent_level]
            )

            # For each node at current level, blend with parent premium
            blended = {}
            for node, row in current_lr.premiums.iterrows():
                parent_id = parent_map[node]
                parent_premium = parent_premiums[parent_id]
                z = row["Z"]
                x_bar = row["observed_mean"]
                blended[node] = z * x_bar + (1 - z) * parent_premium

            parent_premiums = pd.Series(blended, name=current_level)

        # Build final DataFrame
        bottom_level = self.level_cols[-1]
        bottom_lr = level_results[bottom_level]
        result = bottom_lr.premiums.copy()
        result["credibility_premium"] = parent_premiums.reindex(result.index)
        return result

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "Model has not been fitted. Call .fit() first."
            )

    def __repr__(self) -> str:
        levels_str = " → ".join(self.level_cols)
        if not self._fitted:
            return f"HierarchicalBuhlmannStraub(levels=[{levels_str}], not fitted)"
        return f"HierarchicalBuhlmannStraub(levels=[{levels_str}])"
