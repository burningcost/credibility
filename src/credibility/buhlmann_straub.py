"""
Bühlmann-Straub credibility model.

This module implements the classic non-parametric Bühlmann-Straub estimator
(Bühlmann & Straub, 1970) with unbiased structural parameter estimation as
described in Bühlmann & Gisler (2005).

The model is the actuarial standard for blending a group's own loss experience
with the collective portfolio mean, weighted by exposure. It is the correct
tool when groups (schemes, territories, NCD classes) have different volumes
of data and you want to avoid both over-crediting thin experience and
under-crediting credible books.

Mathematical background
-----------------------
Each group i has observations X_{ij} (loss rate in period j) with exposure
weights w_{ij}. The model assumes:

    E[X_{ij} | theta_i] = mu(theta_i)
    Var[X_{ij} | theta_i] = sigma²(theta_i) / w_{ij}

where theta_i is a latent risk parameter drawn from the portfolio distribution.

Three structural parameters govern the blend:

    mu  = E[mu(theta)]       collective mean loss rate
    v   = E[sigma²(theta)]   EPV: expected process variance (within-group noise)
    a   = Var[mu(theta)]     VHM: variance of hypothetical means (between-group signal)

The credibility factor and premium for group i are:

    K   = v / a              Bühlmann's k (noise-to-signal ratio)
    Z_i = w_i / (w_i + K)   credibility factor in [0, 1]
    P_i = Z_i * X̄_i + (1 - Z_i) * mu_hat

A large K means the portfolio is noisy and homogeneous — trust the collective.
A small K means groups genuinely differ — trust their own experience.

References
----------
Bühlmann, H. & Straub, E. (1970). Glaubwürdigkeit für Schadensätze.
    Mitteilungen VSVM, 70, 111–133.
Bühlmann, H. & Gisler, A. (2005). A Course in Credibility Theory and its
    Applications. Springer.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from ._validation import check_duplicate_periods, validate_panel_data


class BuhlmannStraub:
    """
    Bühlmann-Straub credibility model with non-parametric structural parameter
    estimation.

    Fit the model to a panel dataset — one row per (group, period) — where
    each row has a loss rate and an exposure weight. The model estimates the
    structural parameters v (within-group variance) and a (between-group
    variance) from the data itself, then computes a credibility factor Z_i
    for each group.

    The result is a credibility premium for each group that blends its own
    weighted-average loss rate with the portfolio mean, with the blend governed
    entirely by the ratio K = v/a and the group's total exposure.

    Parameters
    ----------
    truncate_a : bool, default True
        If True (the default), truncate a_hat at zero when the between-group
        variance estimate is negative. This sets K → ∞ and Z_i → 0 for all
        groups, so every group gets the collective mean. If False, a negative
        a_hat raises an error instead.

    Examples
    --------
    >>> import pandas as pd
    >>> from credibility import BuhlmannStraub
    >>>
    >>> df = pd.DataFrame({
    ...     "scheme": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
    ...     "year":   [2021, 2022, 2023] * 3,
    ...     "loss_rate": [0.55, 0.60, 0.58, 0.80, 0.75, 0.82, 0.40, 0.42, 0.38],
    ...     "exposure": [1000, 1200, 1100, 300, 350, 320, 5000, 4800, 5200],
    ... })
    >>> bs = BuhlmannStraub()
    >>> bs.fit(df, group_col="scheme", period_col="year",
    ...        loss_col="loss_rate", weight_col="exposure")
    >>> bs.summary()
    """

    def __init__(self, truncate_a: bool = True) -> None:
        self.truncate_a = truncate_a

        # Fitted attributes — set by .fit()
        self._mu_hat: Optional[float] = None
        self._v_hat: Optional[float] = None
        self._a_hat: Optional[float] = None
        self._k: Optional[float] = None
        self._z: Optional[pd.Series] = None
        self._premiums: Optional[pd.DataFrame] = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Scikit-learn style fit
    # ------------------------------------------------------------------

    def fit(
        self,
        data: pd.DataFrame,
        group_col: str = "group",
        period_col: str = "period",
        loss_col: str = "loss",
        weight_col: str = "weight",
    ) -> "BuhlmannStraub":
        """
        Estimate structural parameters and credibility factors from panel data.

        Parameters
        ----------
        data:
            A pandas DataFrame with one row per (group, period). Columns need
            not be in any particular order.
        group_col:
            Column identifying the group (e.g. scheme ID, territory, NCD class).
        period_col:
            Column identifying the time period (e.g. accident year). Used only
            to count periods per group; the ordering does not matter.
        loss_col:
            Column containing the loss rate or loss ratio. This should be losses
            per unit of exposure — not total losses. If you have total losses,
            divide by the exposure column before fitting.
        weight_col:
            Column containing the exposure weight (e.g. earned car years,
            earned premium, policy count). Must be strictly positive.

        Returns
        -------
        self
            Returns the fitted estimator so calls can be chained.
        """
        validate_panel_data(data, group_col, period_col, loss_col, weight_col)
        check_duplicate_periods(data, group_col, period_col)

        self._group_col = group_col
        self._period_col = period_col
        self._loss_col = loss_col
        self._weight_col = weight_col

        # Build group-level summary: w_i, X_bar_i, T_i
        groups = self._build_group_summary(data, group_col, period_col, loss_col, weight_col)
        self._groups = groups

        # Estimate structural parameters
        mu_hat, v_hat, a_hat_raw = self._estimate_structural_params(
            data, groups, group_col, period_col, loss_col, weight_col
        )

        # Handle negative a_hat
        if a_hat_raw <= 0:
            if self.truncate_a:
                warnings.warn(
                    f"Between-group variance estimate a_hat = {a_hat_raw:.6g} ≤ 0. "
                    "Truncating to zero. This means the model finds no detectable "
                    "heterogeneity between groups — all groups will receive the "
                    "collective mean as their credibility premium (Z_i = 0). "
                    "Consider whether your data genuinely lacks between-group "
                    "variation, or whether you have too few groups to estimate a reliably.",
                    stacklevel=2,
                )
                a_hat = 0.0
                k = np.inf
            else:
                raise ValueError(
                    f"Between-group variance estimate a_hat = {a_hat_raw:.6g} ≤ 0. "
                    "Set truncate_a=True to handle this automatically."
                )
        else:
            a_hat = a_hat_raw
            k = v_hat / a_hat

        # Credibility factors
        w = groups["w_i"].values
        if np.isinf(k):
            z_values = np.zeros(len(groups))
        else:
            z_values = w / (w + k)

        z = pd.Series(z_values, index=groups.index, name="Z")

        # Credibility premiums
        x_bar = groups["x_bar_i"]
        premiums = pd.DataFrame(
            {
                "group": groups.index,
                "exposure": groups["w_i"],
                "observed_mean": x_bar.values,
                "Z": z_values,
                "credibility_premium": z_values * x_bar.values + (1 - z_values) * mu_hat,
                "complement": mu_hat,
            }
        ).set_index("group")

        self._mu_hat = float(mu_hat)
        self._v_hat = float(v_hat)
        self._a_hat = float(a_hat)
        self._a_hat_raw = float(a_hat_raw)
        self._k = float(k) if not np.isinf(k) else np.inf
        self._z = z
        self._premiums = premiums
        self._fitted = True

        return self

    # ------------------------------------------------------------------
    # Structural parameter properties
    # ------------------------------------------------------------------

    @property
    def mu_hat_(self) -> float:
        """
        Collective mean — the grand weighted average loss rate across all groups.

        This is the complement: what every group's credibility premium regresses
        toward as exposure shrinks.
        """
        self._check_fitted()
        return self._mu_hat  # type: ignore[return-value]

    @property
    def v_hat_(self) -> float:
        """
        EPV — Expected value of Process Variance.

        Measures within-group year-to-year volatility, averaged across groups.
        High v_hat means individual group experience is noisy — lean on the
        collective mean.
        """
        self._check_fitted()
        return self._v_hat  # type: ignore[return-value]

    @property
    def a_hat_(self) -> float:
        """
        VHM — Variance of Hypothetical Means.

        Measures how much the true underlying loss rates differ between groups.
        High a_hat means groups are genuinely heterogeneous — trust their own
        experience. Truncated at zero if truncate_a=True.
        """
        self._check_fitted()
        return self._a_hat  # type: ignore[return-value]

    @property
    def k_(self) -> float:
        """
        Bühlmann's k — the credibility parameter.

        k = v / a. This is the exposure a group needs to achieve Z = 0.5 (equal
        weight on own experience and collective mean). A group with exposure w_i
        has Z_i = w_i / (w_i + k).

        Returns np.inf when a_hat is zero (all groups get the collective mean).
        """
        self._check_fitted()
        return self._k  # type: ignore[return-value]

    @property
    def z_(self) -> pd.Series:
        """
        Credibility factors — one per group, indexed by group identifier.

        Z_i = w_i / (w_i + k), ranging from 0 (no credibility, use collective
        mean) to 1 (full credibility, use group's own experience).

        A group needs exposure w_i = k to achieve Z = 0.5.
        """
        self._check_fitted()
        return self._z  # type: ignore[return-value]

    @property
    def premiums_(self) -> pd.DataFrame:
        """
        Credibility premiums — one row per group.

        Columns:

        - ``exposure``: total exposure weight for the group
        - ``observed_mean``: exposure-weighted average loss rate
        - ``Z``: credibility factor
        - ``credibility_premium``: Z * observed_mean + (1 - Z) * mu_hat
        - ``complement``: mu_hat (the collective mean complement)
        """
        self._check_fitted()
        return self._premiums  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        """
        Return a formatted summary of the fit.

        The table shows structural parameters at the top and per-group credibility
        results below, formatted for an actuarial audience.

        Returns
        -------
        pd.DataFrame
            Per-group results. Structural parameters are printed to stdout.
        """
        self._check_fitted()

        print("Bühlmann-Straub Credibility Model")
        print("=" * 42)
        print(f"  Collective mean    mu  = {self._mu_hat:.6g}")
        print(f"  Process variance   v   = {self._v_hat:.6g}   (EPV, within-group)")
        print(f"  Between-group var  a   = {self._a_hat:.6g}   (VHM, between-group)")
        if np.isinf(self._k):
            print(f"  Credibility param  k   = inf   (a ≤ 0, all Z = 0)")
        else:
            print(f"  Credibility param  k   = {self._k:.6g}   (v / a)")
        print()
        print("  Interpretation: a group needs exposure = k to achieve Z = 0.50")
        if self._a_hat_raw <= 0:
            print(f"  (raw a_hat before truncation: {self._a_hat_raw:.6g})")
        print()

        tbl = self._premiums.copy()
        tbl.columns = ["Exposure", "Obs. Mean", "Z", "Cred. Premium", "Complement"]
        return tbl

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_group_summary(
        data: pd.DataFrame,
        group_col: str,
        period_col: str,
        loss_col: str,
        weight_col: str,
    ) -> pd.DataFrame:
        """
        Compute per-group summary statistics needed for parameter estimation.

        Returns a DataFrame indexed by group_col with columns:
            w_i     : total exposure (sum of weights)
            x_bar_i : exposure-weighted mean loss rate
            T_i     : number of periods
        """

        def group_stats(grp: pd.DataFrame) -> pd.Series:
            w_ij = grp[weight_col].values
            x_ij = grp[loss_col].values
            w_i = w_ij.sum()
            x_bar_i = (w_ij * x_ij).sum() / w_i
            T_i = len(grp)
            return pd.Series({"w_i": w_i, "x_bar_i": x_bar_i, "T_i": T_i})

        return data.groupby(group_col).apply(group_stats)

    @staticmethod
    def _estimate_structural_params(
        data: pd.DataFrame,
        groups: pd.DataFrame,
        group_col: str,
        period_col: str,
        loss_col: str,
        weight_col: str,
    ) -> tuple[float, float, float]:
        """
        Compute unbiased estimates of mu, v, and a.

        These are the standard Bühlmann-Straub (1970) non-parametric estimators,
        as presented in Bühlmann & Gisler (2005), Chapter 4.

        Parameters
        ----------
        data:
            Full panel data.
        groups:
            Per-group summary from _build_group_summary.

        Returns
        -------
        mu_hat, v_hat, a_hat_raw
            a_hat_raw may be negative — the caller decides how to handle it.
        """
        w = groups["w_i"].values
        x_bar = groups["x_bar_i"].values
        T = groups["T_i"].values
        r = len(groups)

        # Collective mean: grand weighted average
        w_total = w.sum()
        mu_hat = (w * x_bar).sum() / w_total

        # v_hat: within-group weighted mean squared deviation
        # v_hat = (1 / sum_i(T_i - 1)) * sum_i sum_j [ w_{ij} * (X_{ij} - X_bar_i)^2 ]
        denom_v = (T - 1).sum()  # degrees of freedom for v

        numerator_v = 0.0
        for group_id, grp_meta in groups.iterrows():
            grp = data[data[group_col] == group_id]
            w_ij = grp[weight_col].values
            x_ij = grp[loss_col].values
            x_bar_i = grp_meta["x_bar_i"]
            numerator_v += (w_ij * (x_ij - x_bar_i) ** 2).sum()

        v_hat = numerator_v / denom_v if denom_v > 0 else 0.0

        # a_hat: between-group variance
        # c = w_total - sum_i(w_i^2) / w_total
        # s2 = sum_i [ w_i * (x_bar_i - mu_hat)^2 ]
        # a_hat = (s2 - (r - 1) * v_hat) / c
        c = w_total - (w ** 2).sum() / w_total
        s2 = (w * (x_bar - mu_hat) ** 2).sum()
        a_hat_raw = (s2 - (r - 1) * v_hat) / c

        return float(mu_hat), float(v_hat), float(a_hat_raw)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "Model has not been fitted. Call .fit() first."
            )

    def __repr__(self) -> str:
        if not self._fitted:
            return "BuhlmannStraub(not fitted)"
        return (
            f"BuhlmannStraub("
            f"mu={self._mu_hat:.4g}, "
            f"v={self._v_hat:.4g}, "
            f"a={self._a_hat:.4g}, "
            f"k={self._k:.4g})"
        )
