"""
Tests for BuhlmannStraub.

The Hachemeister dataset is the standard benchmark for Bühlmann-Straub
credibility. It contains average bodily injury claim amounts for 5 US states
over 12 quarters, with claim counts as exposure weights.

Reference values are computed directly from the Bühlmann-Straub non-parametric
estimators (Bühlmann & Straub, 1970; Bühlmann & Gisler, 2005):

    mu_hat = 1832.816   (grand weighted mean across all states)
    v_hat  = 136793600.7 (EPV: within-state quarter-to-quarter variance)
    a_hat  = 100301.9    (VHM: between-state variance)
    k      = 1363.818    (v / a)
"""

import warnings

import numpy as np
import polars as pl
import pytest

from credibility import BuhlmannStraub


# ---------------------------------------------------------------------------
# Helper: look up a value from premiums_ by group id
# ---------------------------------------------------------------------------

def _premium(premiums: pl.DataFrame, group_id, col: str):
    """Extract a scalar from premiums_ for a given group and column."""
    return premiums.filter(pl.col("group") == group_id)[col][0]


# ---------------------------------------------------------------------------
# Hachemeister benchmark
# ---------------------------------------------------------------------------

class TestHachemeisterBenchmark:
    """Verify structural parameters against computed reference values."""

    @pytest.fixture(autouse=True)
    def fit(self, hachemeister_df, hachemeister_reference):
        self.ref = hachemeister_reference
        self.bs = BuhlmannStraub()
        self.bs.fit(
            hachemeister_df,
            group_col="state",
            period_col="period",
            loss_col="ratio",
            weight_col="weight",
        )

    def test_mu_hat(self):
        """Collective mean should match computed reference within 0.01%."""
        assert abs(self.bs.mu_hat_ - self.ref["mu_hat"]) / self.ref["mu_hat"] < 0.0001

    def test_v_hat(self):
        """EPV (within-group variance) should match computed reference within 0.01%."""
        assert abs(self.bs.v_hat_ - self.ref["v_hat"]) / self.ref["v_hat"] < 0.0001

    def test_a_hat(self):
        """VHM (between-group variance) should match computed reference within 0.1%."""
        assert abs(self.bs.a_hat_ - self.ref["a_hat"]) / self.ref["a_hat"] < 0.001

    def test_k(self):
        """Bühlmann's k = v/a should match computed reference within 0.1%."""
        assert abs(self.bs.k_ - self.ref["k"]) / self.ref["k"] < 0.001

    def test_credibility_premiums(self):
        """Credibility premiums should match computed reference within 0.01%."""
        premiums = self.bs.premiums_
        for state, expected in self.ref["premiums"].items():
            actual = _premium(premiums, state, "credibility_premium")
            rel_error = abs(actual - expected) / expected
            assert rel_error < 0.001, (
                f"State {state}: expected {expected:.3f}, got {actual:.3f} "
                f"(relative error {rel_error:.4%})"
            )

    def test_z_between_zero_and_one(self):
        """All credibility factors must be in [0, 1]."""
        z_vals = self.bs.z_["Z"]
        assert (z_vals >= 0).all() and (z_vals <= 1).all()

    def test_state1_highest_z(self):
        """State 1 has the highest total exposure and should have the highest Z."""
        z = self.bs.z_
        assert z.sort("Z", descending=True)["group"][0] == 1

    def test_state4_lowest_z(self):
        """State 4 has the lowest total exposure and should have the lowest Z."""
        z = self.bs.z_
        assert z.sort("Z")["group"][0] == 4

    def test_premiums_dataframe_shape(self):
        """Premiums DataFrame should have 5 rows (one per state) and expected columns."""
        premiums = self.bs.premiums_
        assert len(premiums) == 5
        assert "credibility_premium" in premiums.columns
        assert "Z" in premiums.columns
        assert "observed_mean" in premiums.columns
        assert "exposure" in premiums.columns

    def test_premiums_are_weighted_average_of_group_and_collective(self):
        """
        For each group: premium = Z * observed_mean + (1 - Z) * mu_hat.
        This is a direct check of the formula rather than the reference values.
        """
        premiums = self.bs.premiums_
        mu = self.bs.mu_hat_
        for row in premiums.iter_rows(named=True):
            expected = row["Z"] * row["observed_mean"] + (1 - row["Z"]) * mu
            assert abs(row["credibility_premium"] - expected) < 1e-6

    def test_summary_returns_dataframe(self, capsys):
        """summary() should print to stdout and return a Polars DataFrame."""
        result = self.bs.summary()
        captured = capsys.readouterr()
        assert "Bühlmann-Straub" in captured.out
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 5

    def test_repr_after_fit(self):
        assert "BuhlmannStraub(" in repr(self.bs)
        assert "not fitted" not in repr(self.bs)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_negative_a_hat_truncated(self):
        """When groups have the same loss rate, a_hat goes negative. Truncation should warn."""
        df = pl.DataFrame({
            "group": ["A", "A", "A", "B", "B", "B"],
            "period": [1, 2, 3, 1, 2, 3],
            "loss": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "weight": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        })
        bs = BuhlmannStraub(truncate_a=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bs.fit(df, group_col="group", period_col="period",
                   loss_col="loss", weight_col="weight")
            assert any("a_hat" in str(warning.message) for warning in w)

        assert bs.a_hat_ == 0.0
        assert np.isinf(bs.k_)
        assert (bs.z_["Z"] == 0.0).all()
        cp = bs.premiums_["credibility_premium"].to_numpy()
        np.testing.assert_allclose(cp, bs.mu_hat_)

    def test_negative_a_hat_raises_when_not_truncating(self):
        """truncate_a=False should raise ValueError when a_hat <= 0."""
        df = pl.DataFrame({
            "group": ["A", "A", "B", "B"],
            "period": [1, 2, 1, 2],
            "loss": [1.0, 1.0, 1.0, 1.0],
            "weight": [100.0, 100.0, 100.0, 100.0],
        })
        bs = BuhlmannStraub(truncate_a=False)
        with pytest.raises(ValueError, match="a_hat"):
            bs.fit(df, group_col="group", period_col="period",
                   loss_col="loss", weight_col="weight")

    def test_all_single_period_raises(self):
        """If all groups have exactly one period, v cannot be estimated."""
        df = pl.DataFrame({
            "group": ["A", "B", "C"],
            "period": [1, 1, 1],
            "loss": [1.0, 1.2, 0.8],
            "weight": [100.0, 200.0, 150.0],
        })
        bs = BuhlmannStraub()
        with pytest.raises(ValueError, match="one period"):
            bs.fit(df, group_col="group", period_col="period",
                   loss_col="loss", weight_col="weight")

    def test_single_period_group_warns(self):
        """A group with only one period should trigger a warning but not fail."""
        df = pl.DataFrame({
            "group": ["A", "A", "A", "B", "B", "B", "C"],
            "period": [1, 2, 3, 1, 2, 3, 1],
            "loss": [1.0, 1.1, 0.9, 1.5, 1.4, 1.6, 1.2],
            "weight": [100.0, 110.0, 90.0, 50.0, 60.0, 55.0, 80.0],
        })
        bs = BuhlmannStraub()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bs.fit(df, group_col="group", period_col="period",
                   loss_col="loss", weight_col="weight")
            single_period_warns = [x for x in w if "one period" in str(x.message)]
            assert len(single_period_warns) > 0

        assert bs._fitted
        assert "C" in bs.z_["group"].to_list()

    def test_single_group_raises(self):
        """At least 2 groups are required."""
        df = pl.DataFrame({
            "group": ["A", "A", "A"],
            "period": [1, 2, 3],
            "loss": [1.0, 1.1, 0.9],
            "weight": [100.0, 110.0, 90.0],
        })
        bs = BuhlmannStraub()
        with pytest.raises(ValueError, match="2 groups"):
            bs.fit(df, group_col="group", period_col="period",
                   loss_col="loss", weight_col="weight")

    def test_missing_column_raises(self):
        df = pl.DataFrame({"group": ["A"], "period": [1], "loss": [1.0]})
        bs = BuhlmannStraub()
        with pytest.raises(ValueError, match="Columns not found"):
            bs.fit(df, group_col="group", period_col="period",
                   loss_col="loss", weight_col="weight")

    def test_negative_weight_raises(self):
        df = pl.DataFrame({
            "group": ["A", "A", "B", "B"],
            "period": [1, 2, 1, 2],
            "loss": [1.0, 1.1, 0.9, 1.0],
            "weight": [100.0, -10.0, 100.0, 100.0],
        })
        bs = BuhlmannStraub()
        with pytest.raises(ValueError, match="non-positive"):
            bs.fit(df, group_col="group", period_col="period",
                   loss_col="loss", weight_col="weight")

    def test_not_fitted_raises(self):
        bs = BuhlmannStraub()
        with pytest.raises(RuntimeError, match="fit"):
            _ = bs.mu_hat_
        with pytest.raises(RuntimeError, match="fit"):
            _ = bs.z_
        with pytest.raises(RuntimeError, match="fit"):
            bs.summary()

    def test_repr_before_fit(self):
        bs = BuhlmannStraub()
        assert "not fitted" in repr(bs)

    def test_accepts_pandas_input(self):
        """pandas DataFrames are converted to Polars internally; output is still Polars."""
        pd = pytest.importorskip("pandas")
        df_pd = pd.DataFrame({
            "group": ["A", "A", "A", "B", "B", "B"],
            "period": [1, 2, 3, 1, 2, 3],
            "loss": [1.0, 1.1, 0.9, 1.5, 1.4, 1.6],
            "weight": [100.0, 110.0, 90.0, 50.0, 60.0, 55.0],
        })
        bs = BuhlmannStraub()
        bs.fit(df_pd, group_col="group", period_col="period",
               loss_col="loss", weight_col="weight")
        assert isinstance(bs.premiums_, pl.DataFrame)
        assert isinstance(bs.z_, pl.DataFrame)


# ---------------------------------------------------------------------------
# Equal weights: reduces to standard Bühlmann (unweighted)
# ---------------------------------------------------------------------------

class TestEqualWeights:
    """
    When all weights are equal, the Bühlmann-Straub model reduces to the
    unweighted Bühlmann model. The credibility factor becomes n / (n + k)
    where n is the number of periods.
    """

    def test_equal_weights_z_depends_only_on_n_periods(self):
        """Groups with equal weight and same number of periods get the same Z."""
        df = pl.DataFrame({
            "group": ["A", "A", "A", "B", "B", "B"],
            "period": [1, 2, 3, 1, 2, 3],
            "loss": [1.0, 1.5, 2.0, 0.5, 0.8, 0.6],
            "weight": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        })
        bs = BuhlmannStraub()
        bs.fit(df, group_col="group", period_col="period",
               loss_col="loss", weight_col="weight")

        z_a = bs.z_.filter(pl.col("group") == "A")["Z"][0]
        z_b = bs.z_.filter(pl.col("group") == "B")["Z"][0]
        assert abs(z_a - z_b) < 1e-10

    def test_equal_weights_z_formula(self):
        """With equal weights, Z_i = w_i / (w_i + k) = n_i / (n_i + k)."""
        df = pl.DataFrame({
            "group": ["A", "A", "A", "B", "B"],
            "period": [1, 2, 3, 1, 2],
            "loss": [1.0, 1.5, 2.0, 0.5, 0.8],
            "weight": [1.0, 1.0, 1.0, 1.0, 1.0],
        })
        bs = BuhlmannStraub()
        bs.fit(df, group_col="group", period_col="period",
               loss_col="loss", weight_col="weight")

        k = bs.k_
        z_a = bs.z_.filter(pl.col("group") == "A")["Z"][0]
        z_b = bs.z_.filter(pl.col("group") == "B")["Z"][0]
        assert abs(z_a - 3 / (3 + k)) < 1e-10
        assert abs(z_b - 2 / (2 + k)) < 1e-10
        assert z_a > z_b


# ---------------------------------------------------------------------------
# Mathematical properties
# ---------------------------------------------------------------------------

class TestMathematicalProperties:

    @pytest.fixture(autouse=True)
    def fit(self, hachemeister_df):
        self.bs = BuhlmannStraub()
        self.bs.fit(
            hachemeister_df,
            group_col="state",
            period_col="period",
            loss_col="ratio",
            weight_col="weight",
        )

    def test_premiums_are_credibility_weighted_average(self):
        """
        Verify: credibility premium = Z * obs_mean + (1-Z) * mu_hat.
        This is the defining formula of the model.
        """
        mu = self.bs.mu_hat_
        premiums = self.bs.premiums_
        for row in premiums.iter_rows(named=True):
            expected = row["Z"] * row["observed_mean"] + (1 - row["Z"]) * mu
            assert abs(row["credibility_premium"] - expected) < 1e-6

    def test_z_increases_with_exposure(self):
        """Higher exposure implies higher Z (monotone in exposure when k is fixed)."""
        premiums = self.bs.premiums_
        k = self.bs.k_
        for row in premiums.iter_rows(named=True):
            z_expected = row["exposure"] / (row["exposure"] + k)
            assert abs(row["Z"] - z_expected) < 1e-10

    def test_k_equals_v_over_a(self):
        """k = v / a by definition."""
        assert abs(self.bs.k_ - self.bs.v_hat_ / self.bs.a_hat_) < 1e-4

    def test_premiums_group_column_values(self):
        """Premiums DataFrame 'group' column should contain all state identifiers."""
        assert set(self.bs.premiums_["group"].to_list()) == {1, 2, 3, 4, 5}

    def test_z_group_column_values(self):
        """z_ DataFrame 'group' column should contain all state identifiers."""
        assert set(self.bs.z_["group"].to_list()) == {1, 2, 3, 4, 5}

    def test_state1_highest_mean_gets_premium_above_mu(self):
        """State 1 has the highest observed mean (2070), so its premium should be above mu_hat."""
        mu = self.bs.mu_hat_
        state1_premium = _premium(self.bs.premiums_, 1, "credibility_premium")
        assert state1_premium > mu

    def test_state4_lowest_mean_gets_premium_below_mu(self):
        """State 4 has the lowest observed mean (1356), so its premium should be below mu_hat."""
        mu = self.bs.mu_hat_
        state4_premium = _premium(self.bs.premiums_, 4, "credibility_premium")
        assert state4_premium < mu
