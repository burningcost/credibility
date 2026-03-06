"""
Tests for HierarchicalBuhlmannStraub.

Uses a synthetic 3-level dataset (region -> district -> sector) to verify
that the model:
- Fits without error
- Produces sensible variance components at each level
- Computes premiums that reflect the hierarchy (sectors in high-loss regions
  get premiums pulled toward their region's mean)
- Handles edge cases correctly
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from credibility import BuhlmannStraub, HierarchicalBuhlmannStraub


class TestHierarchicalFitsWithoutError:

    def test_basic_fit(self, hierarchical_df):
        model = HierarchicalBuhlmannStraub(level_cols=["region", "district", "sector"])
        model.fit(hierarchical_df, period_col="period",
                  loss_col="loss_rate", weight_col="exposure")
        assert model._fitted

    def test_two_level_fit(self, hierarchical_df):
        """Two-level hierarchy (equivalent to standard B-S with an extra layer)."""
        model = HierarchicalBuhlmannStraub(level_cols=["region", "district"])
        model.fit(hierarchical_df, period_col="period",
                  loss_col="loss_rate", weight_col="exposure")
        assert model._fitted

    def test_repr_after_fit(self, hierarchical_df):
        model = HierarchicalBuhlmannStraub(level_cols=["region", "district", "sector"])
        model.fit(hierarchical_df, period_col="period",
                  loss_col="loss_rate", weight_col="exposure")
        assert "not fitted" not in repr(model)
        assert "region" in repr(model)

    def test_repr_before_fit(self):
        model = HierarchicalBuhlmannStraub(level_cols=["region", "district"])
        assert "not fitted" in repr(model)


class TestHierarchicalOutputShape:

    @pytest.fixture(autouse=True)
    def fit(self, hierarchical_df):
        self.df = hierarchical_df
        self.model = HierarchicalBuhlmannStraub(
            level_cols=["region", "district", "sector"]
        )
        self.model.fit(hierarchical_df, period_col="period",
                       loss_col="loss_rate", weight_col="exposure")

    def test_level_results_keys(self):
        keys = set(self.model.level_results_.keys())
        assert keys == {"region", "district", "sector"}

    def test_premiums_at_sector_has_12_rows(self):
        """12 sectors in the synthetic dataset."""
        premiums = self.model.premiums_at("sector")
        assert len(premiums) == 12

    def test_premiums_at_district_has_4_rows(self):
        premiums = self.model.premiums_at("district")
        assert len(premiums) == 4

    def test_premiums_at_region_has_2_rows(self):
        premiums = self.model.premiums_at("region")
        assert len(premiums) == 2

    def test_bottom_premiums_equals_premiums_at_sector(self):
        bottom = self.model.premiums_
        sector = self.model.premiums_at("sector")
        pd.testing.assert_frame_equal(bottom, sector)

    def test_premiums_columns(self):
        premiums = self.model.premiums_at("sector")
        assert "credibility_premium" in premiums.columns
        assert "Z" in premiums.columns
        assert "observed_mean" in premiums.columns

    def test_z_between_zero_and_one_at_all_levels(self):
        for level in ["region", "district", "sector"]:
            lr = self.model.level_results_[level]
            assert (lr.z >= 0).all() and (lr.z <= 1).all(), (
                f"Z values out of [0, 1] at level '{level}'"
            )

    def test_summary_runs_without_error(self, capsys):
        self.model.summary()
        captured = capsys.readouterr()
        assert "Hierarchical" in captured.out
        assert "region" in captured.out
        assert "district" in captured.out
        assert "sector" in captured.out


class TestHierarchicalVarianceComponents:

    @pytest.fixture(autouse=True)
    def fit(self, hierarchical_df):
        self.df = hierarchical_df
        self.model = HierarchicalBuhlmannStraub(
            level_cols=["region", "district", "sector"]
        )
        self.model.fit(hierarchical_df, period_col="period",
                       loss_col="loss_rate", weight_col="exposure")

    def test_structural_params_finite(self):
        """All structural parameters should be finite numbers."""
        for level in ["region", "district", "sector"]:
            lr = self.model.level_results_[level]
            assert np.isfinite(lr.mu_hat)
            assert np.isfinite(lr.v_hat)
            assert np.isfinite(lr.a_hat)

    def test_v_hat_positive_at_sector_level(self):
        """Within-sector variance should be positive (we have 3 periods per sector)."""
        lr = self.model.level_results_["sector"]
        assert lr.v_hat > 0

    def test_level_results_accessible(self):
        for level in ["region", "district", "sector"]:
            lr = self.model.level_results_[level]
            assert hasattr(lr, "mu_hat")
            assert hasattr(lr, "v_hat")
            assert hasattr(lr, "a_hat")
            assert hasattr(lr, "k")
            assert hasattr(lr, "z")
            assert hasattr(lr, "premiums")


class TestHierarchicalPremiumProperties:

    @pytest.fixture(autouse=True)
    def fit(self, hierarchical_df):
        self.df = hierarchical_df
        self.model = HierarchicalBuhlmannStraub(
            level_cols=["region", "district", "sector"]
        )
        self.model.fit(hierarchical_df, period_col="period",
                       loss_col="loss_rate", weight_col="exposure")

    def test_r1_sectors_above_r2_sectors_on_average(self):
        """
        Region R1 has a +0.05 true effect vs R2 -0.05.
        Credibility premiums should reflect this on average.
        """
        premiums = self.model.premiums_
        r1_sectors = [s for s in premiums.index if s.startswith("R1")]
        r2_sectors = [s for s in premiums.index if s.startswith("R2")]
        r1_mean = premiums.loc[r1_sectors, "credibility_premium"].mean()
        r2_mean = premiums.loc[r2_sectors, "credibility_premium"].mean()
        assert r1_mean > r2_mean, (
            f"R1 mean premium {r1_mean:.4f} should exceed R2 {r2_mean:.4f}"
        )

    def test_premiums_not_all_equal(self):
        """If the model is working, premiums should differ across sectors."""
        premiums = self.model.premiums_["credibility_premium"]
        assert premiums.std() > 0

    def test_premiums_all_positive(self):
        """Credibility premiums should be positive loss rates."""
        premiums = self.model.premiums_["credibility_premium"]
        assert (premiums > 0).all()


class TestHierarchicalEdgeCases:

    def test_only_one_level_raises(self):
        with pytest.raises(ValueError, match="two levels"):
            HierarchicalBuhlmannStraub(level_cols=["region"])

    def test_missing_level_column_raises(self, hierarchical_df):
        model = HierarchicalBuhlmannStraub(level_cols=["region", "district", "nonexistent"])
        with pytest.raises(ValueError, match="Columns not found"):
            model.fit(hierarchical_df, period_col="period",
                      loss_col="loss_rate", weight_col="exposure")

    def test_invalid_premiums_at_level_raises(self, hierarchical_df):
        model = HierarchicalBuhlmannStraub(level_cols=["region", "district"])
        model.fit(hierarchical_df, period_col="period",
                  loss_col="loss_rate", weight_col="exposure")
        with pytest.raises(ValueError, match="not one of the fitted levels"):
            model.premiums_at("sector")

    def test_not_fitted_raises(self):
        model = HierarchicalBuhlmannStraub(level_cols=["region", "district"])
        with pytest.raises(RuntimeError, match="fit"):
            _ = model.level_results_
        with pytest.raises(RuntimeError, match="fit"):
            _ = model.premiums_

    def test_non_strict_hierarchy_raises(self):
        """If a child appears under multiple parents, raise a clear error."""
        df = pd.DataFrame({
            "region": ["R1", "R1", "R2", "R2"],
            "district": ["D1", "D1", "D1", "D1"],  # D1 appears in both regions
            "period": [1, 2, 1, 2],
            "loss_rate": [0.5, 0.6, 0.7, 0.8],
            "exposure": [100.0, 100.0, 100.0, 100.0],
        })
        model = HierarchicalBuhlmannStraub(level_cols=["region", "district"])
        with pytest.raises(ValueError, match="strict"):
            model.fit(df, period_col="period", loss_col="loss_rate", weight_col="exposure")


class TestHierarchicalConsistencyWithBuhlmannStraub:
    """
    A two-level hierarchical model with only one region should be broadly
    consistent with a flat BuhlmannStraub model on the same data, since
    the top-level blending adds negligible information.
    """

    def test_sector_level_premiums_order_matches_flat_model(self, hierarchical_df):
        """
        The ordering of sector premiums from the hierarchical model should broadly
        match the ordering from a flat BuhlmannStraub model (same sectors).
        """
        # Flat model: just sector as group
        bs = BuhlmannStraub()
        bs.fit(
            hierarchical_df,
            group_col="sector",
            period_col="period",
            loss_col="loss_rate",
            weight_col="exposure",
        )

        # Hierarchical model
        hier = HierarchicalBuhlmannStraub(level_cols=["region", "district", "sector"])
        hier.fit(hierarchical_df, period_col="period",
                 loss_col="loss_rate", weight_col="exposure")

        flat_premiums = bs.premiums_["credibility_premium"].sort_values()
        hier_premiums = hier.premiums_["credibility_premium"].sort_values()

        # The top 3 and bottom 3 sectors by premium should have significant overlap
        flat_top3 = set(flat_premiums.tail(3).index)
        hier_top3 = set(hier_premiums.tail(3).index)
        overlap = len(flat_top3 & hier_top3)
        assert overlap >= 2, (
            f"Top-3 sector overlap too low: flat={flat_top3}, hier={hier_top3}"
        )
