"""
credibility — Bühlmann-Straub credibility models for insurance pricing.

This package implements the Bühlmann-Straub (1970) credibility model and its
hierarchical extension (Jewell, 1975) for UK non-life insurance pricing teams.

The core problem: you have multiple groups (schemes, territories, NCD classes)
with varying amounts of loss experience. How much should each group's own
experience influence its technical rate? Credibility theory gives a rigorous,
data-driven answer.

Quick start::

    from credibility import BuhlmannStraub

    bs = BuhlmannStraub()
    bs.fit(df, group_col="scheme", period_col="year",
           loss_col="loss_rate", weight_col="exposure")

    bs.summary()          # structural parameters + per-group table
    bs.z_                 # credibility factors by scheme
    bs.premiums_          # full results DataFrame

For hierarchical multi-level structures::

    from credibility import HierarchicalBuhlmannStraub

    model = HierarchicalBuhlmannStraub(level_cols=["region", "district", "sector"])
    model.fit(df, period_col="year", loss_col="loss_rate", weight_col="exposure")
    model.premiums_at("sector")

"""

from .buhlmann_straub import BuhlmannStraub
from .hierarchical import HierarchicalBuhlmannStraub, LevelResult

__all__ = [
    "BuhlmannStraub",
    "HierarchicalBuhlmannStraub",
    "LevelResult",
]

__version__ = "0.1.0"
