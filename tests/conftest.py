"""
Shared test fixtures for the credibility test suite.

The primary benchmark dataset is the Hachemeister dataset - 5 US states,
12 quarters of bodily injury claim severity with claim count weights.
It appears in Hachemeister (1975) and is the canonical test case for
Bühlmann-Straub credibility. The R package actuar ships it as
``data(hachemeister)`` and uses it in all cm() examples.

Reference values below are computed directly from the Bühlmann-Straub
estimators on this dataset and verified against independent Python
calculation. The Hachemeister dataset contains average claim amounts
(not loss ratios), with the number of claims as exposure weights.

Per-group weighted means:
    State 1: X_bar = 2070.336, exposure = 108722
    State 2: X_bar = 1510.944, exposure = 20200
    State 3: X_bar = 1606.249, exposure = 27533
    State 4: X_bar = 1356.491, exposure = 9959
    State 5: X_bar = 1605.640, exposure = 36712

Structural parameters (Buhlmann-Straub non-parametric estimators):
    mu_hat = 1832.816   (grand weighted mean)
    v_hat  = 136793600.7 (EPV, within-group)
    a_hat  = 100301.9    (VHM, between-group)
    k      = 1363.818    (v / a)

Credibility premiums:
    State 1: 2067.394
    State 2: 1531.301
    State 3: 1616.943
    State 4: 1413.864
    State 5: 1613.778
"""

import numpy as np
import polars as pl
import pytest


# ---------------------------------------------------------------------------
# Hachemeister dataset
# ---------------------------------------------------------------------------
# 5 US states, 12 quarters. ratio = average bodily injury claim amount.
# weight = number of claims.
# Source: Hachemeister (1975), reproduced in actuar package.
# ---------------------------------------------------------------------------

HACHEMEISTER_RATIOS = {
    1: [1738, 1642, 1794, 2051, 2079, 2234, 2032, 2035, 2115, 2262, 2267, 2517],
    2: [1364, 1408, 1597, 1444, 1342, 1675, 1470, 1448, 1464, 1831, 1612, 1471],
    3: [1759, 1685, 1479, 1763, 1674, 1517, 1448, 1464, 1634, 1519, 1656, 1657],
    4: [1223, 1146, 1010, 1247, 1492, 1290, 1332, 1349, 1393, 1396, 1677, 1716],
    5: [1456, 1499, 1609, 1741, 1482, 1572, 1606, 1573, 1613, 1741, 1659, 1685],
}

HACHEMEISTER_WEIGHTS = {
    1: [7861, 9251, 8706, 8792, 9593, 9077, 9470, 9364, 8742, 8921, 9145, 9800],
    2: [1622, 1742, 1523, 1515, 1622, 1602, 1764, 1861, 1698, 1765, 1721, 1765],
    3: [2592, 2466, 2321, 2127, 2122, 2339, 2291, 2145, 2289, 2203, 2219, 2419],
    4: [802, 900, 774, 836, 801, 796, 936, 814, 828, 805, 813, 854],
    5: [2755, 2668, 2910, 3097, 3295, 2976, 3197, 3198, 2976, 3070, 3089, 3481],
}


@pytest.fixture
def hachemeister_df() -> pl.DataFrame:
    """
    Return the Hachemeister dataset as a tidy long-format Polars DataFrame.

    Columns:
        state  : int (1-5)
        period : int (1-12, representing quarters)
        ratio  : float -- average bodily injury claim amount per claim
        weight : float -- number of claims
    """
    rows_state = []
    rows_period = []
    rows_ratio = []
    rows_weight = []
    for state in range(1, 6):
        for period in range(1, 13):
            rows_state.append(state)
            rows_period.append(period)
            rows_ratio.append(float(HACHEMEISTER_RATIOS[state][period - 1]))
            rows_weight.append(float(HACHEMEISTER_WEIGHTS[state][period - 1]))
    return pl.DataFrame({
        "state": rows_state,
        "period": rows_period,
        "ratio": rows_ratio,
        "weight": rows_weight,
    })


# Reference values computed from the Buhlmann-Straub estimators directly.
# These are the mathematically correct values for this dataset.
HACHEMEISTER_MU_HAT = 1832.816
HACHEMEISTER_V_HAT = 136793600.7
HACHEMEISTER_A_HAT = 100301.9
HACHEMEISTER_K = 1363.818

HACHEMEISTER_PREMIUMS = {
    1: 2067.394,
    2: 1531.301,
    3: 1616.943,
    4: 1413.864,
    5: 1613.778,
}


@pytest.fixture
def hachemeister_reference():
    """Return the reference values as a dict."""
    return {
        "mu_hat": HACHEMEISTER_MU_HAT,
        "v_hat": HACHEMEISTER_V_HAT,
        "a_hat": HACHEMEISTER_A_HAT,
        "k": HACHEMEISTER_K,
        "premiums": HACHEMEISTER_PREMIUMS,
    }


# ---------------------------------------------------------------------------
# Synthetic 3-level hierarchical dataset
# ---------------------------------------------------------------------------

@pytest.fixture
def hierarchical_df() -> pl.DataFrame:
    """
    Synthetic 3-level dataset: region -> district -> sector, 3 periods each.

    Designed to have clear between-level variation so that a_hat is positive
    at all levels and the hierarchical model produces meaningful credibility
    factors.

    Structure:
        2 regions
        2 districts per region (4 total)
        3 sectors per district (12 total)
        3 periods per sector
        = 108 rows
    """
    rng = np.random.default_rng(42)

    region_effects = {"R1": 0.05, "R2": -0.05}
    district_effects = {
        "R1_D1": 0.03, "R1_D2": -0.03,
        "R2_D1": 0.02, "R2_D2": -0.02,
    }

    base_loss = 0.65
    rows_region = []
    rows_district = []
    rows_sector = []
    rows_period = []
    rows_loss_rate = []
    rows_exposure = []

    for region, r_eff in region_effects.items():
        for d_idx in [1, 2]:
            district = f"{region}_D{d_idx}"
            d_eff = district_effects[district]
            for s_idx in [1, 2, 3]:
                sector = f"{district}_S{s_idx}"
                s_eff = rng.normal(0, 0.02)
                for period in [2021, 2022, 2023]:
                    exposure = rng.uniform(200, 2000)
                    noise = rng.normal(0, 0.05)
                    loss_rate = max(0.1, base_loss + r_eff + d_eff + s_eff + noise)
                    rows_region.append(region)
                    rows_district.append(district)
                    rows_sector.append(sector)
                    rows_period.append(period)
                    rows_loss_rate.append(loss_rate)
                    rows_exposure.append(round(exposure))

    return pl.DataFrame({
        "region": rows_region,
        "district": rows_district,
        "sector": rows_sector,
        "period": rows_period,
        "loss_rate": rows_loss_rate,
        "exposure": rows_exposure,
    })
