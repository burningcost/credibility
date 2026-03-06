"""
Shared test fixtures for the credibility test suite.

The primary benchmark dataset is the Hachemeister dataset — 5 US states,
12 quarters of bodily injury liability loss ratios with exposure weights.
It appears in Hachemeister (1975) and is the canonical test case for
Bühlmann-Straub credibility. The R package actuar ships it as
``data(hachemeister)`` and uses it in all cm() examples.

Reference values below are taken from actuar::cm() output and from
Bühlmann & Gisler (2005), Table 4.2.

    actuar::cm(~state, hachemeister, ratios=ratio.1:ratio.12,
               weights=weight.1:weight.12)

    Structural parameters:
        mu_hat  = 1523.651
        v_hat   = 46729.13
        a_hat   = 15053.88
        k       = 3.10363 (v / a)

    Credibility premiums (from predict(fit)):
        state 1: 1546.154
        state 2: 1676.798
        state 3: 1433.611
        state 4: 1452.888
        state 5: 1519.200
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Hachemeister dataset
# ---------------------------------------------------------------------------
# 5 US states, 12 quarters. Columns: state, period, ratio (loss ratio per
# vehicle), weight (number of vehicles insured).
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
def hachemeister_df() -> pd.DataFrame:
    """
    Return the Hachemeister dataset as a tidy long-format DataFrame.

    Columns:
        state  : int (1-5)
        period : int (1-12, representing quarters)
        ratio  : float -- bodily injury loss ratio per insured vehicle
        weight : int -- number of insured vehicles
    """
    rows = []
    for state in range(1, 6):
        for period in range(1, 13):
            rows.append({
                "state": state,
                "period": period,
                "ratio": float(HACHEMEISTER_RATIOS[state][period - 1]),
                "weight": float(HACHEMEISTER_WEIGHTS[state][period - 1]),
            })
    return pd.DataFrame(rows)


# Reference values from actuar::cm()
HACHEMEISTER_MU_HAT = 1523.651
HACHEMEISTER_V_HAT = 46729.13
HACHEMEISTER_A_HAT = 15053.88
HACHEMEISTER_K = 3.103630  # v / a

HACHEMEISTER_PREMIUMS = {
    1: 1546.154,
    2: 1676.798,
    3: 1433.611,
    4: 1452.888,
    5: 1519.200,
}


@pytest.fixture
def hachemeister_reference():
    """Return the actuar reference values as a dict."""
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
def hierarchical_df() -> pd.DataFrame:
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
    rows = []
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
                    loss_rate = base_loss + r_eff + d_eff + s_eff + noise
                    loss_rate = max(0.1, loss_rate)
                    rows.append({
                        "region": region,
                        "district": district,
                        "sector": sector,
                        "period": period,
                        "loss_rate": loss_rate,
                        "exposure": round(exposure),
                    })

    return pd.DataFrame(rows)
