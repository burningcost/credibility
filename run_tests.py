"""
Test runner for the credibility package on Databricks serverless compute.

Installs polars and the package inline, then runs assertions against the
Hachemeister benchmark and the synthetic hierarchical dataset.

Version 0.2.0 — Polars-native (pandas is bridge-only).
"""

# ============================================================
# Step 1: Install dependencies
# ============================================================
import subprocess, sys, os, uuid, warnings
import numpy as np

result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "hatchling", "polars>=0.20", "--quiet"],
    capture_output=True, text=True
)
if result.returncode != 0:
    print("dependency install error:", result.stderr[:400])
else:
    print("Dependencies installed")

import polars as pl

# ============================================================
# Step 2: Write package files to a unique temp directory
# ============================================================
pkg_dir = f"/tmp/credibility_{uuid.uuid4().hex[:8]}"
os.makedirs(f"{pkg_dir}/src/credibility", exist_ok=True)

# ---- _validation.py ----
VALIDATION_SRC = r'''
from __future__ import annotations
import warnings
from typing import Union
import numpy as np
import polars as pl


def _to_polars(data):
    if isinstance(data, pl.DataFrame):
        return data
    try:
        return pl.from_pandas(data)
    except Exception as exc:
        raise TypeError(
            f"data must be a polars.DataFrame or pandas.DataFrame. Got {type(data).__name__}."
        ) from exc


def validate_panel_data(data, group_col, period_col, loss_col, weight_col):
    required = {group_col, period_col, loss_col, weight_col}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Columns not found in data: {missing}")
    if data.is_empty():
        raise ValueError("data is empty.")
    for col in [group_col, period_col, loss_col, weight_col]:
        n_null = data[col].null_count()
        if n_null > 0:
            raise ValueError(f"Column '{col}' contains {n_null} null value(s).")
    if (data[weight_col] <= 0).any():
        raise ValueError(f"Column '{weight_col}' contains non-positive values.")
    loss_vals = data[loss_col].to_numpy()
    if not np.isfinite(loss_vals).all():
        raise ValueError(f"Column '{loss_col}' contains non-finite values.")
    periods_per_group = (
        data.group_by(group_col)
        .agg(pl.col(period_col).n_unique().alias("n_periods"))
    )
    single_period_groups = (
        periods_per_group.filter(pl.col("n_periods") == 1)[group_col].to_list()
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
            f"{len(single_period_groups)} group(s) have only one period: "
            f"{single_period_groups[:5]}.",
            stacklevel=4,
        )
    if n_groups < 2:
        raise ValueError(
            f"At least 2 groups are required to estimate between-group variance (a). "
            f"Found {n_groups} group."
        )


def check_duplicate_periods(data, group_col, period_col):
    n_dupes = data.height - data.unique(subset=[group_col, period_col]).height
    if n_dupes > 0:
        warnings.warn(
            f"{n_dupes} duplicate (group, period) row(s) found.",
            stacklevel=4,
        )
'''

# ---- buhlmann_straub.py ----
BS_SRC = r'''
from __future__ import annotations
import warnings
import numpy as np
import polars as pl
from ._validation import _to_polars, check_duplicate_periods, validate_panel_data


class BuhlmannStraub:
    def __init__(self, truncate_a=True):
        self.truncate_a = truncate_a
        self._fitted = False

    def fit(self, data, group_col="group", period_col="period",
            loss_col="loss", weight_col="weight"):
        data = _to_polars(data)
        validate_panel_data(data, group_col, period_col, loss_col, weight_col)
        check_duplicate_periods(data, group_col, period_col)
        self._group_col = group_col
        groups = self._build_group_summary(data, group_col, period_col, loss_col, weight_col)
        self._groups = groups
        mu_hat, v_hat, a_hat_raw = self._estimate_structural_params(
            data, groups, group_col, period_col, loss_col, weight_col
        )
        if a_hat_raw <= 0:
            if self.truncate_a:
                warnings.warn(
                    f"Between-group variance estimate a_hat = {a_hat_raw:.6g} <= 0. "
                    "Truncating to zero. This means the model finds no detectable "
                    "heterogeneity between groups -- all groups will receive the "
                    "collective mean as their credibility premium (Z_i = 0). "
                    "Consider whether your data genuinely lacks between-group "
                    "variation, or whether you have too few groups to estimate a reliably.",
                    stacklevel=2,
                )
                a_hat, k = 0.0, np.inf
            else:
                raise ValueError(
                    f"Between-group variance estimate a_hat = {a_hat_raw:.6g} <= 0. "
                    "Set truncate_a=True to handle this automatically."
                )
        else:
            a_hat = a_hat_raw
            k = v_hat / a_hat

        w = groups["w_i"].to_numpy()
        z_values = np.zeros(len(groups)) if np.isinf(k) else w / (w + k)
        group_ids = groups["group"].to_list()
        x_bar = groups["x_bar_i"].to_numpy()

        z_df = pl.DataFrame({"group": group_ids, "Z": z_values})
        premiums = pl.DataFrame({
            "group": group_ids,
            "exposure": w,
            "observed_mean": x_bar,
            "Z": z_values,
            "credibility_premium": z_values * x_bar + (1 - z_values) * mu_hat,
            "complement": np.full(len(groups), mu_hat),
        })
        self._mu_hat = float(mu_hat)
        self._v_hat = float(v_hat)
        self._a_hat = float(a_hat)
        self._a_hat_raw = float(a_hat_raw)
        self._k = float(k) if not np.isinf(k) else np.inf
        self._z = z_df
        self._premiums = premiums
        self._fitted = True
        return self

    @property
    def mu_hat_(self): self._check_fitted(); return self._mu_hat
    @property
    def v_hat_(self): self._check_fitted(); return self._v_hat
    @property
    def a_hat_(self): self._check_fitted(); return self._a_hat
    @property
    def k_(self): self._check_fitted(); return self._k
    @property
    def z_(self): self._check_fitted(); return self._z
    @property
    def premiums_(self): self._check_fitted(); return self._premiums

    def summary(self):
        self._check_fitted()
        print("Buhlmann-Straub Credibility Model")
        print("=" * 42)
        print(f"  mu  = {self._mu_hat:.6g}")
        print(f"  v   = {self._v_hat:.6g}")
        print(f"  a   = {self._a_hat:.6g}")
        if np.isinf(self._k):
            print("  k   = inf")
        else:
            print(f"  k   = {self._k:.6g}")
        tbl = self._premiums.rename({
            "exposure": "Exposure", "observed_mean": "Obs. Mean",
            "credibility_premium": "Cred. Premium", "complement": "Complement",
        })
        return tbl

    @staticmethod
    def _build_group_summary(data, group_col, period_col, loss_col, weight_col):
        return (
            data.group_by(group_col)
            .agg([
                pl.col(weight_col).sum().alias("w_i"),
                (
                    (pl.col(weight_col) * pl.col(loss_col)).sum()
                    / pl.col(weight_col).sum()
                ).alias("x_bar_i"),
                pl.len().alias("T_i"),
            ])
            .rename({group_col: "group"})
            .sort("group")
        )

    @staticmethod
    def _estimate_structural_params(data, groups, group_col, period_col, loss_col, weight_col):
        w = groups["w_i"].to_numpy()
        x_bar = groups["x_bar_i"].to_numpy()
        T = groups["T_i"].to_numpy()
        group_ids = groups["group"].to_list()
        r = len(groups)
        w_total = w.sum()
        mu_hat = (w * x_bar).sum() / w_total
        denom_v = (T - 1).sum()

        x_bar_lf = pl.DataFrame({group_col: group_ids, "_x_bar_i": x_bar}).lazy()
        numerator_v = (
            data.lazy()
            .join(x_bar_lf, on=group_col, how="left")
            .with_columns(
                (pl.col(weight_col) * (pl.col(loss_col) - pl.col("_x_bar_i")) ** 2).alias("_sq_dev")
            )
            .select(pl.col("_sq_dev").sum())
            .collect()
            ["_sq_dev"][0]
        )
        v_hat = float(numerator_v) / denom_v if denom_v > 0 else 0.0
        c = w_total - (w ** 2).sum() / w_total
        s2 = (w * (x_bar - mu_hat) ** 2).sum()
        a_hat_raw = (s2 - (r - 1) * v_hat) / c
        return float(mu_hat), float(v_hat), float(a_hat_raw)

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call .fit() first.")

    def __repr__(self):
        if not self._fitted:
            return "BuhlmannStraub(not fitted)"
        return f"BuhlmannStraub(mu={self._mu_hat:.4g}, v={self._v_hat:.4g}, a={self._a_hat:.4g}, k={self._k:.4g})"
'''

# ---- hierarchical.py ----
HIER_SRC = r'''
from __future__ import annotations
import numpy as np
import polars as pl
from ._validation import _to_polars
from .buhlmann_straub import BuhlmannStraub


class LevelResult:
    def __init__(self, level_name, mu_hat, v_hat, a_hat, k, z, premiums):
        self.level_name = level_name
        self.mu_hat = mu_hat; self.v_hat = v_hat; self.a_hat = a_hat
        self.k = k; self.z = z; self.premiums = premiums
    def __repr__(self):
        return f"LevelResult(level='{self.level_name}', mu={self.mu_hat:.4g}, a={self.a_hat:.4g})"


class HierarchicalBuhlmannStraub:
    def __init__(self, level_cols, truncate_a=True):
        if len(level_cols) < 2:
            raise ValueError("At least two levels are required for a hierarchical model. "
                             "For a single-level model, use BuhlmannStraub.")
        self.level_cols = level_cols
        self.truncate_a = truncate_a
        self._fitted = False

    def fit(self, data, period_col="period", loss_col="loss", weight_col="weight"):
        data = _to_polars(data)
        required = set(self.level_cols) | {period_col, loss_col, weight_col}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Columns not found in data: {missing}")
        self._validate_hierarchy(data)
        level_results = {}
        bottom_level = self.level_cols[-1]
        bs = BuhlmannStraub(truncate_a=self.truncate_a)
        bs.fit(data, group_col=bottom_level, period_col=period_col,
               loss_col=loss_col, weight_col=weight_col)
        level_results[bottom_level] = LevelResult(
            bottom_level, bs.mu_hat_, bs.v_hat_, bs.a_hat_, bs.k_, bs.z_, bs.premiums_)

        for depth in range(len(self.level_cols) - 2, -1, -1):
            parent_col = self.level_cols[depth]
            child_col = self.level_cols[depth + 1]
            child_summary = (
                data.select([parent_col, child_col, weight_col, loss_col])
                .group_by([parent_col, child_col])
                .agg([
                    pl.col(weight_col).sum().alias("exposure"),
                    (
                        (pl.col(weight_col) * pl.col(loss_col)).sum()
                        / pl.col(weight_col).sum()
                    ).alias("loss_rate"),
                ])
            )
            bs2 = BuhlmannStraub(truncate_a=self.truncate_a)
            bs2.fit(child_summary, group_col=parent_col, period_col=child_col,
                    loss_col="loss_rate", weight_col="exposure")
            level_results[parent_col] = LevelResult(
                parent_col, bs2.mu_hat_, bs2.v_hat_, bs2.a_hat_, bs2.k_, bs2.z_, bs2.premiums_)

        self._level_results = level_results
        self._bottom_premiums = self._compute_top_down_premiums(data, level_results)
        self._fitted = True
        return self

    @property
    def level_results_(self): self._check_fitted(); return self._level_results
    @property
    def premiums_(self): self._check_fitted(); return self._bottom_premiums

    def premiums_at(self, level):
        self._check_fitted()
        if level not in self.level_cols:
            raise ValueError(f"'{level}' is not one of the fitted levels: {self.level_cols}")
        return self._level_results[level].premiums

    def summary(self):
        self._check_fitted()
        print("Hierarchical Buhlmann-Straub Credibility Model")
        print("=" * 50)
        print(f"Levels: {' -> '.join(self.level_cols)}")
        for level in self.level_cols:
            lr = self._level_results[level]
            k_str = "inf" if np.isinf(lr.k) else f"{lr.k:.6g}"
            print(f"  {level}: mu={lr.mu_hat:.4g}, v={lr.v_hat:.4g}, a={lr.a_hat:.4g}, k={k_str}")

    def _validate_hierarchy(self, data):
        for depth in range(len(self.level_cols) - 1):
            parent = self.level_cols[depth]
            child = self.level_cols[depth + 1]
            parents_per_child = (
                data.select([parent, child]).unique()
                .group_by(child)
                .agg(pl.col(parent).n_unique().alias("n_parents"))
            )
            ambiguous = parents_per_child.filter(pl.col("n_parents") > 1)
            if ambiguous.height > 0:
                examples = ambiguous[child].to_list()[:3]
                raise ValueError(
                    f"Hierarchy is not strict at level '{child}' -> '{parent}': "
                    f"{ambiguous.height} child node(s) appear under multiple parents. "
                    f"Examples: {examples}"
                )

    def _compute_top_down_premiums(self, data, level_results):
        top_level = self.level_cols[0]
        top_lr = level_results[top_level]
        parent_premiums_df = top_lr.premiums.select([
            pl.col("group"),
            pl.col("credibility_premium").alias("parent_premium"),
        ])
        for depth in range(1, len(self.level_cols)):
            current_level = self.level_cols[depth]
            parent_level = self.level_cols[depth - 1]
            current_lr = level_results[current_level]
            child_to_parent = (
                data.select([current_level, parent_level])
                .unique()
                .rename({current_level: "group", parent_level: "parent_id"})
            )
            child_with_parent_premium = (
                child_to_parent
                .join(parent_premiums_df.rename({"group": "parent_id"}),
                      on="parent_id", how="left")
            )
            blended = (
                current_lr.premiums.select(["group", "Z", "observed_mean"])
                .join(child_with_parent_premium.select(["group", "parent_premium"]),
                      on="group", how="left")
                .with_columns(
                    (pl.col("Z") * pl.col("observed_mean")
                     + (1 - pl.col("Z")) * pl.col("parent_premium")).alias("blended_premium")
                )
                .select(["group", "blended_premium"])
                .rename({"blended_premium": "parent_premium"})
            )
            parent_premiums_df = blended

        bottom_level = self.level_cols[-1]
        bottom_lr = level_results[bottom_level]
        result = (
            bottom_lr.premiums.drop("credibility_premium")
            .join(parent_premiums_df.rename({"parent_premium": "credibility_premium"}),
                  on="group", how="left")
        )
        return result

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call .fit() first.")

    def __repr__(self):
        levels_str = " -> ".join(self.level_cols)
        suffix = "" if self._fitted else ", not fitted"
        return f"HierarchicalBuhlmannStraub(levels=[{levels_str}]{suffix})"
'''

INIT_SRC = '''
from .buhlmann_straub import BuhlmannStraub
from .hierarchical import HierarchicalBuhlmannStraub, LevelResult
__all__ = ["BuhlmannStraub", "HierarchicalBuhlmannStraub", "LevelResult"]
__version__ = "0.2.0"
'''

PYPROJECT_SRC = '''
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
[project]
name = "credibility"
version = "0.2.0"
requires-python = ">=3.9"
dependencies = ["numpy>=1.21", "polars>=0.20"]
[tool.hatch.build.targets.wheel]
packages = ["src/credibility"]
'''

# Write all files
files = {
    f"{pkg_dir}/src/credibility/_validation.py": VALIDATION_SRC,
    f"{pkg_dir}/src/credibility/buhlmann_straub.py": BS_SRC,
    f"{pkg_dir}/src/credibility/hierarchical.py": HIER_SRC,
    f"{pkg_dir}/src/credibility/__init__.py": INIT_SRC,
    f"{pkg_dir}/pyproject.toml": PYPROJECT_SRC,
}
for path, content in files.items():
    with open(path, "w") as f:
        f.write(content)

print(f"Written to {pkg_dir}")

result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", pkg_dir, "--quiet"],
    capture_output=True, text=True
)
if result.returncode != 0:
    print("Install error:", result.stderr[:500])
else:
    print("Package installed successfully")

sys.path.insert(0, f"{pkg_dir}/src")

# ============================================================
# Step 3: Run tests
# ============================================================
from credibility import BuhlmannStraub, HierarchicalBuhlmannStraub

print("\n" + "=" * 60)
print("RUNNING TESTS (credibility v0.2.0 — Polars-native)")
print("=" * 60)

passed = 0
failed = 0
errors = []


def check(name, condition, msg=""):
    global passed, failed
    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name}: {msg}")
        failed += 1
        errors.append(f"{name}: {msg}")


# --- Build Hachemeister dataset as Polars DataFrame ---
RATIOS = {
    1: [1738, 1642, 1794, 2051, 2079, 2234, 2032, 2035, 2115, 2262, 2267, 2517],
    2: [1364, 1408, 1597, 1444, 1342, 1675, 1470, 1448, 1464, 1831, 1612, 1471],
    3: [1759, 1685, 1479, 1763, 1674, 1517, 1448, 1464, 1634, 1519, 1656, 1657],
    4: [1223, 1146, 1010, 1247, 1492, 1290, 1332, 1349, 1393, 1396, 1677, 1716],
    5: [1456, 1499, 1609, 1741, 1482, 1572, 1606, 1573, 1613, 1741, 1659, 1685],
}
WEIGHTS = {
    1: [7861, 9251, 8706, 8792, 9593, 9077, 9470, 9364, 8742, 8921, 9145, 9800],
    2: [1622, 1742, 1523, 1515, 1622, 1602, 1764, 1861, 1698, 1765, 1721, 1765],
    3: [2592, 2466, 2321, 2127, 2122, 2339, 2291, 2145, 2289, 2203, 2219, 2419],
    4: [802, 900, 774, 836, 801, 796, 936, 814, 828, 805, 813, 854],
    5: [2755, 2668, 2910, 3097, 3295, 2976, 3197, 3198, 2976, 3070, 3089, 3481],
}
states, periods, ratios, weights = [], [], [], []
for state in range(1, 6):
    for period in range(1, 13):
        states.append(state)
        periods.append(period)
        ratios.append(float(RATIOS[state][period - 1]))
        weights.append(float(WEIGHTS[state][period - 1]))

hdf = pl.DataFrame({
    "state": states, "period": periods,
    "ratio": ratios, "weight": weights,
})

REF = {
    "mu_hat": 1832.816, "v_hat": 136793600.7, "a_hat": 100301.9, "k": 1363.818,
    "premiums": {1: 2067.394, 2: 1531.301, 3: 1616.943, 4: 1413.864, 5: 1613.778},
}


def get_premium(premiums_df, group_id, col):
    return premiums_df.filter(pl.col("group") == group_id)[col][0]


# === Hachemeister Benchmark ===
print("\n--- Hachemeister Benchmark ---")
bs = BuhlmannStraub()
bs.fit(hdf, group_col="state", period_col="period", loss_col="ratio", weight_col="weight")

check("premiums_ is pl.DataFrame", isinstance(bs.premiums_, pl.DataFrame))
check("z_ is pl.DataFrame", isinstance(bs.z_, pl.DataFrame))
check("mu_hat matches reference within 0.1%",
      abs(bs.mu_hat_ - REF["mu_hat"]) / REF["mu_hat"] < 0.001,
      f"got {bs.mu_hat_:.4f}")
check("v_hat matches reference within 0.5%",
      abs(bs.v_hat_ - REF["v_hat"]) / REF["v_hat"] < 0.005,
      f"got {bs.v_hat_:.2f}")
check("a_hat matches reference within 0.5%",
      abs(bs.a_hat_ - REF["a_hat"]) / REF["a_hat"] < 0.005,
      f"got {bs.a_hat_:.2f}")
check("k matches reference within 0.5%",
      abs(bs.k_ - REF["k"]) / REF["k"] < 0.005,
      f"got {bs.k_:.6f}")

for state, expected in REF["premiums"].items():
    actual = get_premium(bs.premiums_, state, "credibility_premium")
    check(f"state {state} premium within 0.2% of reference",
          abs(actual - expected) / expected < 0.002,
          f"got {actual:.3f}, expected {expected:.3f}")

z_vals = bs.z_["Z"]
check("all Z in [0, 1]", (z_vals >= 0).all() and (z_vals <= 1).all())
check("state 1 has highest Z",
      bs.z_.sort("Z", descending=True)["group"][0] == 1)
check("state 4 has lowest Z",
      bs.z_.sort("Z")["group"][0] == 4)
check("k = v/a", abs(bs.k_ - bs.v_hat_ / bs.a_hat_) < 1e-8)

for row in bs.premiums_.iter_rows(named=True):
    expected = row["Z"] * row["observed_mean"] + (1 - row["Z"]) * bs.mu_hat_
    check(f"group {row['group']}: premium = Z*Xbar + (1-Z)*mu",
          abs(row["credibility_premium"] - expected) < 1e-6)

print(f"\n  mu = {bs.mu_hat_:.3f}  (actuar: {REF['mu_hat']:.3f})")
print(f"  v  = {bs.v_hat_:.2f}  (actuar: {REF['v_hat']:.2f})")
print(f"  a  = {bs.a_hat_:.2f}  (actuar: {REF['a_hat']:.2f})")
print(f"  k  = {bs.k_:.6f}  (actuar: {REF['k']:.6f})")
print()
print(bs.premiums_.select(["group", "Z", "credibility_premium"]))

# === Edge Cases ===
print("\n--- Edge Cases ---")

df_same = pl.DataFrame({
    "group": ["A", "A", "A", "B", "B", "B"],
    "period": [1, 2, 3, 1, 2, 3],
    "loss": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "weight": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
})
bs2 = BuhlmannStraub(truncate_a=True)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    bs2.fit(df_same, group_col="group", period_col="period", loss_col="loss", weight_col="weight")
check("negative a_hat triggers warning", any("a_hat" in str(x.message) for x in w))
check("a_hat truncated to 0", bs2.a_hat_ == 0.0)
check("k = inf when a=0", np.isinf(bs2.k_))
check("all Z=0 when a=0", (bs2.z_["Z"] == 0.0).all())
check("all premiums = mu when Z=0",
      np.allclose(bs2.premiums_["credibility_premium"].to_numpy(), bs2.mu_hat_))

bs3 = BuhlmannStraub(truncate_a=False)
try:
    bs3.fit(df_same, group_col="group", period_col="period", loss_col="loss", weight_col="weight")
    check("truncate_a=False raises ValueError", False)
except ValueError as e:
    check("truncate_a=False raises ValueError", "a_hat" in str(e))

df_single = pl.DataFrame({
    "group": ["A", "B", "C"], "period": [1, 1, 1],
    "loss": [1.0, 1.2, 0.8], "weight": [100.0, 200.0, 150.0],
})
try:
    BuhlmannStraub().fit(df_single, group_col="group", period_col="period",
                         loss_col="loss", weight_col="weight")
    check("all single-period raises ValueError", False)
except ValueError as e:
    check("all single-period raises ValueError", "one period" in str(e))

df_one = pl.DataFrame({
    "group": ["A", "A", "A"], "period": [1, 2, 3],
    "loss": [1.0, 1.1, 0.9], "weight": [100.0, 110.0, 90.0],
})
try:
    BuhlmannStraub().fit(df_one, group_col="group", period_col="period",
                         loss_col="loss", weight_col="weight")
    check("single group raises ValueError", False)
except ValueError as e:
    check("single group raises ValueError", "2 groups" in str(e))

# === Equal Weights ===
print("\n--- Equal Weights ---")
df_eq = pl.DataFrame({
    "group": ["A", "A", "A", "B", "B"],
    "period": [1, 2, 3, 1, 2],
    "loss": [1.0, 1.5, 2.0, 0.5, 0.8],
    "weight": [1.0, 1.0, 1.0, 1.0, 1.0],
})
bs_eq = BuhlmannStraub()
bs_eq.fit(df_eq, group_col="group", period_col="period", loss_col="loss", weight_col="weight")
k = bs_eq.k_
z_a = bs_eq.z_.filter(pl.col("group") == "A")["Z"][0]
z_b = bs_eq.z_.filter(pl.col("group") == "B")["Z"][0]
check("equal weights Z_A = 3/(3+k)", abs(z_a - 3 / (3 + k)) < 1e-10)
check("equal weights Z_B = 2/(2+k)", abs(z_b - 2 / (2 + k)) < 1e-10)
check("A has higher Z than B", z_a > z_b)

# === Hierarchical Model ===
print("\n--- Hierarchical Model ---")
rng = np.random.default_rng(42)
base_loss = 0.65
region_effects = {"R1": 0.05, "R2": -0.05}
district_effects = {"R1_D1": 0.03, "R1_D2": -0.03, "R2_D1": 0.02, "R2_D2": -0.02}
r_list, d_list, s_list, p_list, lr_list, ex_list = [], [], [], [], [], []
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
                r_list.append(region); d_list.append(district); s_list.append(sector)
                p_list.append(period)
                lr_list.append(max(0.1, base_loss + r_eff + d_eff + s_eff + noise))
                ex_list.append(round(exposure))

hdf_hier = pl.DataFrame({
    "region": r_list, "district": d_list, "sector": s_list,
    "period": p_list, "loss_rate": lr_list, "exposure": ex_list,
})

model = HierarchicalBuhlmannStraub(level_cols=["region", "district", "sector"])
model.fit(hdf_hier, period_col="period", loss_col="loss_rate", weight_col="exposure")

check("hierarchical model fitted", model._fitted)
check("premiums_ is pl.DataFrame", isinstance(model.premiums_, pl.DataFrame))
check("level_results_ has 3 keys",
      set(model.level_results_.keys()) == {"region", "district", "sector"})
check("12 sector premiums", len(model.premiums_at("sector")) == 12)
check("4 district premiums", len(model.premiums_at("district")) == 4)
check("2 region premiums", len(model.premiums_at("region")) == 2)
check("premiums_ groups match premiums_at(sector)",
      set(model.premiums_["group"].to_list()) == set(model.premiums_at("sector")["group"].to_list()))

for level in ["region", "district", "sector"]:
    z_vals = model.level_results_[level].z["Z"]
    check(f"Z in [0,1] at {level}", (z_vals >= 0).all() and (z_vals <= 1).all())

premiums = model.premiums_
r1_mean = premiums.filter(pl.col("group").str.starts_with("R1"))["credibility_premium"].mean()
r2_mean = premiums.filter(pl.col("group").str.starts_with("R2"))["credibility_premium"].mean()
check("R1 sectors > R2 on average",
      r1_mean > r2_mean, f"R1={r1_mean:.4f}, R2={r2_mean:.4f}")

df_bad = pl.DataFrame({
    "region": ["R1", "R1", "R2", "R2"],
    "district": ["D1", "D1", "D1", "D1"],
    "period": [1, 2, 1, 2],
    "loss_rate": [0.5, 0.6, 0.7, 0.8],
    "exposure": [100.0, 100.0, 100.0, 100.0],
})
try:
    HierarchicalBuhlmannStraub(["region", "district"]).fit(
        df_bad, period_col="period", loss_col="loss_rate", weight_col="exposure")
    check("non-strict hierarchy raises ValueError", False)
except ValueError as e:
    check("non-strict hierarchy raises ValueError", "strict" in str(e))

try:
    HierarchicalBuhlmannStraub(["region"])
    check("single level raises ValueError", False)
except ValueError as e:
    check("single level raises ValueError", "two levels" in str(e))

model.summary()

# === Summary ===
print("\n" + "=" * 60)
print(f"RESULTS: {passed} passed, {failed} failed")
if errors:
    print("\nFailed tests:")
    for e in errors:
        print(f"  - {e}")
print("=" * 60)

try:
    summary = f"PASSED: {passed}, FAILED: {failed}"
    if errors:
        summary += " | Failures: " + "; ".join(errors[:3])
    dbutils.notebook.exit(summary)
except NameError:
    pass  # Not in Databricks environment
