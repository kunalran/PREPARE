"""
imputation_benchmark.py — Benchmark all 10 imputation strategies from
imputation_strategies.md against each other on expanded agmarknet data.

Pipeline:
  1. Load expanded CSVs from data2_expanded/
  2. Filter to dense mandi×year combos (≥50% data present)
  3. Mask 10% of known values as a held-out test set
  4. Apply each of the 10 methods independently
  5. Score RMSE / MAE / MAPE on imputed-vs-truth for masked cells
  6. Output summary to imputation/benchmark_results.md

Usage:  ./venv/bin/python imputation/imputation_benchmark.py
"""

import os
import sys
import warnings
import time
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data2_expanded")
OUT_DIR = Path("imputation")
MASK_FRACTION = 0.10
RANDOM_STATE = 42
TARGET_COLS = ["Modal_Price", "Arrival_Quantity"]

# Forward fill cap (days)
FFILL_CAP = 7
# Rolling window size
ROLLING_WINDOW = 15
# Spatial decay lambda
DECAY_LAMBDA = 0.15
# SVD components
SVD_RANK = 20
SVD_ITERS = 30


# ═════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING & PREPROCESSING
# ═════════════════════════════════════════════════════════════════════

def load_and_filter(csv_path: Path) -> pd.DataFrame:
    """Load one expanded CSV and filter to dense (≥50% present) mandi-year combos."""
    print(f"  Loading {csv_path.name}...")
    df = pd.read_csv(
        csv_path,
        parse_dates=["Date"],
        usecols=["State", "District", "Market", "Commodity", "Date",
                 "Day_of_Week", "Arrival_Quantity", "Modal_Price"],
        low_memory=False,
    )
    df = df.sort_values(["State", "Market", "Date"]).reset_index(drop=True)
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.dayofweek  # 0=Mon, 6=Sun

    # Compute data-present % per (State, Market, Year)
    grp = df.groupby(["State", "Market", "Year"])["Modal_Price"]
    n_total = grp.transform("size")
    n_present = grp.transform(lambda x: x.notna().sum())
    df["_pct_present"] = n_present / n_total * 100

    before = len(df)
    df = df[df["_pct_present"] >= 50].copy()
    df.drop(columns=["_pct_present"], inplace=True)
    after = len(df)

    n_mandis = df.groupby(["State", "Market"]).ngroups
    print(f"    Rows: {before:,} → {after:,} (dense ≥50%)  |  Mandis: {n_mandis}")
    return df


def mask_ground_truth(df: pd.DataFrame, target_col: str) -> tuple:
    """
    Randomly mask MASK_FRACTION of the known (non-NaN) values for `target_col`.
    Returns (masked_df, truth_series) where truth_series has the hidden values
    and NaN everywhere else.
    """
    rng = np.random.RandomState(RANDOM_STATE)
    known_idx = df.index[df[target_col].notna()]
    mask_size = int(len(known_idx) * MASK_FRACTION)
    mask_idx = rng.choice(known_idx, size=mask_size, replace=False)

    truth = pd.Series(np.nan, index=df.index, name=target_col)
    truth.loc[mask_idx] = df.loc[mask_idx, target_col]

    masked_df = df.copy()
    masked_df.loc[mask_idx, target_col] = np.nan
    return masked_df, truth


# ═════════════════════════════════════════════════════════════════════
# SECTION 2 — 10 IMPUTATION METHODS  (each returns a full Series)
# ═════════════════════════════════════════════════════════════════════

# --- Method 1: Capped Forward Fill (LOCF) ---
def impute_capped_ffill(df: pd.DataFrame, col: str) -> pd.Series:
    """Forward fill within each mandi, capped at FFILL_CAP days."""
    result = df[col].copy()
    for _, idx in df.groupby(["State", "Market"]).groups.items():
        grp = df.loc[idx].sort_values("Date")
        filled = grp[col].ffill(limit=FFILL_CAP)
        result.loc[grp.index] = filled.values
    return result


# --- Method 2: Simple Moving Average (Rolling Window) ---
def impute_rolling_mean(df: pd.DataFrame, col: str) -> pd.Series:
    """Centered rolling window mean, per mandi, window=ROLLING_WINDOW."""
    result = df[col].copy()
    for _, idx in df.groupby(["State", "Market"]).groups.items():
        grp = df.loc[idx].sort_values("Date")
        series = grp[col]
        rolling = series.rolling(window=ROLLING_WINDOW, center=True, min_periods=1).mean()
        # Only fill where original was NaN
        fill_mask = series.isna()
        filled = series.copy()
        filled[fill_mask] = rolling[fill_mask]
        result.loc[grp.index] = filled.values
    return result


# --- Method 3: State-Level Daily Median ---
def impute_state_median(df: pd.DataFrame, col: str) -> pd.Series:
    """Fill NaN with the daily median of the same State+Crop."""
    result = df[col].copy()
    medians = df.groupby(["Date", "Commodity", "State"])[col].transform("median")
    mask = result.isna()
    result[mask] = medians[mask]
    return result


# --- Method 4: Historical Seasonality (Month + DayOfWeek) ---
def impute_seasonal(df: pd.DataFrame, col: str) -> pd.Series:
    """Fill NaN with mandi's historical Month+DayOfWeek median."""
    result = df[col].copy()
    medians = df.groupby(["State", "Market", "Month", "DayOfWeek"])[col].transform("median")
    mask = result.isna()
    result[mask] = medians[mask]
    return result


# --- Method 5: Global Crop Median ---
def impute_global_median(df: pd.DataFrame, col: str) -> pd.Series:
    """Fill all NaN with the overall crop median."""
    result = df[col].copy()
    med = df[col].median()
    result.fillna(med, inplace=True)
    return result


# --- Method 6: Forward Fill with Spatial Decay ---
def impute_ffill_decay(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Blend last-known value with state daily average using exponential decay.
    w = exp(-lambda * t),  imputed = w*P + (1-w)*S
    """
    result = df[col].copy()
    # Pre-compute state daily averages
    state_daily = df.groupby(["Date", "State"])[col].transform("mean")

    for _, idx in df.groupby(["State", "Market"]).groups.items():
        grp = df.loc[idx].sort_values("Date")
        series = grp[col].copy()
        s_daily = state_daily.loc[grp.index]

        last_val = np.nan
        days_since = 0
        filled = series.copy()

        for i, (ix, val) in enumerate(series.items()):
            if pd.notna(val):
                last_val = val
                days_since = 0
            else:
                days_since += 1
                if pd.notna(last_val):
                    w = np.exp(-DECAY_LAMBDA * days_since)
                    s = s_daily.loc[ix] if pd.notna(s_daily.loc[ix]) else last_val
                    filled.loc[ix] = w * last_val + (1 - w) * s
                elif pd.notna(s_daily.loc[ix]):
                    filled.loc[ix] = s_daily.loc[ix]

        result.loc[grp.index] = filled.values
    return result


# --- Method 7: Calendar & Seasonality Indexing (Day-of-Week Focus) ---
def impute_dow_ratio(df: pd.DataFrame, col: str) -> pd.Series:
    """
    For each mandi, compute the ratio of each weekday's avg to the
    overall mandi avg. Impute missing values as: weekly_context * dow_ratio.
    """
    result = df[col].copy()

    for _, idx in df.groupby(["State", "Market"]).groups.items():
        grp = df.loc[idx].sort_values("Date")
        series = grp[col]
        dow = grp["DayOfWeek"]
        overall_mean = series.mean()

        if pd.isna(overall_mean) or overall_mean == 0:
            continue

        # Per-weekday ratio
        dow_means = series.groupby(dow).mean()
        dow_ratios = dow_means / overall_mean
        dow_ratios = dow_ratios.fillna(1.0)

        # Weekly rolling context (7-day centered mean of known values)
        weekly = series.rolling(7, center=True, min_periods=1).mean()

        filled = series.copy()
        for ix in series.index[series.isna()]:
            d = dow.loc[ix]
            ctx = weekly.loc[ix]
            if pd.notna(ctx):
                ratio = dow_ratios.get(d, 1.0)
                filled.loc[ix] = ctx * ratio

        result.loc[grp.index] = filled.values
    return result


# --- Method 8: 4-Step Smoothing & Spline Pipeline ---
def impute_spline_pipeline(df: pd.DataFrame, col: str) -> pd.Series:
    """
    1. Outlier flagging (>6x or <1/6 of prev week avg)
    2. Year-wise cubic spline on mandis with >50% data
    3. Window smoothing (±15% of 15-day extrema)
    4. Linear interpolation for leftovers
    """
    result = df[col].copy()

    for _, idx in df.groupby(["State", "Market"]).groups.items():
        grp = df.loc[idx].sort_values("Date")
        series = grp[col].copy()
        dates_num = (grp["Date"] - grp["Date"].min()).dt.days.values

        # Step 1: Outlier flagging
        for i in range(7, len(series)):
            val = series.iloc[i]
            if pd.isna(val):
                continue
            prev_week = series.iloc[max(0, i - 7):i].dropna()
            if len(prev_week) == 0:
                continue
            avg = prev_week.mean()
            if avg > 0 and (val > 6 * avg or val < avg / 6):
                series.iloc[i] = np.nan

        # Step 2: Year-wise cubic spline
        for year in grp["Date"].dt.year.unique():
            yr_mask = grp["Date"].dt.year.values == year
            yr_series = series[yr_mask]
            yr_dates = dates_num[yr_mask]
            known = yr_series.notna()
            if known.sum() < 4:  # need ≥4 points for cubic spline
                continue
            pct_known = known.sum() / len(yr_series)
            if pct_known < 0.5:
                continue
            try:
                cs = CubicSpline(yr_dates[known.values], yr_series[known].values,
                                 extrapolate=False)
                splined = cs(yr_dates)
                nan_mask = yr_series.isna()
                yr_series_filled = yr_series.copy()
                yr_series_filled[nan_mask] = splined[nan_mask.values]
                # Clamp negatives
                yr_series_filled = yr_series_filled.clip(lower=0)
                series[yr_mask] = yr_series_filled
            except Exception:
                pass

        # Step 3: Window-based smoothing check
        for i in range(len(series)):
            val = series.iloc[i]
            if pd.isna(val) or grp[col].iloc[i] == val:
                continue  # skip original values and NaN
            window = series.iloc[max(0, i - 7):i + 8].dropna()
            if len(window) < 2:
                continue
            lo, hi = window.min(), window.max()
            if hi > 0 and (val > hi * 1.15 or val < lo * 0.85):
                series.iloc[i] = np.nan

        # Step 4: Linear interpolation
        series = series.interpolate(method="linear", limit_direction="both")

        result.loc[grp.index] = series.values
    return result


# --- Method 9: Random Forest (MissForest-style) ---
def impute_random_forest(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Train a Random Forest on known values using encoded features.
    Features: State, Market, Month, DayOfWeek, Year, plus lag-7-day mean.
    """
    result = df[col].copy()
    work = df[["State", "Market", "Month", "DayOfWeek", "Year", col]].copy()

    # Encode categoricals
    le_state = LabelEncoder()
    le_market = LabelEncoder()
    work["State_enc"] = le_state.fit_transform(work["State"])
    work["Market_enc"] = le_market.fit_transform(work["Market"])

    # Lag feature: 7-day rolling mean per mandi
    lag_means = df.groupby(["State", "Market"])[col].transform(
        lambda x: x.rolling(7, min_periods=1).mean().shift(1)
    )
    work["lag7_mean"] = lag_means.fillna(df[col].median())

    features = ["State_enc", "Market_enc", "Month", "DayOfWeek", "Year", "lag7_mean"]

    known = work[col].notna()
    missing = work[col].isna()

    if known.sum() < 100 or missing.sum() == 0:
        return result

    X_train = work.loc[known, features].values
    y_train = work.loc[known, col].values
    X_pred = work.loc[missing, features].values

    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rf.fit(X_train, y_train)
    preds = rf.predict(X_pred)
    preds = np.clip(preds, 0, None)  # no negatives

    result.loc[missing[missing].index] = preds
    return result


# --- Method 10: Matrix Factorization (Truncated SVD / SoftImpute) ---
def impute_svd(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Pivot to Date × Mandi matrix, iterative SVD (SoftImpute style).
    """
    result = df[col].copy()

    # Create mandi key
    df_work = df[["State", "Market", "Date", col]].copy()
    df_work["mandi_key"] = df_work["State"] + "||" + df_work["Market"]

    # Pivot
    pivot = df_work.pivot_table(index="Date", columns="mandi_key", values=col, aggfunc="first")

    # SoftImpute-style iterative SVD
    mat = pivot.values.copy().astype(float)
    observed = ~np.isnan(mat)
    # Initialize NaN with column means
    col_means = np.nanmean(mat, axis=0)
    col_means = np.where(np.isnan(col_means), 0, col_means)
    for j in range(mat.shape[1]):
        mat[np.isnan(mat[:, j]), j] = col_means[j]

    rank = min(SVD_RANK, min(mat.shape) - 1)
    if rank < 1:
        return result

    for _ in range(SVD_ITERS):
        U, s, Vt = np.linalg.svd(mat, full_matrices=False)
        s_trunc = np.zeros_like(s)
        s_trunc[:rank] = s[:rank]
        approx = U @ np.diag(s_trunc) @ Vt
        # Only update unobserved entries
        mat[~observed] = approx[~observed]

    # Clip negatives
    mat = np.clip(mat, 0, None)

    # Map back
    filled_pivot = pd.DataFrame(mat, index=pivot.index, columns=pivot.columns)

    for mandi_key in filled_pivot.columns:
        parts = mandi_key.split("||")
        if len(parts) != 2:
            continue
        state, market = parts
        idx = df.index[(df["State"] == state) & (df["Market"] == market)]
        dates = df.loc[idx, "Date"]
        for ix, d in zip(idx, dates):
            if pd.isna(result.loc[ix]) and d in filled_pivot.index:
                result.loc[ix] = filled_pivot.loc[d, mandi_key]

    return result


# ═════════════════════════════════════════════════════════════════════
# SECTION 3 — EVALUATION ENGINE
# ═════════════════════════════════════════════════════════════════════

def score(truth: pd.Series, imputed: pd.Series) -> dict:
    """Compute RMSE, MAE, MAPE on indices where truth is known."""
    mask = truth.notna()
    t = truth[mask].values
    p = imputed[mask].values

    # Only evaluate where imputation actually produced a value
    valid = ~np.isnan(p)
    if valid.sum() == 0:
        return {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan,
                "Coverage": 0.0, "N_Evaluated": 0}

    t = t[valid]
    p = p[valid]

    rmse = np.sqrt(np.mean((t - p) ** 2))
    mae = np.mean(np.abs(t - p))
    # MAPE: avoid division by zero
    nonzero = t != 0
    if nonzero.sum() > 0:
        mape = np.mean(np.abs((t[nonzero] - p[nonzero]) / t[nonzero])) * 100
    else:
        mape = np.nan

    coverage = valid.sum() / mask.sum() * 100  # % of masked cells imputed

    return {"RMSE": round(rmse, 2), "MAE": round(mae, 2),
            "MAPE": round(mape, 2), "Coverage": round(coverage, 1),
            "N_Evaluated": int(valid.sum())}


# All methods registry
METHODS = OrderedDict([
    ("1_Capped_FFill",         impute_capped_ffill),
    ("2_Rolling_Mean",         impute_rolling_mean),
    ("3_State_Median",         impute_state_median),
    ("4_Seasonal",             impute_seasonal),
    ("5_Global_Median",        impute_global_median),
    ("6_FFill_Decay",          impute_ffill_decay),
    ("7_DOW_Ratio",            impute_dow_ratio),
    ("8_Spline_Pipeline",      impute_spline_pipeline),
    ("9_Random_Forest",        impute_random_forest),
    ("10_SVD_Matrix",          impute_svd),
])


def benchmark_one_crop(csv_path: Path) -> list[dict]:
    """Run the full benchmark for one crop file."""
    crop = csv_path.stem.replace("agmarknet_", "").replace("_data_expanded", "").title()
    print(f"\n{'━' * 65}")
    print(f"  BENCHMARKING: {crop}")
    print(f"{'━' * 65}")

    df = load_and_filter(csv_path)
    if len(df) == 0:
        print(f"  ⚠ No dense data for {crop}, skipping.")
        return []

    results = []

    for target_col in TARGET_COLS:
        print(f"\n  ── Target: {target_col} ──")

        # Check we have enough known values
        n_known = df[target_col].notna().sum()
        if n_known < 100:
            print(f"    ⚠ Only {n_known} known values, skipping.")
            continue

        masked_df, truth = mask_ground_truth(df, target_col)
        n_masked = truth.notna().sum()
        print(f"    Known: {n_known:,} | Masked: {n_masked:,} ({MASK_FRACTION*100:.0f}%)")

        for method_name, method_fn in METHODS.items():
            t0 = time.time()
            try:
                imputed = method_fn(masked_df, target_col)
                elapsed = time.time() - t0
                metrics = score(truth, imputed)
                metrics["Method"] = method_name
                metrics["Crop"] = crop
                metrics["Target"] = target_col
                metrics["Time_s"] = round(elapsed, 1)
                results.append(metrics)
                print(f"    {method_name:<25}  RMSE={metrics['RMSE']:>10}  "
                      f"MAE={metrics['MAE']:>10}  MAPE={metrics['MAPE']:>7}%  "
                      f"Cov={metrics['Coverage']:>5}%  ({elapsed:.1f}s)")
            except Exception as e:
                elapsed = time.time() - t0
                print(f"    {method_name:<25}  ❌ ERROR: {e}  ({elapsed:.1f}s)")
                results.append({
                    "Method": method_name, "Crop": crop, "Target": target_col,
                    "RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan,
                    "Coverage": 0.0, "N_Evaluated": 0, "Time_s": round(elapsed, 1),
                })

    return results


# ═════════════════════════════════════════════════════════════════════
# SECTION 4 — OUTPUT
# ═════════════════════════════════════════════════════════════════════

def write_results(all_results: list[dict]):
    """Write benchmark results to CSV and markdown."""
    OUT_DIR.mkdir(exist_ok=True)
    results_df = pd.DataFrame(all_results)
    results_df = results_df[["Crop", "Target", "Method", "RMSE", "MAE", "MAPE",
                              "Coverage", "N_Evaluated", "Time_s"]]

    # Save CSV
    csv_path = OUT_DIR / "benchmark_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n  CSV saved: {csv_path}")

    # Build Markdown
    md = []
    md.append("# Imputation Benchmark Results")
    md.append("")
    md.append("> Auto-generated by `imputation_benchmark.py`.")
    md.append(f"> Masked {MASK_FRACTION*100:.0f}% of known values, evaluated imputed vs ground truth.")
    md.append(f"> Only mandi×year combos with ≥50% data were included.")
    md.append("")

    # Overall summary: average across all crops, per target
    for target in TARGET_COLS:
        sub = results_df[results_df["Target"] == target]
        if sub.empty:
            continue

        md.append(f"## {target}")
        md.append("")

        # Per-crop tables
        for crop in sub["Crop"].unique():
            crop_sub = sub[sub["Crop"] == crop].sort_values("RMSE")
            md.append(f"### {crop}")
            md.append("")
            md.append("| Rank | Method | RMSE | MAE | MAPE (%) | Coverage (%) | Time (s) |")
            md.append("|------|--------|------|-----|----------|--------------|----------|")
            for rank, (_, row) in enumerate(crop_sub.iterrows(), 1):
                md.append(f"| {rank} | {row['Method']} | {row['RMSE']} | "
                          f"{row['MAE']} | {row['MAPE']} | {row['Coverage']} | {row['Time_s']} |")
            md.append("")

        # Cross-crop average
        avg = sub.groupby("Method")[["RMSE", "MAE", "MAPE", "Coverage", "Time_s"]].mean()
        avg = avg.sort_values("RMSE").round(2)
        md.append(f"### Cross-Crop Average — {target}")
        md.append("")
        md.append("| Rank | Method | Avg RMSE | Avg MAE | Avg MAPE (%) | Avg Coverage (%) | Avg Time (s) |")
        md.append("|------|--------|----------|---------|--------------|------------------|--------------|")
        for rank, (method, row) in enumerate(avg.iterrows(), 1):
            md.append(f"| {rank} | {method} | {row['RMSE']} | {row['MAE']} | "
                      f"{row['MAPE']} | {row['Coverage']} | {row['Time_s']} |")
        md.append("")

    md_path = OUT_DIR / "benchmark_results.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"  Markdown saved: {md_path}")


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  IMPUTATION STRATEGY BENCHMARK")
    print(f"  Mask fraction: {MASK_FRACTION*100:.0f}%  |  Methods: {len(METHODS)}")
    print("=" * 65)

    csv_files = sorted(DATA_DIR.glob("*_expanded.csv"))
    if not csv_files:
        print(f"No expanded CSVs in {DATA_DIR}/. Run expand_data.py first.")
        sys.exit(1)

    all_results = []
    for csv_path in csv_files:
        crop_results = benchmark_one_crop(csv_path)
        all_results.extend(crop_results)

    if all_results:
        write_results(all_results)
        print(f"\n{'=' * 65}")
        print("  BENCHMARK COMPLETE")
        print(f"{'=' * 65}")
    else:
        print("No results generated.")


if __name__ == "__main__":
    main()
