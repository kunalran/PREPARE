"""
data4_eda_and_impute.py
───────────────────────
1. EDA on data4/ (expanded + weather + location data)
2. Filter to mandis with ≥50% Modal_Price data per year; drop the rest
3. Drop Arrival_Quantity column
4. Impute Modal_Price using DOW_Ratio (Method 7)
5. Save imputed files to data4_imputed/
6. Evaluate accuracy via 10% masked test and output .md report

Usage:  ./venv/bin/python data4_eda_and_impute.py
"""

import time, warnings, sys
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = Path("data4")
OUT_DIR  = Path("data4_imputed")
REPORT   = Path("data4_imputed/imputation_report.md")
SEEDS    = [42, 123, 456, 789, 2025]
MASK_FRACTION = 0.10
TARGET = "Modal_Price"

WEATHER_COLS = ["t", "tp", "ssr", "r"]
LOCATION_COLS = ["latitude", "longitude"]


# ═══════════════════════════════════════════════════════════════
# EDA
# ═══════════════════════════════════════════════════════════════

def eda_one_crop(df: pd.DataFrame, crop: str) -> str:
    """Return markdown EDA for one crop."""
    lines = []
    def p(s=""): lines.append(s); print(s)

    p(f"## {crop}")
    p()

    # Overview
    n_mandis = df.groupby(["State","Market"]).ngroups
    n_states = df["State"].nunique()
    dmin, dmax = df["Date"].min(), df["Date"].max()
    n_days = df["Date"].nunique()
    total = len(df)
    price_present = df[TARGET].notna().sum()
    price_missing = df[TARGET].isna().sum()
    price_pct = round(price_missing/total*100, 1)
    arr_present = df["Arrival_Quantity"].notna().sum()

    p("### Overview")
    p()
    p(f"| Metric | Value |")
    p(f"|--------|-------|")
    p(f"| Total rows | {total:,} |")
    p(f"| Unique mandis | {n_mandis:,} |")
    p(f"| Unique states | {n_states} |")
    p(f"| Date range | {dmin} to {dmax} ({n_days} days) |")
    p(f"| Modal_Price present | {price_present:,} ({round(price_present/total*100,1)}%) |")
    p(f"| Modal_Price missing | {price_missing:,} ({price_pct}%) |")
    p(f"| Arrival_Quantity present | {arr_present:,} ({round(arr_present/total*100,1)}%) |")
    p()

    # Weather columns
    p("### Weather & Location Data")
    p()
    p("| Column | Description | Non-null | Mean | Min | Max | Std |")
    p("|--------|------------|----------|------|-----|-----|-----|")
    for col, desc in [("latitude","Latitude (°)"), ("longitude","Longitude (°)"),
                       ("t","Temperature (°C)"), ("tp","Total Precipitation (m)"),
                       ("ssr","Surface Solar Radiation (MJ/m²)"), ("r","Relative Humidity (%)")]:
        s = df[col].dropna()
        nn = len(s)
        p(f"| {col} | {desc} | {nn:,} ({round(nn/total*100,1)}%) | {s.mean():.2f} | {s.min():.2f} | {s.max():.2f} | {s.std():.2f} |")
    p()

    # Missing weather check
    weather_null = sum(df[c].isna().sum() for c in WEATHER_COLS)
    loc_null = sum(df[c].isna().sum() for c in LOCATION_COLS)
    p(f"- **Weather NaN total:** {weather_null:,} across {len(WEATHER_COLS)} columns")
    p(f"- **Location NaN total:** {loc_null:,} across {len(LOCATION_COLS)} columns")
    p()

    # Price by day of week
    p("### Modal_Price Missing by Day of Week")
    p()
    df_tmp = df.copy()
    df_tmp["dow"] = df_tmp["Date"].dt.dayofweek
    dow_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    dow_stats = df_tmp.groupby("dow")[TARGET].agg(total="size", present=lambda x: x.notna().sum())
    dow_stats["missing_pct"] = ((dow_stats["total"] - dow_stats["present"]) / dow_stats["total"] * 100).round(1)

    p("| Day | Total | Present | % Missing |")
    p("|-----|-------|---------|-----------|")
    for i in range(7):
        row = dow_stats.loc[i]
        p(f"| {dow_names[i]} | {int(row['total']):,} | {int(row['present']):,} | {row['missing_pct']}% |")
    p()

    # Per-year density
    p("### Per-Year Data Density")
    p()
    df_tmp["Year"] = df_tmp["Date"].dt.year
    yr_stats = df_tmp.groupby("Year")[TARGET].agg(total="size", present=lambda x: x.notna().sum())
    yr_stats["mandis"] = df_tmp.groupby("Year").apply(lambda x: x.groupby(["State","Market"]).ngroups)
    yr_stats["pct"] = (yr_stats["present"]/yr_stats["total"]*100).round(1)

    p("| Year | Mandis | Total Rows | Present | % Present |")
    p("|------|--------|-----------|---------|-----------|")
    for yr, row in yr_stats.iterrows():
        p(f"| {yr} | {int(row['mandis'])} | {int(row['total']):,} | {int(row['present']):,} | {row['pct']}% |")
    p()

    # Density filter preview
    grp = df_tmp.groupby(["State","Market","Year"])[TARGET]
    pct_df = pd.DataFrame({
        "total": grp.size(),
        "present": grp.apply(lambda x: x.notna().sum()),
    }).reset_index()
    pct_df["pct"] = pct_df["present"] / pct_df["total"] * 100
    dense = pct_df[pct_df["pct"] >= 50]
    n_dense_mandis = dense.groupby(["State","Market"]).ngroups if len(dense) > 0 else 0
    n_dense_rows = 0
    if len(dense) > 0:
        dense_keys = dense[["State","Market","Year"]].drop_duplicates()
        for _, rr in dense_keys.iterrows():
            n_dense_rows += ((df["State"]==rr["State"])&(df["Market"]==rr["Market"])&
                             (df_tmp["Year"]==rr["Year"])).sum()

    p("### ≥50% Density Filter Preview")
    p()
    p(f"- **Mandis passing ≥50% filter:** {n_dense_mandis:,} / {n_mandis:,}")
    p(f"- **Rows after filter:** ~{n_dense_rows:,} / {total:,}")
    p()
    p("---")
    p()

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# FILTERING & IMPUTATION
# ═══════════════════════════════════════════════════════════════

def filter_dense(df, min_pct=50):
    """Keep only mandi×year combos with ≥min_pct% Modal_Price present."""
    df["Year"] = df["Date"].dt.year
    grp = df.groupby(["State","Market","Year"])[TARGET]
    n_total = grp.transform("size")
    n_present = grp.transform(lambda x: x.notna().sum())
    df["_pct"] = n_present / n_total * 100
    df = df[df["_pct"] >= min_pct].copy()
    df.drop(columns=["_pct"], inplace=True)
    return df


def impute_dow_ratio(df, col):
    """DOW_Ratio imputation (Method 7)."""
    result = df[col].copy()
    df_tmp = df.copy()
    df_tmp["DayOfWeek"] = df_tmp["Date"].dt.dayofweek

    for _, idx in df_tmp.groupby(["State","Market"]).groups.items():
        grp = df_tmp.loc[idx].sort_values("Date")
        s = grp[col]
        dow = grp["DayOfWeek"]
        om = s.mean()
        if pd.isna(om) or om == 0:
            continue
        dr = (s.groupby(dow).mean() / om).fillna(1.0)
        wk = s.rolling(7, center=True, min_periods=1).mean()
        filled = s.copy()
        for ix in s.index[s.isna()]:
            ctx = wk.loc[ix]
            if pd.notna(ctx):
                filled.loc[ix] = ctx * dr.get(dow.loc[ix], 1.0)
        result.loc[grp.index] = filled.values
    return result


def mask_and_score(df, col, seed):
    """Mask 10% of known values, impute, score."""
    rng = np.random.RandomState(seed)
    known = df.index[df[col].notna()]
    mask_idx = rng.choice(known, size=int(len(known)*MASK_FRACTION), replace=False)
    truth = pd.Series(np.nan, index=df.index)
    truth.loc[mask_idx] = df.loc[mask_idx, col]
    masked = df.copy()
    masked.loc[mask_idx, col] = np.nan

    imputed = impute_dow_ratio(masked, col)

    mask = truth.notna()
    t, p_arr = truth[mask].values, imputed[mask].values
    valid = ~np.isnan(p_arr)
    if valid.sum() == 0:
        return None
    t, p_arr = t[valid], p_arr[valid]
    errors = np.abs(t - p_arr)

    rmse = np.sqrt(np.mean((t-p_arr)**2))
    mae = np.mean(errors)
    median_ae = np.median(errors)
    nz = t != 0
    mape = np.mean(np.abs((t[nz]-p_arr[nz])/t[nz]))*100 if nz.sum()>0 else np.nan
    p90 = np.percentile(errors, 90)
    p99 = np.percentile(errors, 99)
    cov = valid.sum()/mask.sum()*100

    return {"RMSE": rmse, "MAE": mae, "MedianAE": median_ae, "MAPE": mape,
            "P90": p90, "P99": p99, "Coverage": cov, "N_Masked": int(mask.sum()),
            "N_Evaluated": int(valid.sum())}


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  DATA4 — EDA + IMPUTATION PIPELINE")
    print("=" * 70)

    csv_files = sorted(DATA_DIR.glob("*_final.csv"))
    if not csv_files:
        print(f"No files in {DATA_DIR}/"); sys.exit(1)

    OUT_DIR.mkdir(exist_ok=True)

    eda_md = []
    eda_md.append("# Data4 — EDA & Imputation Report")
    eda_md.append("")
    eda_md.append("> Data4 contains expanded date×mandi grids enriched with **weather** (temperature,")
    eda_md.append("> precipitation, solar radiation, humidity) and **location** (lat/lon) data.")
    eda_md.append("> This report covers EDA findings and DOW_Ratio imputation accuracy.")
    eda_md.append("")

    # ── Part 1: EDA ──
    eda_md.append("# Part 1: Exploratory Data Analysis")
    eda_md.append("")

    crop_data = []
    for csv_path in csv_files:
        crop = csv_path.stem.replace("agmarknet_","").replace("_data_final","").title()
        print(f"\n{'━'*60}")
        print(f"  EDA: {crop}")
        print(f"{'━'*60}")

        df = pd.read_csv(csv_path, parse_dates=["Date"], low_memory=False)
        eda_section = eda_one_crop(df, crop)
        eda_md.append(eda_section)
        crop_data.append((csv_path, crop, df))

    # ── Part 2: Filter + Drop + Impute ──
    eda_md.append("# Part 2: Imputation Pipeline")
    eda_md.append("")
    eda_md.append("**Steps:**")
    eda_md.append("1. Filter to mandis with ≥50% Modal_Price data per year")
    eda_md.append("2. Drop `Arrival_Quantity` and `Arrival_Unit` columns")
    eda_md.append("3. Impute Modal_Price using **DOW_Ratio** (Method 7)")
    eda_md.append("4. Evaluate accuracy: mask 10% of known values × 5 random seeds")
    eda_md.append("")

    imputation_results = []

    for csv_path, crop, df in crop_data:
        print(f"\n{'━'*60}")
        print(f"  IMPUTING: {crop}")
        print(f"{'━'*60}")

        # Filter
        before = len(df)
        n_mandis_before = df.groupby(["State","Market"]).ngroups
        df = filter_dense(df, 50)
        after = len(df)
        n_mandis_after = df.groupby(["State","Market"]).ngroups

        if after == 0:
            print(f"  ⚠ No dense mandis for {crop}, skipping.")
            eda_md.append(f"### {crop}")
            eda_md.append(f"⚠ No mandis with ≥50% data — skipped.")
            eda_md.append("")
            continue

        print(f"  Filtered: {before:,} → {after:,} rows  |  {n_mandis_before:,} → {n_mandis_after:,} mandis")

        # Drop Arrival_Quantity
        cols_to_drop = [c for c in ["Arrival_Quantity", "Arrival_Unit"] if c in df.columns]
        df = df.drop(columns=cols_to_drop)
        print(f"  Dropped: {cols_to_drop}")

        # Accuracy evaluation (mask & score across seeds)
        n_known = df[TARGET].notna().sum()
        print(f"  Known prices: {n_known:,}  |  Running accuracy test ({len(SEEDS)} seeds)...")

        seed_scores = []
        for seed in SEEDS:
            s = mask_and_score(df, TARGET, seed)
            if s: seed_scores.append(s)

        if seed_scores:
            avg = {k: round(np.mean([s[k] for s in seed_scores]), 2) for k in seed_scores[0]}
            std_rmse = round(np.std([s["RMSE"] for s in seed_scores]), 2)
            print(f"  RMSE: {avg['RMSE']}±{std_rmse}  MAE: {avg['MAE']}  MAPE: {avg['MAPE']}%  Coverage: {avg['Coverage']}%")
        else:
            avg = None

        # Now do the actual imputation on the full dataset
        t0 = time.time()
        df[TARGET] = impute_dow_ratio(df, TARGET)
        elapsed = round(time.time() - t0, 1)

        remaining_nan = df[TARGET].isna().sum()
        total_after = len(df)
        fill_pct = round((total_after - remaining_nan) / total_after * 100, 2)

        print(f"  Imputed in {elapsed}s  |  Remaining NaN: {remaining_nan:,} / {total_after:,} ({round(remaining_nan/total_after*100,1)}%)")

        # Drop Year helper column if present
        if "Year" in df.columns:
            df = df.drop(columns=["Year"])

        # Save
        out_path = OUT_DIR / csv_path.name.replace("_final", "_imputed")
        df.to_csv(out_path, index=False)
        out_size = out_path.stat().st_size / 1e6
        print(f"  Saved: {out_path} ({out_size:.1f} MB)")

        # Collect results
        imputation_results.append({
            "Crop": crop, "Mandis_Before": n_mandis_before,
            "Mandis_After": n_mandis_after, "Rows_Before": before,
            "Rows_After": after, "Known_Prices": n_known,
            "Remaining_NaN": remaining_nan, "Fill_Pct": fill_pct,
            "Time_s": elapsed, "Scores": avg, "RMSE_std": std_rmse if seed_scores else 0,
        })

    # ── Build report tables ──
    eda_md.append("## Summary of Filtering & Imputation")
    eda_md.append("")
    eda_md.append("| Crop | Mandis (before→after) | Rows (before→after) | Known Prices | Filled % | Remaining NaN | Time |")
    eda_md.append("|------|-----------------------|---------------------|--------------|----------|---------------|------|")
    for r in imputation_results:
        eda_md.append(f"| {r['Crop']} | {r['Mandis_Before']:,}→{r['Mandis_After']:,} | "
                      f"{r['Rows_Before']:,}→{r['Rows_After']:,} | {r['Known_Prices']:,} | "
                      f"{r['Fill_Pct']}% | {r['Remaining_NaN']:,} | {r['Time_s']}s |")
    eda_md.append("")

    eda_md.append("## Imputation Accuracy (DOW_Ratio, Method 7)")
    eda_md.append("")
    eda_md.append(f"> Evaluated by masking {MASK_FRACTION*100:.0f}% of known Modal_Price values across "
                  f"**{len(SEEDS)} random seeds**, then comparing imputed values to ground truth.")
    eda_md.append("")
    eda_md.append("| Crop | RMSE (±std) | MAE | Median AE | MAPE (%) | P90 Error | P99 Error | Coverage |")
    eda_md.append("|------|-------------|-----|-----------|----------|-----------|-----------|----------|")
    for r in imputation_results:
        s = r["Scores"]
        if s:
            eda_md.append(f"| {r['Crop']} | {s['RMSE']}±{r['RMSE_std']} | {s['MAE']} | "
                          f"{s['MedianAE']} | {s['MAPE']} | {s['P90']} | {s['P99']} | {s['Coverage']}% |")
    eda_md.append("")

    eda_md.append("## Output Files")
    eda_md.append("")
    eda_md.append("All imputed files saved to `data4_imputed/`:")
    eda_md.append("")
    for r in imputation_results:
        fname = f"agmarknet_{r['Crop'].lower()}_data_imputed.csv"
        eda_md.append(f"- `{fname}` — {r['Mandis_After']:,} mandis, {r['Rows_After']:,} rows")
    eda_md.append("")
    eda_md.append("**Columns in output files:**")
    eda_md.append("State, District, Market, Commodity_Group, Commodity, Date, Day_of_Week, "
                  "Modal_Price *(imputed)*, Price_Unit, latitude, longitude, t, tp, ssr, r")
    eda_md.append("")
    eda_md.append("> `Arrival_Quantity` and `Arrival_Unit` have been dropped as per requirement.")

    # Write report
    REPORT.write_text("\n".join(eda_md), encoding="utf-8")
    print(f"\n{'='*70}")
    print(f"  Report: {REPORT}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
