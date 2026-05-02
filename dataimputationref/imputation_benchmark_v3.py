"""
imputation_benchmark_v3.py — Price-only benchmark at 50% and 75% thresholds.
Focused version: Modal_Price only, 5 seeds, 10 methods.
Outputs: imputation/benchmark_v3_results.md + .csv
"""

import sys, time, warnings
from pathlib import Path
from collections import OrderedDict
import numpy as np, pandas as pd
from scipy.interpolate import CubicSpline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

DATA_DIR = Path("data2_expanded")
OUT_DIR = Path("imputation")
MASK_FRACTION = 0.10
THRESHOLDS = [50, 75]
SEEDS = [42, 123, 456, 789, 2025]
TARGET = "Modal_Price"
FFILL_CAP, ROLLING_WINDOW, DECAY_LAMBDA, SVD_RANK, SVD_ITERS = 7, 15, 0.15, 20, 30


# ── Data ──
def load_and_filter(csv_path, min_pct):
    df = pd.read_csv(csv_path, parse_dates=["Date"],
        usecols=["State","District","Market","Commodity","Date",
                 "Day_of_Week","Arrival_Quantity","Modal_Price"], low_memory=False)
    df = df.sort_values(["State","Market","Date"]).reset_index(drop=True)
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    grp = df.groupby(["State","Market","Year"])[TARGET]
    n_total = grp.transform("size")
    n_present = grp.transform(lambda x: x.notna().sum())
    df["_pct"] = n_present / n_total * 100
    df = df[df["_pct"] >= min_pct].copy()
    df.drop(columns=["_pct"], inplace=True)
    return df

def mask_ground_truth(df, seed):
    rng = np.random.RandomState(seed)
    known = df.index[df[TARGET].notna()]
    mask_idx = rng.choice(known, size=int(len(known)*MASK_FRACTION), replace=False)
    truth = pd.Series(np.nan, index=df.index)
    truth.loc[mask_idx] = df.loc[mask_idx, TARGET]
    masked = df.copy()
    masked.loc[mask_idx, TARGET] = np.nan
    return masked, truth


# ── 10 Methods ──
def impute_capped_ffill(df, col):
    r = df[col].copy()
    for _, idx in df.groupby(["State","Market"]).groups.items():
        g = df.loc[idx].sort_values("Date")
        r.loc[g.index] = g[col].ffill(limit=FFILL_CAP).values
    return r

def impute_rolling_mean(df, col):
    r = df[col].copy()
    for _, idx in df.groupby(["State","Market"]).groups.items():
        g = df.loc[idx].sort_values("Date"); s = g[col]
        roll = s.rolling(ROLLING_WINDOW, center=True, min_periods=1).mean()
        f = s.copy(); f[s.isna()] = roll[s.isna()]
        r.loc[g.index] = f.values
    return r

def impute_state_median(df, col):
    r = df[col].copy()
    med = df.groupby(["Date","Commodity","State"])[col].transform("median")
    r[r.isna()] = med[r.isna()]
    return r

def impute_seasonal(df, col):
    r = df[col].copy()
    med = df.groupby(["State","Market","Month","DayOfWeek"])[col].transform("median")
    r[r.isna()] = med[r.isna()]
    return r

def impute_global_median(df, col):
    r = df[col].copy(); r.fillna(df[col].median(), inplace=True); return r

def impute_ffill_decay(df, col):
    r = df[col].copy()
    sd = df.groupby(["Date","State"])[col].transform("mean")
    for _, idx in df.groupby(["State","Market"]).groups.items():
        g = df.loc[idx].sort_values("Date"); s = g[col].copy(); sdg = sd.loc[g.index]
        lv, ds = np.nan, 0; f = s.copy()
        for ix, val in s.items():
            if pd.notna(val): lv, ds = val, 0
            else:
                ds += 1
                if pd.notna(lv):
                    w = np.exp(-DECAY_LAMBDA * ds)
                    sv = sdg.loc[ix] if pd.notna(sdg.loc[ix]) else lv
                    f.loc[ix] = w * lv + (1 - w) * sv
                elif pd.notna(sdg.loc[ix]): f.loc[ix] = sdg.loc[ix]
        r.loc[g.index] = f.values
    return r

def impute_dow_ratio(df, col):
    r = df[col].copy()
    for _, idx in df.groupby(["State","Market"]).groups.items():
        g = df.loc[idx].sort_values("Date"); s, dow = g[col], g["DayOfWeek"]
        om = s.mean()
        if pd.isna(om) or om == 0: continue
        dr = (s.groupby(dow).mean() / om).fillna(1.0)
        wk = s.rolling(7, center=True, min_periods=1).mean()
        f = s.copy()
        for ix in s.index[s.isna()]:
            ctx = wk.loc[ix]
            if pd.notna(ctx): f.loc[ix] = ctx * dr.get(dow.loc[ix], 1.0)
        r.loc[g.index] = f.values
    return r

def impute_spline_pipeline(df, col):
    r = df[col].copy()
    for _, idx in df.groupby(["State","Market"]).groups.items():
        g = df.loc[idx].sort_values("Date"); s = g[col].copy()
        dn = (g["Date"] - g["Date"].min()).dt.days.values
        for i in range(7, len(s)):
            v = s.iloc[i]
            if pd.isna(v): continue
            pw = s.iloc[max(0,i-7):i].dropna()
            if len(pw) == 0: continue
            a = pw.mean()
            if a > 0 and (v > 6*a or v < a/6): s.iloc[i] = np.nan
        for yr in g["Date"].dt.year.unique():
            ym = g["Date"].dt.year.values == yr
            ys, yd = s[ym], dn[ym]; kn = ys.notna()
            if kn.sum() < 4 or kn.sum()/len(ys) < 0.5: continue
            try:
                cs = CubicSpline(yd[kn.values], ys[kn].values, extrapolate=False)
                sp = cs(yd); nm = ys.isna(); ysf = ys.copy()
                ysf[nm] = sp[nm.values]; s[ym] = ysf.clip(lower=0)
            except: pass
        for i in range(len(s)):
            v = s.iloc[i]
            if pd.isna(v) or g[col].iloc[i] == v: continue
            w = s.iloc[max(0,i-7):i+8].dropna()
            if len(w) < 2: continue
            if w.max() > 0 and (v > w.max()*1.15 or v < w.min()*0.85): s.iloc[i] = np.nan
        s = s.interpolate(method="linear", limit_direction="both")
        r.loc[g.index] = s.values
    return r

def impute_random_forest(df, col):
    r = df[col].copy()
    wk = df[["State","Market","Month","DayOfWeek","Year",col]].copy()
    le_s, le_m = LabelEncoder(), LabelEncoder()
    wk["S"] = le_s.fit_transform(wk["State"]); wk["M"] = le_m.fit_transform(wk["Market"])
    lag = df.groupby(["State","Market"])[col].transform(lambda x: x.rolling(7,min_periods=1).mean().shift(1))
    wk["lag7"] = lag.fillna(df[col].median())
    feats = ["S","M","Month","DayOfWeek","Year","lag7"]
    kn, ms = wk[col].notna(), wk[col].isna()
    if kn.sum() < 100 or ms.sum() == 0: return r
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf.fit(wk.loc[kn, feats].values, wk.loc[kn, col].values)
    r.loc[ms[ms].index] = np.clip(rf.predict(wk.loc[ms, feats].values), 0, None)
    return r

def impute_svd(df, col):
    r = df[col].copy()
    dw = df[["State","Market","Date",col]].copy()
    dw["key"] = dw["State"] + "||" + dw["Market"]
    pv = dw.pivot_table(index="Date", columns="key", values=col, aggfunc="first")
    mat = pv.values.copy().astype(float); obs = ~np.isnan(mat)
    cm = np.nanmean(mat, axis=0); cm = np.where(np.isnan(cm), 0, cm)
    for j in range(mat.shape[1]): mat[np.isnan(mat[:,j]), j] = cm[j]
    rank = min(SVD_RANK, min(mat.shape)-1)
    if rank < 1: return r
    for _ in range(SVD_ITERS):
        U,s,Vt = np.linalg.svd(mat, full_matrices=False)
        st = np.zeros_like(s); st[:rank] = s[:rank]
        mat[~obs] = (U @ np.diag(st) @ Vt)[~obs]
    mat = np.clip(mat, 0, None)
    fp = pd.DataFrame(mat, index=pv.index, columns=pv.columns)
    for mk in fp.columns:
        parts = mk.split("||")
        if len(parts) != 2: continue
        st, mkt = parts
        idx = df.index[(df["State"]==st)&(df["Market"]==mkt)]
        for ix, d in zip(idx, df.loc[idx,"Date"]):
            if pd.isna(r.loc[ix]) and d in fp.index: r.loc[ix] = fp.loc[d, mk]
    return r

METHODS = OrderedDict([
    ("1_Capped_FFill", impute_capped_ffill), ("2_Rolling_Mean", impute_rolling_mean),
    ("3_State_Median", impute_state_median), ("4_Seasonal", impute_seasonal),
    ("5_Global_Median", impute_global_median), ("6_FFill_Decay", impute_ffill_decay),
    ("7_DOW_Ratio", impute_dow_ratio), ("8_Spline_Pipeline", impute_spline_pipeline),
    ("9_Random_Forest", impute_random_forest), ("10_SVD_Matrix", impute_svd),
])


# ── Scoring ──
def score(truth, imputed):
    mask = truth.notna(); t, p = truth[mask].values, imputed[mask].values
    valid = ~np.isnan(p)
    if valid.sum() == 0:
        return {"RMSE":np.nan,"MAE":np.nan,"MAPE":np.nan,"MedianAE":np.nan,"Coverage":0.0}
    t, p = t[valid], p[valid]
    errors = t - p
    abs_errors = np.abs(errors)
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(abs_errors)
    median_ae = np.median(abs_errors)
    nz = t != 0
    mape = np.mean(np.abs(errors[nz]/t[nz]))*100 if nz.sum()>0 else np.nan
    cov = valid.sum()/mask.sum()*100

    # Error distribution stats (for RMSE vs MAE analysis)
    p90 = np.percentile(abs_errors, 90)
    p99 = np.percentile(abs_errors, 99)
    max_err = abs_errors.max()

    return {"RMSE":rmse, "MAE":mae, "MedianAE":median_ae, "MAPE":mape,
            "Coverage":cov, "P90_Error":p90, "P99_Error":p99, "Max_Error":max_err}


# ── Main ──
def main():
    print("="*70)
    print("  PRICE-ONLY BENCHMARK v3 (Modal_Price, 50% & 75% thresholds)")
    print(f"  Seeds: {len(SEEDS)}  |  Mask: {MASK_FRACTION*100:.0f}%  |  Methods: {len(METHODS)}")
    print("="*70)

    csv_files = sorted(DATA_DIR.glob("*_expanded.csv"))
    if not csv_files: print(f"No files in {DATA_DIR}/"); sys.exit(1)

    all_results = []

    for threshold in THRESHOLDS:
        print(f"\n{'━'*70}")
        print(f"  THRESHOLD: ≥{threshold}% data present")
        print(f"{'━'*70}")

        for csv_path in csv_files:
            crop = csv_path.stem.replace("agmarknet_","").replace("_data_expanded","").title()
            df = load_and_filter(csv_path, threshold)
            if len(df) == 0:
                print(f"    {crop}: ⚠ No dense data, skipping.")
                continue
            n_mandis = df.groupby(["State","Market"]).ngroups
            n_known = df[TARGET].notna().sum()
            if n_known < 100:
                print(f"    {crop}: ⚠ Only {n_known} known values, skipping.")
                continue

            print(f"\n    {crop} (≥{threshold}%)  |  {n_mandis:,} mandis  |  {n_known:,} known prices")

            for method_name, method_fn in METHODS.items():
                seed_scores = []
                t0 = time.time()
                for seed in SEEDS:
                    masked_df, truth = mask_ground_truth(df, seed)
                    try:
                        imputed = method_fn(masked_df, TARGET)
                        seed_scores.append(score(truth, imputed))
                    except Exception as e:
                        seed_scores.append({"RMSE":np.nan,"MAE":np.nan,"MedianAE":np.nan,
                                            "MAPE":np.nan,"Coverage":0.0,
                                            "P90_Error":np.nan,"P99_Error":np.nan,"Max_Error":np.nan})
                elapsed = time.time() - t0

                def avg_metric(key):
                    vals = [s[key] for s in seed_scores if not np.isnan(s[key])]
                    return round(np.mean(vals),2) if vals else np.nan
                def std_metric(key):
                    vals = [s[key] for s in seed_scores if not np.isnan(s[key])]
                    return round(np.std(vals),2) if vals else np.nan

                row = {
                    "Crop": crop, "Threshold": threshold, "Method": method_name,
                    "N_Mandis": n_mandis,
                    "RMSE_mean": avg_metric("RMSE"), "RMSE_std": std_metric("RMSE"),
                    "MAE_mean": avg_metric("MAE"), "MAE_std": std_metric("MAE"),
                    "MedianAE_mean": avg_metric("MedianAE"),
                    "MAPE_mean": avg_metric("MAPE"),
                    "Coverage": avg_metric("Coverage"),
                    "P90_Error": avg_metric("P90_Error"),
                    "P99_Error": avg_metric("P99_Error"),
                    "Max_Error": avg_metric("Max_Error"),
                    "Time_s": round(elapsed,1),
                }
                all_results.append(row)
                print(f"      {method_name:<24}  RMSE={row['RMSE_mean']:>8}±{row['RMSE_std']:<6}  "
                      f"MAE={row['MAE_mean']:>8}  MedAE={row['MedianAE_mean']:>8}  "
                      f"P90={row['P90_Error']:>8}  ({elapsed:.1f}s)")

    if not all_results: print("No results."); return

    rdf = pd.DataFrame(all_results)
    rdf.to_csv(OUT_DIR / "benchmark_v3_results.csv", index=False)

    # ── Build Markdown ──
    md = []
    md.append("# Price Imputation Benchmark v3 — Modal_Price Only")
    md.append("")
    md.append(f"> Masked {MASK_FRACTION*100:.0f}% of known values, **{len(SEEDS)} random seeds**.")
    md.append("> Thresholds: **≥50%** and **≥75%** data presence.")
    md.append("> Target: **Modal_Price** only.")
    md.append("")

    # ── RMSE vs MAE explanation ──
    md.append("## Understanding RMSE vs MAE Divergence")
    md.append("")
    md.append("### Why can Method A have lower RMSE but higher MAE than Method B?")
    md.append("")
    md.append("**RMSE** (Root Mean Square Error) and **MAE** (Mean Absolute Error) measure different things:")
    md.append("")
    md.append("| Metric | Formula | What it emphasizes |")
    md.append("|--------|---------|-------------------|")
    md.append("| **MAE** | mean(|error|) | Treats all errors equally — a ₹10 error and a ₹1000 error contribute proportionally |")
    md.append("| **RMSE** | √mean(error²) | Squares errors first — a ₹1000 error contributes **100×** more than a ₹10 error |")
    md.append("")
    md.append("**Example:** Consider two methods imputing 5 masked prices:")
    md.append("")
    md.append("| | Method A errors | Method B errors |")
    md.append("|---|----------------|----------------|")
    md.append("| Errors | 50, 50, 50, 50, 50 | 10, 10, 10, 10, 200 |")
    md.append("| **MAE** | **50** | **48** ← lower |")
    md.append("| **RMSE** | **50** ← lower | **90** |")
    md.append("")
    md.append("Method B has lower MAE (most predictions are very accurate), but its single large outlier blows up the RMSE.")
    md.append("")
    md.append("**What this means for imputation:**")
    md.append("")
    md.append("- **Lower MAE, higher RMSE** → The method is accurate *on average* but occasionally makes catastrophic errors (large outliers)")
    md.append("- **Lower RMSE, higher MAE** → The method avoids extreme errors but is slightly less accurate on typical predictions")
    md.append("- **Median AE** is included below to show the \"typical\" error unaffected by outliers")
    md.append("- **P90 / P99 errors** show the tail — how bad the worst 10% and 1% of predictions are")
    md.append("")
    md.append("> **Recommendation:** For forecasting models, **RMSE is preferred** because downstream models are sensitive to large errors. A few catastrophic imputation errors can poison an entire training batch.")
    md.append("")

    # ── Results tables ──
    for threshold in THRESHOLDS:
        label = f"≥{threshold}%"
        ts = rdf[rdf["Threshold"] == threshold]
        if ts.empty: continue

        md.append(f"## Results — {label} Data Present")
        md.append("")

        for crop in ts["Crop"].unique():
            cs = ts[ts["Crop"] == crop].copy()
            n_mandis = cs["N_Mandis"].iloc[0]
            cs_rmse = cs.sort_values("RMSE_mean")
            cs_mae = cs.sort_values("MAE_mean")

            md.append(f"### {crop} ({label}, {n_mandis:,} mandis)")
            md.append("")

            # Side-by-side: RMSE ranking vs MAE ranking
            md.append("#### Ranked by RMSE")
            md.append("")
            md.append("| Rank | Method | RMSE (±std) | MAE | Median AE | MAPE% | P90 Err | P99 Err | Coverage | Time |")
            md.append("|------|--------|-------------|-----|-----------|-------|---------|---------|----------|------|")
            for rank, (_, row) in enumerate(cs_rmse.iterrows(), 1):
                md.append(f"| {rank} | {row['Method']} | {row['RMSE_mean']}±{row['RMSE_std']} | "
                          f"{row['MAE_mean']} | {row['MedianAE_mean']} | {row['MAPE_mean']} | "
                          f"{row['P90_Error']} | {row['P99_Error']} | {row['Coverage']}% | {row['Time_s']}s |")
            md.append("")

            # Check for RMSE/MAE rank disagreement
            rmse_top = cs_rmse["Method"].iloc[0]
            mae_top = cs_mae["Method"].iloc[0]
            if rmse_top != mae_top:
                md.append(f"> **⚡ Rank disagreement:** RMSE winner = **{rmse_top}**, MAE winner = **{mae_top}**. "
                          f"This means {mae_top} is more accurate on *typical* prices, but {rmse_top} avoids large outlier errors.")
                md.append("")

        # Cross-crop average
        avg = ts.groupby("Method")[["RMSE_mean","MAE_mean","MedianAE_mean","MAPE_mean",
                                      "P90_Error","P99_Error","Coverage","Time_s"]].mean()
        avg_rmse = avg.sort_values("RMSE_mean").round(2)

        md.append(f"### Cross-Crop Average — {label}")
        md.append("")
        md.append("| Rank | Method | Avg RMSE | Avg MAE | Avg Median AE | Avg MAPE% | Avg P90 | Avg P99 | Time |")
        md.append("|------|--------|----------|---------|---------------|-----------|---------|---------|------|")
        for rank, (method, row) in enumerate(avg_rmse.iterrows(), 1):
            md.append(f"| {rank} | {method} | {row['RMSE_mean']} | {row['MAE_mean']} | "
                      f"{row['MedianAE_mean']} | {row['MAPE_mean']} | {row['P90_Error']} | "
                      f"{row['P99_Error']} | {row['Time_s']}s |")
        md.append("")

    # ── Threshold comparison ──
    md.append("## Threshold Comparison — How RMSE Changes with Data Density")
    md.append("")
    md.append("| Method | ≥50% RMSE | ≥75% RMSE | Improvement |")
    md.append("|--------|-----------|-----------|-------------|")
    for method in METHODS.keys():
        r50 = rdf[(rdf["Method"]==method)&(rdf["Threshold"]==50)]["RMSE_mean"].mean()
        r75 = rdf[(rdf["Method"]==method)&(rdf["Threshold"]==75)]["RMSE_mean"].mean()
        if not np.isnan(r50) and not np.isnan(r75):
            imp = round((r50 - r75) / r50 * 100, 1)
            md.append(f"| {method} | {r50:.1f} | {r75:.1f} | {imp}% |")
    md.append("")

    md_path = OUT_DIR / "benchmark_v3_results.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\n{'='*70}")
    print(f"  CSV:  {OUT_DIR / 'benchmark_v3_results.csv'}")
    print(f"  MD:   {md_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
