"""
apply_dow_ratio_imputation.py
─────────────────────────────
Production script to apply DOW Ratio imputation to any crop dataset.

Steps:
  1. Reads an expanded CSV from an input directory
  2. Filters to mandis with ≥50% Modal_Price data per year (drops the rest)
  3. Drops the Arrival_Quantity column (if present)
  4. Imputes Modal_Price using the DOW Ratio (Method 7) strategy
  5. Saves the imputed CSV to an output directory

Usage:
    ./venv/bin/python final_imputation_strat/apply_dow_ratio_imputation.py \\
        --input-dir  data4 \\
        --output-dir data4_imputed \\
        --min-pct    50

    # Or process a single file:
    ./venv/bin/python final_imputation_strat/apply_dow_ratio_imputation.py \\
        --input-file data4/agmarknet_onion_data_final.csv \\
        --output-dir data4_imputed \\
        --min-pct    50
"""

import argparse
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def filter_dense_mandis(df: pd.DataFrame, target: str, min_pct: float) -> pd.DataFrame:
    """
    Keep only mandi×year combinations where at least `min_pct`%
    of the target column values are non-null.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: State, Market, Date, and `target`.
    target : str
        Column name to check density for.
    min_pct : float
        Minimum percentage of non-null values required (0–100).

    Returns
    -------
    pd.DataFrame
        Filtered dataframe (rows belonging to sparse mandis are removed).
    """
    df = df.copy()
    df["_Year"] = pd.to_datetime(df["Date"]).dt.year

    grp = df.groupby(["State", "Market", "_Year"])[target]
    n_total = grp.transform("size")
    n_present = grp.transform(lambda x: x.notna().sum())
    df["_pct"] = n_present / n_total * 100

    df = df[df["_pct"] >= min_pct].copy()
    df.drop(columns=["_pct", "_Year"], inplace=True)
    return df


def impute_dow_ratio(df: pd.DataFrame, col: str = "Modal_Price") -> pd.Series:
    """
    DOW Ratio imputation: fills missing prices using a 7-day centered
    rolling mean adjusted by the historical day-of-week price ratio
    for each mandi.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: State, Market, Date, and `col`.
    col : str
        Target column to impute.

    Returns
    -------
    pd.Series
        Imputed column (same index as df).
    """
    result = df[col].copy()
    df_work = df.copy()
    df_work["_DayOfWeek"] = pd.to_datetime(df_work["Date"]).dt.dayofweek

    for _, idx in df_work.groupby(["State", "Market"]).groups.items():
        grp = df_work.loc[idx].sort_values("Date")
        series = grp[col]
        dow = grp["_DayOfWeek"]

        # Step 1: Overall mean for this mandi
        overall_mean = series.mean()
        if pd.isna(overall_mean) or overall_mean == 0:
            continue

        # Step 2: Day-of-week ratios
        dow_ratios = (series.groupby(dow).mean() / overall_mean).fillna(1.0)

        # Step 3: 7-day centered rolling mean (local context)
        rolling_ctx = series.rolling(7, center=True, min_periods=1).mean()

        # Step 4: Fill missing values
        filled = series.copy()
        for ix in series.index[series.isna()]:
            ctx = rolling_ctx.loc[ix]
            if pd.notna(ctx):
                filled.loc[ix] = ctx * dow_ratios.get(dow.loc[ix], 1.0)

        result.loc[grp.index] = filled.values

    return result


def process_one_file(
    input_path: Path,
    output_dir: Path,
    min_pct: float = 50,
    target: str = "Modal_Price",
    drop_arrival: bool = True,
) -> dict:
    """
    Full pipeline for one CSV: load → filter → drop → impute → save.

    Returns a summary dict with stats.
    """
    crop = input_path.stem
    print(f"\n{'━'*60}")
    print(f"  Processing: {crop}")
    print(f"{'━'*60}")

    # Load
    df = pd.read_csv(input_path, parse_dates=["Date"], low_memory=False)
    df = df.sort_values(["State", "Market", "Date"]).reset_index(drop=True)
    rows_before = len(df)
    mandis_before = df.groupby(["State", "Market"]).ngroups
    print(f"  Loaded: {rows_before:,} rows, {mandis_before:,} mandis")

    # Filter
    df = filter_dense_mandis(df, target, min_pct)
    rows_after = len(df)
    mandis_after = df.groupby(["State", "Market"]).ngroups

    if rows_after == 0:
        print(f"  ⚠ No mandis with ≥{min_pct}% data — skipping.")
        return {"crop": crop, "status": "skipped", "reason": "no dense mandis"}

    print(f"  Filtered: {rows_before:,} → {rows_after:,} rows | "
          f"{mandis_before:,} → {mandis_after:,} mandis")

    # Drop Arrival columns
    if drop_arrival:
        cols_to_drop = [c for c in ["Arrival_Quantity", "Arrival_Unit"] if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            print(f"  Dropped columns: {cols_to_drop}")

    # Stats before imputation
    known_before = df[target].notna().sum()
    missing_before = df[target].isna().sum()

    # Impute
    t0 = time.time()
    df[target] = impute_dow_ratio(df, target)
    elapsed = round(time.time() - t0, 1)

    # Stats after imputation
    remaining_nan = df[target].isna().sum()
    filled = missing_before - remaining_nan
    fill_pct = round(filled / missing_before * 100, 1) if missing_before > 0 else 100.0
    total_fill_pct = round((rows_after - remaining_nan) / rows_after * 100, 2)

    print(f"  Imputed {filled:,} / {missing_before:,} missing values ({fill_pct}%) in {elapsed}s")
    print(f"  Remaining NaN: {remaining_nan:,} / {rows_after:,} ({round(remaining_nan/rows_after*100,1)}%)")
    print(f"  Total data coverage: {total_fill_pct}%")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    out_name = input_path.name.replace("_final", "_imputed").replace("_expanded", "_imputed")
    out_path = output_dir / out_name
    df.to_csv(out_path, index=False)
    out_size = out_path.stat().st_size / 1e6
    print(f"  Saved: {out_path} ({out_size:.1f} MB)")

    return {
        "crop": crop, "status": "done",
        "mandis_before": mandis_before, "mandis_after": mandis_after,
        "rows_before": rows_before, "rows_after": rows_after,
        "known_before": known_before, "filled": filled,
        "remaining_nan": remaining_nan,
        "fill_pct": fill_pct, "total_coverage": total_fill_pct,
        "time_s": elapsed,
    }


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Apply DOW Ratio imputation to crop price CSVs."
    )
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Directory containing input CSVs (processes all *.csv)")
    parser.add_argument("--input-file", type=str, default=None,
                        help="Single input CSV file to process")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save imputed CSVs")
    parser.add_argument("--min-pct", type=float, default=50,
                        help="Minimum %% of Modal_Price data per mandi per year (default: 50)")
    parser.add_argument("--target", type=str, default="Modal_Price",
                        help="Column to impute (default: Modal_Price)")
    parser.add_argument("--keep-arrival", action="store_true",
                        help="Keep Arrival_Quantity column (default: drop it)")

    args = parser.parse_args()

    if args.input_dir is None and args.input_file is None:
        print("Error: specify --input-dir or --input-file")
        sys.exit(1)

    # Collect files
    if args.input_file:
        files = [Path(args.input_file)]
    else:
        files = sorted(Path(args.input_dir).glob("*.csv"))

    if not files:
        print("No CSV files found."); sys.exit(1)

    output_dir = Path(args.output_dir)

    print("=" * 60)
    print(f"  DOW RATIO IMPUTATION PIPELINE")
    print(f"  Files: {len(files)} | Min density: ≥{args.min_pct}%")
    print(f"  Target: {args.target} | Output: {output_dir}/")
    print("=" * 60)

    results = []
    for f in files:
        r = process_one_file(f, output_dir, args.min_pct, args.target,
                             drop_arrival=not args.keep_arrival)
        results.append(r)

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for r in results:
        if r["status"] == "skipped":
            print(f"  {r['crop']}: SKIPPED ({r['reason']})")
        else:
            print(f"  {r['crop']}: {r['mandis_after']} mandis, "
                  f"{r['filled']:,} filled, {r['remaining_nan']:,} remaining NaN, "
                  f"{r['total_coverage']}% coverage ({r['time_s']}s)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
