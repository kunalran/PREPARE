"""
expand_data.py — Expand cleaned agmarknet data to a full date × mandi grid.
For each crop, every mandi gets a row for every date in the crop's global
date range. Dates where no data was originally recorded have NaN for
Arrival_Quantity and Modal_Price.
Saves expanded CSVs to data2_expanded/.
"""

import pandas as pd
import numpy as np
from pathlib import Path

INPUT_DIR = Path("data_cleaned")
OUTPUT_DIR = Path("data_expanded")


def expand_crop(csv_path: Path, output_path: Path) -> dict:
    """Expand one crop CSV to full date × mandi grid. Returns stats dict."""
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    crop_name = csv_path.stem.replace("agmarknet_", "").replace("_data_cleaned", "").title()

    original_rows = len(df)
    date_min = df["Date"].min()
    date_max = df["Date"].max()
    full_dates = pd.date_range(date_min, date_max, freq="D")

    # Build unique mandi list (State + District + Market + Commodity_Group + Commodity)
    mandi_cols = ["State", "District", "Market", "Commodity_Group", "Commodity"]
    mandis = df[mandi_cols].drop_duplicates().reset_index(drop=True)

    # Also keep Arrival_Unit and Price_Unit (constant per mandi)
    unit_info = df.groupby(mandi_cols).agg(
        Arrival_Unit=("Arrival_Unit", "first"),
        Price_Unit=("Price_Unit", "first"),
    ).reset_index()

    n_mandis = len(mandis)
    n_dates = len(full_dates)
    expected_rows = n_mandis * n_dates

    print(f"  Mandis: {n_mandis:,}  |  Date range: {date_min.date()} to {date_max.date()} ({n_dates:,} days)")
    print(f"  Original rows: {original_rows:,}  |  Expanded grid: {expected_rows:,}")

    # Create the full grid via cross-merge
    dates_df = pd.DataFrame({"Date": full_dates})
    grid = mandis.merge(dates_df, how="cross")

    # Merge original data onto the grid
    merge_cols = mandi_cols + ["Date"]
    value_cols = ["Arrival_Quantity", "Modal_Price"]
    expanded = grid.merge(
        df[merge_cols + value_cols],
        on=merge_cols,
        how="left"
    )

    # Merge unit info
    expanded = expanded.merge(unit_info, on=mandi_cols, how="left")

    # Sort
    expanded = expanded.sort_values(["State", "Market", "Date"]).reset_index(drop=True)

    # Add day-of-week column for downstream EDA convenience
    expanded["Day_of_Week"] = expanded["Date"].dt.day_name()

    # Reorder columns
    col_order = [
        "State", "District", "Market", "Commodity_Group", "Commodity",
        "Date", "Day_of_Week", "Arrival_Quantity", "Arrival_Unit",
        "Modal_Price", "Price_Unit"
    ]
    expanded = expanded[col_order]

    # Save
    expanded.to_csv(output_path, index=False)
    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    missing_rows = expanded["Arrival_Quantity"].isna().sum()
    present_rows = expanded["Arrival_Quantity"].notna().sum()
    missing_pct = round(missing_rows / len(expanded) * 100, 2)

    stats = {
        "crop": crop_name,
        "n_mandis": n_mandis,
        "n_dates": n_dates,
        "original_rows": original_rows,
        "expanded_rows": len(expanded),
        "present_rows": present_rows,
        "missing_rows": missing_rows,
        "missing_pct": missing_pct,
        "file_size_mb": round(file_size_mb, 1),
    }

    print(f"  Present: {present_rows:,}  |  Missing (NaN): {missing_rows:,} ({missing_pct}%)")
    print(f"  Saved: {output_path} ({file_size_mb:.1f} MB)")
    return stats


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    csv_files = sorted(INPUT_DIR.glob("*_cleaned.csv"))
    if not csv_files:
        print(f"No cleaned CSVs in {INPUT_DIR}/. Run clean_data.py first.")
        return

    print("=" * 70)
    print("EXPANDING CLEANED DATA TO FULL DATE × MANDI GRID")
    print("=" * 70)

    all_stats = []
    for csv_path in csv_files:
        out_name = csv_path.stem.replace("_cleaned", "_expanded") + ".csv"
        out_path = OUTPUT_DIR / out_name
        crop_label = csv_path.stem.replace("agmarknet_", "").replace("_data_cleaned", "").title()
        print(f"\n{'─' * 50}")
        print(f"  {crop_label}")
        print(f"{'─' * 50}")
        stats = expand_crop(csv_path, out_path)
        all_stats.append(stats)

    print(f"\n{'=' * 70}")
    print("EXPANSION COMPLETE")
    print(f"{'=' * 70}")
    print(f"{'Crop':<14} {'Original':>10} {'Expanded':>12} {'Missing':>10} {'%':>7} {'Size':>8}")
    for s in all_stats:
        print(f"{s['crop']:<14} {s['original_rows']:>10,} {s['expanded_rows']:>12,} "
              f"{s['missing_rows']:>10,} {s['missing_pct']:>6}% {s['file_size_mb']:>6.1f}MB")

    return all_stats


if __name__ == "__main__":
    main()
