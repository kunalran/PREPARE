"""
clean_data.py — Clean scraped agmarknet CSVs from data/
Removes repeated headers, title rows, note/footer rows, blank lines.
Standardizes column names, parses dates, casts numeric types, drops duplicates.
Outputs cleaned files to data2_cleaned/.
"""

import os
import pandas as pd
from pathlib import Path

INPUT_DIR = Path("data")
OUTPUT_DIR = Path("data_cleaned")

STANDARD_COLUMNS = [
    "State", "District", "Market", "Commodity_Group", "Commodity",
    "Date", "Arrival_Quantity", "Arrival_Unit", "Modal_Price", "Price_Unit"
]

def is_junk_line(line: str) -> bool:
    """Return True if the line is a scraping artifact (header, title, note, blank)."""
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.startswith(",,,,,"):
        return True
    if stripped.startswith("State,District,Market"):
        return True
    if stripped.startswith("Note:"):
        return True
    if stripped.startswith("Weighted Average"):
        return True
    return False

def clean_single_file(input_path: Path, output_path: Path) -> dict:
    """Clean a single CSV file and return cleaning stats."""
    stats = {
        "input_file": input_path.name,
        "raw_lines": 0,
        "title_rows_removed": 0,
        "header_rows_removed": 0,
        "note_rows_removed": 0,
        "blank_rows_removed": 0,
        "data_rows_kept": 0,
        "duplicates_removed": 0,
        "final_rows": 0,
    }

    # --- Pass 1: Line-by-line filtering ---
    clean_lines = []
    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stats["raw_lines"] += 1
            stripped = line.strip()
            if not stripped:
                stats["blank_rows_removed"] += 1
                continue
            if stripped.startswith(",,,,,"):
                stats["title_rows_removed"] += 1
                continue
            if stripped.startswith("State,District,Market"):
                stats["header_rows_removed"] += 1
                continue
            if stripped.startswith("Note:"):
                stats["note_rows_removed"] += 1
                continue
            if stripped.startswith("Weighted Average"):
                stats["note_rows_removed"] += 1
                continue
            clean_lines.append(stripped)

    stats["data_rows_kept"] = len(clean_lines)

    # --- Pass 2: Parse with pandas ---
    # Write temp in-memory CSV with standard header
    import io
    csv_text = ",".join(STANDARD_COLUMNS) + "\n" + "\n".join(clean_lines)
    df = pd.read_csv(io.StringIO(csv_text))

    # Parse date
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="coerce")

    # Cast numeric columns
    df["Arrival_Quantity"] = pd.to_numeric(df["Arrival_Quantity"], errors="coerce")
    df["Modal_Price"] = pd.to_numeric(df["Modal_Price"], errors="coerce")

    # Drop exact duplicates
    before_dedup = len(df)
    df = df.drop_duplicates()
    stats["duplicates_removed"] = before_dedup - len(df)

    # Drop rows where Date failed to parse (likely residual junk)
    bad_dates = df["Date"].isna().sum()
    if bad_dates > 0:
        print(f"  ⚠ Dropping {bad_dates} rows with unparseable dates")
        df = df.dropna(subset=["Date"])

    # Sort
    df = df.sort_values(["Date", "State", "Market"]).reset_index(drop=True)

    stats["final_rows"] = len(df)

    # Save
    df.to_csv(output_path, index=False)
    return stats

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    csv_files = sorted(INPUT_DIR.glob("agmarknet_*.csv"))
    if not csv_files:
        print("No CSV files found in", INPUT_DIR)
        return

    print("=" * 70)
    print("AGMARKNET DATA CLEANING")
    print("=" * 70)

    all_stats = []
    for csv_path in csv_files:
        out_path = OUTPUT_DIR / csv_path.name.replace(".csv", "_cleaned.csv")
        print(f"\n{'─' * 50}")
        print(f"Cleaning: {csv_path.name}")
        stats = clean_single_file(csv_path, out_path)
        all_stats.append(stats)

        print(f"  Raw lines:           {stats['raw_lines']:>10,}")
        print(f"  Title rows removed:  {stats['title_rows_removed']:>10,}")
        print(f"  Header rows removed: {stats['header_rows_removed']:>10,}")
        print(f"  Note rows removed:   {stats['note_rows_removed']:>10,}")
        print(f"  Blank rows removed:  {stats['blank_rows_removed']:>10,}")
        print(f"  Data rows kept:      {stats['data_rows_kept']:>10,}")
        print(f"  Duplicates removed:  {stats['duplicates_removed']:>10,}")
        print(f"  Final rows:          {stats['final_rows']:>10,}")
        print(f"  → Saved to: {out_path}")

    print(f"\n{'=' * 70}")
    print("CLEANING COMPLETE")
    print(f"{'=' * 70}")
    for s in all_stats:
        print(f"  {s['input_file']:<40} {s['raw_lines']:>8,} → {s['final_rows']:>8,} rows")

    return all_stats

if __name__ == "__main__":
    main()
