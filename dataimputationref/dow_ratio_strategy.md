# DOW Ratio Imputation Strategy — Explained

## Overview

The **Day-of-Week (DOW) Ratio** strategy is a calendar-aware imputation method that fills missing crop prices by combining a local rolling average with a day-of-week adjustment factor. It was selected as the best-performing imputation method across all crops and density thresholds in our benchmark tests.

### Why it works

Indian agricultural markets (mandis) exhibit strong day-of-week patterns:
- **Sundays** have 77–91% missing data (most mandis are closed)
- **Saturdays** show elevated missingness (~66–75%)
- **Weekdays** are relatively stable (~63–67% missing)

When a mandi *does* report on a low-activity day, the price often differs systematically from the weekly average. The DOW Ratio method captures this pattern: it computes the historical ratio of each day's average price to the overall mandi average, then uses that ratio to scale a local rolling mean when filling gaps.

---

## Algorithm (Step-by-Step)

For each **mandi** (grouped by `State` + `Market`):

### Step 1: Compute the Overall Mean Price

```
overall_mean = mean of all known Modal_Price values for this mandi
```

If `overall_mean` is NaN or 0, skip this mandi (no basis for imputation).

### Step 2: Compute Day-of-Week Ratios

For each day of the week (0=Mon, 1=Tue, ..., 6=Sun):

```
dow_ratio[day] = mean(prices on that day) / overall_mean
```

If a day has no data at all, `dow_ratio[day]` defaults to `1.0`.

**Intuition:** If Monday prices average ₹2,100 and the overall mean is ₹2,000, then `dow_ratio[Monday] = 1.05`.

### Step 3: Compute a Centered 7-Day Rolling Mean

```
rolling_context = centered 7-day rolling mean (min_periods=1)
```

This gives the "local context" — what prices looked like around the missing date.

### Step 4: Fill Missing Values

For each missing price at index `ix`:

```
if rolling_context[ix] is available:
    imputed_value = rolling_context[ix] × dow_ratio[day_of_week(ix)]
```

If no rolling context is available (e.g., the mandi has a very long gap), the cell remains NaN.

---

## Visual Example

Consider a mandi in Maharashtra selling Onion:

| Date       | Day | Actual Price | Rolling Mean | DOW Ratio | Imputed |
|------------|-----|-------------|-------------|-----------|---------|
| 2024-03-04 | Mon | ₹2,100      | ₹2,050      | 1.05      | —       |
| 2024-03-05 | Tue | ₹2,000      | ₹2,040      | 1.00      | —       |
| 2024-03-06 | Wed | **NaN**     | ₹2,030      | 0.98      | **₹1,989** |
| 2024-03-07 | Thu | ₹2,050      | ₹2,020      | 1.02      | —       |
| 2024-03-08 | Fri | ₹2,100      | ₹2,010      | 1.01      | —       |
| 2024-03-09 | Sat | **NaN**     | ₹1,980      | 0.95      | **₹1,881** |
| 2024-03-10 | Sun | **NaN**     | ₹1,950      | 0.90      | **₹1,755** |

Notice how Saturday and Sunday imputed values are systematically lower, reflecting the historical pattern.

---

## Performance (Benchmark Results)

From our multi-threshold benchmark with 5 random seeds:

| Crop   | RMSE (≥50%) | MAE    | MAPE  | Coverage |
|--------|-------------|--------|-------|----------|
| Onion  | 275.38      | 131.63 | 6.07% | 99.75%   |
| Potato | 197.55      | 80.73  | 5.39% | 99.77%   |
| Tomato | 452.43      | 220.97 | 9.71% | 99.74%   |
| Wheat  | 112.97      | 48.98  | 1.95% | 99.47%   |

**Consistently ranked #1 or #2** across all crops and thresholds (50%, 56%, 62%, 69%, 75%), outperforming Rolling Mean, Random Forest, Spline Pipeline, and 7 other methods.

---

## Code Snippet

```python
import pandas as pd
import numpy as np

def impute_dow_ratio(df: pd.DataFrame, col: str = "Modal_Price") -> pd.Series:
    """
    DOW Ratio imputation: fills missing prices using a 7-day centered rolling
    mean adjusted by the historical day-of-week price ratio for each mandi.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: State, Market, Date, and `col` (target).
        Should be sorted by State, Market, Date.
    col : str
        The target column to impute (default: "Modal_Price").

    Returns
    -------
    pd.Series
        The imputed column (same index as df).
    """
    result = df[col].copy()
    df_tmp = df.copy()
    df_tmp["DayOfWeek"] = pd.to_datetime(df_tmp["Date"]).dt.dayofweek

    for _, idx in df_tmp.groupby(["State", "Market"]).groups.items():
        grp = df_tmp.loc[idx].sort_values("Date")
        series = grp[col]
        dow = grp["DayOfWeek"]

        # Step 1: Overall mean
        overall_mean = series.mean()
        if pd.isna(overall_mean) or overall_mean == 0:
            continue

        # Step 2: Day-of-week ratios
        dow_ratios = (series.groupby(dow).mean() / overall_mean).fillna(1.0)

        # Step 3: 7-day centered rolling mean
        rolling_ctx = series.rolling(7, center=True, min_periods=1).mean()

        # Step 4: Fill missing
        filled = series.copy()
        for ix in series.index[series.isna()]:
            ctx = rolling_ctx.loc[ix]
            if pd.notna(ctx):
                filled.loc[ix] = ctx * dow_ratios.get(dow.loc[ix], 1.0)

        result.loc[grp.index] = filled.values

    return result
```

---

## Key Properties

| Property | Value |
|----------|-------|
| **Type** | Temporal + Calendar-aware |
| **Scope** | Per-mandi (does not borrow from other mandis) |
| **Speed** | ~1–2 seconds per crop |
| **Coverage** | ~99.5–99.8% (may leave NaN if mandi has zero known values) |
| **Uses future data?** | Yes (centered rolling mean) — valid for imputation, not for forecasting |
| **Hyperparameters** | Window size (7 days) |

## Limitations

1. **Cannot fill mandis with zero history** — if a mandi has no price data at all, the rolling mean is NaN everywhere
2. **Uses future data** — the centered rolling window looks both backwards and forwards, which is appropriate for filling historical gaps but not for real-time prediction
3. **Assumes stable DOW patterns** — if a mandi's trading days change (e.g., it starts closing on Mondays), the historical ratios become stale
