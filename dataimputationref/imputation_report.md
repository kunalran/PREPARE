# Data4 — EDA & Imputation Report

> Data4 contains expanded date×mandi grids enriched with **weather** (temperature,
> precipitation, solar radiation, humidity) and **location** (lat/lon) data.
> This report covers EDA findings and DOW_Ratio imputation accuracy.

# Part 1: Exploratory Data Analysis

## Onion

### Overview

| Metric | Value |
|--------|-------|
| Total rows | 1,615,504 |
| Unique mandis | 1,474 |
| Unique states | 28 |
| Date range | 2023-01-01 00:00:00 to 2025-12-31 00:00:00 (1096 days) |
| Modal_Price present | 499,856 (30.9%) |
| Modal_Price missing | 1,115,648 (69.1%) |
| Arrival_Quantity present | 499,856 (30.9%) |

### Weather & Location Data

| Column | Description | Non-null | Mean | Min | Max | Std |
|--------|------------|----------|------|-----|-----|-----|
| latitude | Latitude (°) | 1,615,504 (100.0%) | 21.92 | 7.00 | 34.51 | 7.15 |
| longitude | Longitude (°) | 1,615,504 (100.0%) | 79.01 | 69.85 | 94.72 | 4.61 |
| t | Temperature (°C) | 1,615,504 (100.0%) | 25.56 | -14.85 | 41.95 | 5.73 |
| tp | Total Precipitation (m) | 1,615,504 (100.0%) | 0.15 | 0.00 | 17.55 | 0.40 |
| ssr | Surface Solar Radiation (MJ/m²) | 1,615,504 (100.0%) | 0.62 | 0.02 | 1.15 | 0.19 |
| r | Relative Humidity (%) | 1,615,504 (100.0%) | 69.04 | 12.47 | 99.94 | 16.32 |

- **Weather NaN total:** 0 across 4 columns
- **Location NaN total:** 0 across 2 columns

### Modal_Price Missing by Day of Week

| Day | Total | Present | % Missing |
|-----|-------|---------|-----------|
| Mon | 231,418 | 76,966 | 66.7% |
| Tue | 231,418 | 77,927 | 66.3% |
| Wed | 231,418 | 76,716 | 66.8% |
| Thu | 229,944 | 76,260 | 66.8% |
| Fri | 229,944 | 76,249 | 66.8% |
| Sat | 229,944 | 69,599 | 69.7% |
| Sun | 231,418 | 46,139 | 80.1% |

### Per-Year Data Density

| Year | Mandis | Total Rows | Present | % Present |
|------|--------|-----------|---------|-----------|
| 2023 | 1474 | 538,010 | 185,868 | 34.5% |
| 2024 | 1474 | 539,484 | 173,953 | 32.2% |
| 2025 | 1474 | 538,010 | 140,035 | 26.0% |

### ≥50% Density Filter Preview

- **Mandis passing ≥50% filter:** 764 / 1,474
- **Rows after filter:** ~470,964 / 1,615,504

---

## Potato

### Overview

| Metric | Value |
|--------|-------|
| Total rows | 1,478,220 |
| Unique mandis | 1,387 |
| Unique states | 29 |
| Date range | 2023-01-01 00:00:00 to 2025-11-30 00:00:00 (1065 days) |
| Modal_Price present | 505,189 (34.2%) |
| Modal_Price missing | 973,031 (65.8%) |
| Arrival_Quantity present | 505,189 (34.2%) |

### Weather & Location Data

| Column | Description | Non-null | Mean | Min | Max | Std |
|--------|------------|----------|------|-----|-----|-----|
| latitude | Latitude (°) | 1,478,220 (100.0%) | 22.35 | 7.00 | 34.51 | 7.24 |
| longitude | Longitude (°) | 1,478,220 (100.0%) | 79.67 | 69.85 | 95.00 | 4.93 |
| t | Temperature (°C) | 1,478,220 (100.0%) | 25.63 | -14.85 | 41.95 | 5.81 |
| tp | Total Precipitation (m) | 1,478,220 (100.0%) | 0.16 | 0.00 | 17.55 | 0.41 |
| ssr | Surface Solar Radiation (MJ/m²) | 1,478,220 (100.0%) | 0.62 | 0.02 | 1.15 | 0.19 |
| r | Relative Humidity (%) | 1,478,220 (100.0%) | 69.74 | 12.47 | 99.94 | 16.06 |

- **Weather NaN total:** 0 across 4 columns
- **Location NaN total:** 0 across 2 columns

### Modal_Price Missing by Day of Week

| Day | Total | Present | % Missing |
|-----|-------|---------|-----------|
| Mon | 210,976 | 76,230 | 63.9% |
| Tue | 210,976 | 77,216 | 63.4% |
| Wed | 210,976 | 77,059 | 63.5% |
| Thu | 210,976 | 76,242 | 63.9% |
| Fri | 210,976 | 76,966 | 63.5% |
| Sat | 210,976 | 71,907 | 65.9% |
| Sun | 212,364 | 49,569 | 76.7% |

### Per-Year Data Density

| Year | Mandis | Total Rows | Present | % Present |
|------|--------|-----------|---------|-----------|
| 2023 | 1387 | 506,620 | 193,774 | 38.2% |
| 2024 | 1387 | 508,008 | 195,682 | 38.5% |
| 2025 | 1387 | 463,592 | 115,733 | 25.0% |

### ≥50% Density Filter Preview

- **Mandis passing ≥50% filter:** 713 / 1,387
- **Rows after filter:** ~459,012 / 1,478,220

---

## Sugarcane

### Overview

| Metric | Value |
|--------|-------|
| Total rows | 18,924 |
| Unique mandis | 19 |
| Unique states | 3 |
| Date range | 2023-01-09 00:00:00 to 2025-09-30 00:00:00 (996 days) |
| Modal_Price present | 273 (1.4%) |
| Modal_Price missing | 18,651 (98.6%) |
| Arrival_Quantity present | 273 (1.4%) |

### Weather & Location Data

| Column | Description | Non-null | Mean | Min | Max | Std |
|--------|------------|----------|------|-----|-----|-----|
| latitude | Latitude (°) | 18,924 (100.0%) | 22.47 | 19.57 | 23.63 | 0.98 |
| longitude | Longitude (°) | 18,924 (100.0%) | 82.18 | 76.42 | 83.97 | 1.96 |
| t | Temperature (°C) | 18,924 (100.0%) | 25.91 | 11.77 | 39.66 | 4.78 |
| tp | Total Precipitation (m) | 18,924 (100.0%) | 0.18 | 0.00 | 7.73 | 0.42 |
| ssr | Surface Solar Radiation (MJ/m²) | 18,924 (100.0%) | 0.62 | 0.04 | 1.02 | 0.19 |
| r | Relative Humidity (%) | 18,924 (100.0%) | 64.58 | 14.20 | 99.94 | 19.78 |

- **Weather NaN total:** 0 across 4 columns
- **Location NaN total:** 0 across 2 columns

### Modal_Price Missing by Day of Week

| Day | Total | Present | % Missing |
|-----|-------|---------|-----------|
| Mon | 2,717 | 42 | 98.5% |
| Tue | 2,717 | 40 | 98.5% |
| Wed | 2,698 | 39 | 98.6% |
| Thu | 2,698 | 45 | 98.3% |
| Fri | 2,698 | 44 | 98.4% |
| Sat | 2,698 | 36 | 98.7% |
| Sun | 2,698 | 27 | 99.0% |

### Per-Year Data Density

| Year | Mandis | Total Rows | Present | % Present |
|------|--------|-----------|---------|-----------|
| 2023 | 19 | 6,783 | 91 | 1.3% |
| 2024 | 19 | 6,954 | 84 | 1.2% |
| 2025 | 19 | 5,187 | 98 | 1.9% |

### ≥50% Density Filter Preview

- **Mandis passing ≥50% filter:** 0 / 19
- **Rows after filter:** ~0 / 18,924

---

## Tomato

### Overview

| Metric | Value |
|--------|-------|
| Total rows | 1,650,576 |
| Unique mandis | 1,505 |
| Unique states | 27 |
| Date range | 2023-01-01 00:00:00 to 2025-12-31 00:00:00 (1096 days) |
| Modal_Price present | 497,359 (30.1%) |
| Modal_Price missing | 1,153,217 (69.9%) |
| Arrival_Quantity present | 497,359 (30.1%) |

### Weather & Location Data

| Column | Description | Non-null | Mean | Min | Max | Std |
|--------|------------|----------|------|-----|-----|-----|
| latitude | Latitude (°) | 1,650,576 (100.0%) | 21.90 | 7.00 | 34.07 | 7.32 |
| longitude | Longitude (°) | 1,650,576 (100.0%) | 79.85 | 69.85 | 95.00 | 5.26 |
| t | Temperature (°C) | 1,650,576 (100.0%) | 25.50 | -14.85 | 41.95 | 5.74 |
| tp | Total Precipitation (m) | 1,650,576 (100.0%) | 0.16 | 0.00 | 17.55 | 0.41 |
| ssr | Surface Solar Radiation (MJ/m²) | 1,650,576 (100.0%) | 0.62 | 0.02 | 1.13 | 0.19 |
| r | Relative Humidity (%) | 1,650,576 (100.0%) | 70.04 | 12.47 | 99.94 | 15.83 |

- **Weather NaN total:** 0 across 4 columns
- **Location NaN total:** 0 across 2 columns

### Modal_Price Missing by Day of Week

| Day | Total | Present | % Missing |
|-----|-------|---------|-----------|
| Mon | 236,442 | 75,303 | 68.2% |
| Tue | 236,442 | 76,116 | 67.8% |
| Wed | 236,442 | 76,042 | 67.8% |
| Thu | 234,936 | 76,062 | 67.6% |
| Fri | 234,936 | 76,718 | 67.3% |
| Sat | 234,936 | 69,337 | 70.5% |
| Sun | 236,442 | 47,781 | 79.8% |

### Per-Year Data Density

| Year | Mandis | Total Rows | Present | % Present |
|------|--------|-----------|---------|-----------|
| 2023 | 1505 | 549,690 | 178,738 | 32.5% |
| 2024 | 1505 | 551,196 | 171,160 | 31.1% |
| 2025 | 1505 | 549,690 | 147,461 | 26.8% |

### ≥50% Density Filter Preview

- **Mandis passing ≥50% filter:** 693 / 1,505
- **Rows after filter:** ~492,870 / 1,650,576

---

## Wheat

### Overview

| Metric | Value |
|--------|-------|
| Total rows | 1,513,365 |
| Unique mandis | 1,421 |
| Unique states | 16 |
| Date range | 2023-01-01 00:00:00 to 2025-11-30 00:00:00 (1065 days) |
| Modal_Price present | 369,815 (24.4%) |
| Modal_Price missing | 1,143,550 (75.6%) |
| Arrival_Quantity present | 369,815 (24.4%) |

### Weather & Location Data

| Column | Description | Non-null | Mean | Min | Max | Std |
|--------|------------|----------|------|-----|-----|-----|
| latitude | Latitude (°) | 1,513,365 (100.0%) | 24.47 | 9.50 | 31.90 | 4.07 |
| longitude | Longitude (°) | 1,513,365 (100.0%) | 77.14 | 68.97 | 88.56 | 3.10 |
| t | Temperature (°C) | 1,513,365 (100.0%) | 25.82 | 4.12 | 41.99 | 5.74 |
| tp | Total Precipitation (m) | 1,513,365 (100.0%) | 0.13 | 0.00 | 13.88 | 0.38 |
| ssr | Surface Solar Radiation (MJ/m²) | 1,513,365 (100.0%) | 0.63 | 0.02 | 1.09 | 0.19 |
| r | Relative Humidity (%) | 1,513,365 (100.0%) | 64.69 | 12.47 | 99.97 | 18.51 |

- **Weather NaN total:** 0 across 4 columns
- **Location NaN total:** 0 across 2 columns

### Modal_Price Missing by Day of Week

| Day | Total | Present | % Missing |
|-----|-------|---------|-----------|
| Mon | 215,992 | 58,672 | 72.8% |
| Tue | 215,992 | 59,250 | 72.6% |
| Wed | 215,992 | 58,669 | 72.8% |
| Thu | 215,992 | 58,888 | 72.7% |
| Fri | 215,992 | 61,365 | 71.6% |
| Sat | 215,992 | 53,130 | 75.4% |
| Sun | 217,413 | 19,841 | 90.9% |

### Per-Year Data Density

| Year | Mandis | Total Rows | Present | % Present |
|------|--------|-----------|---------|-----------|
| 2023 | 1421 | 518,665 | 145,896 | 28.1% |
| 2024 | 1421 | 520,086 | 142,963 | 27.5% |
| 2025 | 1421 | 474,614 | 80,956 | 17.1% |

### ≥50% Density Filter Preview

- **Mandis passing ≥50% filter:** 475 / 1,421
- **Rows after filter:** ~295,535 / 1,513,365

---

# Part 2: Imputation Pipeline

**Steps:**
1. Filter to mandis with ≥50% Modal_Price data per year
2. Drop `Arrival_Quantity` and `Arrival_Unit` columns
3. Impute Modal_Price using **DOW_Ratio** (Method 7)
4. Evaluate accuracy: mask 10% of known values × 5 random seeds

### Sugarcane
⚠ No mandis with ≥50% data — skipped.

## Summary of Filtering & Imputation

| Crop | Mandis (before→after) | Rows (before→after) | Known Prices | Filled % | Remaining NaN | Time |
|------|-----------------------|---------------------|--------------|----------|---------------|------|
| Onion | 1,474→764 | 1,615,504→470,964 | 318,744 | 84.71% | 72,006 | 1.4s |
| Potato | 1,387→713 | 1,478,220→459,012 | 330,418 | 90.3% | 44,526 | 1.4s |
| Tomato | 1,505→693 | 1,650,576→492,870 | 328,581 | 84.54% | 76,192 | 1.5s |
| Wheat | 1,421→475 | 1,513,365→295,535 | 196,342 | 92.7% | 21,575 | 1.1s |

## Imputation Accuracy (DOW_Ratio, Method 7)

> Evaluated by masking 10% of known Modal_Price values across **5 random seeds**, then comparing imputed values to ground truth.

| Crop | RMSE (±std) | MAE | Median AE | MAPE (%) | P90 Error | P99 Error | Coverage |
|------|-------------|-----|-----------|----------|-----------|-----------|----------|
| Onion | 275.38±23.43 | 131.63 | 67.14 | 6.07 | 300.74 | 957.09 | 99.75% |
| Potato | 197.55±25.08 | 80.73 | 37.12 | 5.39 | 186.01 | 635.37 | 99.77% |
| Tomato | 452.43±33.03 | 220.97 | 110.54 | 9.71 | 522.33 | 1602.81 | 99.74% |
| Wheat | 112.97±8.33 | 48.98 | 21.9 | 1.95 | 116.16 | 379.82 | 99.47% |

## Output Files

All imputed files saved to `data4_imputed/`:

- `agmarknet_onion_data_imputed.csv` — 764 mandis, 470,964 rows
- `agmarknet_potato_data_imputed.csv` — 713 mandis, 459,012 rows
- `agmarknet_tomato_data_imputed.csv` — 693 mandis, 492,870 rows
- `agmarknet_wheat_data_imputed.csv` — 475 mandis, 295,535 rows

**Columns in output files:**
State, District, Market, Commodity_Group, Commodity, Date, Day_of_Week, Modal_Price *(imputed)*, Price_Unit, latitude, longitude, t, tp, ssr, r

> `Arrival_Quantity` and `Arrival_Unit` have been dropped as per requirement.