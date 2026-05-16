# Price Imputation Benchmark v3 — Modal_Price Only

> Masked 10% of known values, **5 random seeds**.
> Thresholds: **≥50%** and **≥75%** data presence.
> Target: **Modal_Price** only.

## Understanding RMSE vs MAE Divergence

### Why can Method A have lower RMSE but higher MAE than Method B?

**RMSE** (Root Mean Square Error) and **MAE** (Mean Absolute Error) measure different things:

| Metric | Formula | What it emphasizes |
|--------|---------|-------------------|
| **MAE** | mean(|error|) | Treats all errors equally — a ₹10 error and a ₹1000 error contribute proportionally |
| **RMSE** | √mean(error²) | Squares errors first — a ₹1000 error contributes **100×** more than a ₹10 error |

**Example:** Consider two methods imputing 5 masked prices:

| | Method A errors | Method B errors |
|---|----------------|----------------|
| Errors | 50, 50, 50, 50, 50 | 10, 10, 10, 10, 200 |
| **MAE** | **50** | **48** ← lower |
| **RMSE** | **50** ← lower | **90** |

Method B has lower MAE (most predictions are very accurate), but its single large outlier blows up the RMSE.

**What this means for imputation:**

- **Lower MAE, higher RMSE** → The method is accurate *on average* but occasionally makes catastrophic errors (large outliers)
- **Lower RMSE, higher MAE** → The method avoids extreme errors but is slightly less accurate on typical predictions
- **Median AE** is included below to show the "typical" error unaffected by outliers
- **P90 / P99 errors** show the tail — how bad the worst 10% and 1% of predictions are

> **Recommendation:** For forecasting models, **RMSE is preferred** because downstream models are sensitive to large errors. A few catastrophic imputation errors can poison an entire training batch.

## Results — ≥50% Data Present

### Onion (≥50%, 764 mandis)

#### Ranked by RMSE

| Rank | Method | RMSE (±std) | MAE | Median AE | MAPE% | P90 Err | P99 Err | Coverage | Time |
|------|--------|-------------|-----|-----------|-------|---------|---------|----------|------|
| 1 | 7_DOW_Ratio | 275.38±23.43 | 131.63 | 67.14 | 6.07 | 300.74 | 957.09 | 99.75% | 8.3s |
| 2 | 2_Rolling_Mean | 279.15±19.83 | 132.31 | 64.15 | 6.18 | 314.19 | 970.44 | 99.98% | 2.0s |
| 3 | 9_Random_Forest | 310.25±15.82 | 154.6 | 77.21 | 7.29 | 366.59 | 1112.98 | 100.0% | 15.5s |
| 4 | 1_Capped_FFill | 337.25±22.44 | 119.63 | 20.0 | 5.65 | 300.0 | 1171.62 | 99.03% | 1.0s |
| 5 | 8_Spline_Pipeline | 340.84±29.28 | 130.9 | 47.35 | 6.36 | 309.75 | 1240.62 | 100.0% | 80.7s |
| 6 | 6_FFill_Decay | 351.62±17.43 | 143.81 | 63.86 | 6.81 | 340.5 | 1109.85 | 100.0% | 10.2s |
| 7 | 10_SVD_Matrix | 448.09±39.36 | 181.86 | 106.73 | 8.62 | 382.74 | 1092.95 | 100.0% | 97.1s |
| 8 | 4_Seasonal | 834.1±13.74 | 542.37 | 398.0 | 26.24 | 1322.42 | 2294.45 | 99.28% | 0.3s |
| 9 | 3_State_Median | 887.88±37.95 | 358.11 | 200.0 | 16.28 | 800.0 | 2001.98 | 98.98% | 0.3s |
| 10 | 5_Global_Median | 1477.18±27.85 | 980.46 | 800.0 | 47.84 | 1921.4 | 3900.0 | 100.0% | 0.1s |

> **⚡ Rank disagreement:** RMSE winner = **7_DOW_Ratio**, MAE winner = **1_Capped_FFill**. This means 1_Capped_FFill is more accurate on *typical* prices, but 7_DOW_Ratio avoids large outlier errors.

### Potato (≥50%, 713 mandis)

#### Ranked by RMSE

| Rank | Method | RMSE (±std) | MAE | Median AE | MAPE% | P90 Err | P99 Err | Coverage | Time |
|------|--------|-------------|-----|-----------|-------|---------|---------|----------|------|
| 1 | 7_DOW_Ratio | 197.55±25.08 | 80.73 | 37.12 | 5.39 | 186.01 | 635.37 | 99.77% | 8.5s |
| 2 | 2_Rolling_Mean | 197.67±27.67 | 74.71 | 29.92 | 5.01 | 181.47 | 603.27 | 99.98% | 2.0s |
| 3 | 9_Random_Forest | 217.25±28.87 | 86.12 | 34.54 | 5.82 | 207.55 | 704.4 | 100.0% | 15.3s |
| 4 | 1_Capped_FFill | 233.16±30.26 | 72.58 | 10.0 | 4.81 | 200.0 | 809.17 | 99.32% | 1.0s |
| 5 | 8_Spline_Pipeline | 245.31±25.92 | 84.39 | 22.79 | 5.71 | 208.01 | 826.21 | 100.0% | 78.5s |
| 6 | 6_FFill_Decay | 272.81±29.17 | 96.18 | 39.86 | 6.37 | 216.33 | 790.6 | 100.0% | 9.6s |
| 7 | 10_SVD_Matrix | 327.39±37.81 | 113.47 | 59.1 | 7.43 | 244.11 | 727.79 | 100.0% | 90.8s |
| 8 | 4_Seasonal | 679.63±20.8 | 443.23 | 300.0 | 30.05 | 1050.0 | 1620.0 | 99.37% | 0.3s |
| 9 | 3_State_Median | 788.2±45.12 | 272.4 | 150.0 | 17.24 | 600.0 | 1403.9 | 98.99% | 0.3s |
| 10 | 5_Global_Median | 1252.28±32.15 | 750.71 | 573.8 | 49.85 | 1600.0 | 3440.0 | 100.0% | 0.1s |

> **⚡ Rank disagreement:** RMSE winner = **7_DOW_Ratio**, MAE winner = **1_Capped_FFill**. This means 1_Capped_FFill is more accurate on *typical* prices, but 7_DOW_Ratio avoids large outlier errors.

### Tomato (≥50%, 693 mandis)

#### Ranked by RMSE

| Rank | Method | RMSE (±std) | MAE | Median AE | MAPE% | P90 Err | P99 Err | Coverage | Time |
|------|--------|-------------|-----|-----------|-------|---------|---------|----------|------|
| 1 | 7_DOW_Ratio | 452.43±33.03 | 220.97 | 110.54 | 9.71 | 522.33 | 1602.81 | 99.74% | 9.0s |
| 2 | 2_Rolling_Mean | 466.27±17.32 | 236.96 | 117.66 | 10.38 | 568.39 | 1696.6 | 99.96% | 2.0s |
| 3 | 8_Spline_Pipeline | 506.08±17.34 | 215.64 | 87.77 | 9.6 | 522.21 | 1864.38 | 100.0% | 83.7s |
| 4 | 1_Capped_FFill | 510.24±15.74 | 210.5 | 52.0 | 9.17 | 500.0 | 2000.0 | 98.86% | 1.0s |
| 5 | 6_FFill_Decay | 552.05±38.23 | 238.08 | 102.86 | 10.81 | 563.54 | 1877.85 | 100.0% | 11.2s |
| 6 | 9_Random_Forest | 552.65±28.37 | 279.0 | 139.37 | 12.34 | 660.7 | 1995.18 | 100.0% | 16.4s |
| 7 | 10_SVD_Matrix | 679.41±36.07 | 317.39 | 186.85 | 15.1 | 678.36 | 1947.46 | 100.0% | 94.2s |
| 8 | 3_State_Median | 1265.57±52.67 | 528.08 | 300.0 | 24.5 | 1151.0 | 3480.0 | 99.51% | 0.3s |
| 9 | 4_Seasonal | 1337.06±34.14 | 745.42 | 475.5 | 33.07 | 1629.0 | 4852.12 | 99.36% | 0.4s |
| 10 | 5_Global_Median | 2081.61±43.01 | 1246.6 | 900.0 | 63.51 | 2535.9 | 7334.3 | 100.0% | 0.2s |

> **⚡ Rank disagreement:** RMSE winner = **7_DOW_Ratio**, MAE winner = **1_Capped_FFill**. This means 1_Capped_FFill is more accurate on *typical* prices, but 7_DOW_Ratio avoids large outlier errors.

### Wheat (≥50%, 475 mandis)

#### Ranked by RMSE

| Rank | Method | RMSE (±std) | MAE | Median AE | MAPE% | P90 Err | P99 Err | Coverage | Time |
|------|--------|-------------|-----|-----------|-------|---------|---------|----------|------|
| 1 | 2_Rolling_Mean | 107.05±9.12 | 46.2 | 20.69 | 1.84 | 111.48 | 355.23 | 99.94% | 1.3s |
| 2 | 7_DOW_Ratio | 112.97±8.33 | 48.98 | 21.9 | 1.95 | 116.16 | 379.82 | 99.47% | 6.6s |
| 3 | 9_Random_Forest | 114.57±8.38 | 52.32 | 25.02 | 2.08 | 124.37 | 375.48 | 100.0% | 9.7s |
| 4 | 6_FFill_Decay | 136.77±7.64 | 57.52 | 24.46 | 2.27 | 136.95 | 446.19 | 100.0% | 7.1s |
| 5 | 10_SVD_Matrix | 140.04±7.12 | 66.03 | 36.67 | 2.64 | 143.19 | 464.14 | 100.0% | 46.9s |
| 6 | 1_Capped_FFill | 144.45±10.52 | 53.89 | 20.0 | 2.14 | 134.71 | 463.76 | 98.98% | 0.7s |
| 7 | 8_Spline_Pipeline | 171.37±19.89 | 60.78 | 20.79 | 2.42 | 148.28 | 537.33 | 100.0% | 51.6s |
| 8 | 4_Seasonal | 197.77±4.54 | 140.98 | 113.3 | 5.72 | 297.7 | 559.06 | 98.89% | 0.2s |
| 9 | 3_State_Median | 206.62±6.93 | 107.34 | 59.3 | 4.15 | 243.02 | 778.23 | 99.5% | 0.2s |
| 10 | 5_Global_Median | 279.66±5.22 | 195.51 | 151.2 | 7.71 | 371.0 | 856.5 | 100.0% | 0.1s |

### Cross-Crop Average — ≥50%

| Rank | Method | Avg RMSE | Avg MAE | Avg Median AE | Avg MAPE% | Avg P90 | Avg P99 | Time |
|------|--------|----------|---------|---------------|-----------|---------|---------|------|
| 1 | 7_DOW_Ratio | 259.58 | 120.58 | 59.18 | 5.78 | 281.31 | 893.77 | 8.1s |
| 2 | 2_Rolling_Mean | 262.53 | 122.54 | 58.1 | 5.85 | 293.88 | 906.38 | 1.82s |
| 3 | 9_Random_Forest | 298.68 | 143.01 | 69.04 | 6.88 | 339.8 | 1047.01 | 14.22s |
| 4 | 1_Capped_FFill | 306.27 | 114.15 | 25.5 | 5.44 | 283.68 | 1111.14 | 0.92s |
| 5 | 8_Spline_Pipeline | 315.9 | 122.93 | 44.68 | 6.02 | 297.06 | 1117.14 | 73.62s |
| 6 | 6_FFill_Decay | 328.31 | 133.9 | 57.76 | 6.56 | 314.33 | 1056.12 | 9.52s |
| 7 | 10_SVD_Matrix | 398.73 | 169.69 | 97.34 | 8.45 | 362.1 | 1058.08 | 82.25s |
| 8 | 4_Seasonal | 762.14 | 468.0 | 321.7 | 23.77 | 1074.78 | 2331.41 | 0.3s |
| 9 | 3_State_Median | 787.07 | 316.48 | 177.32 | 15.54 | 698.5 | 1916.03 | 0.28s |
| 10 | 5_Global_Median | 1272.68 | 793.32 | 606.25 | 42.23 | 1607.08 | 3882.7 | 0.12s |

## Results — ≥75% Data Present

### Onion (≥75%, 304 mandis)

#### Ranked by RMSE

| Rank | Method | RMSE (±std) | MAE | Median AE | MAPE% | P90 Err | P99 Err | Coverage | Time |
|------|--------|-------------|-----|-----------|-------|---------|---------|----------|------|
| 1 | 7_DOW_Ratio | 209.3±9.2 | 109.83 | 57.0 | 5.44 | 252.26 | 822.53 | 99.93% | 2.3s |
| 2 | 2_Rolling_Mean | 212.5±9.6 | 107.54 | 50.09 | 5.23 | 263.48 | 821.67 | 100.0% | 0.8s |
| 3 | 6_FFill_Decay | 230.35±8.44 | 108.43 | 47.08 | 5.57 | 254.68 | 911.81 | 100.0% | 2.4s |
| 4 | 8_Spline_Pipeline | 234.09±11.86 | 100.81 | 35.51 | 5.15 | 245.69 | 956.05 | 100.0% | 27.4s |
| 5 | 1_Capped_FFill | 237.61±11.38 | 92.67 | 20.0 | 4.63 | 214.0 | 994.45 | 99.5% | 0.4s |
| 6 | 9_Random_Forest | 239.9±9.9 | 124.23 | 59.98 | 6.05 | 297.96 | 942.16 | 100.0% | 5.3s |
| 7 | 10_SVD_Matrix | 271.96±10.2 | 149.86 | 86.59 | 7.62 | 329.83 | 1012.42 | 100.0% | 15.6s |
| 8 | 3_State_Median | 457.93±5.83 | 282.3 | 177.0 | 15.17 | 656.64 | 1752.75 | 98.0% | 0.1s |
| 9 | 4_Seasonal | 612.82±4.22 | 391.28 | 200.0 | 18.8 | 1060.0 | 2000.0 | 99.44% | 0.1s |
| 10 | 5_Global_Median | 1146.42±4.76 | 826.02 | 600.0 | 41.89 | 1938.6 | 3680.0 | 100.0% | 0.0s |

> **⚡ Rank disagreement:** RMSE winner = **7_DOW_Ratio**, MAE winner = **1_Capped_FFill**. This means 1_Capped_FFill is more accurate on *typical* prices, but 7_DOW_Ratio avoids large outlier errors.

### Potato (≥75%, 364 mandis)

#### Ranked by RMSE

| Rank | Method | RMSE (±std) | MAE | Median AE | MAPE% | P90 Err | P99 Err | Coverage | Time |
|------|--------|-------------|-----|-----------|-------|---------|---------|----------|------|
| 1 | 2_Rolling_Mean | 118.2±5.63 | 53.48 | 22.18 | 3.92 | 132.3 | 451.74 | 100.0% | 1.0s |
| 2 | 7_DOW_Ratio | 126.55±5.7 | 59.91 | 28.98 | 4.32 | 134.15 | 488.42 | 99.98% | 3.3s |
| 3 | 9_Random_Forest | 133.08±4.59 | 62.02 | 25.93 | 4.6 | 149.05 | 520.92 | 100.0% | 7.6s |
| 4 | 1_Capped_FFill | 143.88±4.47 | 50.39 | 10.0 | 3.73 | 100.0 | 590.0 | 99.69% | 0.5s |
| 5 | 6_FFill_Decay | 145.37±4.5 | 67.17 | 29.44 | 5.03 | 158.07 | 563.89 | 100.0% | 3.4s |
| 6 | 8_Spline_Pipeline | 153.36±9.19 | 59.5 | 16.39 | 4.43 | 150.74 | 619.05 | 100.0% | 38.3s |
| 7 | 10_SVD_Matrix | 172.19±8.35 | 85.7 | 45.6 | 6.38 | 192.52 | 639.56 | 100.0% | 24.8s |
| 8 | 3_State_Median | 328.7±2.91 | 206.33 | 119.0 | 15.6 | 500.0 | 1250.0 | 98.29% | 0.1s |
| 9 | 4_Seasonal | 577.43±3.01 | 404.12 | 248.0 | 28.91 | 1031.0 | 1500.0 | 99.65% | 0.2s |
| 10 | 5_Global_Median | 796.17±4.07 | 573.11 | 417.0 | 40.69 | 1204.0 | 2780.0 | 100.0% | 0.1s |

> **⚡ Rank disagreement:** RMSE winner = **2_Rolling_Mean**, MAE winner = **1_Capped_FFill**. This means 1_Capped_FFill is more accurate on *typical* prices, but 2_Rolling_Mean avoids large outlier errors.

### Tomato (≥75%, 259 mandis)

#### Ranked by RMSE

| Rank | Method | RMSE (±std) | MAE | Median AE | MAPE% | P90 Err | P99 Err | Coverage | Time |
|------|--------|-------------|-----|-----------|-------|---------|---------|----------|------|
| 1 | 7_DOW_Ratio | 401.03±15.88 | 195.73 | 90.02 | 8.92 | 461.23 | 1603.63 | 99.91% | 2.0s |
| 2 | 2_Rolling_Mean | 405.77±18.42 | 198.08 | 85.97 | 8.99 | 486.83 | 1656.39 | 99.99% | 0.7s |
| 3 | 6_FFill_Decay | 422.16±13.54 | 187.03 | 71.06 | 8.91 | 450.66 | 1763.64 | 100.0% | 2.2s |
| 4 | 8_Spline_Pipeline | 426.84±24.67 | 170.56 | 58.11 | 8.1 | 411.15 | 1692.62 | 100.0% | 23.8s |
| 5 | 1_Capped_FFill | 430.66±11.27 | 167.82 | 46.0 | 7.69 | 475.36 | 1910.0 | 99.43% | 0.3s |
| 6 | 9_Random_Forest | 455.05±18.06 | 226.28 | 103.82 | 10.45 | 535.49 | 1886.73 | 100.0% | 4.8s |
| 7 | 10_SVD_Matrix | 525.17±20.36 | 272.07 | 145.52 | 13.77 | 608.42 | 1987.7 | 100.0% | 12.5s |
| 8 | 3_State_Median | 822.31±11.26 | 435.07 | 202.0 | 22.26 | 1000.0 | 3600.0 | 97.15% | 0.1s |
| 9 | 4_Seasonal | 1131.38±18.42 | 645.54 | 345.0 | 28.99 | 1501.1 | 4589.14 | 99.36% | 0.1s |
| 10 | 5_Global_Median | 2100.49±34.57 | 1270.42 | 708.0 | 57.67 | 3160.0 | 8320.0 | 100.0% | 0.0s |

> **⚡ Rank disagreement:** RMSE winner = **7_DOW_Ratio**, MAE winner = **1_Capped_FFill**. This means 1_Capped_FFill is more accurate on *typical* prices, but 7_DOW_Ratio avoids large outlier errors.

### Wheat (≥75%, 139 mandis)

#### Ranked by RMSE

| Rank | Method | RMSE (±std) | MAE | Median AE | MAPE% | P90 Err | P99 Err | Coverage | Time |
|------|--------|-------------|-----|-----------|-------|---------|---------|----------|------|
| 1 | 2_Rolling_Mean | 72.82±13.97 | 27.91 | 12.15 | 1.14 | 65.93 | 223.54 | 100.0% | 0.4s |
| 2 | 7_DOW_Ratio | 76.66±14.55 | 29.31 | 13.59 | 1.2 | 67.17 | 228.32 | 99.95% | 1.2s |
| 3 | 9_Random_Forest | 78.27±14.47 | 32.02 | 15.11 | 1.31 | 75.83 | 239.91 | 100.0% | 2.2s |
| 4 | 6_FFill_Decay | 91.4±13.44 | 33.35 | 14.31 | 1.36 | 76.82 | 281.51 | 100.0% | 1.2s |
| 5 | 1_Capped_FFill | 96.14±13.31 | 29.39 | 10.0 | 1.2 | 72.0 | 301.68 | 99.67% | 0.2s |
| 6 | 8_Spline_Pipeline | 98.81±11.18 | 33.0 | 10.78 | 1.36 | 77.13 | 321.12 | 100.0% | 12.2s |
| 7 | 10_SVD_Matrix | 115.4±11.27 | 52.28 | 27.8 | 2.16 | 114.24 | 386.21 | 100.0% | 4.9s |
| 8 | 3_State_Median | 140.13±9.82 | 80.5 | 44.5 | 3.29 | 203.63 | 484.72 | 99.68% | 0.1s |
| 9 | 4_Seasonal | 149.78±6.03 | 93.24 | 51.95 | 3.84 | 232.57 | 513.25 | 99.47% | 0.1s |
| 10 | 5_Global_Median | 210.59±5.66 | 158.99 | 125.8 | 6.46 | 356.96 | 529.67 | 100.0% | 0.0s |

### Cross-Crop Average — ≥75%

| Rank | Method | Avg RMSE | Avg MAE | Avg Median AE | Avg MAPE% | Avg P90 | Avg P99 | Time |
|------|--------|----------|---------|---------------|-----------|---------|---------|------|
| 1 | 2_Rolling_Mean | 202.32 | 96.75 | 42.6 | 4.82 | 237.14 | 788.34 | 0.72s |
| 2 | 7_DOW_Ratio | 203.38 | 98.7 | 47.4 | 4.97 | 228.7 | 785.72 | 2.2s |
| 3 | 6_FFill_Decay | 222.32 | 99.0 | 40.47 | 5.22 | 235.06 | 880.21 | 2.3s |
| 4 | 9_Random_Forest | 226.58 | 111.14 | 51.21 | 5.6 | 264.58 | 897.43 | 4.97s |
| 5 | 1_Capped_FFill | 227.07 | 85.07 | 21.5 | 4.31 | 215.34 | 949.03 | 0.35s |
| 6 | 8_Spline_Pipeline | 228.27 | 90.97 | 30.2 | 4.76 | 221.18 | 897.21 | 25.42s |
| 7 | 10_SVD_Matrix | 271.18 | 139.98 | 76.38 | 7.48 | 311.25 | 1006.47 | 14.45s |
| 8 | 3_State_Median | 437.27 | 251.05 | 135.62 | 14.08 | 590.07 | 1771.87 | 0.1s |
| 9 | 4_Seasonal | 617.85 | 383.54 | 211.24 | 20.13 | 956.17 | 2150.6 | 0.12s |
| 10 | 5_Global_Median | 1063.42 | 707.14 | 462.7 | 36.68 | 1664.89 | 3827.42 | 0.02s |

## Threshold Comparison — How RMSE Changes with Data Density

| Method | ≥50% RMSE | ≥75% RMSE | Improvement |
|--------|-----------|-----------|-------------|
| 1_Capped_FFill | 306.3 | 227.1 | 25.9% |
| 2_Rolling_Mean | 262.5 | 202.3 | 22.9% |
| 3_State_Median | 787.1 | 437.3 | 44.4% |
| 4_Seasonal | 762.1 | 617.9 | 18.9% |
| 5_Global_Median | 1272.7 | 1063.4 | 16.4% |
| 6_FFill_Decay | 328.3 | 222.3 | 32.3% |
| 7_DOW_Ratio | 259.6 | 203.4 | 21.6% |
| 8_Spline_Pipeline | 315.9 | 228.3 | 27.7% |
| 9_Random_Forest | 298.7 | 226.6 | 24.1% |
| 10_SVD_Matrix | 398.7 | 271.2 | 32.0% |
