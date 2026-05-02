# Agmarknet Data Imputation Strategies 

**Objective:** Impute missing values for `Arrival_Quantity` and `Modal_Price` across a highly sparse, expanded date × mandi grid (2023-2025). The dataset features significant missingness (e.g., Wheat at 75.6%, Sugarcane at 98.6%) and extreme consecutive missing streaks (1,000+ days). The imputation must respect regional agricultural realities, seasonal harvest cycles, and local supply chain shocks to prepare the data for downstream forecasting models enriched with weather and location data.

---

## Tier 1: Simple Baselines (Low Compute, High Interpretability)
*Used to establish benchmark metrics before testing complex algorithms.*

### 1. Capped Forward Fill (LOCF)
* **Method:** Carry the last known observation forward to fill missing gaps.
* **Implementation:** Group by `Mandi` and `Crop`. Apply forward fill (`ffill`) but strictly cap it at a maximum of 3 to 7 days. This prevents ancient prices from polluting multi-year missing streaks.

### 2. Simple Moving Average (SMA) / Rolling Window
* **Method:** Fill gaps using the average of surrounding days.
* **Implementation:** Use a centered rolling window (e.g., 7 days or 15 days). If a value is `NaN`, replace it with the mean of the window. Requires capping to avoid smoothing over massive gaps.

### 3. Broad Categorical Medians (State-Level Fallback)
* **Method:** Impute missing mandi prices using the median price of all other active mandis in the same State on that exact day.
* **Implementation:** Group data by `Date`, `Crop`, and `State`. Calculate the daily median. Map these State-level medians back to the missing rows for individual mandis.

### 4. Simple Historical Seasonality
* **Method:** Use historical calendar norms to fill data.
* **Implementation:** Group by `Mandi`, `Crop`, and `Month` (or `Month` + `DayOfWeek`). Calculate the historical mean/median. Replace missing values with the corresponding historical average for that specific time of year.

### 5. Global Crop Median (Structural Fallback)
* **Method:** The absolute last resort to ensure no `NaN` values remain.
* **Implementation:** Calculate the overall median `Modal_Price` and `Arrival_Quantity` for each `Crop` across the entire 3-year dataset. Fill any remaining holes.

---

## Tier 2: Advanced Spatial & Temporal Proxies

### 6. Forward Fill with Spatial Decay
* **Method:** Blend temporal proximity with spatial reality. As the missing gap gets longer, the weight of the last known price decays, and the weight of the State-level daily average increases.
* **Implementation:** * Calculate Days Since Last Observation ($t$).
    * Calculate State Daily Average ($S$).
    * Let Last Known Price = $P$.
    * Imputed Value = $(w * P) + ((1 - w) * S)$, where $w$ is a decay factor (e.g., $w = e^{-\lambda t}$).

### 7. Calendar & Seasonality Indexing (Day-of-Week Focus)
* **Method:** Adjust for extreme reporting drop-offs on specific days (e.g., Sundays). 
* **Implementation:** Calculate the ratio of Sunday prices to the weekly average for a specific mandi. Impute missing Sundays by taking the surrounding week's average and applying the historical Sunday ratio.

---

## Tier 3: The Literature-Based Hybrid Pipeline
*Based on the specific multi-step methodology for manual data-entry correction.*

### 8. The 4-Step Smoothing & Spline Pipeline
* **Method:** A structured approach to filter outliers and mathematically connect data points.
* **Implementation:**
    1.  **Outlier Flagging:** Check if the current value is > 6x or < 1/6th of the previous week's average. If so, flag as `NaN`. If the previous week is empty, backtrack to the nearest ground-truth value.
    2.  **Year-Wise Spline:** For mandis with >50% data availability in a given year, apply a year-wise cubic spline imputation.
    3.  **Window-Based Smoothing:** For each spline-imputed point, check a 15-day window (7 days before, 7 days after). If the imputed value is beyond +/- 15% of the window's extrema, flag it as an outlier. Expand the window if no ground-truth values exist inside it.
    4.  **Linear Interpolation:** Use basic linear interpolation to fill any remaining `NaN` values.

---

## Tier 4: Machine Learning & Multi-Variate (Predictive)

### 9. Random Forest / Iterative Imputation (MissForest)
* **Method:** Treat the missing price/quantity as a target variable to be predicted by an ML model.
* **Implementation:** Use features like `State`, `Month`, `DayOfWeek`, and historical lags. The model learns non-linear relationships (e.g., how Wheat behaves in Punjab during April versus November).

### 10. Matrix Factorization (SoftImpute / SVD)
* **Method:** Treat the dataset as a Date × Mandi matrix and learn latent collaborative factors.
* **Implementation:** Pivot the data so Dates are rows and Mandis are columns. Apply SoftImpute or Truncated SVD to fill the matrix based on correlated market behaviors, regardless of geographical borders.