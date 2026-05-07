# PREPARE Price Forecasting Report

## Scope

This report summarizes the forecasting experiments run on the `final_data_hourly` dataset for the 15-day forecasting horizon across onion, potato, tomato, and wheat.

## 1. Baseline Models Tried

- Strict per-crop `HistGradientBoostingRegressor` baseline.
- Naive baselines:
  - current price
  - 7-day rolling mean
  - 28-day rolling mean
- Tabular target-shift and normalization variants:
  - `delta_current`
  - `delta_current + series_mean_center`
  - `series_zscore`
- Clustered tabular model:
  - mandi clusters from mean arrival, arrival variability, and mean price
  - cluster ID used as an input feature
- Graph models:
  - GraphWaveNet
  - GraphWaveNet+
  - SAGMM-inspired graph mixture model
  - final GAT-GRU graph model with correlation-heavy edges

## 2. Strict Baseline Results

- Onion: `R2 = 0.6281`
- Potato: `R2 = 0.9255`
- Tomato: `R2 = 0.2113`
- Wheat: `R2 = 0.5915`

## 3. Naive Baseline Results

Best naive baseline for each crop:

- Onion: `current_price = 0.7247`
- Potato: `current_price = 0.9007`
- Tomato: `roll_mean_28 = 0.4063`
- Wheat: `roll_mean_7 = 0.6043`

Interpretation:

- The strict model beat naive only for potato.
- Onion, tomato, and wheat required a better problem formulation.

## 4. Tabular Variant Results

### `delta_current`

- Onion: `0.6256`
- Potato: `0.9256`
- Tomato: `0.2615`
- Wheat: `0.6212`

### `delta_current + series_mean_center`

- Onion: `0.7257`
- Potato: `0.9229`
- Tomato: `0.2674`
- Wheat: `0.6381`

### `series_zscore`

- Onion: `0.7917`
- Potato: `0.8474`
- Tomato: `-0.1414`
- Wheat: `0.0696`

Interpretation:

- `delta_current + series_mean_center` was the strongest general tabular improvement.
- `series_zscore` helped onion but hurt the other crops.

## 5. Clustered Tabular Results

- Onion: `0.8078`
- Potato: `0.9047`
- Tomato: `0.3366`
- Wheat: `0.5633`

Interpretation:

- Clustering gave a major improvement for onion.
- It improved tomato over the strict baseline.
- It was worse than the best non-clustered variants for potato and wheat.

## 6. Early Graph Models

Tried:

- GraphWaveNet
- GraphWaveNet+
- SAGMM-inspired model

Outcome:

- All early graph models performed poorly and were not competitive with the tabular baselines.

Main reasons identified:

- weak edge construction
- thinner feature set than the tabular pipeline
- direct graph forecasting instead of anchored residual forecasting
- evaluation mismatch before restricting to observed validation rows

## 7. Final Graph Model

Final graph model:

- GAT-GRU mandi graph model
- one node per mandi series
- richer tabular-style lag, rolling, state, national, weather, and calendar features
- correlation + geo + state + cluster edges
- residual prediction around a crop-specific anchor

Crop-specific anchors:

- Onion: `delta_current`
- Potato: `delta_current`
- Tomato: `delta_roll28`
- Wheat: `delta_roll7`

Reason:

- each crop used the strongest simple anchor found from the naive baseline experiments

## 8. Final Graph Results

- Onion: `0.8544`
- Potato: `0.9287`
- Tomato: `0.4051`
- Wheat: `0.6057`

Additional graph diagnostics:

- Onion `best_alpha = 0.75`
- Potato `best_alpha = 0.85`
- Tomato `best_alpha = 0.0`
- Wheat `best_alpha = 0.1`

Interpretation:

- Onion and potato used the graph correction strongly.
- Tomato fell back almost entirely to the 28-day rolling anchor.
- Wheat used only a very small graph correction.

## 9. Best Current Result by Crop

- Onion: graph GAT-GRU `0.8544`
- Potato: graph GAT-GRU `0.9287`
- Tomato: naive `roll_mean_28 = 0.4063` and graph `0.4051` are effectively tied
- Wheat: tabular `delta_current + series_mean_center = 0.6381`

## 10. Main Conclusions

- Graph modeling now clearly improves onion and slightly improves potato.
- Tomato remains dominated by a smoothed anchor baseline.
- Wheat improves over the strict baseline with graph, but the best tabular model is still stronger.
- The key change that made graph modeling viable was switching to anchored residual prediction with richer features and correlation-based edges.

## 11. Remaining Work

- Controlled graph ablations:
  - anchor-only
  - temporal-only
  - geo-only
  - correlation-only
  - shuffled-graph
- These are needed to isolate exactly why graph structure helps onion and potato.
