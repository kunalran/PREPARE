# PREPARE Final Experiment Report

## Location

- Canonical results root: `newtests/targeted_rebuild/results/`
- Final 15-day merged comparison table: `newtests/targeted_rebuild/results/final_comparison_15d.csv`
- Graph ablation summary: `newtests/targeted_rebuild/results/gat_gru_ablations/ablation_summary.csv`
- Deep GAT radius-threshold summary: `newtests/targeted_rebuild/results/gat_gru_radius_ablation/radius_ablation_summary.csv`
- Onion low-radius refinement summary: `newtests/targeted_rebuild/results/gat_gru_radius_refine/radius_refine_summary.csv`

## Scope

This report consolidates the completed experiments on the imputed dataset in `final_data_hourly_dow_imputed/` for onion, potato, tomato, and wheat. The final comparison emphasizes the 15-day forecasting horizon because that is where the graph models and most of the specialized experiments were compared directly.

## Experiment Families Completed

- Explicit baselines:
  - previous-day price
  - current price
  - 7-day rolling mean
  - 28-day rolling mean
- Simple tabular model:
  - numeric HistGradientBoosting baseline
- Tabular non-graph variants:
  - anchored HistGB with `delta_current`
  - cross-crop HistGB with paired crop features
  - density-bucket HistGB
  - volume-cluster HistGB
- Graph-style non-deep variants:
  - neighbor-blend threshold sweep
  - graph-augmented HistGB threshold sweep
- Deep graph models:
  - GraphWaveNet+
  - Graph GAT-GRU
- Deep graph ablations on GAT-GRU:
  - graph mode: `full_graph`, `geo_only`, `corr_only`, `temporal_only`, `shuffled_graph`
  - target mode: `delta_current`, `delta_roll7`, `delta_roll28`
  - `k_neighbors`: `6`, `10`, `14`, `18`
  - edge-weight presets: corr-heavy, geo-heavy, state/cluster-heavy

## Best 15-Day Result By Crop

- Onion:
  - best overall: graph GAT-GRU with `delta_roll28`
  - `R2 = 0.8132`, `WAPE = 12.44%`
- Potato:
  - best overall: graph GAT-GRU with `delta_roll28`
  - `R2 = 0.9374`, `WAPE = 9.25%`
- Tomato:
  - best overall: graph GAT-GRU with `delta_roll28`
  - `R2 = 0.3791`, `WAPE = 23.21%`
- Wheat:
  - best overall: graph GAT-GRU with `delta_current`
  - `R2 = 0.7026`, `WAPE = 1.54%`

## Comparison To Yesterday-Price Baseline

15-day `previous_day_price` baseline:

- Onion: `R2 = 0.6508`, `WAPE = 16.80%`
- Potato: `R2 = 0.8937`, `WAPE = 10.12%`
- Tomato: `R2 = 0.2588`, `WAPE = 25.89%`
- Wheat: `R2 = 0.6209`, `WAPE = 1.65%`

Best completed model versus that baseline:

- Onion:
  - yesterday baseline: `0.6508`
  - best overall model: `0.8132`
  - gain: `+0.1624`
- Potato:
  - yesterday baseline: `0.8937`
  - best overall model: `0.9374`
  - gain: `+0.0437`
- Tomato:
  - yesterday baseline: `0.2588`
  - best overall model: `0.3791`
  - gain: `+0.1203`
- Wheat:
  - yesterday baseline: `0.6209`
  - best overall model: `0.7026`
  - gain: `+0.0817`

Conclusion:

- Every crop beat the previous-day baseline.
- The largest gain over the previous-day baseline was on onion.
- The smallest gain was on potato, where the naive baseline was already very strong.

## One-Day Prediction Results

One-day `previous_day_price` baseline:

- Onion: `R2 = 0.9180`, `WAPE = 5.70%`
- Potato: `R2 = 0.9518`, `WAPE = 5.11%`
- Tomato: `R2 = 0.8448`, `WAPE = 9.87%`
- Wheat: `R2 = 0.6859`, `WAPE = 1.43%`

Best completed non-graph learned model at one day:

- Onion:
  - `histgb_delta_current_mean_center`
  - `R2 = 0.9475`, `WAPE = 5.19%`
- Potato:
  - `histgb_cross_crop_delta_current_mean_center`
  - `R2 = 0.9603`, `WAPE = 5.64%`
- Tomato:
  - `numeric_histgb_simple`
  - `R2 = 0.8966`, `WAPE = 8.15%`
- Wheat:
  - `numeric_histgb_simple`
  - `R2 = 0.7873`, `WAPE = 1.27%`

Interpretation:

- One-day prediction is much easier than 15-day prediction for all crops.
- Even the yesterday-price baseline is already very strong at one day.
- The learned non-graph models still beat that one-day baseline on `R2` for all four crops.
- Deep graph models were not run on the one-day horizon in this experiment pass, so the one-day section is a non-graph comparison.

## Graph Vs Non-Graph

Best graph result vs best non-graph result at 15 days:

- Onion:
  - best graph: `0.8132`
  - best non-graph: `0.7788` from `roll_mean_28`
  - graph advantage: `+0.0344`
- Potato:
  - best graph: `0.9374`
  - best non-graph: `0.9336` from `roll_mean_28`
  - graph advantage: `+0.0038`
- Tomato:
  - best graph: `0.3791`
  - best non-graph: `0.3791` from `roll_mean_28`
  - effectively tied
- Wheat:
  - best graph: `0.7026`
  - best non-graph: `0.6663` from `roll_mean_7`
  - graph advantage: `+0.0362`

Interpretation:

- Graph models are best overall on all four crops if we compare the best completed graph run against the best completed non-graph run.
- The gain is strong enough to matter for onion and wheat.
- The gain is real but small for potato.
- For tomato, the graph winner is not meaningfully better than the best non-graph baseline.

## What Actually Helped

### 1. Target formulation mattered more than most graph hyperparameters

- `delta_roll28` was the best GAT-GRU target for onion, potato, and tomato.
- `delta_current` remained best for wheat.
- This target choice changed results more than the `k_neighbors` or edge-weight tweaks.

### 2. Graph structure mattered by crop, but not uniformly

From the graph-mode ablations:

- Onion:
  - `shuffled_graph` slightly beat `full_graph`
  - conclusion: exact edge semantics do not appear important
- Potato:
  - `temporal_only` and `shuffled_graph` were both very competitive
  - conclusion: graph structure is not essential here
- Tomato:
  - `full_graph` beat `shuffled_graph`
  - conclusion: graph structure may help somewhat, but gains remain limited
- Wheat:
  - `full_graph` clearly beat `temporal_only`, `corr_only`, `geo_only`, and `shuffled_graph`
  - conclusion: real graph structure is genuinely useful for wheat

### 3. Explicit radius-threshold graphs inside the deep GAT did not change the story much

We also added a true radius-based geo graph inside the GAT-GRU and tested `75km`, `150km`, and `300km` directly inside the deep model, using the best target mode per crop.

Results:

- Onion:
  - `75km = 0.8131`
  - `150km = 0.8129`
  - `300km = 0.8121`
  - best prior GAT result without explicit radius threshold: `0.8132`
- Potato:
  - `75km = 0.9380`
  - `150km = 0.9380`
  - `300km = 0.9380`
  - slightly better than the earlier best `0.9374`
- Tomato:
  - `75km = 0.3912`
  - `150km = 0.3910`
  - `300km = 0.3905`
  - clearly better than the earlier best `0.3791`
- Wheat:
  - `75km = 0.6893`
  - `150km = 0.6894`
  - `300km = 0.6895`
  - worse than the earlier best `0.7026`

Interpretation:

- Radius thresholding helped tomato clearly.
- It helped potato only marginally.
- It did not materially help onion.
- It hurt wheat versus the best non-radius full-graph setup.
- Tighter radii were mildly better for onion and tomato; potato and wheat were almost flat across radius choices.

### 3b. Refined low-radius check for onion

Because onion had looked slightly better at the tighter end of the first radius sweep, we also tested smaller onion radii directly inside the deep GAT:

- `10km = 0.8133`
- `25km = 0.8134`
- `50km = 0.8132`
- prior non-radius best: `0.8132`
- prior coarse-radius best (`75km`): `0.8131`

Interpretation:

- Onion does prefer a tighter radius than the broad `150-300km` graphs.
- The best refined onion result was `25km`, but the gain is extremely small.
- Practically, onion should be treated as flat across `10-75km`, not as a case with a strong threshold optimum.

### 4. Simple non-graph learned models did not beat the strong naive baselines

Best non-graph learned tabular results at 15 days:

- Onion:
  - best learned non-graph: cross-crop HistGB `0.7075`
  - best baseline: `roll_mean_28 = 0.7788`
- Potato:
  - best learned non-graph: anchored HistGB `0.9320`
  - best baseline: `roll_mean_28 = 0.9336`
- Tomato:
  - best learned non-graph: anchored HistGB `0.2640`
  - best baseline: `roll_mean_28 = 0.3791`
- Wheat:
  - best learned non-graph: cross-crop HistGB `0.6598`
  - best baseline: `roll_mean_7 = 0.6663`

Conclusion:

- The strong non-graph baseline is not the tabular learner. It is the naive rolling baseline.
- The graph work was mainly valuable because it was able to beat those rolling baselines, especially on onion and wheat.

## Cross-Crop Features

Cross-crop pairings were chosen from the strongest daily national price correlations:

- Onion <-> Wheat
- Potato -> Onion
- Tomato -> Potato
- Wheat -> Onion

Observed effect:

- modest help for onion and wheat
- neutral to slightly negative for potato
- negative for tomato

Cross-crop features were not the main source of gains.

## Threshold Distance Sweeps

Threshold-based graph-style experiments tried `75km`, `150km`, and `300km`.

Observed pattern:

- Onion favored the tighter graph, especially `75km`
- Potato slightly preferred wider thresholds
- Tomato showed little stable benefit from threshold choice
- Wheat degraded when the graph became too broad in the neighbor-blend model

These threshold sweeps were useful diagnostics, but the deep GAT-GRU target choice mattered more than the threshold-style graph features. When thresholding was moved into the deep GAT itself, it improved tomato and slightly improved potato, but did not improve onion and worsened wheat.

## Recommended Final Model Per Crop

- Onion:
  - Graph GAT-GRU, `target_mode=delta_roll28`
  - radius threshold is optional; `75km` was effectively tied with the best non-radius run
- Potato:
  - Graph GAT-GRU, `target_mode=delta_roll28`
  - radius threshold can be used, but the gain is tiny
- Tomato:
  - Graph GAT-GRU, `target_mode=delta_roll28`, preferably with a tighter radius such as `75km`
- Wheat:
  - Graph GAT-GRU, `full_graph`, `target_mode=delta_current`
  - keep the non-radius graph construction; radius-thresholding hurt

## Other Graph Models Worth Trying

These are the highest-value next graph directions, in priority order:

- DCRNN:
  - strong candidate for wheat because it explicitly models diffusion over directed or weighted graphs
- MTGNN:
  - useful if we want to learn part of the graph instead of fixing it, especially for onion and potato where edge semantics were weak
- STGCN:
  - lighter than the current GAT-GRU and worth testing as a cleaner spatiotemporal baseline
- GraphSAGE or GCN over a temporal encoder:
  - a simpler graph head on top of a TCN/GRU encoder would test whether attention is overkill
- Heterogeneous mandi graph:
  - add state, district, or crop-level hub nodes instead of only mandi-to-mandi edges
- Dynamic learned sparse graph:
  - learn edges from lagged price co-movement rather than using fixed geo/corr mixes

What not to prioritize first:

- more minor `k_neighbors` tuning on the current GAT-GRU
- more small edge-weight retuning on the same architecture

Those did not move results enough to justify being the next major effort.

## Bottom Line

- Graph beats non-graph overall, but the story is crop-specific.
- The deepest useful finding is not “graphs always win.” It is:
  - wheat clearly benefits from meaningful graph structure
  - onion benefits from graph-style modeling, but not necessarily meaningful edges
  - potato improves a bit, but the gains are small relative to strong rolling baselines
  - tomato improved once the graph model used `delta_roll28` and also improved further with radius-thresholded geo edges, but it is still a relatively hard crop
