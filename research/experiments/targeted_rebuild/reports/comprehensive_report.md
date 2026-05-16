# PREPARE Comprehensive Results Report

## Scope

This report covers the work requested in `newtests/newtests.txt` using the imputed dataset in `final_data_hourly_dow_imputed/`.

Crops covered:

- onion
- potato
- tomato
- wheat

Forecast horizons emphasized here:

- `1d`, because the request asked for one-day prediction results
- `15d`, because the graph runs and the specialized comparisons were done there

Primary result folders:

- `newtests/targeted_rebuild/results/baseline_metrics/`
- `newtests/targeted_rebuild/results/simple_numeric_metrics/`
- `newtests/targeted_rebuild/results/anchored_histgb_metrics/`
- `newtests/targeted_rebuild/results/cross_crop_histgb_metrics/`
- `newtests/targeted_rebuild/results/density_variant_metrics/`
- `newtests/targeted_rebuild/results/volume_variant_metrics/`
- `newtests/targeted_rebuild/results/graph_neighbor_blend_metrics/`
- `newtests/targeted_rebuild/results/graph_histgb_metrics/`
- `newtests/targeted_rebuild/results/graph_wavenet_plus_mps/`
- `newtests/targeted_rebuild/results/graph_gat_gru_mps/`
- `newtests/targeted_rebuild/results/gat_gru_ablations/`
- `newtests/targeted_rebuild/results/gat_gru_radius_ablation/`
- `newtests/targeted_rebuild/results/gat_gru_radius_refine/`
- `newtests/targeted_rebuild/results/inventory/`

## Work Completed Against `newtests.txt`

Requested item: compute a baseline R2 by predicting the previous day’s price at every day

- Done in `baseline_metrics/baseline_metrics.csv`
- Reported below for `1d` and `15d`

Requested item: check how many mandis are present for each crop and in what region

- Done in:
  - `inventory/crop_inventory.csv`
  - `inventory/state_mandi_counts.csv`
  - `inventory/district_mandi_counts.csv`

Requested item: try out simple models

- Done in `simple_numeric_metrics/simple_numeric_metrics.csv`

Requested item: try other models

- Done with:
  - anchored HistGradientBoosting
  - cross-crop HistGradientBoosting
  - density-bucket HistGradientBoosting
  - volume-cluster HistGradientBoosting
  - graph-augmented HistGradientBoosting
  - neighbor-blend graph baseline
  - GraphWaveNet+
  - GAT-GRU

Requested item: experiment with shifting the price values or subtracting a common baseline and then predicting prices

- Done in `anchored_histgb_metrics/anchored_histgb_metrics.csv`
- Also revisited in GAT-GRU target-mode tests with `delta_current`, `delta_roll7`, and `delta_roll28`

Requested item: try out graph based models

- Done in:
  - `graph_neighbor_blend_metrics/graph_neighbor_blend_metrics.csv`
  - `graph_histgb_metrics/graph_histgb_metrics.csv`
  - `graph_wavenet_plus_mps/graph_training_summary.json`
  - `graph_gat_gru_mps/graph_training_summary.json`

Requested item: experiment with different distance thresholds for graph nodes on different graph models for all crops

- Done in:
  - `graph_neighbor_blend_metrics/graph_neighbor_blend_metrics.csv`
  - `graph_histgb_metrics/graph_histgb_metrics.csv`
  - `gat_gru_radius_ablation/radius_ablation_summary.csv`
  - `gat_gru_radius_refine/radius_refine_summary.csv`

Requested item: try out models that cluster or differentiate dense vs sparse mandis

- Done in `density_variant_metrics/density_variant_metrics.csv`

Requested item: try out grouping mandis based on output volume

- Done in `volume_variant_metrics/volume_variant_metrics.csv`

Requested item: test models that have prices of one other crop with them as a feature

- Done in:
  - `cross_crop_histgb_metrics/cross_crop_histgb_metrics.csv`
  - `cross_crop_pairings.csv`
  - `inventory/cross_crop_daily_correlations.csv`

## Data Inventory And Regional Coverage

### Crop inventory

| crop | mandis | rows | date_range | top_state |
| --- | --- | --- | --- | --- |
| onion | 764 | 470964 | 2023-01-01 to 2025-12-31 | Uttar Pradesh (170) |
| potato | 713 | 459012 | 2023-01-01 to 2025-11-30 | Uttar Pradesh (182) |
| tomato | 693 | 492870 | 2023-01-01 to 2025-12-31 | Uttar Pradesh (165) |
| wheat | 475 | 295535 | 2023-01-01 to 2025-11-30 | Madhya Pradesh (167) |

### Regional coverage notes

- Onion has the widest mandi coverage in the dataset.
- Wheat has the smallest mandi network and the fewest rows.
- Uttar Pradesh dominates mandi counts for onion, potato, and tomato.
- Madhya Pradesh dominates mandi counts for wheat.

### Strongest cross-crop national price correlation

| crop | strongest_other_crop | corr |
| --- | --- | --- |
| onion | wheat | 0.7641 |
| potato | onion | 0.6781 |
| tomato | potato | 0.2174 |
| wheat | onion | 0.7641 |

### Mean geographic graph density

| crop | 75km_mean_neighbors | 150km_mean_neighbors | 300km_mean_neighbors |
| --- | --- | --- | --- |
| onion | 13.60 | 41.06 | 112.26 |
| potato | 12.20 | 37.71 | 100.83 |
| tomato | 11.43 | 35.09 | 96.31 |
| wheat | 9.04 | 35.04 | 111.05 |

This matters later because the graph threshold sweeps were not being applied to equally dense networks. Wheat starts from the sparsest local graph at `75km`.

## One-Day Results

What was done:

- Baseline models for all crops
- Simple numeric HistGradientBoosting
- Shifted or baseline-subtracted HistGradientBoosting
- Cross-crop HistGradientBoosting

What was not done at `1d`:

- Deep graph runs were not executed for the `1d` horizon in this experiment pass
- Dense-sparse, volume, and threshold graph comparisons were kept at `15d`

### One-day comparison by crop

| crop | previous_day_r2 | best_baseline | best_simple | best_shifted | best_cross_crop | best_completed_1d |
| --- | --- | --- | --- | --- | --- | --- |
| onion | 0.9180 | current_price (0.9535) | numeric_histgb_simple (0.9392) | histgb_delta_current_mean_center (0.9475) | histgb_cross_crop_delta_current_mean_center (0.9467) | current_price (0.9535) |
| potato | 0.9518 | current_price (0.9638) | numeric_histgb_simple (0.9543) | histgb_delta_current_mean_center (0.9602) | histgb_cross_crop_delta_current_mean_center (0.9603) | current_price (0.9638) |
| tomato | 0.8448 | current_price (0.8876) | numeric_histgb_simple (0.8966) | histgb_delta_current_mean_center (0.8962) | histgb_cross_crop_delta_current_mean_center (0.8961) | numeric_histgb_simple (0.8966) |
| wheat | 0.6859 | roll_mean_7 (0.7530) | numeric_histgb_simple (0.7873) | histgb_delta_current_mean_center (0.7500) | histgb_cross_crop_delta_current_mean_center (0.7531) | numeric_histgb_simple (0.7873) |

### One-day findings

- The previous-day baseline is already strong at `1d` for every crop.
- Onion and potato are so persistent at `1d` that the strongest baseline is still `current_price`.
- Tomato and wheat benefit from learned non-graph models at `1d`.
- Cross-crop features do not materially change the `1d` result relative to the shifted non-graph model family.

## Fifteen-Day Results

What was done:

- Baseline models
- Simple numeric HistGradientBoosting
- Shifted or baseline-subtracted HistGradientBoosting
- Cross-crop HistGradientBoosting
- Dense-sparse mandi differentiation
- Volume grouping
- Graph-style non-deep models
- Deep graph models
- Graph ablations
- Graph radius-threshold sweeps

### Fifteen-day comparison by crop

| crop | previous_day_r2 | best_baseline | best_simple | best_shifted | best_cross_crop | dense_sparse | volume_grouping | best_graph_style | best_deep_graph |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| onion | 0.6508 | roll_mean_28 (0.7788) | numeric_histgb_simple (-0.0535) | histgb_delta_current_mean_center (0.7010) | histgb_cross_crop_delta_current_mean_center (0.7075) | density_bucket_histgb (0.6796) | volume_cluster_histgb (0.6981) | graph_histgb_augmented (0.7014) | target_mode__delta_roll28 (0.8132) |
| potato | 0.8937 | roll_mean_28 (0.9336) | numeric_histgb_simple (0.9017) | histgb_delta_current_mean_center (0.9320) | histgb_cross_crop_delta_current_mean_center (0.9319) | density_bucket_histgb (0.9317) | volume_cluster_histgb (0.9310) | graph_histgb_augmented (0.9312) | target_mode__delta_roll28 (0.9374) |
| tomato | 0.2588 | roll_mean_28 (0.3791) | numeric_histgb_simple (0.2647) | histgb_delta_current_mean_center (0.2640) | histgb_cross_crop_delta_current_mean_center (0.2376) | density_bucket_histgb (0.2489) | volume_cluster_histgb (0.2501) | graph_histgb_augmented (0.2568) | target_mode__delta_roll28 (0.3791) |
| wheat | 0.6209 | roll_mean_7 (0.6663) | numeric_histgb_simple (0.6417) | histgb_delta_current_mean_center (0.6478) | histgb_cross_crop_delta_current_mean_center (0.6598) | density_bucket_histgb (0.6439) | volume_cluster_histgb (0.6538) | graph_neighbor_blend (0.6630) | full_graph_delta_current_default (0.7026) |

### Fifteen-day findings

- Onion and wheat get their best results from deep graph models.
- Potato gets a small but real gain from deep graph models over the strongest rolling baseline.
- Tomato is the hardest crop at `15d`. The final deep graph result ties the strongest non-graph rolling baseline before radius refinement, and only improves after thresholding is pushed directly into the GAT graph.
- The simple numeric model is poor at `15d`, especially on onion.

## Baseline Test

What was done:

- Previous-day price baseline
- Current-price baseline
- Rolling mean `7d`
- Rolling mean `28d`

### Previous-day baseline requested in `newtests.txt`

`1d` previous-day baseline:

- onion: `R2 = 0.9180`
- potato: `R2 = 0.9518`
- tomato: `R2 = 0.8448`
- wheat: `R2 = 0.6859`

`15d` previous-day baseline:

- onion: `R2 = 0.6508`
- potato: `R2 = 0.8937`
- tomato: `R2 = 0.2588`
- wheat: `R2 = 0.6209`

Best baseline at `1d`:

- onion: `current_price = 0.9535`
- potato: `current_price = 0.9638`
- tomato: `current_price = 0.8876`
- wheat: `roll_mean_7 = 0.7530`

Best baseline at `15d`:

- onion: `roll_mean_28 = 0.7788`
- potato: `roll_mean_28 = 0.9336`
- tomato: `roll_mean_28 = 0.3791`
- wheat: `roll_mean_7 = 0.6663`

Result:

- Every final model family had to clear a strong naive baseline, not just the previous-day baseline.
- At `15d`, the rolling baselines are the hard baseline to beat.

## Simple Model Test

What was done:

- One simple non-graph numeric HistGradientBoosting model across `1d` to `15d`

Result:

- `1d`:
  - onion: `0.9392`
  - potato: `0.9543`
  - tomato: `0.8966`
  - wheat: `0.7873`
- `15d`:
  - onion: `-0.0535`
  - potato: `0.9017`
  - tomato: `0.2647`
  - wheat: `0.6417`

Conclusion:

- This model is competitive only at `1d`.
- It is not a strong `15d` solution, especially for onion and tomato.

## Shifted Or Baseline-Subtracted Target Test

What was done:

- Anchored HistGradientBoosting using `delta_current` with series mean centering
- GAT-GRU target-mode sweep across:
  - `delta_current`
  - `delta_roll7`
  - `delta_roll28`

Result for anchored HistGradientBoosting:

- `1d` best anchored scores:
  - onion: `0.9475`
  - potato: `0.9602`
  - tomato: `0.8962`
  - wheat: `0.7500`
- `15d` best anchored scores:
  - onion: `0.7010`
  - potato: `0.9320`
  - tomato: `0.2640`
  - wheat: `0.6478`

Result for GAT target modes:

- onion: best at `15d` with `delta_roll28`
- potato: best at `15d` with `delta_roll28`
- tomato: best at `15d` with `delta_roll28`
- wheat: best at `15d` with `delta_current`

Conclusion:

- The shift or baseline formulation matters more than most GAT hyperparameter changes.
- `delta_roll28` is the correct target transform for onion, potato, and tomato.
- Wheat is the exception.

## Cross-Crop Feature Test

What was done:

- Added one paired crop price feature using the strongest national daily correlation rule
- Pairings recorded in `cross_crop_pairings.csv`

Pairings used:

- onion with wheat
- potato with onion
- tomato with potato
- wheat with onion

Result:

- `1d`:
  - cross-crop is close to anchored but not better in a meaningful way
- `15d`:
  - onion improves slightly over anchored: `0.7075` vs `0.7010`
  - potato is flat relative to anchored: `0.9319` vs `0.9320`
  - tomato gets worse: `0.2376` vs `0.2640`
  - wheat improves slightly over anchored: `0.6598` vs `0.6478`

Conclusion:

- Cross-crop information is useful only where the empirical correlation is strong enough and stable enough.
- Onion and wheat benefit modestly.
- Potato is neutral.
- Tomato does not benefit.

## Dense Versus Sparse Mandi Test

What was done:

- Density-bucket HistGradientBoosting at `15d`
- Supported by graph density summaries in `inventory/graph_density_summary.csv`

Result:

- onion: `0.6796`
- potato: `0.9317`
- tomato: `0.2489`
- wheat: `0.6439`

Conclusion:

- Density-aware partitioning does not beat the strongest rolling baseline.
- It is useful as a diagnostic and slightly competitive for potato, but not as the best final model.

## Output Volume Grouping Test

What was done:

- Volume-cluster HistGradientBoosting at `15d`

Result:

- onion: `0.6981`
- potato: `0.9310`
- tomato: `0.2501`
- wheat: `0.6538`

Conclusion:

- Volume grouping is modestly useful.
- It still does not beat the strongest baseline or the deep graph models.

## Graph-Based Model Test

What was done:

- Neighbor-blend graph baseline
- Graph-augmented HistGradientBoosting
- GraphWaveNet+
- GAT-GRU on Apple Silicon MPS

### Non-deep graph-style models at `15d`

Best result by crop from neighbor-blend or graph-augmented HistGradientBoosting:

- onion: `graph_histgb_augmented = 0.7014`
- potato: `graph_histgb_augmented = 0.9312`
- tomato: `graph_histgb_augmented = 0.2568`
- wheat: `graph_neighbor_blend = 0.6630`

### Deep graph models at `15d`

GraphWaveNet+:

- onion: `0.1614`
- potato: `0.9131`
- tomato: `0.2761`
- wheat: `0.3627`

Initial full-graph GAT-GRU with `delta_current`:

- onion: `0.8048`
- potato: `0.9206`
- tomato: `0.3089`
- wheat: `0.7026`

Best GAT-GRU after ablations:

- onion: `target_mode__delta_roll28 = 0.8132`
- potato: `target_mode__delta_roll28 = 0.9374`
- tomato: `target_mode__delta_roll28 = 0.3791`
- wheat: `full_graph_delta_current_default = 0.7026`

Conclusion:

- GraphWaveNet+ is not the right architecture here.
- GAT-GRU is the strongest model family in the experiment set.
- The graph advantage is large on onion and wheat, small on potato, and limited on tomato unless radius thresholding is added directly inside the GAT graph.

## Threshold Distance Test

What was done:

- Threshold sweeps at `75km`, `150km`, and `300km` for:
  - neighbor-blend
  - graph-augmented HistGradientBoosting
  - GAT-GRU radius graphs
- Onion refinement at `10km`, `25km`, and `50km`

### Graph-style threshold sweep pattern

Neighbor-blend:

- onion favored `75km`
- potato slightly favored `300km`
- tomato stayed weak across all thresholds
- wheat collapsed at `300km`

Graph-augmented HistGradientBoosting:

- onion favored `75km`
- potato slightly favored `300km`
- tomato slightly favored `300km`
- wheat favored `75km`

### Deep GAT radius sweep

| crop | geo_radius_km | target_mode | r2 | wape_pct |
| --- | --- | --- | --- | --- |
| onion | 75 | delta_roll28 | 0.8131 | 12.4419 |
| onion | 150 | delta_roll28 | 0.8129 | 12.4524 |
| onion | 300 | delta_roll28 | 0.8121 | 12.4756 |
| potato | 75 | delta_roll28 | 0.9380 | 9.1856 |
| potato | 150 | delta_roll28 | 0.9380 | 9.1879 |
| potato | 300 | delta_roll28 | 0.9380 | 9.1914 |
| tomato | 75 | delta_roll28 | 0.3912 | 22.9404 |
| tomato | 150 | delta_roll28 | 0.3910 | 22.9453 |
| tomato | 300 | delta_roll28 | 0.3905 | 22.9535 |
| wheat | 75 | delta_current | 0.6893 | 1.6013 |
| wheat | 150 | delta_current | 0.6894 | 1.6008 |
| wheat | 300 | delta_current | 0.6895 | 1.6012 |

### Onion low-radius refinement

| crop | geo_radius_km | r2 | wape_pct |
| --- | --- | --- | --- |
| onion | 10 | 0.8133 | 12.4347 |
| onion | 25 | 0.8134 | 12.4329 |
| onion | 50 | 0.8132 | 12.4397 |

Conclusion:

- Onion slightly prefers a tighter radius, but the gain is too small to be treated as a major effect.
- Potato is almost flat across radius choices.
- Tomato is the crop that clearly benefits from radius thresholding inside the GAT.
- Wheat does not benefit from radius thresholding and is best with the original non-radius full-graph setup.

## Graph Ablation Test

What was done:

- GAT graph-mode sweep:
  - `full_graph`
  - `geo_only`
  - `corr_only`
  - `temporal_only`
  - `shuffled_graph`
- GAT target-mode sweep
- GAT `k_neighbors` sweep
- GAT edge-weight sweep

Result:

- Onion:
  - `shuffled_graph` stayed very close to `full_graph`
  - edge semantics do not appear critical
- Potato:
  - `temporal_only`, `corr_only`, and `shuffled_graph` stayed close to `full_graph`
  - graph structure is not the main source of gain
- Tomato:
  - `full_graph` beats `shuffled_graph`
  - graph structure helps somewhat
- Wheat:
  - `full_graph` clearly beats the graph ablations
  - graph structure is genuinely useful
- `k_neighbors` changes did not beat the best target-mode choices
- edge-weight presets did not beat the best target-mode choices

Conclusion:

- The largest graph modeling lever is target formulation, not edge-weight tuning.
- Real graph structure matters most for wheat.

## Graph Versus Non-Graph

Best `15d` graph result versus best `15d` non-graph result:

- onion:
  - graph: `0.8132`
  - non-graph: `0.7788`
  - graph gain: `+0.0344`
- potato:
  - graph: `0.9374`
  - non-graph: `0.9336`
  - graph gain: `+0.0038`
- tomato:
  - graph before radius refinement: `0.3791`
  - non-graph: `0.3791`
  - tie before refinement
  - graph after radius refinement: `0.3912`
- wheat:
  - graph: `0.7026`
  - non-graph: `0.6663`
  - graph gain: `+0.0362`

Conclusion:

- Graph is clearly better for onion and wheat.
- Graph is slightly better for potato.
- Tomato only shows a real graph gain after radius thresholding is moved directly into the GAT graph.

## Final Model Recommendation

For `1d`:

- onion: `current_price`
- potato: `current_price`
- tomato: `numeric_histgb_simple`
- wheat: `numeric_histgb_simple`

For `15d`:

- onion: GAT-GRU with `delta_roll28`, `25km` radius is the best refined setting
- potato: GAT-GRU with `delta_roll28`, any of `75km`, `150km`, or `300km` are effectively equivalent
- tomato: GAT-GRU with `delta_roll28` and `75km` radius
- wheat: GAT-GRU with `delta_current` and the original non-radius full graph

## Main Findings

1. The previous-day baseline is not the difficult baseline at `15d`. The rolling baselines are.
2. Simple tabular models are fine at `1d` but not enough at `15d`.
3. The most important modeling change for the deep graph model is target definition.
4. Cross-crop features only help a little, and only on some crops.
5. Dense-sparse and volume grouping help as diagnostics but do not produce the best final models.
6. The best overall `15d` family is GAT-GRU.
7. The benefit of graph structure is crop-specific.
8. Radius thresholding inside the GAT is useful mainly for tomato.
