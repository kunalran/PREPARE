# PREPARE Final Master Report

## Scope

This report consolidates the full experiment history under `newtests/`: the original `targeted_rebuild` sweep, the later focused `1d` / `15d` tuned-model expansion, and the tomato / wheat 15-day follow-up tests.

Primary outputs referenced here:

- `newtests/targeted_rebuild/results/`
- `newtests/focused_1d_15d_execution/results/`
- `newtests/tomato_wheat_15d_followups/results/`
- `newtests/reports/same_day_previous_price_baseline.csv`

## Method Catalog

| method_family | what_it_is | tests_done | primary_outputs |
| --- | --- | --- | --- |
| Explicit baselines | Naive forecast rules using lagged or smoothed price directly. | previous_day_price, current_price, roll_mean_7, roll_mean_28 across 1..15 days for all 4 crops. | targeted_rebuild/results/baseline_metrics/ |
| Same-day persistence baseline | New report-only check: predict current-day observed price from previous day's price_lag_1. | Computed on the same trailing 90-day validation windows used for the 1-day experiments. | reports/same_day_previous_price_baseline.csv |
| Simple numeric HistGB | Single per-crop HistGradientBoosting regressor over numeric engineered features. | 1..15 days for all 4 crops. | targeted_rebuild/results/simple_numeric_metrics/ |
| Anchored HistGB | Predict a correction around an anchor such as current price or rolling mean, then invert back. | 1..15 days for all 4 crops; target encodings revisited later in GAT-GRU ablations. | targeted_rebuild/results/anchored_histgb_metrics/ |
| Cross-crop HistGB | Anchored HistGB with another crop's daily price features added based on national correlation pairing. | 1..15 days for all 4 crops plus pairing analysis. | targeted_rebuild/results/cross_crop_histgb_metrics/ |
| Classical linear models | LinearRegression, Ridge, and ElasticNet on the same tabular engineered feature space. | Focused 1-day and 15-day runs for all 4 crops. | targeted_rebuild/results/classical_ml_metrics/ |
| Density / volume variants | HistGB variants that split markets by graph density or output-volume clusters. | 15-day specialized runs for all 4 crops. | targeted_rebuild/results/density_variant_metrics/ and volume_variant_metrics/ |
| Graph-style non-deep models | Neighbor-blend baseline and graph-augmented HistGB using threshold-based neighbor signals. | 15-day threshold sweeps at 75km / 150km / 300km for all 4 crops. | targeted_rebuild/results/graph_neighbor_blend_metrics/ and graph_histgb_metrics/ |
| Deep graph models | GraphWaveNet+ and GAT-GRU sequence models over mandi graphs. | 15-day runs for all 4 crops. | targeted_rebuild/results/graph_wavenet_plus_mps/ and graph_gat_gru_mps/ |
| GAT-GRU ablations | Target-mode, graph-mode, k-neighbor, edge-weight, and radius-threshold sweeps inside the deep graph model. | 15-day ablations plus radius sweeps 75/150/300km and onion/tomato low-radius refinement 10/25/50km. | targeted_rebuild/results/gat_gru_ablations/, gat_gru_radius_ablation/, gat_gru_radius_refine/ |
| Focused tuned tabular expansion | XGBoost, LightGBM, and ExtraTrees on 1-day and 15-day horizons with compact deterministic tuning. | XGBoost 2 configs, LightGBM 2 configs, ExtraTrees 4 configs across all 4 crops at 1d and 15d. | focused_1d_15d_execution/results/expanded_model_metrics/ and tuning_metrics/ |
| Focused TCN expansion | Temporal convolutional network over windowed price / arrival / calendar sequences. | Two tuned configs: 21x32 and 28x48 window/channel setups for all 4 crops at 1d and 15d. | focused_1d_15d_execution/results/expanded_model_metrics/ and tuning_metrics/ |
| Tomato / wheat 15d follow-ups | Targeted local-anchor, weighted-anchor, regime, long-window TCN, and wheat graph-refinement tests. | Tomato local delta_roll14, delta_roll28, weighted 70/30 anchor, horizon-focused local; wheat regime delta_current, weighted 70/30, horizon-focused regime; TCN 42x64 and 56x96; wheat graph refine. | tomato_wheat_15d_followups/results/ |

## Baseline Definitions

- `previous_day_price`: use `price_lag_1` to predict the future target at the selected horizon.
- `current_price`: use today's `Modal_Price_CausalFilled` to predict the future target at the selected horizon.
- `roll_mean_7` / `roll_mean_28`: use trailing rolling mean price anchors.
- `same_day_previous_price`: new in this report. Predict today's observed `Modal_Price` directly from yesterday's `price_lag_1` on the same trailing validation windows used for the 1-day experiments.

### New Same-Day Persistence Baseline

| crop | validation_rows | prev_day_to_current_r2 | prev_day_to_current_wape | prev_day_to_tomorrow_r2 | prev_day_to_tomorrow_wape | prev_day_to_day15_r2 | prev_day_to_day15_wape |
| --- | --- | --- | --- | --- | --- | --- | --- |
| onion | 8678 | 0.9357 | 4.5134 | 0.9180 | 5.7003 | 0.6508 | 16.8034 |
| potato | 4113 | 0.9593 | 4.2117 | 0.9518 | 5.1104 | 0.8937 | 10.1230 |
| tomato | 29374 | 0.8852 | 7.6105 | 0.8448 | 9.8718 | 0.2588 | 25.8861 |
| wheat | 528 | 0.6794 | 1.3729 | 0.6859 | 1.4263 | 0.6209 | 1.6456 |

Interpretation: `prev_day_to_current_r2` is the corrected baseline you asked for. It is not the same metric as the old horizon-specific `previous_day_price` baseline, which predicts tomorrow or day-15 from yesterday.

## One-Day Results

One-day learned-model comparisons combine the original baselines / HistGB / cross-crop / classical ML runs with the later tuned `xgboost`, `lightgbm`, `extratrees`, and `tcn` expansion.

| crop | best_existing_before_expansion | best_focused_new_model | final_best_1d |
| --- | --- | --- | --- |
| onion | current_price (0.9535) | tcn_tuned (0.9490) | current_price (0.9535) |
| potato | elasticnet_a001_l05 (0.9705) | extratrees_tuned (0.9689) | elasticnet_a001_l05 (0.9705) |
| tomato | linear_regression (0.9043) | tcn_tuned (0.9191) | tcn_tuned (0.9191) |
| wheat | numeric_histgb_simple (0.7873) | extratrees_tuned (0.7935) | extratrees_tuned (0.7935) |

Focused 1-day / 15-day tuned expansion R2 values:

| crop | horizon_days | xgboost_tuned | lightgbm_tuned | extratrees_tuned | tcn_tuned |
| --- | --- | --- | --- | --- | --- |
| onion | 1 | 0.9411 | 0.9472 | 0.9414 | 0.9490 |
| potato | 1 | 0.9670 | 0.9652 | 0.9689 | 0.9662 |
| tomato | 1 | 0.8964 | 0.8931 | 0.8957 | 0.9191 |
| wheat | 1 | 0.7897 | 0.7916 | 0.7935 | 0.7432 |
| onion | 15 | -0.0657 | 0.0347 | 0.5272 | 0.5391 |
| potato | 15 | 0.8792 | 0.9166 | 0.9365 | 0.9335 |
| tomato | 15 | 0.1883 | 0.2698 | 0.3675 | 0.1813 |
| wheat | 15 | 0.6425 | 0.6386 | 0.6764 | 0.5313 |

## Fifteen-Day Results

Fifteen-day comparisons combine the original `targeted_rebuild` final comparison, classical ML, graph ablations, radius sweeps / refinements, the focused tuned-model expansion, and the tomato / wheat follow-up wave.

| crop | best_existing_before_expansion | best_focused_new_model | best_followup_model | final_best_15d |
| --- | --- | --- | --- | --- |
| onion | onion__radius_25km (0.8134) | tcn_tuned (0.5391) | onion__radius_25km (0.8134) | onion__radius_25km (0.8134) |
| potato | potato__radius_300km (0.9380) | extratrees_tuned (0.9365) | potato__radius_300km (0.9380) | potato__radius_300km (0.9380) |
| tomato | tomato__radius_75km (0.3912) | extratrees_tuned (0.3675) | tomato_radius_10km_delta_roll28_existing (0.3914) | tomato_radius_10km_delta_roll28_existing (0.3914) |
| wheat | full_graph_delta_current_default (0.7026) | extratrees_tuned (0.6764) | full_graph_delta_current_default (0.7026) | wheat_best_existing_graph_reference (0.7026) |

Selected graph-family 15-day checkpoints:

| crop | experiment_name | r2 | wape_pct |
| --- | --- | --- | --- |
| onion | onion__radius_25km | 0.8134 | 12.4329 |
| onion | target_mode__delta_roll28 | 0.8132 | 12.4376 |
| onion | full_graph_delta_current_default | 0.8048 | 12.3363 |
| onion | graph_gat_gru_mps | 0.8048 | 12.3363 |
| onion | graph_wavenet_plus_mps | 0.1614 | 27.4447 |
| potato | potato__radius_300km | 0.9380 | 9.1914 |
| potato | target_mode__delta_roll28 | 0.9374 | 9.2530 |
| potato | full_graph_delta_current_default | 0.9206 | 10.2063 |
| potato | graph_gat_gru_mps | 0.9206 | 10.2063 |
| potato | graph_wavenet_plus_mps | 0.9131 | 10.5173 |
| tomato | tomato_radius_10km_delta_roll28_existing | 0.3914 | 22.9382 |
| tomato | target_mode__delta_roll28 | 0.3791 | 23.2115 |
| tomato | full_graph_delta_current_default | 0.3089 | 25.3554 |
| tomato | graph_gat_gru_mps | 0.3089 | 25.3554 |
| tomato | graph_wavenet_plus_mps | 0.2761 | 26.9353 |
| wheat | full_graph_delta_current_default | 0.7026 | 1.5430 |
| wheat | graph_gat_gru_mps | 0.7026 | 1.5430 |
| wheat | wheat_graph_refine_longwindow | 0.6967 | 1.5867 |
| wheat | target_mode__delta_roll28 | 0.5333 | 2.1854 |
| wheat | graph_wavenet_plus_mps | 0.3627 | 2.6307 |

Tomato / wheat 15-day follow-up results:

| crop | experiment_name | r2 | wape_pct |
| --- | --- | --- | --- |
| tomato | tomato_radius_10km_delta_roll28_existing | 0.3914 | 22.9382 |
| tomato | tomato_radius_25km_delta_roll28_existing | 0.3914 | 22.9382 |
| tomato | tomato_horizon_focus_local | 0.3605 | 24.4739 |
| tomato | tomato_local_delta_roll28 | 0.3193 | 25.0664 |
| tomato | tomato_local_weighted_anchor7030 | 0.3142 | 25.2074 |
| tomato | tomato_local_delta_roll14 | 0.2950 | 25.6891 |
| tomato | tomato_tcn_longwindow_candidate2 | 0.1715 | 29.7413 |
| tomato | tomato_tcn_longwindow_candidate1 | 0.0994 | 30.4185 |
| wheat | wheat_best_existing_graph_reference | 0.7026 | 1.5430 |
| wheat | wheat_tcn_longwindow_candidate2 | 0.5791 | 2.1827 |
| wheat | wheat_regime_weighted_anchor7030 | 0.5729 | 2.0919 |
| wheat | wheat_regime_delta_current | 0.5471 | 2.1686 |
| wheat | wheat_horizon_focus_regime | 0.5428 | 2.1854 |
| wheat | wheat_tcn_longwindow_candidate1 | 0.4866 | 2.5838 |

## What Performed Best

Final best completed `1d` result by crop:

| crop | final_best_1d |
| --- | --- |
| onion | current_price (0.9535) |
| potato | elasticnet_a001_l05 (0.9705) |
| tomato | tcn_tuned (0.9191) |
| wheat | extratrees_tuned (0.7935) |

Final best completed `15d` result by crop:

| crop | final_best_15d |
| --- | --- |
| onion | onion__radius_25km (0.8134) |
| potato | potato__radius_300km (0.9380) |
| tomato | tomato_radius_10km_delta_roll28_existing (0.3914) |
| wheat | wheat_best_existing_graph_reference (0.7026) |

## Key Findings

- `1d`: onion stayed baseline-dominated, potato favored classical linear / elastic-net style models, tomato favored the focused TCN, and wheat favored focused ExtraTrees.
- `15d`: the graph family still won overall. Onion, potato, and tomato favored `delta_roll28` graph targets; wheat favored `delta_current` on the full graph.
- Focused tuned tree models were useful, especially `extratrees_tuned`, but they did not displace the graph frontier at 15 days.
- The tomato / wheat follow-up tabular and long-window TCN tests did not beat the existing 15-day graph winners.
- Runtime note: Focused TCN runs used `cpu`. MPS built=True, available=False.

## Files Created By This Report

- Report: `newtests/reports/final_master_report.md`
- Corrected baseline CSV: `newtests/reports/same_day_previous_price_baseline.csv`
