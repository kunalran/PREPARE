# PREPARE New Tests Progress Report

## Completed Outputs

- Full crop inventory and mandi-region summaries written under `newtests/targeted_rebuild/results/inventory/`
- Cross-crop pairing rule written to `newtests/targeted_rebuild/results/cross_crop_pairings.csv`
- Full `1-15` day explicit-baseline sweep written to `newtests/targeted_rebuild/results/baseline_metrics/`
- Reproducible experiment runner and report builder added under `newtests/scripts/`

## Dataset Inventory

- Onion: 764 mandis, 470,964 rows, 398,958 non-null price rows, date range `2023-01-01` to `2025-12-31`
- Potato: 713 mandis, 459,012 rows, 414,486 non-null price rows, date range `2023-01-01` to `2025-11-30`
- Tomato: 693 mandis, 492,870 rows, 416,678 non-null price rows, date range `2023-01-01` to `2025-12-31`
- Wheat: 475 mandis, 295,535 rows, 273,960 non-null price rows, date range `2023-01-01` to `2025-11-30`

## Mandi Connectivity Summary

- Onion is the densest crop graph at shorter thresholds: mean neighbors `13.6` within `75km`, `41.1` within `150km`
- Wheat is the sparsest at `75km`: mean neighbors `9.0`
- At `300km`, wheat and onion are similarly dense: mean neighbors `111.0` and `112.3`
- Several crops still have isolated mandis at `75km`; onion and potato also still have isolated mandis at `150km`

## Cross-Crop Pairing Rule

Pair each crop with the other crop that has the highest daily national-mean price correlation:

- Onion -> Wheat (`0.7690`)
- Potato -> Onion (`0.6816`)
- Tomato -> Potato (`0.2437`)
- Wheat -> Onion (`0.7690`)

## Baseline Results

Best 15-day explicit baseline by crop:

- Onion: `roll_mean_28`, `R2 = 0.7788`, `WAPE = 13.55%`
- Potato: `roll_mean_28`, `R2 = 0.9336`, `WAPE = 9.41%`
- Tomato: `roll_mean_28`, `R2 = 0.3791`, `WAPE = 23.21%`
- Wheat: `roll_mean_7`, `R2 = 0.6663`, `WAPE = 1.62%`

Best baseline pattern across horizons:

- Onion: `current_price` is best on `11/15` horizons; `roll_mean_28` takes the longer horizons
- Potato: `roll_mean_28` is best on `11/15` horizons
- Tomato: `current_price` is best on `10/15` horizons; `roll_mean_28` becomes best deeper into the horizon
- Wheat: `roll_mean_7` is best on `13/15` horizons

Selected baseline checkpoints:

- Onion:
  - `1d`: `current_price`, `R2 = 0.9535`
  - `7d`: `current_price`, `R2 = 0.8393`
  - `15d`: `roll_mean_28`, `R2 = 0.7788`
- Potato:
  - `1d`: `current_price`, `R2 = 0.9638`
  - `7d`: `roll_mean_28`, `R2 = 0.9373`
  - `15d`: `roll_mean_28`, `R2 = 0.9336`
- Tomato:
  - `1d`: `current_price`, `R2 = 0.8876`
  - `7d`: `current_price`, `R2 = 0.6572`
  - `15d`: `roll_mean_28`, `R2 = 0.3791`
- Wheat:
  - `1d`: `roll_mean_7`, `R2 = 0.7530`
  - `7d`: `roll_mean_7`, `R2 = 0.6946`
  - `15d`: `roll_mean_7`, `R2 = 0.6663`

## Current Conclusions

- A single naive baseline is not dominant across crops.
- Longer-horizon smoothing is already very strong for onion, potato, and tomato; this raises the bar for learned models.
- Wheat behaves differently from the vegetables: a shorter rolling window is consistently stronger than the 28-day smoother.
- The mandi graph is materially denser for onion and potato than for wheat at short thresholds, which supports testing whether graph-aware methods help more on those crops.
- Tomato’s weak cross-crop correlation relative to the others suggests cross-crop features may help less there than on onion, potato, or wheat.

## Remaining Heavy Runs

The following experiment families are implemented in `newtests/scripts/prepare_experiments.py` but are still runtime-heavy on this CPU-only machine:

- Simple numeric HistGB model
- Anchored per-crop HistGB variant
- Cross-crop augmented HistGB variant
- Density-bucket and volume-cluster variants
- Graph neighbor-blend threshold sweep
- Graph-augmented HistGB threshold sweep

Those runners are already wired to write results into grouped subfolders under `newtests/`.
