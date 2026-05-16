# Focused 1-Day / 15-Day Model Expansion

## One-Day Comparison

- onion: best new `tcn_tuned` R2=0.9490 did not beat best existing `current_price` R2=0.9535 (delta -0.0045); overall winner `current_price`
- potato: best new `extratrees_tuned` R2=0.9689 did not beat best existing `elasticnet_a001_l05` R2=0.9705 (delta -0.0016); overall winner `elasticnet_a001_l05`
- tomato: best new `tcn_tuned` R2=0.9191 beat best existing `linear_regression` R2=0.9043 (delta +0.0148); overall winner `tcn_tuned`
- wheat: best new `extratrees_tuned` R2=0.7935 beat best existing `numeric_histgb_simple` R2=0.7873 (delta +0.0062); overall winner `extratrees_tuned`

## Fifteen-Day Comparison

- onion: best new `tcn_tuned` R2=0.5391 did not beat best existing `target_mode__delta_roll28` R2=0.8132 (delta -0.2741); overall winner `target_mode__delta_roll28`
- potato: best new `extratrees_tuned` R2=0.9365 did not beat best existing `target_mode__delta_roll28` R2=0.9374 (delta -0.0010); overall winner `target_mode__delta_roll28`
- tomato: best new `extratrees_tuned` R2=0.3675 did not beat best existing `target_mode__delta_roll28` R2=0.3791 (delta -0.0116); overall winner `target_mode__delta_roll28`
- wheat: best new `extratrees_tuned` R2=0.6764 did not beat best existing `full_graph_delta_current_default` R2=0.7026 (delta -0.0261); overall winner `full_graph_delta_current_default`

## Outputs

- `results/final_comparison/horizon_1_full.csv`
- `results/final_comparison/horizon_15_full.csv`
- `results/final_comparison/horizon_1_best_by_crop.csv`
- `results/final_comparison/horizon_15_best_by_crop.csv`