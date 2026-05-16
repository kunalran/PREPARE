# PREPARE Sem 6 Final Slide Content

## Recommended Story

This final presentation should tell one clear story:

1. The problem is real and high impact.
2. The hard part was not just modeling, but building a usable mandi-level dataset.
3. Strong baselines were difficult to beat, especially at longer horizons.
4. The final graph pipeline wins where it matters most: 15-day forecasting.
5. The project is not just a model paper exercise; it is already integrated into an app.

Keep the final deck to 13-15 core slides, then move extra experiments into appendix slides.

## Recommended Core Deck

### Slide 1 - Title

**Title:** PREPARE: Prediction of Prices in Agriculture

**Subtitle:** Mandi-level 1 to 15 day forecasting for onion, potato, tomato, and wheat

**Include:**
- Group members
- Faculty guide
- One-line value proposition: "Helping farmers and regulators make better selling and intervention decisions"

**Speaker focus:**
- Start with the problem, not the model.

### Slide 2 - Problem and Why It Matters

**Title:** Why agricultural price forecasting matters

**Include:**
- Farmers face price blindness and often sell without reliable forward visibility.
- Regulators need localized forecasts for intervention planning.
- Existing work is often coarse, market-independent, or not consumer-facing.

**Reuse from old slides:**
- Sem 5 final PDF page 3
- Sem 6 midsem PDF page 3

**Speaker focus:**
- Stress that this is a real deployment problem, not just an ML benchmark.

### Slide 3 - What We Built End-to-End

**Title:** End-to-end system overview

**Recommended layout:** left = pipeline, right = deployment/app

**Include:**
- Scraping from Agmarknet
- Data cleaning and date x mandi expansion
- Missing data handling and imputation
- Feature engineering + graph construction
- Forecasting models
- FastAPI + bilingual mobile app integration

**Reuse from old slides:**
- Sem 5 final PDF page 4
- Sem 6 midsem PDF page 4

**Update the wording:**
- Replace "10 day" references with "1 to 15 day"
- Replace "GAT-LSTM" wording with final model wording: "GAT-GRU final graph architecture"

### Slide 4 - Dataset and Scope

**Title:** Data coverage and final project scope

**Figure to use:**
- `final report/figures/data_inventory.png`

**Include:**
- 3 years of data: Jan 2023 to Dec 2025
- 4 final crops: onion, potato, tomato, wheat
- Final retained mandi counts:
  - Onion: 764
  - Potato: 713
  - Tomato: 693
  - Wheat: 475
- Sugarcane dropped because missingness was too extreme

**Key line to say:**
- "Before modeling, we first had to decide what data was actually reliable enough to model."

### Slide 5 - Geographic Coverage and Market Network

**Title:** Nationwide mandi network

**Figure to use:**
- `final report/figures/state_distribution.png`

**Include:**
- Coverage spans 25+ states
- Uttar Pradesh dominates onion, potato, and tomato mandi counts
- Madhya Pradesh dominates wheat
- Wheat is the smallest mandi network; onion is among the densest

**Why this slide matters:**
- It sets up why graph modeling is reasonable in the first place.

### Slide 6 - The Core Data Challenge: Missingness

**Title:** The hardest problem was data sparsity

**Include:**
- Structural missingness after grid expansion: roughly 65-75%
- Sundays often appear missing because mandis are physically closed
- Many mandis have very long missing streaks
- Missingness is administrative and structural, not random noise

**Source material:**
- Final report sections 2.4 and 2.5
- Mid-update reports 2 and 3

**Speaker focus:**
- This is one of the most important non-model slides in the deck.

### Slide 7 - Imputation Pipeline and Why DOW Ratio Won

**Title:** Imputation benchmark across 10 strategies

**Figure to use:**
- `final report/figures/imputation_comparison.png`

**Include:**
- 10 strategies benchmarked
- Threshold testing at 0%, >=50%, >=75% data availability
- DOW Ratio won for the practical >=50% regime
- Rolling mean was strongest only at very dense thresholds
- Spline worked for huge gaps but over-smoothed dense data

**Key conclusion line:**
- "We selected DOW Ratio because it matched the real reporting behavior of the mandi data."

### Slide 8 - Experimental Setup

**Title:** How we evaluated all models

**Include:**
- Validation window: trailing 90-day period
- Forecast horizons: 1 day and 15 day, with broader horizon sweeps for baselines
- Metrics: R-squared as primary, plus WAPE, RMSE, and MAE
- Comparison philosophy: every advanced model had to beat strong non-learning baselines, not just weak references

**Good visual layout:**
- Left: evaluation pipeline
- Right: compact bullets for metrics and horizon setup

### Slide 9 - Feature Space

**Title:** What information each model could use

**Include:**
- Price history: lagged prices, rolling means, rolling variability
- Arrival information: arrivals and transformed arrival features
- Calendar information: day-of-week, month, seasonality markers
- Spatial information: mandi identity, state/district context, latitude/longitude
- Weather information: hourly ERA5-derived weather aggregated into daily signals
- Graph information: neighborhood prices and graph connectivity for graph-aware models

**Key message:**
- The task is not univariate forecasting; it is a spatiotemporal forecasting problem with market, weather, and spatial context.

### Slide 10 - Experiment Slide 1: Non-Graph Search

**Title:** Experiment map: non-graph models

**Include:**
- **Explicit baselines:** previous-day price, current price, 7-day rolling mean, 28-day rolling mean
  - Tried first to establish how hard the task already is without learning
- **Simple numeric HistGB**
  - Tried as a fast non-deep benchmark and sanity check
- **Anchored HistGB**
  - Predict a correction around a baseline such as current price or rolling mean
  - Tried because direct raw-price prediction can be unstable
- **Cross-crop HistGB**
  - Add one paired crop as an input feature
  - Tried to test whether related crops act as leading indicators
- **Density-bucket / volume-cluster variants**
  - Separate mandis by graph density or output volume
  - Tried to test whether sparse and dense markets need different behavior

**Speaker focus:**
- This slide should make it clear that a broad non-graph search was completed before committing to graph models.

### Slide 11 - Experiment Slide 2: Graph Search and Ablation Logic

**Title:** Experiment map: graph models and ablations

**Include:**
- **Neighbor-blend and graph-augmented HistGB**
  - Add neighborhood information without full deep graph learning
  - Tried to isolate whether graph information helps even in simpler models
- **TCN**
  - Temporal sequence model over recent windows
  - Tried because some crops may benefit more from temporal pattern extraction than explicit graph structure
- **GraphWaveNet+**
  - Adaptive deep graph model from spatiotemporal forecasting literature
  - Tried as a strong graph benchmark
- **GAT-GRU**
  - Graph attention encoder with recurrent temporal decoding
  - Tried to jointly model inter-mandi influence and temporal evolution while predicting anchored residuals

**Ablation knobs to summarize on this slide:**
- **Target formulation:** `delta_current`, `delta_roll7`, `delta_roll28`
- **Graph mode:** full graph, temporal-only, geo-only, corr-only, shuffled graph
- **Radius threshold:** 75 km, 150 km, 300 km, plus finer onion refinement
- **Secondary tuning:** edge-weight presets and `k` neighbors

**Key message:**
- The graph search was not a single lucky model run; it was a structured comparison over target choice, graph structure, and locality assumptions.

### Slide 12 - Ablation Inferences

**Title:** Key ablation inferences

**Recommended layout:** 4 compact tables or 2x2 grid of mini-tables

**Table 1 - Target formulation**

| Crop | Best target mode | Inference |
| --- | --- | --- |
| Onion | `delta_roll28` | Longer anchor stabilized multi-day movement |
| Potato | `delta_roll28` | Smoother anchor matched persistent dynamics |
| Tomato | `delta_roll28` | Stronger anchoring helped the most volatile crop |
| Wheat | `delta_current` | Current price remained the best reference |

**Table 2 - Radius threshold**

| Crop | Best radius behavior | Inference |
| --- | --- | --- |
| Onion | Best around `25 km`, but `10-75 km` nearly flat | Tight local graph helps slightly, not decisively |
| Potato | Nearly flat across `75-300 km` | Reach matters little once target is right |
| Tomato | Best at lower radius setting | Useful graph signal is local |
| Wheat | Full graph beats radius-thresholded variants | Longer-range structure is useful |

**Table 3 - Graph structure**

| Crop | Structural takeaway |
| --- | --- |
| Onion | Temporal signal plus target design explain much of the gain; exact edge semantics matter less |
| Potato | Graph gains are real but small because the crop is already highly persistent |
| Tomato | Graph helps only after careful target and radius design |
| Wheat | Real graph structure matters the most clearly |

**Table 4 - Cross-crop results**

| Crop | Paired crop | Corr. | Cross-crop takeaway |
| --- | --- | ---: | --- |
| Onion | Wheat | `0.769` | Useful secondary signal |
| Potato | Onion | `0.682` | Useful secondary signal |
| Tomato | Potato | `0.244` | Weak relationship, little practical gain |
| Wheat | Onion | `0.769` | Useful secondary signal |

**One-line summary to say aloud:**
- "The biggest gains came from choosing the right target, using graph structure where it truly helped, and recognizing that crop behavior differs a lot across onion, potato, tomato, and wheat."

### Slide 13 - One-Day Results

**Title:** One-day forecasting is strong, but not the main challenge

**Figure to use:**
- `final report/figures/focused_1d_models.png`

**If you want to avoid `current_price`, `ElasticNet`, `ExtraTrees`, and `Ridge`, use these callouts instead:**
- Onion: best filtered 1d = tuned TCN, `R2 = 0.9490`
- Potato: best filtered 1d = Linear Regression, `R2 = 0.9701`
- Tomato: best filtered 1d = tuned TCN, `R2 = 0.9191`
- Wheat: best filtered 1d = tuned LightGBM, `R2 = 0.7916`

**Message:**
- One-day prediction is already very easy because prices are persistent.
- The real benchmark of model quality is what happens at 15 days.
- These filtered winners are useful if you want the results slide to emphasize learned models other than `current_price`, `ElasticNet`, and `ExtraTrees`.

**How the HistGB family performed:**
- **1-day:** HistGB variants were strong but usually not the very top filtered winner. Anchored HistGB reached `0.9475` on onion, `0.9602` on potato, `0.8962` on tomato, and `0.7500` on wheat.
- **15-day:** HistGB variants were more mixed. The best entries were:
  - Onion: cross-crop HistGB, `R2 = 0.7075`
  - Potato: anchored HistGB, `R2 = 0.9320`
  - Tomato: simple numeric HistGB, `R2 = 0.2647`
  - Wheat: cross-crop HistGB, `R2 = 0.6598`
- **Takeaway:** HistGB was a strong non-graph family, especially for onion and potato, but it did not displace the final graph winners at 15 days.

### Slide 14 - Fifteen-Day Results

**Title:** At 15 days, the graph family wins

**Primary figure to use:**
- `final report/figures/model_family_15d.png`

**Optional supporting figure:**
- `final report/figures/best_model_r2.png`

**Include final best 15d results:**
- Onion: GAT-GRU, R2 0.8134
- Potato: GAT-GRU, R2 0.9380
- Tomato: GAT-GRU, R2 0.3914
- Wheat: GAT-GRU, R2 0.7026

**Speaker focus:**
- This is the main result slide of the deck.

**Recommended filtered summary table for this slide or the next one:**

| Crop | Previous-day -> Current-day baseline R2 | Best 1-day result | Best 15-day result |
| --- | ---: | --- | --- |
| Onion | 0.9357 | Tuned TCN, `R2 = 0.9490` | GAT-GRU (`delta_roll28`, 25 km), `R2 = 0.8134` |
| Potato | 0.9593 | Linear Regression, `R2 = 0.9701` | GAT-GRU (`delta_roll28`, 300 km / flat 75-300 km), `R2 = 0.9380` |
| Tomato | 0.8852 | Tuned TCN, `R2 = 0.9191` | GAT-GRU (`delta_roll28`, best refined low-radius run), `R2 = 0.3914` |
| Wheat | 0.6794 | Tuned LightGBM, `R2 = 0.7916` | GAT-GRU (`delta_current`, full graph), `R2 = 0.7026` |

**Important wording note:**
- Label the first column explicitly as the **same-day persistence baseline** so the panel does not confuse it with the 1-day or 15-day horizon-specific `previous_day_price` baseline.
- These 1-day winners are the best results after excluding `current_price`, `ElasticNet`, `ExtraTrees`, and `Ridge`.

### Slide 15 - Strong Baselines Were Hard to Beat

**Title:** We compared against strong rolling and persistence baselines

**Include the best 15d baseline for each crop:**
- Onion: roll_mean_28, R2 0.7788
- Potato: roll_mean_28, R2 0.9336
- Tomato: roll_mean_28, R2 0.3791
- Wheat: roll_mean_7, R2 0.6663

**Also include graph improvement over previous-day 15d baseline:**
- Onion: +0.1626 over previous-day baseline
- Potato: +0.0443 over previous-day baseline
- Tomato: +0.1326 over previous-day baseline
- Wheat: +0.0817 over previous-day baseline

**Why this slide matters:**
- It prevents the audience from thinking the gains came from beating weak comparators.

### Slide 16 - Graph vs Non-Graph

**Title:** Graph vs non-graph at 15 days

**Figure to use:**
- `final report/figures/graph_vs_nongraph.png`

**Include:**
- Onion: graph `0.8134` vs non-graph `0.7788`, gain about `+0.034`
- Potato: graph `0.9380` vs non-graph `0.9336`, gain about `+0.004`
- Tomato: graph `0.3914` vs non-graph `0.3791`, gain about `+0.012`
- Wheat: graph `0.7026` vs non-graph `0.6663`, gain about `+0.036`

**Interpretation bullets:**
- Onion and wheat benefit most clearly
- Potato has a small but real graph gain
- Tomato remains hard, but graph thresholding still helps

### Slide 17 - Deployment: App Integration

**Title:** From model to usable product

**Include:**
- Bilingual app interface
- Map-based mandi selection
- Crop selection
- Day-horizon slider from 1 to 15 days
- Backend served using FastAPI
- Final trained models integrated into prediction flow

**Reuse from old slides:**
- Sem 5 final PDF pages 16 and 18 if you have screenshots/assets separately
- Sem 6 midsem PDF page 7 for app feature wording

**Speaker focus:**
- This is where you show that the work is actually usable.

### Slide 18 - Final Takeaways

**Title:** What we learned from PREPARE

**Include:**
- Strong baselines matter: the 28-day rolling mean was already very hard to beat for onion, potato, and tomato, while wheat behaved differently and preferred shorter anchors
- Data work was as important as model work: handling structural missingness correctly was essential, and DOW Ratio turned out to be the most practical imputation strategy
- Anchored residual prediction worked better than raw-price prediction for most crops: `delta_roll28` was best for onion, potato, and tomato, while wheat preferred `delta_current`
- Graph models helped most when the crop actually had exploitable spatial structure: gains were clearest for onion and wheat, smaller for potato, and harder to realize for tomato
- The final system is more than a model: we built a full mandi-level pipeline from scraping and cleaning through forecasting and app deployment

**Recommended closing message:**
- "The biggest lesson was that good forecasting here comes from the right problem framing, the right data pipeline, and graph structure only where it truly helps."

### Slide 19 - Limitations and Future Work

**Title:** Limitations and what comes next

**Include:**
- **Current limitations**
  - Structural missingness remains a fundamental data bottleneck
  - Strong rolling baselines are still hard to beat on highly persistent crops
  - Tomato remains the hardest crop at 15 days
  - Some graph gains are modest and crop-specific rather than universal
- **Future work**
  - learnable graph structures instead of fixed graph design
  - explainability for farmer and regulator trust
  - adding more crops and broader seasonal coverage
  - policy and game-theoretic extensions for market intervention settings

**If you want a more report-aligned phrasing, use these learned limitations:**
- Cross-crop features were interesting but only gave modest gains, especially when pairwise crop correlation was weak
- Density and volume bucketing were useful diagnostic experiments, but they did not beat the strongest shared-model baselines
- Earlier graph directions such as GraphWaveNet+ and SAGMM-style variants were not competitive in this setting
- The next cohort can extend the system because the codebase, experiment framework, and data pipeline are already modular

**Optional closing line:**
- "PREPARE shows that mandi-level agricultural price forecasting is feasible, and graph-aware modeling improves it further."

## Recommended Appendix Slides

If time allows or if you expect technical questions, keep these ready after the main deck.

### Appendix A - Same-Day vs 1-Day vs 15-Day Baselines

Include these corrected same-day persistence numbers:
- Onion: same-day R2 0.9357
- Potato: same-day R2 0.9593
- Tomato: same-day R2 0.8852
- Wheat: same-day R2 0.6794

Use this to explain price persistence.

### Appendix B - Horizon Decay

**Figure:**
- `final report/figures/horizon_decay.png`

Use this to answer how performance changes from day 1 to day 15.

### Appendix C - WAPE Comparison

**Figure:**
- `final report/figures/wape_comparison.png`

Useful if someone asks whether R2 improvements also translate to absolute error improvement.

### Appendix D - Final Result Summary Table

If the evaluators want one compact summary slide, use this exact structure:

| Crop | Same-day persistence baseline | Horizon-specific previous-day baseline at 15d | Best 1-day result | Best 15-day result |
| --- | --- | --- | --- | --- |
| Onion | 0.9357 | 0.6508 | Tuned TCN, 0.9490 | GAT-GRU, 0.8134 |
| Potato | 0.9593 | 0.8937 | Linear Regression, 0.9701 | GAT-GRU, 0.9380 |
| Tomato | 0.8852 | 0.2588 | Tuned TCN, 0.9191 | GAT-GRU, 0.3914 |
| Wheat | 0.6794 | 0.6209 | Tuned LightGBM, 0.7916 | GAT-GRU, 0.7026 |

This is the safest appendix table because it shows both:
- how persistent the raw series already is on the next day
- how much harder the true 15-day horizon is

### Appendix E - Baseline Comparison

**Figure:**
- `final report/figures/baseline_comparison.png`

Useful to show why `roll_mean_28` and `roll_mean_7` are serious baselines.

### Appendix F - Crop-Wise Recommended Final Models

- Onion: GAT-GRU, `delta_roll28`, radius 25 km
- Potato: GAT-GRU, `delta_roll28`, radius effectively flat across 75-300 km, best recorded at 300 km
- Tomato: GAT-GRU, `delta_roll28`, best recorded at 10 km follow-up / effectively tied with 25 km
- Wheat: GAT-GRU, `delta_current`, full graph better than explicit radius thresholding

### Appendix G - Detailed Technical Inferences

If you expect deeper ML questioning, keep one slide with concise takeaways from the secondary tests:

- **Imputation thresholds:** DOW Ratio dominates in the realistic 50-75% range because the dominant failure mode is weekly reporting structure, not random isolated gaps.
- **Cross-crop features:** helpful mainly when national price correlations are strong, but weak for tomato because its best paired crop relationship is much smaller.
- **Density and volume splits:** useful diagnostic probes, but they did not beat the strongest shared-model baselines, so market heterogeneity alone was not the missing ingredient.
- **Graph neighbor-blend and graph-HistGB:** adding simple neighborhood information was not enough; deeper graph models only became worthwhile when combined with the right target formulation.
- **Potato behavior:** the task is already so persistent that many models cluster closely together, which is why graph gains are real but small.
- **Tomato behavior:** the crop is highly volatile and perishable, so horizon extension is much harder; improvements exist, but every point of gain is expensive.
- **Wheat behavior:** the crop benefits the most from meaningful graph structure and has the lowest WAPE, suggesting a more stable market process with exploitable spatial structure.

## What To Reuse From Old Slides

Reuse conceptually, but update wording and metrics:

- Problem framing:
  - Sem 5 final page 3
  - Sem 6 midsem page 3
- System schematic:
  - Sem 5 final page 4
  - Sem 6 midsem page 4
- App feature framing:
  - Sem 6 midsem page 7
- Data acquisition / EDA / imputation story:
  - Sem 6 midsem pages 9, 11, 12, 13, 15

Do **not** reuse old result slides as-is:
- Sem 5 final pages 8-15 contain older scope, older metrics, tomato/wheat-only result emphasis, and GAT-LSTM-era framing.

## Suggested Final Order If You Need A 10-Minute Version

1. Title
2. Problem
3. End-to-end system
4. Dataset and missingness
5. Imputation benchmark
6. Experimental setup
7. Model families
8. One-day results
9. Fifteen-day results
10. Graph vs non-graph
11. App deployment
12. Takeaways and future work

## Suggested Final Order If You Need A 15-Minute Viva Version

1. Title
2. Problem
3. End-to-end system
4. Dataset scope
5. Geographic coverage
6. Missingness challenge
7. Imputation benchmark
8. Experimental setup
9. Feature space
10. Baseline and tabular experiments
11. Sequence and graph experiments
12. Ablation inferences
13. One-day results
14. Fifteen-day results
15. Baselines vs final models
16. Graph vs non-graph
17. App integration
18. Final takeaways
19. Limitations and future work

## One-Line Thesis For The Whole Presentation

"The main contribution of PREPARE is not only a better 15-day forecasting model, but a full mandi-level forecasting pipeline that turns sparse government data into deployable crop-price predictions."
