# PREPARE

PREPARE is a research repository for forecasting Indian agricultural mandi prices from Agmarknet market data. It focuses on onion, potato, tomato, and wheat and includes data-processing utilities, forecasting baselines, tabular models, graph neural-network experiments, evaluation scripts, saved results, and a small web interface for exploring forecasts.

Repository: [github.com/kunalran/PREPARE](https://github.com/kunalran/PREPARE)

## What is included

The main modeling tracks are:

- naive baselines: current price, 7-day rolling mean, and 28-day rolling mean
- per-crop and pooled HistGradientBoosting models
- anchored residual targets such as `delta_current` and `delta_roll28`
- normalization and mandi-clustering variants
- GraphWaveNet, GraphWaveNet+, and SAGMM-style graph mixture experiments
- GAT-GRU graph models using lag, rolling, geographic, state, national, and weather features
- DOW-ratio imputation experiments and benchmarks

The web app combines a FastAPI prediction API with a React/Vite interface. Users can select a crop, mandi, and supported forecast horizon, then view the saved model's forecast.

## Reported results

The best reported 15-day results currently stored in the repository are:

| Crop | Best approach | R² |
| --- | --- | ---: |
| Onion | GAT-GRU graph model | 0.8544 |
| Potato | GAT-GRU graph model | 0.9287 |
| Tomato | 28-day rolling anchor baseline | 0.4096 |
| Wheat | Tabular residual model (`delta_current` + series mean centering) | 0.6381 |

Wheat also achieved an R² of `0.8261` at the 1-day horizon with the blended wheat model. See the [experiment report](reports/experiment_report.md) and the metric summaries under [`models/`](models/) for more detail.

## Repository structure

```text
PREPARE/
├── training/                 # Model training scripts
├── evaluation/               # Baselines and evaluation scripts
├── scripts/data_processing/  # Cleaning, expansion, weather, and imputation
├── final_data/               # Daily processed data (required locally)
├── final_data_hourly/        # Hourly processed data (required locally)
├── models/                   # Saved models and metric summaries
├── research/                 # Experiment-specific and archival work
├── reports/                  # Reports and supporting documents
├── web_app/                  # FastAPI backend
└── frontend/                 # React/Vite frontend
```

## Quick start

### Prerequisites

- Git
- Python 3.9 (recommended for the pinned dependencies)
- Node.js 18 or newer and npm, if you want to run the web interface

The processed datasets are intentionally not tracked by Git. Before training models or using the prediction app, obtain the project data and place the crop CSV files in `final_data/` and `final_data_hourly/`. Expected names include:

```text
final_data/agmarknet_onion_data_final.csv
final_data/agmarknet_potato_data_final.csv
final_data/agmarknet_tomato_data_final.csv
final_data/agmarknet_wheat_data_final.csv

final_data_hourly/agmarknet_onion_data_final_hourly.csv
final_data_hourly/agmarknet_potato_data_final_hourly.csv
final_data_hourly/agmarknet_tomato_data_final_hourly.csv
final_data_hourly/agmarknet_wheat_data_final_hourly.csv
```

### 1. Clone and install the Python dependencies

Run these commands from a terminal:

```bash
git clone https://github.com/kunalran/PREPARE.git
cd PREPARE

python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Run the web app in development

Start the API from the repository root:

```bash
source .venv/bin/activate
python -m uvicorn web_app.main:app --reload --host 0.0.0.0 --port 8000
```

In a second terminal, start the frontend:

```bash
cd frontend
npm ci
npm run dev
```

Open [http://localhost:5173](http://localhost:5173). Interactive API documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).

For a single-server, production-style local run, build the frontend and then start FastAPI:

```bash
cd frontend
npm ci
npm run build
cd ..

source .venv/bin/activate
python -m uvicorn web_app.main:app --host 0.0.0.0 --port 8000
```

Then open [http://localhost:8000](http://localhost:8000).

## Training and evaluation examples

All scripts expose their available options through `--help`. Run them from the repository root so their default paths resolve correctly.

Train a HistGradientBoosting baseline for one crop and horizon:

```bash
python training/train_per_crop_models.py \
  --crops onion \
  --horizons 15 \
  --data-dir final_data_hourly \
  --output-dir models/per_crop_histgb
```

Train a 15-day GAT-GRU graph model:

```bash
python training/train_graph_gat_gru.py \
  --crops onion \
  --horizon 15 \
  --data-dir final_data_hourly \
  --output-dir models/graph_gat_gru_onion
```

Evaluate naive forecasting baselines:

```bash
PYTHONPATH=training python evaluation/evaluate_naive_baselines.py \
  --data-dir final_data_hourly \
  --horizons 1,7,15
```

Model training can be compute- and memory-intensive. Start with a single crop and horizon, then inspect the relevant script's `--help` output before launching larger experiments.

## Notes

- Saved model artifacts must be present at the paths configured in `web_app/main.py` for predictions to be available.
- The repository retains many historical outputs because they document how the forecasting setup evolved.
- Some folders are archival; the main reusable entry points are under `training/`, `evaluation/`, and `scripts/data_processing/`.
