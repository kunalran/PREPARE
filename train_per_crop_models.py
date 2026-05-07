from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from train_global_price_model import (
    Config,
    load_and_engineer_crop,
    make_pipeline,
    safe_mape,
    safe_wape,
    sample_training_rows,
)


CROP_FILES = {
    "onion": "agmarknet_onion_data_final_hourly.csv",
    "potato": "agmarknet_potato_data_final_hourly.csv",
    "tomato": "agmarknet_tomato_data_final_hourly.csv",
    "wheat": "agmarknet_wheat_data_final_hourly.csv",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train separate crop price forecasting models for 1-15 day horizons."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("final_data_hourly"),
        help="Directory containing agmarknet_*_final_hourly.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/per_crop_histgb"),
        help="Root directory where per-crop models and metrics will be saved.",
    )
    parser.add_argument(
        "--crops",
        type=str,
        default="onion,potato,tomato,wheat",
        help="Comma-separated crops to train. Sugarcane is intentionally excluded.",
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
        help="Comma-separated forecast horizons in days.",
    )
    parser.add_argument(
        "--validation-days",
        type=int,
        default=90,
        help="Number of trailing calendar days reserved for validation.",
    )
    parser.add_argument(
        "--min-series-observations",
        type=int,
        default=30,
        help="Drop market series with fewer observed price points than this.",
    )
    parser.add_argument(
        "--dense-min-pct",
        type=float,
        default=0.0,
        help="Minimum %% non-null Modal_Price per market-year to keep. Use 0 to disable.",
    )
    parser.add_argument(
        "--max-train-rows-per-horizon",
        type=int,
        default=250000,
        help="Optional cap for training rows per horizon.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def selected_crops(raw: str) -> list[str]:
    crops = [c.strip().lower() for c in raw.split(",") if c.strip()]
    invalid = [c for c in crops if c not in CROP_FILES]
    if invalid:
        raise ValueError(
            f"Unsupported crops requested: {invalid}. Valid crops: {sorted(CROP_FILES)}"
        )
    return sorted(dict.fromkeys(crops))


def filter_series(frame: pd.DataFrame, min_series_observations: int) -> pd.DataFrame:
    series_counts = frame.groupby("series_id")["Modal_Price"].count()
    valid_series = series_counts[series_counts >= min_series_observations].index
    filtered = frame[frame["series_id"].isin(valid_series)].copy()
    print(
        f"Kept {len(valid_series):,} series with at least "
        f"{min_series_observations} observed prices."
    )
    print(f"Training frame rows after filtering: {len(filtered):,}")
    return filtered


def feature_columns() -> tuple[list[str], list[str]]:
    categorical_features = ["Commodity", "State", "District", "Market", "series_id"]
    numeric_features = [
        "latitude",
        "longitude",
        "Arrival_Quantity",
        "Modal_Price_CausalFilled",
        "Arrival_Quantity_CausalFilled",
        "arrival_log1p",
        "Modal_Price",
        "price_log1p",
        "price_filled_log1p",
        "temp_mean",
        "temp_min",
        "temp_max",
        "temp_range",
        "rain_sum",
        "solar_sum",
        "solar_peak",
        "rh_mean",
        "rh_min",
        "rh_max",
        "state_price_mean",
        "state_arrival_mean",
        "national_price_mean",
        "national_arrival_mean",
        "month",
        "day_of_week_num",
        "day_of_year",
        "week_of_year",
        "year",
        "series_age_days",
        "is_month_start",
        "is_month_end",
        "day_of_year_sin",
        "day_of_year_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "price_dow_ratio",
        "arrival_dow_ratio",
        "price_lag_1",
        "price_lag_7",
        "price_lag_14",
        "price_lag_28",
        "price_lag_56",
        "price_lag_84",
        "price_lag_112",
        "price_lag_168",
        "arrival_lag_1",
        "arrival_lag_7",
        "arrival_lag_14",
        "arrival_lag_28",
        "arrival_lag_56",
        "arrival_lag_84",
        "arrival_lag_112",
        "arrival_lag_168",
        "price_roll_mean_7",
        "price_roll_mean_14",
        "price_roll_mean_21",
        "price_roll_mean_28",
        "price_roll_mean_56",
        "price_roll_mean_84",
        "price_roll_mean_112",
        "price_roll_mean_168",
        "price_roll_std_7",
        "price_roll_std_14",
        "price_roll_std_28",
        "price_roll_std_56",
        "price_roll_std_84",
        "price_roll_std_112",
        "price_roll_std_168",
        "arrival_roll_mean_7",
        "arrival_roll_mean_14",
        "arrival_roll_mean_21",
        "arrival_roll_mean_28",
        "arrival_roll_mean_56",
        "arrival_roll_mean_84",
        "arrival_roll_mean_112",
        "arrival_roll_mean_168",
        "price_roll_min_28",
        "price_roll_max_28",
        "price_roll_min_56",
        "price_roll_max_56",
        "price_roll_min_84",
        "price_roll_max_84",
        "price_roll_min_168",
        "price_roll_max_168",
        "price_vs_roll7",
        "price_vs_roll28",
        "price_vs_roll84",
        "price_vs_roll168",
        "price_vs_state_mean",
        "price_vs_national_mean",
        "arrival_vs_roll28",
        "arrival_vs_state_mean",
        "arrival_vs_national_mean",
        "price_minus_state_mean",
        "price_minus_national_mean",
        "price_trend_7_28",
        "price_trend_14_56",
        "price_trend_28_84",
        "price_trend_28_168",
        "arrival_trend_7_28",
        "arrival_trend_28_84",
        "price_range_28",
        "price_range_84",
        "price_range_168",
        "price_volatility_ratio_28",
        "price_volatility_ratio_84",
        "price_volatility_ratio_168",
    ]
    return categorical_features, numeric_features


def train_one_crop(
    crop_name: str,
    crop_file: Path,
    output_root: Path,
    config: Config,
) -> list[dict]:
    print(f"\n=== Training {crop_name.upper()} models ===")
    data = load_and_engineer_crop(crop_file, config.horizons, config.dense_min_pct)
    data = filter_series(data, config.min_series_observations)

    crop_dir = output_root / crop_name
    crop_dir.mkdir(parents=True, exist_ok=True)
    categorical_features, numeric_features = feature_columns()

    max_date = data["Date"].max()
    metrics: list[dict] = []

    for horizon in config.horizons:
        validation_start = max_date - pd.Timedelta(days=config.validation_days - 1)
        target_col = f"target_{horizon}d"
        horizon_frame = data[data[target_col].notna()].copy()
        if horizon_frame.empty:
            print(f"Skipping {crop_name} {horizon}d: no rows with non-null targets.")
            continue

        train_frame = horizon_frame[horizon_frame["Date"] < validation_start].copy()
        val_frame = horizon_frame[horizon_frame["Date"] >= validation_start].copy()
        if train_frame.empty or val_frame.empty:
            print(f"Skipping {crop_name} {horizon}d: train or validation split is empty.")
            continue

        y_val = val_frame[target_col].to_numpy()
        train_frame = sample_training_rows(
            train_frame,
            config.max_train_rows_per_horizon,
            config.random_state + horizon,
        )

        X_train = train_frame[categorical_features + numeric_features]
        y_train = np.log1p(train_frame[target_col].to_numpy())
        X_val = val_frame[categorical_features + numeric_features]

        model = make_pipeline(categorical_features, numeric_features, horizon)
        print(
            f"Training {crop_name} horizon {horizon}d with "
            f"{len(train_frame):,} train rows and {len(val_frame):,} validation rows ..."
        )
        model.fit(X_train, y_train)

        pred_log = model.predict(X_val)
        preds = np.expm1(pred_log)
        preds = np.clip(preds, a_min=0.0, a_max=None)

        horizon_metrics = {
            "crop": crop_name,
            "horizon_days": horizon,
            "train_rows": int(len(train_frame)),
            "validation_rows": int(len(val_frame)),
            "validation_start": validation_start.strftime("%Y-%m-%d"),
            "validation_end": max_date.strftime("%Y-%m-%d"),
            "mae": float(mean_absolute_error(y_val, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y_val, preds))),
            "r2": float(r2_score(y_val, preds)),
            "mape_pct": safe_mape(y_val, preds),
            "wape_pct": safe_wape(y_val, preds),
        }
        metrics.append(horizon_metrics)

        artifact = {
            "model": model,
            "categorical_features": categorical_features,
            "numeric_features": numeric_features,
            "crop": crop_name,
            "horizon_days": horizon,
            "validation_start": validation_start.strftime("%Y-%m-%d"),
            "validation_end": max_date.strftime("%Y-%m-%d"),
        }
        model_path = crop_dir / f"{crop_name}_price_model_{horizon}d.joblib"
        joblib.dump(artifact, model_path)
        print(
            f"Saved {model_path} | "
            f"MAE={horizon_metrics['mae']:.2f} RMSE={horizon_metrics['rmse']:.2f} "
            f"R2={horizon_metrics['r2']:.4f} WAPE={horizon_metrics['wape_pct']:.2f}%"
        )

    metrics_path = crop_dir / f"{crop_name}_metrics.json"
    ordered_metrics = sorted(metrics, key=lambda row: int(row["horizon_days"]))
    metrics_path.write_text(json.dumps(ordered_metrics, indent=2), encoding="utf-8")
    print(f"Wrote {metrics_path}")
    return ordered_metrics


def main() -> None:
    args = parse_args()
    crops = selected_crops(args.crops)
    horizons = sorted({int(x.strip()) for x in args.horizons.split(",") if x.strip()})

    config = Config(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        horizons=horizons,
        validation_days=args.validation_days,
        min_series_observations=args.min_series_observations,
        dense_min_pct=args.dense_min_pct,
        max_train_rows_per_horizon=args.max_train_rows_per_horizon,
        random_state=args.random_state,
    )

    all_metrics: dict[str, list[dict]] = {}
    for crop_name in crops:
        crop_file = args.data_dir / CROP_FILES[crop_name]
        if not crop_file.exists():
            raise FileNotFoundError(f"Missing crop file: {crop_file}")
        all_metrics[crop_name] = train_one_crop(
            crop_name=crop_name,
            crop_file=crop_file,
            output_root=args.output_dir,
            config=config,
        )

    summary_path = args.output_dir / "per_crop_training_summary.json"
    summary_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
