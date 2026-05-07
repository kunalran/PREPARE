from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from train_global_price_model import Config, load_and_engineer_crop, make_pipeline, safe_mape, safe_wape, sample_training_rows
from train_per_crop_models import CROP_FILES, feature_columns, filter_series, selected_crops
from train_per_crop_variants import apply_series_normalization, make_target_arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train clustered per-crop variants using mandi production-volume clusters."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("final_data_hourly"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/per_crop_clustered"))
    parser.add_argument("--crops", type=str, default="onion,potato,tomato,wheat")
    parser.add_argument("--horizons", type=str, default="15")
    parser.add_argument("--validation-days", type=int, default=90)
    parser.add_argument("--min-series-observations", type=int, default=30)
    parser.add_argument("--dense-min-pct", type=float, default=0.0)
    parser.add_argument("--max-train-rows-per-horizon", type=int, default=250000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--target-mode", choices=["price", "delta_current", "delta_roll28", "logratio_current"], default="delta_current")
    parser.add_argument("--normalization-mode", choices=["none", "series_mean_center", "series_zscore"], default="series_mean_center")
    parser.add_argument("--cluster-count", type=int, default=4)
    return parser.parse_args()


def add_volume_clusters(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    cluster_count: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_frame = train_frame.copy()
    val_frame = val_frame.copy()
    stats = train_frame.groupby("series_id").agg(
        mean_arrival=("Arrival_Quantity_CausalFilled", "mean"),
        std_arrival=("Arrival_Quantity_CausalFilled", "std"),
        mean_price=("Modal_Price_CausalFilled", "mean"),
    )
    stats["std_arrival"] = stats["std_arrival"].fillna(0.0)
    features = np.log1p(stats[["mean_arrival", "std_arrival", "mean_price"]].fillna(0.0).to_numpy())
    k = min(cluster_count, len(stats))
    if k <= 1:
        stats["cluster_id"] = 0
    else:
        labels = KMeans(n_clusters=k, n_init=20, random_state=random_state).fit_predict(features)
        stats["cluster_id"] = labels.astype(int)
    train_frame["cluster_id"] = train_frame["series_id"].map(stats["cluster_id"]).fillna(-1).astype(int).astype(str)
    val_frame["cluster_id"] = val_frame["series_id"].map(stats["cluster_id"]).fillna(-1).astype(int).astype(str)
    return train_frame, val_frame


def train_one_crop(crop_name: str, crop_file: Path, output_root: Path, args: argparse.Namespace) -> list[dict]:
    print(f"\n=== Clustered variant {crop_name.upper()} ===")
    horizons = sorted({int(x.strip()) for x in args.horizons.split(",") if x.strip()})
    config = Config(
        data_dir=args.data_dir,
        output_dir=output_root,
        horizons=horizons,
        validation_days=args.validation_days,
        min_series_observations=args.min_series_observations,
        dense_min_pct=args.dense_min_pct,
        max_train_rows_per_horizon=args.max_train_rows_per_horizon,
        random_state=args.random_state,
    )
    data = load_and_engineer_crop(crop_file, config.horizons, config.dense_min_pct)
    data = filter_series(data, config.min_series_observations)
    categorical_features, numeric_features = feature_columns()
    categorical_features = categorical_features + ["cluster_id"]
    crop_dir = output_root / crop_name
    crop_dir.mkdir(parents=True, exist_ok=True)
    max_date = data["Date"].max()
    metrics: list[dict] = []

    for horizon in horizons:
        validation_start = max_date - pd.Timedelta(days=config.validation_days - 1)
        target_col = f"target_{horizon}d"
        frame = data[data[target_col].notna()].copy()
        train_frame = frame[frame["Date"] < validation_start].copy()
        val_frame = frame[frame["Date"] >= validation_start].copy()
        if train_frame.empty or val_frame.empty:
            continue

        train_frame = sample_training_rows(train_frame, config.max_train_rows_per_horizon, config.random_state + horizon)
        raw_train = train_frame.copy()
        raw_val = val_frame.copy()
        train_frame, val_frame = add_volume_clusters(train_frame, val_frame, args.cluster_count, config.random_state + horizon)
        train_frame, val_frame = apply_series_normalization(
            train_frame,
            val_frame,
            numeric_features,
            args.normalization_mode,
        )
        y_train, y_val, anchor_val = make_target_arrays(
            train_frame,
            val_frame,
            target_col,
            args.target_mode,
            train_anchor_source=raw_train,
            val_anchor_source=raw_val,
        )
        X_train = train_frame[categorical_features + numeric_features]
        X_val = val_frame[categorical_features + numeric_features]
        model = make_pipeline(categorical_features, numeric_features, horizon)
        print(
            f"Training {crop_name} {horizon}d target={args.target_mode} "
            f"norm={args.normalization_mode} clusters={args.cluster_count} "
            f"with {len(train_frame):,} train rows and {len(val_frame):,} validation rows ..."
        )
        model.fit(X_train, y_train)
        pred_raw = model.predict(X_val)
        preds = np.expm1(pred_raw + anchor_val) if args.target_mode != "price" else np.expm1(pred_raw)
        preds = np.clip(preds, a_min=0.0, a_max=None)
        row = {
            "crop": crop_name,
            "horizon_days": horizon,
            "target_mode": args.target_mode,
            "normalization_mode": args.normalization_mode,
            "cluster_count": args.cluster_count,
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
        metrics.append(row)
        artifact = {
            "model": model,
            "categorical_features": categorical_features,
            "numeric_features": numeric_features,
            "crop": crop_name,
            "horizon_days": horizon,
            "target_mode": args.target_mode,
            "normalization_mode": args.normalization_mode,
            "cluster_count": args.cluster_count,
        }
        model_path = crop_dir / f"{crop_name}_{args.target_mode}_{args.normalization_mode}_clusters{args.cluster_count}_{horizon}d.joblib"
        joblib.dump(artifact, model_path)
        print(
            f"Saved {model_path} | MAE={row['mae']:.2f} RMSE={row['rmse']:.2f} "
            f"R2={row['r2']:.4f} WAPE={row['wape_pct']:.2f}%"
        )

    metrics_path = crop_dir / f"{crop_name}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    args = parse_args()
    crops = selected_crops(args.crops)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, list[dict]] = {}
    for crop in crops:
        crop_file = args.data_dir / CROP_FILES[crop]
        summary[crop] = train_one_crop(crop, crop_file, args.output_dir, args)
    summary_path = args.output_dir / "per_crop_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
