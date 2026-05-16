from __future__ import annotations

import argparse
import json
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
from train_per_crop_models import CROP_FILES, feature_columns, filter_series, selected_crops


NORMALIZE_SKIP_PREFIXES = (
    "latitude",
    "longitude",
    "temp_",
    "rain_",
    "solar_",
    "rh_",
    "month",
    "day_of_week",
    "day_of_year",
    "week_of_year",
    "year",
    "is_month",
    "series_age_days",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train no-shortcut per-crop HistGB variants with shifted targets and normalization."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("final_data_hourly"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/per_crop_histgb_variant"),
    )
    parser.add_argument(
        "--crops",
        type=str,
        default="onion,potato,tomato,wheat",
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default="15",
    )
    parser.add_argument("--validation-days", type=int, default=90)
    parser.add_argument("--min-series-observations", type=int, default=30)
    parser.add_argument("--dense-min-pct", type=float, default=0.0)
    parser.add_argument("--max-train-rows-per-horizon", type=int, default=250000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--target-mode",
        choices=["price", "delta_current", "delta_roll28", "logratio_current"],
        default="price",
    )
    parser.add_argument(
        "--normalization-mode",
        choices=["none", "series_mean_center", "series_zscore"],
        default="none",
    )
    return parser.parse_args()


def columns_to_normalize(numeric_features: list[str]) -> list[str]:
    cols: list[str] = []
    for col in numeric_features:
        if any(col.startswith(prefix) for prefix in NORMALIZE_SKIP_PREFIXES):
            continue
        cols.append(col)
    return cols


def apply_series_normalization(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    columns: list[str],
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if mode == "none":
        return train_frame, val_frame

    train_frame = train_frame.copy()
    val_frame = val_frame.copy()

    for col in columns:
        stats = train_frame.groupby("series_id")[col].agg(["mean", "std"])
        stats["mean"] = stats["mean"].fillna(0.0)
        stats["std"] = stats["std"].replace(0.0, np.nan).fillna(1.0)

        train_mean = train_frame["series_id"].map(stats["mean"])
        val_mean = val_frame["series_id"].map(stats["mean"]).fillna(stats["mean"].mean())
        if mode == "series_mean_center":
            train_frame[col] = train_frame[col] - train_mean
            val_frame[col] = val_frame[col] - val_mean
            continue

        train_std = train_frame["series_id"].map(stats["std"])
        val_std = val_frame["series_id"].map(stats["std"]).fillna(1.0)
        train_frame[col] = (train_frame[col] - train_mean) / train_std
        val_frame[col] = (val_frame[col] - val_mean) / val_std

    return train_frame, val_frame


def make_target_arrays(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    target_col: str,
    target_mode: str,
    train_anchor_source: pd.DataFrame | None = None,
    val_anchor_source: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_val_actual = val_frame[target_col].to_numpy()
    train_anchor_source = train_frame if train_anchor_source is None else train_anchor_source
    val_anchor_source = val_frame if val_anchor_source is None else val_anchor_source

    if target_mode == "price":
        y_train = np.log1p(train_frame[target_col].to_numpy())
        anchor_val = np.zeros(len(val_frame), dtype=float)
        return y_train, y_val_actual, anchor_val

    if target_mode == "delta_current":
        train_anchor = np.log1p(
            np.clip(train_anchor_source["Modal_Price_CausalFilled"].to_numpy(), a_min=0.0, a_max=None)
        )
        val_anchor = np.log1p(
            np.clip(val_anchor_source["Modal_Price_CausalFilled"].to_numpy(), a_min=0.0, a_max=None)
        )
        y_train = np.log1p(train_frame[target_col].to_numpy()) - train_anchor
        return y_train, y_val_actual, val_anchor

    if target_mode == "logratio_current":
        train_anchor = np.log1p(
            np.clip(train_anchor_source["Modal_Price_CausalFilled"].to_numpy(), a_min=0.0, a_max=None)
        )
        val_anchor = np.log1p(
            np.clip(val_anchor_source["Modal_Price_CausalFilled"].to_numpy(), a_min=0.0, a_max=None)
        )
        y_train = np.log1p(train_frame[target_col].to_numpy()) - train_anchor
        return y_train, y_val_actual, val_anchor

    if target_mode == "delta_roll28":
        train_roll = np.clip(train_anchor_source["price_roll_mean_28"].to_numpy(), a_min=0.0, a_max=None)
        val_roll = np.clip(val_anchor_source["price_roll_mean_28"].to_numpy(), a_min=0.0, a_max=None)
        train_anchor = np.log1p(train_roll)
        val_anchor = np.log1p(val_roll)
        y_train = np.log1p(train_frame[target_col].to_numpy()) - train_anchor
        return y_train, y_val_actual, val_anchor

    raise ValueError(f"Unsupported target mode: {target_mode}")


def invert_predictions(pred_raw: np.ndarray, anchor_val: np.ndarray, target_mode: str) -> np.ndarray:
    if target_mode == "price":
        preds = np.expm1(pred_raw)
    else:
        preds = np.expm1(pred_raw + anchor_val)
    return np.clip(preds, a_min=0.0, a_max=None)


def train_one_crop(crop_name: str, crop_file: Path, output_root: Path, args: argparse.Namespace) -> list[dict]:
    print(f"\n=== Variant training {crop_name.upper()} ===")
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
    normalize_cols = columns_to_normalize(numeric_features)
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

        raw_train_frame = sample_training_rows(
            train_frame,
            config.max_train_rows_per_horizon,
            config.random_state + horizon,
        )
        raw_val_frame = val_frame.copy()
        train_frame, val_frame = apply_series_normalization(
            raw_train_frame,
            raw_val_frame,
            normalize_cols,
            args.normalization_mode,
        )

        y_train, y_val, anchor_val = make_target_arrays(
            train_frame,
            val_frame,
            target_col,
            args.target_mode,
            train_anchor_source=raw_train_frame,
            val_anchor_source=raw_val_frame,
        )
        X_train = train_frame[categorical_features + numeric_features]
        X_val = val_frame[categorical_features + numeric_features]
        model = make_pipeline(categorical_features, numeric_features, horizon)
        print(
            f"Training {crop_name} {horizon}d target={args.target_mode} "
            f"norm={args.normalization_mode} with {len(train_frame):,} train rows "
            f"and {len(val_frame):,} validation rows ..."
        )
        model.fit(X_train, y_train)
        pred_raw = model.predict(X_val)
        preds = invert_predictions(pred_raw, anchor_val, args.target_mode)

        row = {
            "crop": crop_name,
            "horizon_days": horizon,
            "target_mode": args.target_mode,
            "normalization_mode": args.normalization_mode,
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
        }
        model_path = crop_dir / f"{crop_name}_{args.target_mode}_{args.normalization_mode}_{horizon}d.joblib"
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
