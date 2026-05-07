from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from train_global_price_model import Config, load_training_frame, safe_mape, safe_wape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate previously trained pooled crop price models."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/global_histgb"),
        help="Directory containing global_price_model_*d.joblib and metrics.json",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("final_data_hourly"),
        help="Directory containing agmarknet_*_final_hourly.csv files.",
    )
    parser.add_argument(
        "--validation-days",
        type=int,
        default=90,
        help="Trailing days used for validation.",
    )
    parser.add_argument(
        "--min-series-observations",
        type=int,
        default=30,
        help="Minimum observed prices required per series.",
    )
    parser.add_argument(
        "--dense-min-pct",
        type=float,
        default=None,
        help="Optional minimum %% non-null Modal_Price per market-year to keep.",
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
        help="Comma-separated forecast horizons in days.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write evaluation metrics JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    horizons = sorted({int(x.strip()) for x in args.horizons.split(",") if x.strip()})

    config = Config(
        data_dir=args.data_dir,
        output_dir=args.model_dir,
        horizons=horizons,
        validation_days=args.validation_days,
        min_series_observations=args.min_series_observations,
        dense_min_pct=args.dense_min_pct,
        max_train_rows_per_horizon=None,
        random_state=42,
    )
    data = load_training_frame(config)
    max_date = data["Date"].max()
    validation_start = max_date - np.timedelta64(config.validation_days - 1, "D")

    results: list[dict] = []
    for horizon in horizons:
        model_path = args.model_dir / f"global_price_model_{horizon}d.joblib"
        if not model_path.exists():
            print(f"Skipping {horizon}d: missing {model_path}")
            continue

        artifact = joblib.load(model_path)
        target_col = f"target_{horizon}d"
        frame = data[data[target_col].notna()].copy()
        val_frame = frame[frame["Date"] >= validation_start].copy()
        if val_frame.empty:
            print(f"Skipping {horizon}d: validation frame is empty.")
            continue

        features = artifact["categorical_features"] + artifact["numeric_features"]
        preds_log = artifact["model"].predict(val_frame[features])
        preds = np.expm1(preds_log)
        preds = np.clip(preds, a_min=0.0, a_max=None)
        y_true = val_frame[target_col].to_numpy()

        metrics = {
            "horizon_days": horizon,
            "commodity": "ALL",
            "validation_rows": int(len(val_frame)),
            "mae": float(mean_absolute_error(y_true, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, preds))),
            "r2": float(r2_score(y_true, preds)),
            "mape_pct": safe_mape(y_true, preds),
            "wape_pct": safe_wape(y_true, preds),
        }
        results.append(metrics)
        print(
            f"ALL {horizon:>2}d  val={metrics['validation_rows']:,}  "
            f"MAE={metrics['mae']:.2f}  RMSE={metrics['rmse']:.2f}  "
            f"R2={metrics['r2']:.4f}  WAPE={metrics['wape_pct']:.2f}%"
        )

        for commodity, crop_frame in val_frame.groupby("Commodity", sort=True):
            crop_y_true = crop_frame[target_col].to_numpy()
            crop_preds_log = artifact["model"].predict(crop_frame[features])
            crop_preds = np.expm1(crop_preds_log)
            crop_preds = np.clip(crop_preds, a_min=0.0, a_max=None)

            crop_metrics = {
                "horizon_days": horizon,
                "commodity": str(commodity),
                "validation_rows": int(len(crop_frame)),
                "mae": float(mean_absolute_error(crop_y_true, crop_preds)),
                "rmse": float(np.sqrt(mean_squared_error(crop_y_true, crop_preds))),
                "r2": float(r2_score(crop_y_true, crop_preds)),
                "mape_pct": safe_mape(crop_y_true, crop_preds),
                "wape_pct": safe_wape(crop_y_true, crop_preds),
            }
            results.append(crop_metrics)
            print(
                f"  {str(commodity):<10} {horizon:>2}d  val={crop_metrics['validation_rows']:,}  "
                f"MAE={crop_metrics['mae']:.2f}  RMSE={crop_metrics['rmse']:.2f}  "
                f"R2={crop_metrics['r2']:.4f}  WAPE={crop_metrics['wape_pct']:.2f}%"
            )

    if args.output_json:
        args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
