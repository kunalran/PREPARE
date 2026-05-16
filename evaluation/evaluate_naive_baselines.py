from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.train_global_price_model import Config, load_and_engineer_crop, safe_mape, safe_wape
from training.train_per_crop_models import CROP_FILES, filter_series


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate naive per-crop baselines on the current hourly dataset."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("final_data_hourly"))
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("models/naive_baselines_summary.json"),
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
    return parser.parse_args()


def selected_crops(raw: str) -> list[str]:
    crops = [c.strip().lower() for c in raw.split(",") if c.strip()]
    invalid = [c for c in crops if c not in CROP_FILES]
    if invalid:
        raise ValueError(f"Unsupported crops: {invalid}")
    return sorted(dict.fromkeys(crops))


def baseline_predictions(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "current_price": frame["Modal_Price_CausalFilled"].to_numpy(),
        "roll_mean_7": frame["price_roll_mean_7"].to_numpy(),
        "roll_mean_28": frame["price_roll_mean_28"].to_numpy(),
    }


def main() -> None:
    args = parse_args()
    crops = selected_crops(args.crops)
    horizons = sorted({int(x.strip()) for x in args.horizons.split(",") if x.strip()})
    output: dict[str, dict[str, list[dict]]] = {}

    config = Config(
        data_dir=args.data_dir,
        output_dir=Path("unused"),
        horizons=horizons,
        validation_days=args.validation_days,
        min_series_observations=args.min_series_observations,
        dense_min_pct=args.dense_min_pct,
        max_train_rows_per_horizon=None,
        random_state=42,
    )

    for crop in crops:
        print(f"\n=== Naive baselines for {crop.upper()} ===")
        crop_file = args.data_dir / CROP_FILES[crop]
        data = load_and_engineer_crop(crop_file, horizons, config.dense_min_pct)
        data = filter_series(data, config.min_series_observations)
        max_date = data["Date"].max()
        validation_start = max_date - pd.Timedelta(days=config.validation_days - 1)
        crop_results: dict[str, list[dict]] = {}

        for horizon in horizons:
            target_col = f"target_{horizon}d"
            frame = data[data[target_col].notna()].copy()
            val = frame[frame["Date"] >= validation_start].copy()
            if val.empty:
                continue
            y_val = val[target_col].to_numpy()
            crop_results[str(horizon)] = []
            for baseline_name, preds in baseline_predictions(val).items():
                preds = np.clip(np.nan_to_num(preds, nan=0.0), a_min=0.0, a_max=None)
                result = {
                    "baseline": baseline_name,
                    "crop": crop,
                    "horizon_days": horizon,
                    "validation_rows": int(len(val)),
                    "validation_start": validation_start.strftime("%Y-%m-%d"),
                    "validation_end": max_date.strftime("%Y-%m-%d"),
                    "mae": float(mean_absolute_error(y_val, preds)),
                    "rmse": float(np.sqrt(mean_squared_error(y_val, preds))),
                    "r2": float(r2_score(y_val, preds)),
                    "mape_pct": safe_mape(y_val, preds),
                    "wape_pct": safe_wape(y_val, preds),
                }
                crop_results[str(horizon)].append(result)
                print(
                    f"{crop} {horizon:2d}d {baseline_name:14s} "
                    f"R2={result['r2']:.4f} MAE={result['mae']:.2f} "
                    f"RMSE={result['rmse']:.2f} WAPE={result['wape_pct']:.2f}%"
                )
        output[crop] = crop_results

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output_path}")


if __name__ == "__main__":
    main()
