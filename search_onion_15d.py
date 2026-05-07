from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from train_global_price_model import Config, load_and_engineer_crop, safe_mape, safe_wape
from train_per_crop_models import CROP_FILES, feature_columns


def load_crops(crops: list[str], horizon: int, dense_min_pct: float | None = None) -> pd.DataFrame:
    frames = []
    for crop in crops:
        path = Path("final_data_hourly") / CROP_FILES[crop]
        frame = load_and_engineer_crop(path, [horizon], dense_min_pct=dense_min_pct)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    return data


def filter_series(data: pd.DataFrame, min_obs: int) -> pd.DataFrame:
    series_counts = data.groupby("series_id")["Modal_Price"].size()
    valid_series = series_counts[series_counts >= min_obs].index
    return data[data["series_id"].isin(valid_series)].copy()


def make_xy(frame: pd.DataFrame, horizon: int):
    categorical_features, numeric_features = feature_columns()
    features = categorical_features + numeric_features
    target_col = f"target_{horizon}d"
    X = frame[features].copy()
    # Ordinalize categoricals directly for tree ensembles
    for col in categorical_features:
        X[col] = X[col].astype("category").cat.codes.astype(np.int32)
    X[numeric_features] = X[numeric_features].fillna(X[numeric_features].median())
    y = frame[target_col].to_numpy()
    return X, y, features


def evaluate(name: str, model, train_frame: pd.DataFrame, val_frame: pd.DataFrame, horizon: int) -> dict:
    X_train, y_train_raw, _ = make_xy(train_frame, horizon)
    X_val, y_val, _ = make_xy(val_frame, horizon)
    y_train = np.log1p(y_train_raw)
    model.fit(X_train, y_train)
    pred_log = model.predict(X_val)
    preds = np.clip(np.expm1(pred_log), a_min=0.0, a_max=None)
    return {
        "name": name,
        "mae": float(mean_absolute_error(y_val, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_val, preds))),
        "r2": float(r2_score(y_val, preds)),
        "mape_pct": safe_mape(y_val, preds),
        "wape_pct": safe_wape(y_val, preds),
        "validation_rows": int(len(val_frame)),
    }


def run_experiments() -> list[dict]:
    horizon = 15
    results: list[dict] = []

    experiments = [
        {
            "name": "all_crops_histgb_full_nocap",
            "crops": ["onion", "potato", "tomato", "wheat"],
            "min_obs": 30,
            "dense": None,
            "train_cap": None,
            "model": HistGradientBoostingRegressor(
                learning_rate=0.03,
                max_depth=10,
                max_iter=500,
                min_samples_leaf=20,
                l2_regularization=0.5,
                early_stopping=False,
                random_state=42,
            ),
        },
        {
            "name": "all_crops_histgb_500k",
            "crops": ["onion", "potato", "tomato", "wheat"],
            "min_obs": 30,
            "dense": None,
            "train_cap": 500000,
            "model": HistGradientBoostingRegressor(
                learning_rate=0.03,
                max_depth=10,
                max_iter=500,
                min_samples_leaf=20,
                l2_regularization=0.5,
                early_stopping=False,
                random_state=42,
            ),
        },
        {
            "name": "onion_potato_tomato_histgb_500k",
            "crops": ["onion", "potato", "tomato"],
            "min_obs": 30,
            "dense": None,
            "train_cap": 500000,
            "model": HistGradientBoostingRegressor(
                learning_rate=0.03,
                max_depth=10,
                max_iter=500,
                min_samples_leaf=20,
                l2_regularization=0.5,
                early_stopping=False,
                random_state=42,
            ),
        },
        {
            "name": "onion_only_histgb_full",
            "crops": ["onion"],
            "min_obs": 30,
            "dense": None,
            "train_cap": None,
            "model": HistGradientBoostingRegressor(
                learning_rate=0.03,
                max_depth=12,
                max_iter=600,
                min_samples_leaf=10,
                l2_regularization=0.5,
                early_stopping=False,
                random_state=42,
            ),
        },
        {
            "name": "onion_only_extratrees_full",
            "crops": ["onion"],
            "min_obs": 30,
            "dense": None,
            "train_cap": None,
            "model": ExtraTreesRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=42,
            ),
        },
        {
            "name": "onion_only_randomforest_full",
            "crops": ["onion"],
            "min_obs": 30,
            "dense": None,
            "train_cap": None,
            "model": RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=42,
            ),
        },
    ]

    for spec in experiments:
        print(f"\n=== {spec['name']} ===")
        data = load_crops(spec["crops"], horizon, dense_min_pct=spec["dense"])
        data = filter_series(data, spec["min_obs"])
        max_date = data["Date"].max()
        validation_start = max_date - pd.Timedelta(days=89)
        target_col = f"target_{horizon}d"
        frame = data[data[target_col].notna()].copy()
        train_frame = frame[frame["Date"] < validation_start].copy()
        val_frame = frame[(frame["Date"] >= validation_start) & (frame["Commodity"] == "Onion")].copy()
        if spec["train_cap"] is not None and len(train_frame) > spec["train_cap"]:
            train_frame = train_frame.sample(spec["train_cap"], random_state=42)
        print(f"train={len(train_frame):,} val_onion={len(val_frame):,}")
        result = evaluate(spec["name"], spec["model"], train_frame, val_frame, horizon)
        print(result)
        results.append(result)

    out = Path("models/onion_15d_search_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return results


if __name__ == "__main__":
    run_experiments()
