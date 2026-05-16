from __future__ import annotations

import json
import math
import os
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "training") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "training"))
if str(REPO_ROOT / "evaluation") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from baseline_models import NumericSubsetHistGBRegressor  # type: ignore  # noqa: E402
from training.train_global_price_model import (  # type: ignore  # noqa: E402
    Config,
    load_and_engineer_crop,
    make_pipeline,
    safe_mape,
    safe_wape,
    sample_training_rows,
)
from training.train_per_crop_clustered import add_volume_clusters  # type: ignore  # noqa: E402
from training.train_per_crop_models import CROP_FILES, feature_columns, filter_series  # type: ignore  # noqa: E402
from training.train_per_crop_variants import (  # type: ignore  # noqa: E402
    apply_series_normalization,
    invert_predictions,
    make_target_arrays,
)


IMPUTED_CROP_FILES = {
    "onion": "agmarknet_onion_data_imputed_hourly.csv",
    "potato": "agmarknet_potato_data_imputed_hourly.csv",
    "tomato": "agmarknet_tomato_data_imputed_hourly.csv",
    "wheat": "agmarknet_wheat_data_imputed_hourly.csv",
}

GRAPH_THRESHOLDS_KM = (75.0, 150.0, 300.0)
GRAPH_PRIMARY_THRESHOLD_KM = 150.0


@dataclass(frozen=True)
class ExperimentConfig:
    data_dir: Path
    output_root: Path
    horizons: tuple[int, ...]
    graph_horizon: int
    validation_days: int
    min_series_observations: int
    dense_min_pct: float
    max_train_rows_full: int
    max_train_rows_simple: int
    random_state: int


def ensure_venv() -> None:
    if sys.prefix == sys.base_prefix:
        raise RuntimeError(
            "This script must run inside a virtual environment. "
            "Use newtests/venv/bin/python."
        )


def set_local_runtime_dirs(output_root: Path) -> None:
    cache_dir = output_root / ".cache" / "matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    warnings.filterwarnings("ignore", category=PerformanceWarning)
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message="The previous implementation of stack is deprecated",
    )


def experiment_config() -> ExperimentConfig:
    output_root = REPO_ROOT / "newtests" / "targeted_rebuild"
    return ExperimentConfig(
        data_dir=REPO_ROOT / "final_data_hourly_dow_imputed",
        output_root=output_root,
        horizons=tuple(range(1, 16)),
        graph_horizon=15,
        validation_days=90,
        min_series_observations=30,
        dense_min_pct=0.0,
        max_train_rows_full=10000,
        max_train_rows_simple=20000,
        random_state=42,
    )


def config_to_training(cfg: ExperimentConfig, horizons: list[int], max_rows: int | None) -> Config:
    return Config(
        data_dir=cfg.data_dir,
        output_dir=cfg.output_root / "results",
        horizons=horizons,
        validation_days=cfg.validation_days,
        min_series_observations=cfg.min_series_observations,
        dense_min_pct=cfg.dense_min_pct,
        max_train_rows_per_horizon=max_rows,
        random_state=cfg.random_state,
    )


def imputed_path(cfg: ExperimentConfig, crop: str) -> Path:
    return cfg.data_dir / IMPUTED_CROP_FILES[crop]


def load_crop_frames(cfg: ExperimentConfig) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    train_cfg = config_to_training(cfg, list(cfg.horizons), cfg.max_train_rows_full)
    for crop in sorted(IMPUTED_CROP_FILES):
        print(f"Loading engineered frame for {crop} ...")
        frame = load_and_engineer_crop(
            imputed_path(cfg, crop),
            list(cfg.horizons),
            dense_min_pct=train_cfg.dense_min_pct,
            observed_only=True,
        )
        frame = filter_series(frame, cfg.min_series_observations)
        frames[crop] = frame
    return frames


def load_raw_inventory_frame(path: Path) -> pd.DataFrame:
    usecols = [
        "State",
        "District",
        "Market",
        "Commodity",
        "Date",
        "Arrival_Quantity",
        "Modal_Price",
        "latitude",
        "longitude",
    ]
    return pd.read_csv(path, usecols=usecols, parse_dates=["Date"])


def haversine_km(latlon: np.ndarray) -> np.ndarray:
    earth_radius_km = 6371.0088
    latlon = np.nan_to_num(latlon, nan=0.0, posinf=0.0, neginf=0.0)
    lat = np.radians(latlon[:, 0])
    lon = np.radians(latlon[:, 1])
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))
    return earth_radius_km * c


def build_threshold_weights(meta: pd.DataFrame, threshold_km: float) -> tuple[np.ndarray, np.ndarray]:
    coords = meta[["latitude", "longitude"]].to_numpy(dtype=float)
    dists = haversine_km(coords)
    dists = np.nan_to_num(dists, nan=np.inf, posinf=np.inf, neginf=np.inf)
    adjacency = (dists <= threshold_km).astype(float)
    np.fill_diagonal(adjacency, 0.0)
    degree = adjacency.sum(axis=1)
    denom = np.where(degree > 0.0, degree, 1.0)
    weights = np.nan_to_num(adjacency / denom[:, None], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return weights, degree.astype(int)


def attach_graph_features(frame: pd.DataFrame, thresholds_km: tuple[float, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = (
        frame.groupby("series_id", sort=False)
        .agg(
            Commodity=("Commodity", "first"),
            State=("State", "first"),
            District=("District", "first"),
            Market=("Market", "first"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            mean_arrival=("Arrival_Quantity_CausalFilled", "mean"),
            std_arrival=("Arrival_Quantity_CausalFilled", "std"),
            mean_price=("Modal_Price_CausalFilled", "mean"),
        )
        .sort_index()
    )
    meta["std_arrival"] = meta["std_arrival"].fillna(0.0)

    series_ids = meta.index.tolist()
    dates = pd.Index(sorted(frame["Date"].unique()))
    price_panel = (
        frame.pivot(index="Date", columns="series_id", values="Modal_Price_CausalFilled")
        .reindex(index=dates, columns=series_ids)
        .to_numpy(dtype=float)
    )
    roll28_panel = (
        frame.pivot(index="Date", columns="series_id", values="price_roll_mean_28")
        .reindex(index=dates, columns=series_ids)
        .to_numpy(dtype=float)
    )
    filled_price_panel = np.nan_to_num(price_panel, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    filled_roll28_panel = np.nan_to_num(roll28_panel, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    enriched = frame.copy()
    for threshold in thresholds_km:
        weights, degree = build_threshold_weights(meta, threshold)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            neighbor_price_panel = np.nan_to_num(
                filled_price_panel @ weights.T,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            neighbor_roll28_panel = np.nan_to_num(
                filled_roll28_panel @ weights.T,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        suffix = f"{int(threshold)}km"

        neighbor_price = (
            pd.DataFrame(neighbor_price_panel, index=dates, columns=series_ids)
            .stack(dropna=False)
            .rename(f"neighbor_price_{suffix}")
            .reset_index()
            .rename(columns={"level_0": "Date", "level_1": "series_id"})
        )
        neighbor_roll28 = (
            pd.DataFrame(neighbor_roll28_panel, index=dates, columns=series_ids)
            .stack(dropna=False)
            .rename(f"neighbor_roll28_{suffix}")
            .reset_index()
            .rename(columns={"level_0": "Date", "level_1": "series_id"})
        )
        degree_frame = meta.reset_index()[["series_id"]].copy()
        degree_frame[f"neighbor_degree_{suffix}"] = degree
        meta[f"neighbor_degree_{suffix}"] = degree

        enriched = enriched.merge(neighbor_price, on=["Date", "series_id"], how="left")
        enriched = enriched.merge(neighbor_roll28, on=["Date", "series_id"], how="left")
        enriched = enriched.merge(degree_frame, on="series_id", how="left")

    primary_degree = meta[f"neighbor_degree_{int(GRAPH_PRIMARY_THRESHOLD_KM)}km"]
    median_degree = float(np.median(primary_degree))
    meta["density_bucket"] = np.where(primary_degree > median_degree, "dense", "sparse")
    meta["volume_bucket"] = pd.qcut(
        meta["mean_arrival"].rank(method="first"),
        q=3,
        labels=["low", "medium", "high"],
    ).astype(str)
    enriched = enriched.merge(
        meta.reset_index()[["series_id", "density_bucket", "volume_bucket"]],
        on="series_id",
        how="left",
    )
    return enriched, meta.reset_index()


def build_inventory_outputs(cfg: ExperimentConfig, frames: dict[str, pd.DataFrame]) -> None:
    results_dir = cfg.output_root / "results" / "inventory"
    results_dir.mkdir(parents=True, exist_ok=True)

    inventory_rows: list[dict[str, object]] = []
    state_rows: list[dict[str, object]] = []
    district_rows: list[dict[str, object]] = []
    density_rows: list[dict[str, object]] = []
    pair_input: dict[str, pd.Series] = {}

    for crop in sorted(frames):
        raw = load_raw_inventory_frame(imputed_path(cfg, crop))
        market_frame = raw[["State", "District", "Market", "latitude", "longitude"]].drop_duplicates().copy()
        inventory_rows.append(
            {
                "crop": crop,
                "rows": int(len(raw)),
                "mandis": int(len(market_frame)),
                "date_min": raw["Date"].min().strftime("%Y-%m-%d"),
                "date_max": raw["Date"].max().strftime("%Y-%m-%d"),
                "non_null_price_rows": int(raw["Modal_Price"].notna().sum()),
                "non_null_arrival_rows": int(raw["Arrival_Quantity"].notna().sum()),
            }
        )

        state_counts = (
            raw[["State", "District", "Market"]]
            .drop_duplicates()
            .groupby("State")
            .size()
            .sort_values(ascending=False)
            .reset_index(name="mandis")
        )
        for row in state_counts.itertuples(index=False):
            state_rows.append({"crop": crop, "State": row.State, "mandis": int(row.mandis)})

        district_counts = (
            raw[["State", "District", "Market"]]
            .drop_duplicates()
            .groupby(["State", "District"])
            .size()
            .sort_values(ascending=False)
            .reset_index(name="mandis")
        )
        for row in district_counts.itertuples(index=False):
            district_rows.append(
                {"crop": crop, "State": row.State, "District": row.District, "mandis": int(row.mandis)}
            )

        coords = market_frame[["latitude", "longitude"]].to_numpy(dtype=float)
        dists = haversine_km(coords)
        np.fill_diagonal(dists, np.inf)
        for threshold in GRAPH_THRESHOLDS_KM:
            degree = (dists <= threshold).sum(axis=1)
            density_rows.append(
                {
                    "crop": crop,
                    "threshold_km": threshold,
                    "mean_neighbors": float(np.mean(degree)),
                    "median_neighbors": float(np.median(degree)),
                    "max_neighbors": int(np.max(degree)) if len(degree) else 0,
                    "min_neighbors": int(np.min(degree)) if len(degree) else 0,
                }
            )

        pair_input[crop] = (
            raw.groupby("Date", sort=True)["Modal_Price"]
            .mean()
            .sort_index()
            .rename(crop)
        )

    correlation_rows: list[dict[str, object]] = []
    crops = sorted(pair_input)
    for crop in crops:
        left = pair_input[crop]
        for other in crops:
            if crop == other:
                continue
            joined = pd.concat([left, pair_input[other]], axis=1, join="inner").dropna()
            corr = joined.iloc[:, 0].corr(joined.iloc[:, 1]) if len(joined) >= 2 else np.nan
            correlation_rows.append(
                {
                    "crop": crop,
                    "other_crop": other,
                    "daily_national_price_corr": float(corr) if pd.notna(corr) else np.nan,
                }
            )

    pd.DataFrame(inventory_rows).to_csv(results_dir / "crop_inventory.csv", index=False)
    pd.DataFrame(state_rows).to_csv(results_dir / "state_mandi_counts.csv", index=False)
    pd.DataFrame(district_rows).to_csv(results_dir / "district_mandi_counts.csv", index=False)
    pd.DataFrame(density_rows).to_csv(results_dir / "graph_density_summary.csv", index=False)
    pd.DataFrame(correlation_rows).to_csv(results_dir / "cross_crop_daily_correlations.csv", index=False)


def validation_start_for(frame: pd.DataFrame, validation_days: int) -> pd.Timestamp:
    max_date = frame["Date"].max()
    return max_date - pd.Timedelta(days=validation_days - 1)


def split_train_val(frame: pd.DataFrame, horizon: int, validation_days: int) -> tuple[pd.DataFrame, pd.DataFrame, str, pd.Timestamp, pd.Timestamp]:
    target_col = f"target_{horizon}d"
    subset = frame[frame[target_col].notna()].copy()
    validation_start = validation_start_for(frame, validation_days)
    train_frame = subset[subset["Date"] < validation_start].copy()
    val_frame = subset[subset["Date"] >= validation_start].copy()
    return train_frame, val_frame, target_col, validation_start, frame["Date"].max()


def metric_row(
    *,
    experiment_family: str,
    experiment_name: str,
    crop: str,
    horizon: int,
    validation_rows: int,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
    y_true: np.ndarray,
    preds: np.ndarray,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "experiment_family": experiment_family,
        "experiment_name": experiment_name,
        "crop": crop,
        "horizon_days": horizon,
        "validation_rows": validation_rows,
        "validation_start": validation_start.strftime("%Y-%m-%d"),
        "validation_end": validation_end.strftime("%Y-%m-%d"),
        "mae": float(mean_absolute_error(y_true, preds)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, preds))),
        "r2": float(r2_score(y_true, preds)),
        "mape_pct": safe_mape(y_true, preds),
        "wape_pct": safe_wape(y_true, preds),
    }
    if extra:
        row.update(extra)
    return row


def save_results(cfg: ExperimentConfig, name: str, rows: list[dict[str, object]]) -> Path:
    out_dir = cfg.output_root / "results" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    csv_path = out_dir / f"{name}.csv"
    json_path = out_dir / f"{name}.json"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {csv_path}")
    return csv_path


def select_cross_crop_pairs(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    by_crop = {
        crop: frame.groupby("Date", sort=True)["national_price_mean"].mean().rename(crop)
        for crop, frame in frames.items()
    }
    rows: list[dict[str, object]] = []
    crops = sorted(by_crop)
    for crop in crops:
        best_other = None
        best_corr = -np.inf
        for other in crops:
            if crop == other:
                continue
            joined = pd.concat([by_crop[crop], by_crop[other]], axis=1, join="inner").dropna()
            corr = joined.iloc[:, 0].corr(joined.iloc[:, 1]) if len(joined) >= 2 else np.nan
            corr_value = float(corr) if pd.notna(corr) else float("-inf")
            if corr_value > best_corr:
                best_corr = corr_value
                best_other = other
        rows.append(
            {
                "crop": crop,
                "paired_crop": best_other,
                "daily_national_price_corr": best_corr,
            }
        )
    return pd.DataFrame(rows)


def run_baselines(cfg: ExperimentConfig, frames: dict[str, pd.DataFrame]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for crop, frame in sorted(frames.items()):
        for horizon in cfg.horizons:
            train_frame, val_frame, target_col, validation_start, validation_end = split_train_val(
                frame, horizon, cfg.validation_days
            )
            del train_frame
            if val_frame.empty:
                continue
            y_true = val_frame[target_col].to_numpy()
            baselines = {
                "previous_day_price": val_frame["price_lag_1"].to_numpy(),
                "current_price": val_frame["Modal_Price_CausalFilled"].to_numpy(),
                "roll_mean_7": val_frame["price_roll_mean_7"].to_numpy(),
                "roll_mean_28": val_frame["price_roll_mean_28"].to_numpy(),
            }
            for baseline_name, preds in baselines.items():
                preds = np.clip(np.nan_to_num(preds, nan=0.0), a_min=0.0, a_max=None)
                rows.append(
                    metric_row(
                        experiment_family="baseline",
                        experiment_name=baseline_name,
                        crop=crop,
                        horizon=horizon,
                        validation_rows=len(val_frame),
                        validation_start=validation_start,
                        validation_end=validation_end,
                        y_true=y_true,
                        preds=preds,
                    )
                )
    return rows


def simple_numeric_features() -> list[str]:
    return [
        "latitude",
        "longitude",
        "Arrival_Quantity_CausalFilled",
        "Modal_Price_CausalFilled",
        "arrival_log1p",
        "price_filled_log1p",
        "temp_mean",
        "temp_range",
        "rain_sum",
        "solar_sum",
        "rh_mean",
        "state_price_mean",
        "national_price_mean",
        "month",
        "day_of_week_num",
        "day_of_year_sin",
        "day_of_year_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "price_dow_ratio",
        "arrival_dow_ratio",
        "price_lag_1",
        "price_lag_7",
        "price_lag_28",
        "arrival_lag_7",
        "arrival_lag_28",
        "price_roll_mean_7",
        "price_roll_mean_28",
        "price_roll_std_28",
        "price_vs_roll28",
        "price_trend_7_28",
        "price_range_28",
        "price_volatility_ratio_28",
    ]


def run_simple_numeric_model(cfg: ExperimentConfig, frames: dict[str, pd.DataFrame]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    numeric_features = simple_numeric_features()
    for crop, frame in sorted(frames.items()):
        for horizon in cfg.horizons:
            train_frame, val_frame, target_col, validation_start, validation_end = split_train_val(
                frame, horizon, cfg.validation_days
            )
            if train_frame.empty or val_frame.empty:
                continue
            train_frame = sample_training_rows(train_frame, cfg.max_train_rows_simple, cfg.random_state + horizon)
            model = NumericSubsetHistGBRegressor(
                feature_names=numeric_features,
                learning_rate=0.05,
                max_depth=6,
                max_iter=250,
                min_samples_leaf=40,
                l2_regularization=0.5,
            )
            model.fit(train_frame, train_frame[target_col].to_numpy())
            preds = model.predict(val_frame)
            rows.append(
                metric_row(
                    experiment_family="simple_model",
                    experiment_name="numeric_histgb_simple",
                    crop=crop,
                    horizon=horizon,
                    validation_rows=len(val_frame),
                    validation_start=validation_start,
                    validation_end=validation_end,
                    y_true=val_frame[target_col].to_numpy(),
                    preds=preds,
                    extra={"train_rows": int(len(train_frame))},
                )
            )
    return rows


def run_histgb_variant(
    cfg: ExperimentConfig,
    frames: dict[str, pd.DataFrame],
    *,
    experiment_name: str,
    target_mode: str,
    normalization_mode: str,
    extra_numeric_features: list[str] | None = None,
    extra_categorical_features: list[str] | None = None,
    horizon_filter: tuple[int, ...] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    categorical_features, numeric_features = feature_columns()
    numeric_features = list(numeric_features)
    categorical_features = list(categorical_features)
    if extra_numeric_features:
        numeric_features.extend(extra_numeric_features)
    if extra_categorical_features:
        categorical_features.extend(extra_categorical_features)

    horizons = horizon_filter if horizon_filter is not None else cfg.horizons
    for crop, frame in sorted(frames.items()):
        for horizon in horizons:
            train_frame, val_frame, target_col, validation_start, validation_end = split_train_val(
                frame, horizon, cfg.validation_days
            )
            if train_frame.empty or val_frame.empty:
                continue
            raw_train = sample_training_rows(train_frame, cfg.max_train_rows_full, cfg.random_state + horizon)
            raw_val = val_frame.copy()
            train_norm, val_norm = apply_series_normalization(
                raw_train,
                raw_val,
                numeric_features,
                normalization_mode,
            )
            y_train, y_val, anchor_val = make_target_arrays(
                train_norm,
                val_norm,
                target_col,
                target_mode,
                train_anchor_source=raw_train,
                val_anchor_source=raw_val,
            )
            model = make_pipeline(categorical_features, numeric_features, horizon)
            model.fit(train_norm[categorical_features + numeric_features], y_train)
            pred_raw = model.predict(val_norm[categorical_features + numeric_features])
            preds = invert_predictions(pred_raw, anchor_val, target_mode)
            rows.append(
                metric_row(
                    experiment_family="tabular_variant",
                    experiment_name=experiment_name,
                    crop=crop,
                    horizon=horizon,
                    validation_rows=len(val_norm),
                    validation_start=validation_start,
                    validation_end=validation_end,
                    y_true=y_val,
                    preds=preds,
                    extra={
                        "train_rows": int(len(train_norm)),
                        "target_mode": target_mode,
                        "normalization_mode": normalization_mode,
                    },
                )
            )
    return rows


def add_cross_crop_features(frames: dict[str, pd.DataFrame], pairings: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], list[dict[str, object]]]:
    pairing_map = pairings.set_index("crop")["paired_crop"].to_dict()
    output: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []
    for crop, frame in sorted(frames.items()):
        other = pairing_map[crop]
        other_daily = (
            frames[other]
            .groupby("Date", sort=True)
            .agg(
                paired_crop_national_price_mean=("national_price_mean", "mean"),
                paired_crop_price_roll_mean_28=("price_roll_mean_28", "mean"),
                paired_crop_price_trend_7_28=("price_trend_7_28", "mean"),
            )
            .reset_index()
        )
        enriched = frame.merge(other_daily, on="Date", how="left")
        enriched["paired_crop_name"] = other
        output[crop] = enriched
        rows.append(
            {
                "crop": crop,
                "paired_crop": other,
                "daily_national_price_corr": float(
                    pairings.loc[pairings["crop"] == crop, "daily_national_price_corr"].iloc[0]
                ),
            }
        )
    return output, rows


def run_density_variant(cfg: ExperimentConfig, frames: dict[str, pd.DataFrame]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    categorical_features, numeric_features = feature_columns()
    categorical_features = list(categorical_features) + ["density_bucket"]
    numeric_features = list(numeric_features)
    horizon = cfg.graph_horizon
    for crop, frame in sorted(frames.items()):
        train_frame, val_frame, target_col, validation_start, validation_end = split_train_val(
            frame, horizon, cfg.validation_days
        )
        if train_frame.empty or val_frame.empty:
            continue
        raw_train = sample_training_rows(train_frame, cfg.max_train_rows_full, cfg.random_state + horizon)
        raw_val = val_frame.copy()
        train_norm, val_norm = apply_series_normalization(
            raw_train,
            raw_val,
            numeric_features,
            "series_mean_center",
        )
        y_train, y_val, anchor_val = make_target_arrays(
            train_norm,
            val_norm,
            target_col,
            "delta_current",
            train_anchor_source=raw_train,
            val_anchor_source=raw_val,
        )
        model = make_pipeline(categorical_features, numeric_features, horizon)
        model.fit(train_norm[categorical_features + numeric_features], y_train)
        pred_raw = model.predict(val_norm[categorical_features + numeric_features])
        preds = invert_predictions(pred_raw, anchor_val, "delta_current")
        rows.append(
            metric_row(
                experiment_family="specialized_variant",
                experiment_name="density_bucket_histgb",
                crop=crop,
                horizon=horizon,
                validation_rows=len(val_norm),
                validation_start=validation_start,
                validation_end=validation_end,
                y_true=y_val,
                preds=preds,
                extra={"train_rows": int(len(train_norm))},
            )
        )
    return rows


def run_volume_variant(cfg: ExperimentConfig, frames: dict[str, pd.DataFrame]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    base_categorical, numeric_features = feature_columns()
    categorical_features = list(base_categorical) + ["cluster_id"]
    numeric_features = list(numeric_features)
    horizon = cfg.graph_horizon
    for crop, frame in sorted(frames.items()):
        train_frame, val_frame, target_col, validation_start, validation_end = split_train_val(
            frame, horizon, cfg.validation_days
        )
        if train_frame.empty or val_frame.empty:
            continue
        raw_train = sample_training_rows(train_frame, cfg.max_train_rows_full, cfg.random_state + horizon)
        raw_val = val_frame.copy()
        clustered_train, clustered_val = add_volume_clusters(
            raw_train,
            raw_val,
            cluster_count=4,
            random_state=cfg.random_state + horizon,
        )
        train_norm, val_norm = apply_series_normalization(
            clustered_train,
            clustered_val,
            numeric_features,
            "series_mean_center",
        )
        y_train, y_val, anchor_val = make_target_arrays(
            train_norm,
            val_norm,
            target_col,
            "delta_current",
            train_anchor_source=clustered_train,
            val_anchor_source=clustered_val,
        )
        model = make_pipeline(categorical_features, numeric_features, horizon)
        model.fit(train_norm[categorical_features + numeric_features], y_train)
        pred_raw = model.predict(val_norm[categorical_features + numeric_features])
        preds = invert_predictions(pred_raw, anchor_val, "delta_current")
        rows.append(
            metric_row(
                experiment_family="specialized_variant",
                experiment_name="volume_cluster_histgb",
                crop=crop,
                horizon=horizon,
                validation_rows=len(val_norm),
                validation_start=validation_start,
                validation_end=validation_end,
                y_true=y_val,
                preds=preds,
                extra={"train_rows": int(len(train_norm))},
            )
        )
    return rows


def best_alpha_from_train(
    train_target: np.ndarray,
    own_feature: np.ndarray,
    neighbor_feature: np.ndarray,
) -> float:
    best_alpha = 1.0
    best_rmse = math.inf
    grid = np.linspace(0.0, 1.0, 21)
    for alpha in grid:
        pred = alpha * own_feature + (1.0 - alpha) * neighbor_feature
        rmse = math.sqrt(mean_squared_error(train_target, pred))
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = float(alpha)
    return best_alpha


def run_graph_neighbor_blend(cfg: ExperimentConfig, frames: dict[str, pd.DataFrame]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    horizon = cfg.graph_horizon
    for crop, frame in sorted(frames.items()):
        train_frame, val_frame, target_col, validation_start, validation_end = split_train_val(
            frame, horizon, cfg.validation_days
        )
        if train_frame.empty or val_frame.empty:
            continue
        y_train = train_frame[target_col].to_numpy()
        y_val = val_frame[target_col].to_numpy()
        own_train = train_frame["Modal_Price_CausalFilled"].to_numpy()
        own_val = val_frame["Modal_Price_CausalFilled"].to_numpy()
        for threshold in GRAPH_THRESHOLDS_KM:
            suffix = f"{int(threshold)}km"
            neighbor_col = f"neighbor_price_{suffix}"
            alpha = best_alpha_from_train(
                y_train,
                own_train,
                np.nan_to_num(train_frame[neighbor_col].to_numpy(), nan=0.0),
            )
            preds = alpha * own_val + (1.0 - alpha) * np.nan_to_num(val_frame[neighbor_col].to_numpy(), nan=0.0)
            preds = np.clip(preds, a_min=0.0, a_max=None)
            rows.append(
                metric_row(
                    experiment_family="graph_model",
                    experiment_name="graph_neighbor_blend",
                    crop=crop,
                    horizon=horizon,
                    validation_rows=len(val_frame),
                    validation_start=validation_start,
                    validation_end=validation_end,
                    y_true=y_val,
                    preds=preds,
                    extra={"threshold_km": threshold, "alpha": alpha},
                )
            )
    return rows


def run_graph_augmented_histgb(cfg: ExperimentConfig, frames: dict[str, pd.DataFrame]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    categorical_features, numeric_features = feature_columns()
    categorical_features = list(categorical_features)
    horizon = cfg.graph_horizon
    for crop, frame in sorted(frames.items()):
        for threshold in GRAPH_THRESHOLDS_KM:
            suffix = f"{int(threshold)}km"
            extra_numeric = [
                f"neighbor_price_{suffix}",
                f"neighbor_roll28_{suffix}",
                f"neighbor_degree_{suffix}",
            ]
            features_numeric = list(numeric_features) + extra_numeric
            train_frame, val_frame, target_col, validation_start, validation_end = split_train_val(
                frame, horizon, cfg.validation_days
            )
            if train_frame.empty or val_frame.empty:
                continue
            raw_train = sample_training_rows(train_frame, cfg.max_train_rows_full, cfg.random_state + horizon)
            raw_val = val_frame.copy()
            train_norm, val_norm = apply_series_normalization(
                raw_train,
                raw_val,
                features_numeric,
                "series_mean_center",
            )
            y_train, y_val, anchor_val = make_target_arrays(
                train_norm,
                val_norm,
                target_col,
                "delta_current",
                train_anchor_source=raw_train,
                val_anchor_source=raw_val,
            )
            model = make_pipeline(categorical_features, features_numeric, horizon)
            model.fit(train_norm[categorical_features + features_numeric], y_train)
            pred_raw = model.predict(val_norm[categorical_features + features_numeric])
            preds = invert_predictions(pred_raw, anchor_val, "delta_current")
            rows.append(
                metric_row(
                    experiment_family="graph_model",
                    experiment_name="graph_histgb_augmented",
                    crop=crop,
                    horizon=horizon,
                    validation_rows=len(val_norm),
                    validation_start=validation_start,
                    validation_end=validation_end,
                    y_true=y_val,
                    preds=preds,
                    extra={"threshold_km": threshold, "train_rows": int(len(train_norm))},
                )
            )
    return rows


def build_manifest(cfg: ExperimentConfig, pairings: pd.DataFrame) -> None:
    config_payload = asdict(cfg)
    for key, value in list(config_payload.items()):
        if isinstance(value, Path):
            config_payload[key] = str(value)
    manifest = {
        "config": config_payload,
        "graph_thresholds_km": list(GRAPH_THRESHOLDS_KM),
        "graph_primary_threshold_km": GRAPH_PRIMARY_THRESHOLD_KM,
        "cross_crop_pairings": pairings.to_dict(orient="records"),
    }
    manifest_path = cfg.output_root / "results" / "experiment_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    ensure_venv()
    cfg = experiment_config()
    set_local_runtime_dirs(cfg.output_root)
    (cfg.output_root / "logs").mkdir(parents=True, exist_ok=True)
    (cfg.output_root / "results").mkdir(parents=True, exist_ok=True)

    frames = load_crop_frames(cfg)
    build_inventory_outputs(cfg, frames)

    graph_frames: dict[str, pd.DataFrame] = {}
    graph_meta_rows: list[pd.DataFrame] = []
    for crop, frame in sorted(frames.items()):
        enriched, meta = attach_graph_features(frame, GRAPH_THRESHOLDS_KM)
        meta.insert(0, "crop", crop)
        graph_frames[crop] = enriched
        graph_meta_rows.append(meta)
    pd.concat(graph_meta_rows, ignore_index=True).to_csv(
        cfg.output_root / "results" / "inventory" / "series_graph_metadata.csv",
        index=False,
    )

    pairings = select_cross_crop_pairs(frames)
    cross_crop_frames, cross_pair_rows = add_cross_crop_features(graph_frames, pairings)
    pairings_out = cfg.output_root / "results" / "cross_crop_pairings.csv"
    pd.DataFrame(cross_pair_rows).to_csv(pairings_out, index=False)
    print(f"Wrote {pairings_out}")
    build_manifest(cfg, pairings)

    experiments = [
        ("baseline_metrics", lambda: run_baselines(cfg, frames)),
        ("simple_numeric_metrics", lambda: run_simple_numeric_model(cfg, frames)),
        (
            "anchored_histgb_metrics",
            lambda: run_histgb_variant(
                cfg,
                frames,
                experiment_name="histgb_delta_current_mean_center",
                target_mode="delta_current",
                normalization_mode="series_mean_center",
            ),
        ),
        (
            "cross_crop_histgb_metrics",
            lambda: run_histgb_variant(
                cfg,
                cross_crop_frames,
                experiment_name="histgb_cross_crop_delta_current_mean_center",
                target_mode="delta_current",
                normalization_mode="series_mean_center",
                extra_numeric_features=[
                    "paired_crop_national_price_mean",
                    "paired_crop_price_roll_mean_28",
                    "paired_crop_price_trend_7_28",
                ],
                extra_categorical_features=["paired_crop_name"],
            ),
        ),
        ("density_variant_metrics", lambda: run_density_variant(cfg, graph_frames)),
        ("volume_variant_metrics", lambda: run_volume_variant(cfg, graph_frames)),
        ("graph_neighbor_blend_metrics", lambda: run_graph_neighbor_blend(cfg, graph_frames)),
        ("graph_histgb_metrics", lambda: run_graph_augmented_histgb(cfg, graph_frames)),
    ]

    for name, fn in experiments:
        print(f"Running {name} ...")
        rows = fn()
        print(f"Saving {name} ({len(rows)} rows) ...")
        save_results(cfg, name, rows)

    print("Experiment run complete.")


if __name__ == "__main__":
    main()
