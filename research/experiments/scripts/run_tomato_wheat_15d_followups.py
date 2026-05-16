from __future__ import annotations

import json
import math
import os
import sys
import warnings
import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from mps_preflight import select_torch_device, write_device_status  # type: ignore
from prepare_experiments import (  # type: ignore
    ExperimentConfig,
    attach_graph_features,
    ensure_venv,
    experiment_config,
    load_crop_frames,
    metric_row,
    save_results,
    set_local_runtime_dirs,
    split_train_val,
)
from run_focused_model_expansion import (  # type: ignore
    TemporalConvRegressor,
    build_sequence_samples,
    train_tcn_model,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = "tomato_wheat_15d_followups"
FOLLOWUP_NAME = "followup_metrics"
FOLLOWUP_TCN_NAME = "followup_tcn_metrics"
GRAPH_REFERENCE_NAME = "graph_reference_metrics"
LOCAL_THRESHOLDS = (10.0, 25.0, 50.0, 150.0)
FOLLOWUP_CROPS = ("tomato", "wheat")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tomato/wheat 15-day follow-up experiments.")
    parser.add_argument("--skip-tabular", action="store_true")
    parser.add_argument("--skip-tcn", action="store_true")
    parser.add_argument("--skip-graph-reference", action="store_true")
    return parser.parse_args()


def followup_config() -> ExperimentConfig:
    base = experiment_config()
    return ExperimentConfig(
        data_dir=base.data_dir,
        output_root=REPO_ROOT / "newtests" / RUN_DIR,
        horizons=(15,),
        graph_horizon=15,
        validation_days=base.validation_days,
        min_series_observations=base.min_series_observations,
        dense_min_pct=base.dense_min_pct,
        max_train_rows_full=base.max_train_rows_full,
        max_train_rows_simple=base.max_train_rows_simple,
        random_state=base.random_state,
    )


def tree_model(random_state: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=160,
                    max_depth=None,
                    min_samples_leaf=1,
                    max_features="sqrt",
                    random_state=random_state,
                    n_jobs=1,
                ),
            ),
        ]
    )


def add_local_aggregate_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    keys = [
        ("State", "state"),
        ("District", "district"),
        (["State", "District"], "state_district"),
    ]
    for grouping, prefix in keys:
        if isinstance(grouping, list):
            group_cols = ["Date"] + grouping
        else:
            group_cols = ["Date", grouping]
        grp = enriched.groupby(group_cols, sort=False)
        count = grp["series_id"].transform("count")

        price_sum = grp["Modal_Price_CausalFilled"].transform("sum")
        roll14_sum = grp["price_roll_mean_14"].transform("sum")
        roll28_sum = grp["price_roll_mean_28"].transform("sum")
        trend_sum = grp["price_trend_7_28"].transform("sum")

        denom = np.where(count.to_numpy() > 1, count.to_numpy() - 1, np.nan)
        enriched[f"{prefix}_other_price_mean"] = np.where(
            np.isfinite(denom),
            (price_sum.to_numpy() - enriched["Modal_Price_CausalFilled"].to_numpy()) / denom,
            np.nan,
        )
        enriched[f"{prefix}_other_roll14_mean"] = np.where(
            np.isfinite(denom),
            (roll14_sum.to_numpy() - enriched["price_roll_mean_14"].to_numpy()) / denom,
            np.nan,
        )
        enriched[f"{prefix}_other_roll28_mean"] = np.where(
            np.isfinite(denom),
            (roll28_sum.to_numpy() - enriched["price_roll_mean_28"].to_numpy()) / denom,
            np.nan,
        )
        enriched[f"{prefix}_other_trend_mean"] = np.where(
            np.isfinite(denom),
            (trend_sum.to_numpy() - enriched["price_trend_7_28"].to_numpy()) / denom,
            np.nan,
        )
        enriched[f"{prefix}_price_std"] = grp["Modal_Price_CausalFilled"].transform("std").fillna(0.0)

    return enriched


def add_weighted_anchor_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    weighted_anchor = 0.7 * enriched["price_roll_mean_28"] + 0.3 * enriched["Modal_Price_CausalFilled"]
    enriched["weighted_anchor_70_30"] = np.clip(weighted_anchor, a_min=0.0, a_max=None)
    enriched["price_vs_weighted_anchor_70_30"] = enriched["Modal_Price_CausalFilled"] - enriched["weighted_anchor_70_30"]
    return enriched


def add_regime_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    quarter = ((enriched["month"].astype(int) - 1) // 3).clip(lower=0, upper=3)
    enriched["quarter_sin"] = np.sin(2.0 * np.pi * quarter / 4.0)
    enriched["quarter_cos"] = np.cos(2.0 * np.pi * quarter / 4.0)
    volatility = enriched["price_volatility_ratio_84"].fillna(0.0)
    enriched["volatility_regime"] = pd.qcut(
        volatility.rank(method="first"),
        q=4,
        labels=False,
        duplicates="drop",
    ).astype(float)
    enriched["trend_regime"] = pd.qcut(
        enriched["price_trend_28_84"].fillna(0.0).rank(method="first"),
        q=4,
        labels=False,
        duplicates="drop",
    ).astype(float)
    return enriched


def safe_cols(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def medium_horizon_feature_set(frame: pd.DataFrame) -> list[str]:
    candidates = [
        "latitude",
        "longitude",
        "arrival_log1p",
        "temp_mean",
        "temp_range",
        "rain_sum",
        "solar_sum",
        "rh_mean",
        "state_price_mean",
        "national_price_mean",
        "month",
        "week_of_year",
        "day_of_year_sin",
        "day_of_year_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "price_dow_ratio",
        "arrival_dow_ratio",
        "price_lag_7",
        "price_lag_14",
        "price_lag_28",
        "price_lag_56",
        "price_lag_84",
        "arrival_lag_7",
        "arrival_lag_28",
        "arrival_lag_84",
        "price_roll_mean_14",
        "price_roll_mean_21",
        "price_roll_mean_28",
        "price_roll_mean_56",
        "price_roll_mean_84",
        "price_roll_std_28",
        "price_roll_std_84",
        "price_vs_roll28",
        "price_vs_roll84",
        "price_trend_14_56",
        "price_trend_28_84",
        "price_range_28",
        "price_range_84",
        "price_volatility_ratio_28",
        "price_volatility_ratio_84",
    ]
    return safe_cols(frame, candidates)


def local_feature_set(frame: pd.DataFrame) -> list[str]:
    candidates = medium_horizon_feature_set(frame) + [
        "neighbor_price_10km",
        "neighbor_roll28_10km",
        "neighbor_degree_10km",
        "neighbor_price_25km",
        "neighbor_roll28_25km",
        "neighbor_degree_25km",
        "state_other_price_mean",
        "state_other_roll14_mean",
        "state_other_roll28_mean",
        "state_other_trend_mean",
        "state_price_std",
        "district_other_price_mean",
        "district_other_roll28_mean",
        "district_price_std",
        "state_district_other_price_mean",
        "state_district_other_roll28_mean",
        "state_district_price_std",
        "weighted_anchor_70_30",
        "price_vs_weighted_anchor_70_30",
    ]
    return safe_cols(frame, candidates)


def regime_feature_set(frame: pd.DataFrame) -> list[str]:
    candidates = medium_horizon_feature_set(frame) + [
        "price_roll_mean_168",
        "price_roll_std_168",
        "price_vs_roll168",
        "price_range_168",
        "price_volatility_ratio_168",
        "quarter_sin",
        "quarter_cos",
        "volatility_regime",
        "trend_regime",
        "is_month_start",
        "is_month_end",
        "year",
    ]
    return safe_cols(frame, candidates)


def compute_anchor(frame: pd.DataFrame, mode: str) -> np.ndarray:
    if mode == "delta_current":
        base = np.clip(frame["Modal_Price_CausalFilled"].to_numpy(dtype=float), a_min=0.0, a_max=None)
    elif mode == "delta_roll14":
        base = np.clip(frame["price_roll_mean_14"].to_numpy(dtype=float), a_min=0.0, a_max=None)
    elif mode == "delta_roll28":
        base = np.clip(frame["price_roll_mean_28"].to_numpy(dtype=float), a_min=0.0, a_max=None)
    elif mode == "delta_weighted_70_30":
        base = np.clip(frame["weighted_anchor_70_30"].to_numpy(dtype=float), a_min=0.0, a_max=None)
    else:
        raise ValueError(f"Unsupported anchor mode: {mode}")
    return np.log1p(base)


def fit_extra_trees_variant(
    frame: pd.DataFrame,
    crop: str,
    experiment_name: str,
    feature_list: list[str],
    anchor_mode: str,
    random_state: int,
) -> dict[str, object]:
    train_frame, val_frame, target_col, validation_start, validation_end = split_train_val(frame, 15, 90)
    if train_frame.empty or val_frame.empty:
        raise RuntimeError(f"{experiment_name} has empty split for {crop}")

    features = safe_cols(frame, feature_list)
    x_train = train_frame[features]
    x_val = val_frame[features]
    train_anchor = compute_anchor(train_frame, anchor_mode)
    val_anchor = compute_anchor(val_frame, anchor_mode)
    train_target_log = np.log1p(train_frame[target_col].to_numpy(dtype=float))
    val_target = val_frame[target_col].to_numpy(dtype=float)
    train_mask = np.isfinite(train_anchor) & np.isfinite(train_target_log)
    val_mask = np.isfinite(val_anchor) & np.isfinite(val_target)
    x_train = x_train.loc[train_mask]
    x_val = x_val.loc[val_mask]
    train_anchor = train_anchor[train_mask]
    val_anchor = val_anchor[val_mask]
    train_target_log = train_target_log[train_mask]
    y_true = val_target[val_mask]
    y_train = train_target_log - train_anchor
    if len(x_train) > 60000:
        sampled = x_train.sample(n=60000, random_state=random_state).index
        x_train = x_train.loc[sampled]
        y_train = pd.Series(y_train, index=train_frame.index[train_mask]).loc[sampled].to_numpy(dtype=float)

    model = tree_model(random_state)
    model.fit(x_train, y_train)
    pred_delta = model.predict(x_val)
    preds = np.expm1(pred_delta + val_anchor)
    preds = np.clip(preds, a_min=0.0, a_max=None)

    artifact_dir = followup_config().output_root / "artifacts" / "tabular" / crop
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"{experiment_name}.joblib"
    joblib.dump({"model": model, "features": features, "anchor_mode": anchor_mode}, artifact_path)

    return metric_row(
        experiment_family="tomato_wheat_followup",
        experiment_name=experiment_name,
        crop=crop,
        horizon=15,
        validation_rows=len(val_frame),
        validation_start=validation_start,
        validation_end=validation_end,
        y_true=y_true,
        preds=preds,
        extra={
            "feature_count": len(features),
            "anchor_mode": anchor_mode,
            "artifact_path": str(artifact_path.relative_to(followup_config().output_root)),
        },
    )


def run_tabular_followups(cfg: ExperimentConfig, frames: dict[str, pd.DataFrame]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for crop in FOLLOWUP_CROPS:
        base_frame = frames[crop]
        enriched, _meta = attach_graph_features(base_frame, LOCAL_THRESHOLDS)
        enriched = add_local_aggregate_features(enriched)
        enriched = add_weighted_anchor_features(enriched)
        enriched = add_regime_features(enriched)

        if crop == "tomato":
            rows.append(
                fit_extra_trees_variant(
                    enriched,
                    crop,
                    "tomato_local_delta_roll14",
                    local_feature_set(enriched),
                    "delta_roll14",
                    cfg.random_state + 11,
                )
            )
            rows.append(
                fit_extra_trees_variant(
                    enriched,
                    crop,
                    "tomato_local_delta_roll28",
                    local_feature_set(enriched),
                    "delta_roll28",
                    cfg.random_state + 12,
                )
            )
            rows.append(
                fit_extra_trees_variant(
                    enriched,
                    crop,
                    "tomato_local_weighted_anchor7030",
                    local_feature_set(enriched),
                    "delta_weighted_70_30",
                    cfg.random_state + 13,
                )
            )
            rows.append(
                fit_extra_trees_variant(
                    enriched,
                    crop,
                    "tomato_horizon_focus_local",
                    medium_horizon_feature_set(enriched)
                    + safe_cols(
                        enriched,
                        [
                            "neighbor_price_10km",
                            "neighbor_roll28_10km",
                            "neighbor_price_25km",
                            "neighbor_roll28_25km",
                            "state_other_roll28_mean",
                            "district_other_roll28_mean",
                        ],
                    ),
                    "delta_roll28",
                    cfg.random_state + 14,
                )
            )
        else:
            rows.append(
                fit_extra_trees_variant(
                    enriched,
                    crop,
                    "wheat_regime_delta_current",
                    regime_feature_set(enriched),
                    "delta_current",
                    cfg.random_state + 21,
                )
            )
            rows.append(
                fit_extra_trees_variant(
                    enriched,
                    crop,
                    "wheat_regime_weighted_anchor7030",
                    regime_feature_set(enriched),
                    "delta_weighted_70_30",
                    cfg.random_state + 22,
                )
            )
            rows.append(
                fit_extra_trees_variant(
                    enriched,
                    crop,
                    "wheat_horizon_focus_regime",
                    medium_horizon_feature_set(enriched) + safe_cols(
                        enriched,
                        [
                            "quarter_sin",
                            "quarter_cos",
                            "volatility_regime",
                            "trend_regime",
                            "price_roll_mean_84",
                            "price_roll_std_84",
                            "price_volatility_ratio_84",
                        ],
                    ),
                    "delta_weighted_70_30",
                    cfg.random_state + 23,
                )
            )
    return rows


def run_tcn_followups(cfg: ExperimentConfig, frames: dict[str, pd.DataFrame]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    device = select_torch_device("auto")
    tcn_grids = [
        {
            "window_size": 42,
            "hidden_channels": 64,
            "kernel_size": 3,
            "dropout": 0.15,
            "learning_rate": 5e-4,
            "weight_decay": 1e-5,
            "epochs": 16,
            "batch_size": 128,
        },
        {
            "window_size": 56,
            "hidden_channels": 96,
            "kernel_size": 3,
            "dropout": 0.20,
            "learning_rate": 3e-4,
            "weight_decay": 1e-5,
            "epochs": 20,
            "batch_size": 96,
        },
    ]

    for crop in FOLLOWUP_CROPS:
        frame = frames[crop]
        best_row: dict[str, object] | None = None
        for candidate_index, params in enumerate(tcn_grids, start=1):
            (
                x_train,
                y_train_delta,
                x_val,
                y_val_delta,
                _y_train_actual,
                y_val_actual,
                _train_anchors,
                val_anchors,
                validation_start,
                validation_end,
                _meta,
            ) = build_sequence_samples(
                frame,
                15,
                cfg.validation_days,
                int(params["window_size"]),
                20000,
                cfg.random_state + candidate_index + 100,
            )
            model, preds = train_tcn_model(
                x_train,
                y_train_delta,
                x_val,
                y_val_actual,
                val_anchors,
                params,
                device,
                cfg.random_state + candidate_index + 100,
            )
            artifact_dir = cfg.output_root / "artifacts" / "tcn" / crop
            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = artifact_dir / f"{crop}_tcn_followup_candidate{candidate_index}.pt"
            torch.save({"state_dict": model.state_dict(), "params": params}, artifact_path)
            row = metric_row(
                experiment_family="tomato_wheat_followup_tcn",
                experiment_name=f"{crop}_tcn_longwindow_candidate{candidate_index}",
                crop=crop,
                horizon=15,
                validation_rows=len(y_val_actual),
                validation_start=validation_start,
                validation_end=validation_end,
                y_true=y_val_actual,
                preds=preds,
                extra={
                    "candidate_index": candidate_index,
                    "device": device,
                    "train_windows": int(len(x_train)),
                    "validation_windows": int(len(x_val)),
                    "params_json": json.dumps(params, sort_keys=True),
                    "artifact_path": str(artifact_path.relative_to(cfg.output_root)),
                },
            )
            rows.append(row)
            if best_row is None or float(row["r2"]) > float(best_row["r2"]):
                best_row = row
        assert best_row is not None
    return rows


def run_graph_reference_rows(cfg: ExperimentConfig) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    tomato_radius_files = [
        (
            "tomato_radius_10km_delta_roll28_existing",
            REPO_ROOT / "newtests" / "targeted_rebuild" / "results" / "gat_gru_radius_refine" / "tomato__radius_10km" / "graph_training_summary.json",
        ),
        (
            "tomato_radius_25km_delta_roll28_existing",
            REPO_ROOT / "newtests" / "targeted_rebuild" / "results" / "gat_gru_radius_refine" / "tomato__radius_25km" / "graph_training_summary.json",
        ),
    ]
    for experiment_name, path in tomato_radius_files:
        data = json.loads(path.read_text(encoding="utf-8"))["tomato"]
        rows.append(
            {
                "experiment_family": "existing_graph_reference",
                "experiment_name": experiment_name,
                "crop": "tomato",
                "horizon_days": 15,
                "validation_rows": data["validation_windows"],
                "validation_start": data["validation_start"],
                "validation_end": data["validation_end"],
                "mae": data["mae"],
                "rmse": data["rmse"],
                "r2": data["r2"],
                "mape_pct": np.nan,
                "wape_pct": data["wape_pct"],
                "source_file": str(path.relative_to(REPO_ROOT)),
                "note": "Reused existing tighter-radius tomato graph result.",
            }
        )

    wheat_ablation = pd.read_csv(
        REPO_ROOT / "newtests" / "targeted_rebuild" / "results" / "gat_gru_ablations" / "ablation_summary.csv"
    )
    wheat_best = wheat_ablation[wheat_ablation["crop"] == "wheat"].sort_values("r2", ascending=False).iloc[0]
    rows.append(
        {
            "experiment_family": "existing_graph_reference",
            "experiment_name": "wheat_best_existing_graph_reference",
            "crop": "wheat",
            "horizon_days": 15,
            "validation_rows": wheat_best.get("validation_windows", np.nan),
            "validation_start": wheat_best.get("validation_start"),
            "validation_end": wheat_best.get("validation_end"),
            "mae": wheat_best["mae"],
            "rmse": wheat_best["rmse"],
            "r2": wheat_best["r2"],
            "mape_pct": np.nan,
            "wape_pct": wheat_best["wape_pct"],
            "source_file": "newtests/targeted_rebuild/results/gat_gru_ablations/ablation_summary.csv",
            "note": "Existing wheat graph ablation winner before new refinement run.",
        }
    )
    return rows


def save_manifest(cfg: ExperimentConfig) -> None:
    payload = asdict(cfg)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    out_path = cfg.output_root / "results" / "followup_manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    ensure_venv()
    args = parse_args()
    cfg = followup_config()
    set_local_runtime_dirs(cfg.output_root)
    warnings.filterwarnings("ignore")
    write_device_status(cfg.output_root / "results" / "environment_status" / "device_status.json")

    frames = load_crop_frames(cfg)
    if not args.skip_tabular:
        followup_rows = run_tabular_followups(cfg, frames)
        save_results(cfg, FOLLOWUP_NAME, followup_rows)
    if not args.skip_tcn:
        tcn_rows = run_tcn_followups(cfg, frames)
        save_results(cfg, FOLLOWUP_TCN_NAME, tcn_rows)
    if not args.skip_graph_reference:
        graph_rows = run_graph_reference_rows(cfg)
        save_results(cfg, GRAPH_REFERENCE_NAME, graph_rows)
    save_manifest(cfg)
    print(f"Completed follow-up experiments in {cfg.output_root}")


if __name__ == "__main__":
    main()
