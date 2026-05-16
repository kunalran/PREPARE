from __future__ import annotations

import argparse
import json
import math
import os
import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterGrid
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from mps_preflight import select_torch_device, write_device_status  # type: ignore
from prepare_experiments import (  # type: ignore
    ExperimentConfig,
    IMPUTED_CROP_FILES,
    ensure_venv,
    experiment_config,
    load_crop_frames,
    metric_row,
    run_baselines,
    sample_training_rows,
    save_results,
    set_local_runtime_dirs,
    simple_numeric_features,
    split_train_val,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = "focused_1d_15d_expansion"
BASELINE_NAME = "baseline_metrics"
EXPANDED_NAME = "expanded_model_metrics"
TUNING_NAME = "tuning_metrics"
MODEL_STATUS_NAME = "model_status"
SUPPORTED_MODELS = ("xgboost", "lightgbm", "extratrees", "mlp", "tcn")
DEFAULT_MODELS = ("xgboost", "lightgbm", "extratrees", "tcn")
TABULAR_MODELS = {"xgboost", "lightgbm", "extratrees", "mlp"}
SEQUENCE_FEATURES = [
    "Modal_Price_CausalFilled",
    "Arrival_Quantity_CausalFilled",
    "price_lag_1",
    "price_lag_7",
    "price_roll_mean_7",
    "price_roll_mean_28",
    "price_trend_7_28",
    "price_vs_roll28",
    "price_dow_ratio",
    "arrival_dow_ratio",
    "temp_mean",
    "temp_range",
    "rain_sum",
    "solar_sum",
    "rh_mean",
    "state_price_mean",
    "national_price_mean",
    "day_of_year_sin",
    "day_of_year_cos",
    "day_of_week_sin",
    "day_of_week_cos",
]

TABULAR_GRIDS: dict[str, list[dict[str, Any]]] = {
    "xgboost": [
        {
            "n_estimators": 250,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 4,
        },
        {
            "n_estimators": 350,
            "max_depth": 6,
            "learning_rate": 0.04,
            "subsample": 0.90,
            "colsample_bytree": 0.80,
            "min_child_weight": 6,
        },
    ],
    "lightgbm": [
        {
            "n_estimators": 250,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 40,
            "subsample": 0.90,
            "colsample_bytree": 0.90,
        },
        {
            "n_estimators": 350,
            "learning_rate": 0.04,
            "num_leaves": 47,
            "max_depth": 8,
            "min_child_samples": 30,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
        },
    ],
    "extratrees": list(
        ParameterGrid(
            {
                "n_estimators": [300],
                "max_depth": [None, 24],
                "min_samples_leaf": [1, 4],
                "max_features": ["sqrt"],
            }
        )
    ),
    "mlp": [
        {
            "hidden_layer_sizes": (96, 48),
            "alpha": 5e-4,
            "learning_rate_init": 3e-4,
            "max_iter": 120,
            "batch_size": 256,
        },
        {
            "hidden_layer_sizes": (128, 64),
            "alpha": 1e-3,
            "learning_rate_init": 2e-4,
            "max_iter": 150,
            "batch_size": 256,
        },
    ],
}
TCN_GRIDS: list[dict[str, Any]] = [
    {
        "window_size": 21,
        "hidden_channels": 32,
        "kernel_size": 3,
        "dropout": 0.15,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "epochs": 8,
        "batch_size": 128,
    },
    {
        "window_size": 28,
        "hidden_channels": 48,
        "kernel_size": 3,
        "dropout": 0.20,
        "learning_rate": 7e-4,
        "weight_decay": 1e-5,
        "epochs": 10,
        "batch_size": 128,
    },
]


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def _trim(self, tensor: torch.Tensor, length: int) -> torch.Tensor:
        return tensor[..., :length]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        length = inputs.size(-1)
        output = self._trim(self.conv1(inputs), length)
        output = self.dropout(F.relu(output))
        output = self._trim(self.conv2(output), length)
        output = self.dropout(F.relu(output))
        residual = inputs if self.residual is None else self.residual(inputs)
        return F.relu(output + residual)


class TemporalConvRegressor(nn.Module):
    def __init__(self, feature_count: int, hidden_channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            TemporalBlock(feature_count, hidden_channels, kernel_size, dilation=1, dropout=dropout),
            TemporalBlock(hidden_channels, hidden_channels, kernel_size, dilation=2, dropout=dropout),
            TemporalBlock(hidden_channels, hidden_channels, kernel_size, dilation=4, dropout=dropout),
        )
        self.head = nn.Linear(hidden_channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.network(inputs)
        last_step = encoded[:, :, -1]
        return self.head(last_step).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run focused 1-day and 15-day model expansion experiments.")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=os.environ.get("NEWTESTS_RUN_DIR", DEFAULT_RUN_DIR),
        help="Folder name under newtests/ where outputs will be written.",
    )
    parser.add_argument(
        "--crops",
        type=str,
        default="onion,potato,tomato,wheat",
        help="Comma-separated crops to evaluate.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model families to run.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps"],
        default="auto",
        help="Torch device policy for TCN training.",
    )
    parser.add_argument(
        "--allow-missing-deps",
        action="store_true",
        help="Skip xgboost/lightgbm when the packages are not installed.",
    )
    parser.add_argument(
        "--max-train-windows",
        type=int,
        default=15000,
        help="Optional cap on TCN training windows per crop/horizon.",
    )
    return parser.parse_args()


def focused_config(run_dir: str) -> ExperimentConfig:
    base = experiment_config()
    return ExperimentConfig(
        data_dir=base.data_dir,
        output_root=REPO_ROOT / "newtests" / run_dir,
        horizons=(1, 15),
        graph_horizon=15,
        validation_days=base.validation_days,
        min_series_observations=base.min_series_observations,
        dense_min_pct=base.dense_min_pct,
        max_train_rows_full=base.max_train_rows_full,
        max_train_rows_simple=base.max_train_rows_simple,
        random_state=base.random_state,
    )


def selected_crops(raw: str) -> list[str]:
    crops = [crop.strip().lower() for crop in raw.split(",") if crop.strip()]
    invalid = [crop for crop in crops if crop not in IMPUTED_CROP_FILES]
    if invalid:
        raise ValueError(f"Unsupported crops requested: {invalid}. Valid crops: {sorted(IMPUTED_CROP_FILES)}")
    return sorted(dict.fromkeys(crops))


def selected_models(raw: str) -> list[str]:
    models = [model.strip().lower() for model in raw.split(",") if model.strip()]
    invalid = [model for model in models if model not in SUPPORTED_MODELS]
    if invalid:
        raise ValueError(f"Unsupported models requested: {invalid}. Valid models: {sorted(SUPPORTED_MODELS)}")
    return models


def status_path(cfg: ExperimentConfig) -> Path:
    return cfg.output_root / "results" / "environment_status" / "device_status.json"


def model_status_path(cfg: ExperimentConfig) -> Path:
    return cfg.output_root / "results" / MODEL_STATUS_NAME / f"{MODEL_STATUS_NAME}.json"


def save_model_status(cfg: ExperimentConfig, rows: list[dict[str, object]]) -> None:
    out_dir = cfg.output_root / "results" / MODEL_STATUS_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_status_path(cfg)
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def optional_import(model_name: str) -> tuple[type[Any] | None, str]:
    if model_name == "xgboost":
        try:
            from xgboost import XGBRegressor  # type: ignore
        except ImportError as exc:
            return None, str(exc)
        return XGBRegressor, ""
    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor  # type: ignore
        except ImportError as exc:
            return None, str(exc)
        return LGBMRegressor, ""
    return None, ""


def resolve_model_status(requested_models: list[str], allow_missing_deps: bool) -> tuple[list[str], list[dict[str, object]]]:
    runnable: list[str] = []
    status_rows: list[dict[str, object]] = []
    for model_name in requested_models:
        if model_name not in {"xgboost", "lightgbm"}:
            runnable.append(model_name)
            status_rows.append({"model_name": model_name, "status": "ready"})
            continue
        imported, message = optional_import(model_name)
        if imported is not None:
            runnable.append(model_name)
            status_rows.append({"model_name": model_name, "status": "ready"})
            continue
        status_rows.append({"model_name": model_name, "status": "missing_dependency", "detail": message})
        if not allow_missing_deps:
            raise RuntimeError(
                f"{model_name} is not installed in newtests/venv. "
                f"Install with `newtests/venv/bin/pip install -r newtests/requirements_model_expansion.txt`."
            )
    return runnable, status_rows


def make_tabular_estimator(model_name: str, params: dict[str, Any], random_state: int) -> Any:
    if model_name == "xgboost":
        estimator_cls, _ = optional_import(model_name)
        assert estimator_cls is not None
        estimator = estimator_cls(
            objective="reg:squarederror",
            random_state=random_state,
            tree_method="hist",
            n_jobs=1,
            **params,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])

    if model_name == "lightgbm":
        estimator_cls, _ = optional_import(model_name)
        assert estimator_cls is not None
        estimator = estimator_cls(
            objective="regression",
            random_state=random_state,
            n_jobs=1,
            verbose=-1,
            **params,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])

    if model_name == "extratrees":
        estimator = ExtraTreesRegressor(
            random_state=random_state,
            n_jobs=1,
            **params,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])

    if model_name == "mlp":
        estimator = MLPRegressor(
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=12,
            **params,
        )
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clip", FunctionTransformer(lambda value: np.clip(value, -8.0, 8.0))),
                ("model", estimator),
            ]
        )
        return TransformedTargetRegressor(
            regressor=pipeline,
            func=np.log1p,
            inverse_func=np.expm1,
        )

    raise ValueError(f"Unsupported tabular model: {model_name}")


def artifact_dir(cfg: ExperimentConfig, model_name: str, crop: str) -> Path:
    path = cfg.output_root / "artifacts" / model_name / crop
    path.mkdir(parents=True, exist_ok=True)
    return path


def dump_tabular_artifact(
    cfg: ExperimentConfig,
    crop: str,
    horizon: int,
    model_name: str,
    estimator: Any,
    params: dict[str, Any],
) -> str:
    out_path = artifact_dir(cfg, model_name, crop) / f"{crop}_{model_name}_{horizon}d.joblib"
    payload = {"model": estimator, "params": params, "features": simple_numeric_features()}
    joblib.dump(payload, out_path)
    return str(out_path.relative_to(cfg.output_root))


def collect_tabular_rows(
    cfg: ExperimentConfig,
    frames: dict[str, pd.DataFrame],
    models: list[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    features = simple_numeric_features()
    tuning_rows: list[dict[str, object]] = []
    best_rows: list[dict[str, object]] = []

    for crop, frame in sorted(frames.items()):
        for horizon in cfg.horizons:
            train_frame, val_frame, target_col, validation_start, validation_end = split_train_val(
                frame,
                horizon,
                cfg.validation_days,
            )
            if train_frame.empty or val_frame.empty:
                continue
            train_frame = sample_training_rows(train_frame, cfg.max_train_rows_simple, cfg.random_state + horizon)
            x_train = train_frame[features]
            y_train = train_frame[target_col].to_numpy(dtype=float)
            x_val = val_frame[features]
            y_val = val_frame[target_col].to_numpy(dtype=float)

            for model_name in models:
                if model_name not in TABULAR_MODELS:
                    continue
                best_candidate: dict[str, object] | None = None
                best_model: Any = None
                for candidate_index, params in enumerate(TABULAR_GRIDS[model_name], start=1):
                    estimator = make_tabular_estimator(model_name, params, cfg.random_state + horizon)
                    estimator.fit(x_train, y_train)
                    preds = np.clip(estimator.predict(x_val), a_min=0.0, a_max=None)
                    row = metric_row(
                        experiment_family="focused_tabular",
                        experiment_name=f"{model_name}_tuned",
                        crop=crop,
                        horizon=horizon,
                        validation_rows=len(val_frame),
                        validation_start=validation_start,
                        validation_end=validation_end,
                        y_true=y_val,
                        preds=preds,
                        extra={
                            "candidate_index": candidate_index,
                            "train_rows": int(len(train_frame)),
                            "feature_count": len(features),
                            "params_json": json.dumps(params, sort_keys=True),
                        },
                    )
                    tuning_rows.append(row)
                    if best_candidate is None or float(row["r2"]) > float(best_candidate["r2"]):
                        best_candidate = deepcopy(row)
                        best_model = estimator

                if best_candidate is None or best_model is None:
                    continue
                best_params = json.loads(str(best_candidate["params_json"]))
                best_candidate["artifact_path"] = dump_tabular_artifact(
                    cfg,
                    crop,
                    horizon,
                    model_name,
                    best_model,
                    best_params,
                )
                best_rows.append(best_candidate)

    return tuning_rows, best_rows


def validation_start(frame: pd.DataFrame, validation_days: int) -> pd.Timestamp:
    return frame["Date"].max() - pd.Timedelta(days=validation_days - 1)


def sample_arrays(
    samples: list[tuple[np.ndarray, float, float, float]],
    max_rows: int | None,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_rows is not None and len(samples) > max_rows:
        rng = np.random.default_rng(random_state)
        selected = rng.choice(len(samples), size=max_rows, replace=False)
        samples = [samples[idx] for idx in np.sort(selected)]

    x_array = np.stack([sample[0] for sample in samples]).astype(np.float32)
    y_delta = np.asarray([sample[1] for sample in samples], dtype=np.float32)
    y_actual = np.asarray([sample[2] for sample in samples], dtype=np.float32)
    anchor = np.asarray([sample[3] for sample in samples], dtype=np.float32)
    return x_array, np.column_stack([y_delta, anchor]), y_actual


def build_sequence_samples(
    frame: pd.DataFrame,
    horizon: int,
    validation_days: int,
    window_size: int,
    max_train_windows: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Timestamp, pd.Timestamp]:
    target_col = f"target_{horizon}d"
    subset = frame[frame[target_col].notna()].copy()
    val_start = validation_start(frame, validation_days)
    validation_end = frame["Date"].max()

    train_samples: list[tuple[np.ndarray, float, float, float]] = []
    val_samples: list[tuple[np.ndarray, float, float, float]] = []

    for _, group in subset.groupby("series_id", sort=False):
        group = group.sort_values("Date").reset_index(drop=True)
        features = np.nan_to_num(group[SEQUENCE_FEATURES].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        targets = group[target_col].to_numpy(dtype=float)
        current_prices = np.clip(group["Modal_Price_CausalFilled"].to_numpy(dtype=float), a_min=0.0, a_max=None)
        dates = group["Date"].to_numpy()
        for end_idx in range(window_size - 1, len(group)):
            target_value = targets[end_idx]
            if not np.isfinite(target_value):
                continue
            anchor = math.log1p(float(current_prices[end_idx]))
            target_delta = math.log1p(float(target_value)) - anchor
            window = features[end_idx - window_size + 1 : end_idx + 1]
            sample = (window.T, target_delta, float(target_value), anchor)
            if pd.Timestamp(dates[end_idx]) < val_start:
                train_samples.append(sample)
            else:
                val_samples.append(sample)

    if not train_samples or not val_samples:
        raise ValueError("No train/validation windows available for the requested TCN setup.")

    x_train, y_train_bundle, y_train_actual = sample_arrays(train_samples, max_train_windows, random_state)
    x_val, y_val_bundle, y_val_actual = sample_arrays(val_samples, None, random_state)

    feature_mean = x_train.mean(axis=(0, 2), keepdims=True)
    feature_std = x_train.std(axis=(0, 2), keepdims=True)
    feature_std = np.where(feature_std == 0.0, 1.0, feature_std)
    x_train = (x_train - feature_mean) / feature_std
    x_val = (x_val - feature_mean) / feature_std

    train_anchors = y_train_bundle[:, 1]
    val_anchors = y_val_bundle[:, 1]
    y_train_delta = y_train_bundle[:, 0]
    y_val_delta = y_val_bundle[:, 0]

    meta = {
        "feature_mean": feature_mean.squeeze().tolist(),
        "feature_std": feature_std.squeeze().tolist(),
        "window_size": window_size,
        "feature_names": list(SEQUENCE_FEATURES),
    }

    return x_train, y_train_delta, x_val, y_val_delta, y_train_actual, y_val_actual, train_anchors, val_anchors, val_start, validation_end, meta


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_tcn_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val_actual: np.ndarray,
    val_anchors: np.ndarray,
    params: dict[str, Any],
    device: str,
    random_state: int,
) -> tuple[TemporalConvRegressor, np.ndarray]:
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    model = TemporalConvRegressor(
        feature_count=x_train.shape[1],
        hidden_channels=int(params["hidden_channels"]),
        kernel_size=int(params["kernel_size"]),
        dropout=float(params["dropout"]),
    ).to(torch.device(device))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )
    criterion = nn.MSELoss()
    loader = make_loader(x_train, y_train, batch_size=int(params["batch_size"]), shuffle=True)

    for _ in range(int(params["epochs"])):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pred_delta = model(torch.from_numpy(x_val).to(device)).cpu().numpy()
    preds = np.expm1(val_pred_delta + val_anchors)
    preds = np.clip(preds, a_min=0.0, a_max=None)
    if len(preds) != len(y_val_actual):
        raise RuntimeError("TCN prediction length mismatch.")
    return model, preds


def dump_tcn_artifact(
    cfg: ExperimentConfig,
    crop: str,
    horizon: int,
    model: TemporalConvRegressor,
    params: dict[str, Any],
    meta: dict[str, Any],
) -> str:
    out_dir = artifact_dir(cfg, "tcn", crop)
    model_path = out_dir / f"{crop}_tcn_{horizon}d.pt"
    state = {"state_dict": model.state_dict(), "params": params, "meta": meta}
    torch.save(state, model_path)
    return str(model_path.relative_to(cfg.output_root))


def collect_tcn_rows(
    cfg: ExperimentConfig,
    frames: dict[str, pd.DataFrame],
    device: str,
    max_train_windows: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    tuning_rows: list[dict[str, object]] = []
    best_rows: list[dict[str, object]] = []

    for crop, frame in sorted(frames.items()):
        for horizon in cfg.horizons:
            best_candidate: dict[str, object] | None = None
            best_model: TemporalConvRegressor | None = None
            best_meta: dict[str, Any] | None = None
            for candidate_index, params in enumerate(TCN_GRIDS, start=1):
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
                    meta,
                ) = build_sequence_samples(
                    frame,
                    horizon,
                    cfg.validation_days,
                    int(params["window_size"]),
                    max_train_windows,
                    cfg.random_state + candidate_index + horizon,
                )
                model, preds = train_tcn_model(
                    x_train,
                    y_train_delta,
                    x_val,
                    y_val_actual,
                    val_anchors,
                    params,
                    device,
                    cfg.random_state + candidate_index + horizon,
                )
                row = metric_row(
                    experiment_family="focused_temporal",
                    experiment_name="tcn_tuned",
                    crop=crop,
                    horizon=horizon,
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
                        "target_encoding": "log_delta_current",
                    },
                )
                tuning_rows.append(row)
                if best_candidate is None or float(row["r2"]) > float(best_candidate["r2"]):
                    best_candidate = deepcopy(row)
                    best_model = model
                    best_meta = meta

            if best_candidate is None or best_model is None or best_meta is None:
                continue
            best_params = json.loads(str(best_candidate["params_json"]))
            best_candidate["artifact_path"] = dump_tcn_artifact(cfg, crop, horizon, best_model, best_params, best_meta)
            best_rows.append(best_candidate)

    return tuning_rows, best_rows


def save_run_manifest(cfg: ExperimentConfig, args: argparse.Namespace, device: str, crops: list[str], models: list[str]) -> None:
    manifest_dir = cfg.output_root / "results"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(cfg),
        "requested_args": {
            "run_dir": args.run_dir,
            "crops": crops,
            "models": models,
            "device": args.device,
            "allow_missing_deps": args.allow_missing_deps,
            "max_train_windows": args.max_train_windows,
        },
        "resolved_device": device,
    }
    for key, value in list(payload["config"].items()):
        if isinstance(value, Path):
            payload["config"][key] = str(value)
    (manifest_dir / "focused_run_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    ensure_venv()
    args = parse_args()
    crops = selected_crops(args.crops)
    requested_models = selected_models(args.models)
    cfg = focused_config(args.run_dir)
    set_local_runtime_dirs(cfg.output_root)
    (cfg.output_root / "results").mkdir(parents=True, exist_ok=True)
    device_status_output = write_device_status(status_path(cfg))
    print(f"Wrote device status: {device_status_output}")

    runnable_models, model_status = resolve_model_status(requested_models, args.allow_missing_deps)
    save_model_status(cfg, model_status)

    tcn_device = "cpu"
    if "tcn" in runnable_models:
        tcn_device = select_torch_device(args.device)
        print(f"TCN device: {tcn_device}")

    base_frames = load_crop_frames(cfg)
    frames = {crop: base_frames[crop] for crop in crops}
    baseline_rows = run_baselines(cfg, frames)
    save_results(cfg, BASELINE_NAME, baseline_rows)

    tuning_rows: list[dict[str, object]] = []
    expanded_rows: list[dict[str, object]] = []
    tabular_models = [model for model in runnable_models if model in TABULAR_MODELS]
    if tabular_models:
        tabular_tuning, tabular_best = collect_tabular_rows(cfg, frames, tabular_models)
        tuning_rows.extend(tabular_tuning)
        expanded_rows.extend(tabular_best)

    if "tcn" in runnable_models:
        tcn_tuning, tcn_best = collect_tcn_rows(cfg, frames, tcn_device, args.max_train_windows)
        tuning_rows.extend(tcn_tuning)
        expanded_rows.extend(tcn_best)

    save_results(cfg, TUNING_NAME, tuning_rows)
    save_results(cfg, EXPANDED_NAME, expanded_rows)
    save_run_manifest(cfg, args, tcn_device, crops, runnable_models)
    print(f"Completed focused expansion run in {cfg.output_root}")


if __name__ == "__main__":
    main()
