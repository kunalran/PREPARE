from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.train_graph_wavenet import (
    CROP_FILES,
    WindowDataset,
    GraphWaveNetLite,
    build_crop_frame,
    filter_valid_series,
    pivot_feature,
    safe_wape,
)


BASE_FEATURE_COLUMNS = [
    "price_filled_log1p",
    "arrival_log1p",
    "temp_mean",
    "temp_range",
    "rain_sum",
    "solar_sum",
    "rh_mean",
    "price_dow_ratio",
    "arrival_dow_ratio",
    "day_of_year_sin",
    "day_of_year_cos",
    "day_of_week_sin",
    "day_of_week_cos",
]


@dataclass(frozen=True)
class GraphPlusConfig:
    data_dir: Path
    output_dir: Path
    crops: list[str]
    horizon: int
    validation_days: int
    min_series_observations: int
    dense_min_pct: float | None
    input_window: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    hidden_channels: int
    dropout: float
    k_neighbors: int
    cluster_count: int
    geo_edge_weight: float
    state_edge_weight: float
    cluster_edge_weight: float
    random_state: int
    device: str


def parse_args() -> GraphPlusConfig:
    parser = argparse.ArgumentParser(
        description="Train improved GraphWaveNet variants using delta_current and richer mandi edges."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("final_data_hourly"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/graph_wavenet_plus_15d"))
    parser.add_argument("--crops", type=str, default="onion,potato,tomato,wheat")
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--validation-days", type=int, default=90)
    parser.add_argument("--min-series-observations", type=int, default=30)
    parser.add_argument("--dense-min-pct", type=float, default=0.0)
    parser.add_argument("--input-window", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-channels", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--k-neighbors", type=int, default=12)
    parser.add_argument("--cluster-count", type=int, default=4)
    parser.add_argument("--geo-edge-weight", type=float, default=0.6)
    parser.add_argument("--state-edge-weight", type=float, default=0.25)
    parser.add_argument("--cluster-edge-weight", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    crops = [c.strip().lower() for c in args.crops.split(",") if c.strip()]
    invalid = [c for c in crops if c not in CROP_FILES]
    if invalid:
        raise ValueError(f"Unsupported crops: {invalid}. Valid crops: {sorted(CROP_FILES)}")
    return GraphPlusConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        crops=sorted(dict.fromkeys(crops)),
        horizon=args.horizon,
        validation_days=args.validation_days,
        min_series_observations=args.min_series_observations,
        dense_min_pct=args.dense_min_pct,
        input_window=args.input_window,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_channels=args.hidden_channels,
        dropout=args.dropout,
        k_neighbors=args.k_neighbors,
        cluster_count=args.cluster_count,
        geo_edge_weight=args.geo_edge_weight,
        state_edge_weight=args.state_edge_weight,
        cluster_edge_weight=args.cluster_edge_weight,
        random_state=args.random_state,
        device=args.device,
    )


def build_geo_adjacency(coords: np.ndarray, k_neighbors: int) -> np.ndarray:
    n_nodes = len(coords)
    if n_nodes == 1:
        return np.eye(1, dtype=np.float32)
    diffs = coords[:, None, :] - coords[None, :, :]
    dists = np.sqrt((diffs**2).sum(axis=-1))
    np.fill_diagonal(dists, np.inf)
    k = min(k_neighbors, n_nodes - 1)
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for idx in range(n_nodes):
        neighbors = np.argpartition(dists[idx], k)[:k]
        weights = np.exp(-dists[idx, neighbors] / (np.nanmedian(dists[idx, neighbors]) + 1e-6))
        adjacency[idx, neighbors] = weights
    return np.maximum(adjacency, adjacency.T)


def row_normalize(adjacency: np.ndarray) -> np.ndarray:
    adjacency = adjacency + np.eye(len(adjacency), dtype=np.float32)
    degree = adjacency.sum(axis=1, keepdims=True)
    return adjacency / np.clip(degree, 1e-6, None)


def build_metadata(frame: pd.DataFrame, validation_start: pd.Timestamp, cluster_count: int, random_state: int) -> pd.DataFrame:
    train_frame = frame[frame["Date"] < validation_start]
    grouped = train_frame.groupby("series_id", sort=False)
    meta = grouped.agg(
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
        State=("State", "first"),
        mean_arrival=("Arrival_Quantity_CausalFilled", "mean"),
        std_arrival=("Arrival_Quantity_CausalFilled", "std"),
        mean_price=("Modal_Price_CausalFilled", "mean"),
    ).sort_index()
    meta["std_arrival"] = meta["std_arrival"].fillna(0.0)
    feats = np.log1p(meta[["mean_arrival", "std_arrival", "mean_price"]].fillna(0.0).to_numpy())
    k = min(cluster_count, len(meta))
    if k <= 1:
        meta["cluster_id"] = 0
    else:
        meta["cluster_id"] = KMeans(n_clusters=k, n_init=20, random_state=random_state).fit_predict(feats)
    return meta


def make_windows(
    feature_panel: np.ndarray,
    target_panel: np.ndarray,
    anchor_panel: np.ndarray,
    dates: pd.Index,
    validation_start: pd.Timestamp,
    input_window: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    train_samples = []
    val_samples = []
    for end_idx in range(input_window - 1, len(dates)):
        start_idx = end_idx - input_window + 1
        target = target_panel[end_idx]
        anchor = anchor_panel[end_idx]
        mask = np.isfinite(target) & np.isfinite(anchor)
        if not mask.any():
            continue
        sample = (
            np.transpose(feature_panel[start_idx : end_idx + 1], (2, 1, 0)).astype(np.float32),
            np.nan_to_num(target, nan=0.0).astype(np.float32),
            mask.astype(np.float32),
            np.nan_to_num(anchor, nan=0.0).astype(np.float32),
        )
        if dates[end_idx] < validation_start:
            train_samples.append(sample)
        else:
            val_samples.append(sample)
    return train_samples, val_samples


class WindowDatasetPlus(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        x, y, mask, anchor = self.samples[index]
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask), torch.from_numpy(anchor)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = ((pred - target) ** 2) * mask
    return diff.sum() / torch.clamp(mask.sum(), min=1.0)


def collect_predictions(model, loader, static_adj, device):
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for x, y, mask, anchor in loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            anchor = anchor.to(device)
            pred_delta = model(x, static_adj)
            pred = torch.expm1(pred_delta + anchor)
            y_true = torch.expm1(y + anchor)
            pred_np = pred.cpu().numpy()
            y_np = y_true.cpu().numpy()
            mask_np = mask.cpu().numpy().astype(bool)
            preds.append(pred_np[mask_np])
            actuals.append(y_np[mask_np])
    return np.concatenate(actuals), np.concatenate(preds)


def train_one_crop(crop: str, config: GraphPlusConfig) -> dict:
    print(f"\n=== GraphWaveNet+ {crop.upper()} horizon {config.horizon}d ===")
    crop_path = config.data_dir / CROP_FILES[crop]
    frame = build_crop_frame(crop_path, config.horizon, config.dense_min_pct)
    max_date = frame["Date"].max()
    validation_start = max_date - pd.Timedelta(days=config.validation_days - 1)
    frame = filter_valid_series(frame, validation_start, config.min_series_observations)
    meta = build_metadata(frame, validation_start, config.cluster_count, config.random_state)
    frame = frame[frame["series_id"].isin(meta.index)].copy()
    frame["cluster_id"] = frame["series_id"].map(meta["cluster_id"]).astype(float)

    series_ids = meta.index.tolist()
    dates = pd.Index(sorted(frame["Date"].unique()))
    feature_cols = BASE_FEATURE_COLUMNS + ["cluster_id"]
    feature_arrays = [pivot_feature(frame, feature, dates, series_ids) for feature in feature_cols]
    feature_panel = np.stack(feature_arrays, axis=-1)
    target_raw = pivot_feature(frame, f"target_{config.horizon}d", dates, series_ids)
    current_raw = pivot_feature(frame, "Modal_Price_CausalFilled", dates, series_ids)
    target_log = np.where(
        np.isfinite(target_raw) & np.isfinite(current_raw),
        np.log1p(np.clip(target_raw, 0.0, None)) - np.log1p(np.clip(current_raw, 0.0, None)),
        np.nan,
    )
    anchor_panel = np.where(np.isfinite(current_raw), np.log1p(np.clip(current_raw, 0.0, None)), np.nan)

    train_date_mask = dates < validation_start
    # mean-center dynamic features per mandi using training period
    dynamic_idx = [0, 1]
    train_mean_by_node = np.nanmean(feature_panel[train_date_mask][:, :, dynamic_idx], axis=0)
    feature_panel[:, :, dynamic_idx] = feature_panel[:, :, dynamic_idx] - train_mean_by_node[None, :, :]
    feature_mean = np.nanmean(feature_panel[train_date_mask], axis=(0, 1))
    feature_std = np.nanstd(feature_panel[train_date_mask], axis=(0, 1))
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
    feature_panel = (feature_panel - feature_mean) / feature_std
    feature_panel = np.nan_to_num(feature_panel, nan=0.0, posinf=0.0, neginf=0.0)

    geo_adj = build_geo_adjacency(meta[["latitude", "longitude"]].to_numpy(dtype=np.float32), config.k_neighbors)
    state_same = (meta["State"].to_numpy()[:, None] == meta["State"].to_numpy()[None, :]).astype(np.float32)
    cluster_same = (meta["cluster_id"].to_numpy()[:, None] == meta["cluster_id"].to_numpy()[None, :]).astype(np.float32)
    np.fill_diagonal(state_same, 0.0)
    np.fill_diagonal(cluster_same, 0.0)
    adjacency = (
        config.geo_edge_weight * geo_adj
        + config.state_edge_weight * state_same
        + config.cluster_edge_weight * cluster_same
    )
    adjacency = row_normalize(adjacency.astype(np.float32))

    train_samples, val_samples = make_windows(
        feature_panel,
        target_log,
        anchor_panel,
        dates,
        validation_start,
        config.input_window,
    )
    if not train_samples or not val_samples:
        raise RuntimeError(f"Not enough samples for crop {crop}")

    device = torch.device(config.device)
    static_adj = torch.from_numpy(adjacency).to(device)
    train_loader = torch.utils.data.DataLoader(WindowDatasetPlus(train_samples), batch_size=config.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(WindowDatasetPlus(val_samples), batch_size=config.batch_size, shuffle=False)
    model = GraphWaveNetLite(
        num_features=len(feature_cols),
        num_nodes=len(series_ids),
        hidden_channels=config.hidden_channels,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_state = None
    best_val = math.inf
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        for x, y, mask, anchor in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad()
            pred = model(x, static_adj)
            loss = masked_mse(pred, y, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss_sum += float(loss.item())
            train_batches += 1
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for x, y, mask, anchor in val_loader:
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                pred = model(x, static_adj)
                loss = masked_mse(pred, y, mask)
                val_loss_sum += float(loss.item())
                val_batches += 1
        train_loss = train_loss_sum / max(train_batches, 1)
        val_loss = val_loss_sum / max(val_batches, 1)
        print(f"  epoch {epoch:02d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    actual, pred = collect_predictions(model, val_loader, static_adj, device)
    metrics = {
        "crop": crop,
        "horizon_days": config.horizon,
        "train_windows": len(train_samples),
        "validation_windows": len(val_samples),
        "validation_start": validation_start.strftime("%Y-%m-%d"),
        "validation_end": max_date.strftime("%Y-%m-%d"),
        "mae": float(mean_absolute_error(actual, pred)),
        "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
        "r2": float(r2_score(actual, pred)),
        "wape_pct": safe_wape(actual, pred),
    }
    crop_dir = config.output_dir / crop
    crop_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "state_dict": best_state,
        "series_ids": series_ids,
        "feature_columns": feature_cols,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "adjacency": adjacency,
        "metadata": meta.reset_index().to_dict(orient="list"),
        "config": asdict(config),
        "metrics": metrics,
    }
    model_path = crop_dir / f"{crop}_graph_wavenet_plus_{config.horizon}d.joblib"
    joblib.dump(artifact, model_path)
    (crop_dir / f"{crop}_graph_metrics.json").write_text(json.dumps([metrics], indent=2), encoding="utf-8")
    print(
        f"Saved {model_path} | MAE={metrics['mae']:.2f} RMSE={metrics['rmse']:.2f} "
        f"R2={metrics['r2']:.4f} WAPE={metrics['wape_pct']:.2f}%"
    )
    return metrics


def main() -> None:
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.random_state)
    np.random.seed(config.random_state)
    summary = {}
    for crop in config.crops:
        summary[crop] = train_one_crop(crop, config)
    summary_path = config.output_dir / "graph_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
