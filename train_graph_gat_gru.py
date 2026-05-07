from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from train_global_price_model import load_and_engineer_crop
from train_graph_wavenet import safe_wape
from train_per_crop_models import CROP_FILES


TEMPORAL_FEATURE_COLUMNS = [
    "price_filled_log1p",
    "arrival_log1p",
    "temp_mean",
    "temp_range",
    "rain_sum",
    "solar_sum",
    "rh_mean",
    "state_price_mean",
    "national_price_mean",
    "state_arrival_mean",
    "national_arrival_mean",
    "price_dow_ratio",
    "arrival_dow_ratio",
    "price_lag_1",
    "price_lag_7",
    "price_lag_14",
    "price_lag_28",
    "price_lag_56",
    "price_lag_84",
    "price_lag_168",
    "arrival_lag_1",
    "arrival_lag_7",
    "arrival_lag_28",
    "arrival_lag_84",
    "price_roll_mean_7",
    "price_roll_mean_14",
    "price_roll_mean_21",
    "price_roll_mean_28",
    "price_roll_mean_84",
    "price_roll_mean_168",
    "arrival_roll_mean_7",
    "arrival_roll_mean_28",
    "arrival_roll_mean_84",
    "price_roll_std_7",
    "price_roll_std_28",
    "price_roll_std_84",
    "price_vs_roll28",
    "price_vs_roll84",
    "price_vs_state_mean",
    "price_vs_national_mean",
    "arrival_vs_roll28",
    "arrival_vs_state_mean",
    "arrival_vs_national_mean",
    "price_trend_7_28",
    "price_trend_28_84",
    "arrival_trend_7_28",
    "price_range_28",
    "price_range_84",
    "price_volatility_ratio_28",
    "price_volatility_ratio_84",
    "day_of_year_sin",
    "day_of_year_cos",
    "day_of_week_sin",
    "day_of_week_cos",
]
STATIC_FEATURE_COLUMNS = ["latitude", "longitude", "cluster_id"]
CENTER_FEATURE_PREFIXES = (
    "price_filled_log1p",
    "arrival_log1p",
    "state_price_mean",
    "national_price_mean",
    "state_arrival_mean",
    "national_arrival_mean",
    "price_lag_",
    "arrival_lag_",
    "price_roll_mean_",
    "arrival_roll_mean_",
    "price_roll_std_",
    "price_range_",
)


@dataclass(frozen=True)
class GraphGatGruConfig:
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
    gru_layers: int
    dropout: float
    k_neighbors: int
    corr_top_k: int
    cluster_count: int
    target_mode: str
    graph_mode: str
    correction_scale: float
    alpha_grid_step: float
    corr_edge_weight: float
    geo_edge_weight: float
    state_edge_weight: float
    cluster_edge_weight: float
    forward_fill_limit: int
    random_state: int
    device: str


def parse_args() -> GraphGatGruConfig:
    parser = argparse.ArgumentParser(
        description="Train richer graph forecasting models with correlation-heavy edges and GRU+attention."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("final_data_hourly"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/graph_gat_gru_15d"))
    parser.add_argument("--crops", type=str, default="onion,potato,tomato,wheat")
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--validation-days", type=int, default=90)
    parser.add_argument("--min-series-observations", type=int, default=30)
    parser.add_argument("--dense-min-pct", type=float, default=0.0)
    parser.add_argument("--input-window", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--gru-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--k-neighbors", type=int, default=10)
    parser.add_argument("--corr-top-k", type=int, default=16)
    parser.add_argument("--cluster-count", type=int, default=4)
    parser.add_argument(
        "--target-mode",
        choices=["delta_current", "delta_roll7", "delta_roll28"],
        default="delta_current",
    )
    parser.add_argument(
        "--graph-mode",
        choices=["full_graph", "temporal_only", "geo_only", "corr_only", "shuffled_graph"],
        default="full_graph",
    )
    parser.add_argument("--correction-scale", type=float, default=0.25)
    parser.add_argument("--alpha-grid-step", type=float, default=0.05)
    parser.add_argument("--corr-edge-weight", type=float, default=0.55)
    parser.add_argument("--geo-edge-weight", type=float, default=0.2)
    parser.add_argument("--state-edge-weight", type=float, default=0.1)
    parser.add_argument("--cluster-edge-weight", type=float, default=0.15)
    parser.add_argument("--forward-fill-limit", type=int, default=28)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    crops = [c.strip().lower() for c in args.crops.split(",") if c.strip()]
    invalid = [c for c in crops if c not in CROP_FILES]
    if invalid:
        raise ValueError(f"Unsupported crops: {invalid}. Valid crops: {sorted(CROP_FILES)}")
    return GraphGatGruConfig(
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
        gru_layers=args.gru_layers,
        dropout=args.dropout,
        k_neighbors=args.k_neighbors,
        corr_top_k=args.corr_top_k,
        cluster_count=args.cluster_count,
        target_mode=args.target_mode,
        graph_mode=args.graph_mode,
        correction_scale=args.correction_scale,
        alpha_grid_step=args.alpha_grid_step,
        corr_edge_weight=args.corr_edge_weight,
        geo_edge_weight=args.geo_edge_weight,
        state_edge_weight=args.state_edge_weight,
        cluster_edge_weight=args.cluster_edge_weight,
        forward_fill_limit=args.forward_fill_limit,
        random_state=args.random_state,
        device=args.device,
    )


def build_metadata(
    frame: pd.DataFrame,
    validation_start: pd.Timestamp,
    cluster_count: int,
    random_state: int,
) -> pd.DataFrame:
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
    cluster_features = np.log1p(meta[["mean_arrival", "std_arrival", "mean_price"]].fillna(0.0).to_numpy())
    k = min(cluster_count, len(meta))
    if k <= 1:
        meta["cluster_id"] = 0
    else:
        meta["cluster_id"] = KMeans(n_clusters=k, n_init=20, random_state=random_state).fit_predict(cluster_features)
    return meta


def filter_valid_series(
    frame: pd.DataFrame,
    validation_start: pd.Timestamp,
    min_series_observations: int,
) -> pd.DataFrame:
    train_mask = frame["Date"] < validation_start
    counts = frame.loc[train_mask].groupby("series_id")["Modal_Price"].count()
    valid_series = counts[counts >= min_series_observations].index
    filtered = frame[frame["series_id"].isin(valid_series)].copy()
    print(
        f"Kept {len(valid_series):,} series with at least "
        f"{min_series_observations} observed training prices."
    )
    print(f"Graph frame rows after filtering: {len(filtered):,}")
    return filtered


def row_normalize(adjacency: np.ndarray) -> np.ndarray:
    adjacency = adjacency.copy()
    adjacency += np.eye(len(adjacency), dtype=np.float32)
    degree = adjacency.sum(axis=1, keepdims=True)
    return adjacency / np.clip(degree, 1e-6, None)


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


def build_correlation_adjacency(price_history: np.ndarray, top_k: int) -> np.ndarray:
    n_steps, n_nodes = price_history.shape
    if n_nodes == 1:
        return np.eye(1, dtype=np.float32)
    filled = price_history.copy()
    col_means = np.nanmean(filled, axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)
    missing = ~np.isfinite(filled)
    if missing.any():
        filled[missing] = np.take(col_means, np.where(missing)[1])
    if n_steps > 1:
        filled = np.diff(filled, axis=0)
    filled = filled - filled.mean(axis=0, keepdims=True)
    denom = filled.std(axis=0, keepdims=True)
    denom = np.where(denom < 1e-6, 1.0, denom)
    standardized = filled / denom
    corr = standardized.T @ standardized / max(len(standardized) - 1, 1)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.clip(corr, 0.0, None)
    np.fill_diagonal(corr, 0.0)
    k = min(top_k, n_nodes - 1)
    adjacency = np.zeros_like(corr, dtype=np.float32)
    for idx in range(n_nodes):
        if k <= 0:
            continue
        neighbors = np.argpartition(corr[idx], -k)[-k:]
        weights = corr[idx, neighbors]
        adjacency[idx, neighbors] = weights
    return np.maximum(adjacency, adjacency.T)


def pivot_temporal_feature(
    frame: pd.DataFrame,
    feature: str,
    dates: pd.Index,
    series_ids: list[str],
    forward_fill_limit: int,
) -> np.ndarray:
    pivot = frame.pivot(index="Date", columns="series_id", values=feature)
    pivot = pivot.reindex(index=dates, columns=series_ids)
    pivot = pivot.ffill(limit=forward_fill_limit)
    return pivot.to_numpy(dtype=np.float32)


def pivot_target(frame: pd.DataFrame, feature: str, dates: pd.Index, series_ids: list[str]) -> np.ndarray:
    pivot = frame.pivot(index="Date", columns="series_id", values=feature)
    pivot = pivot.reindex(index=dates, columns=series_ids)
    return pivot.to_numpy(dtype=np.float32)


def make_feature_panel(
    frame: pd.DataFrame,
    meta: pd.DataFrame,
    dates: pd.Index,
    series_ids: list[str],
    forward_fill_limit: int,
) -> tuple[np.ndarray, list[str]]:
    feature_arrays = [
        pivot_temporal_feature(frame, feature, dates, series_ids, forward_fill_limit)
        for feature in TEMPORAL_FEATURE_COLUMNS
    ]
    n_dates = len(dates)
    for feature in STATIC_FEATURE_COLUMNS:
        values = meta.loc[series_ids, feature].to_numpy(dtype=np.float32)
        feature_arrays.append(np.broadcast_to(values[None, :], (n_dates, len(series_ids))).copy())
    feature_names = TEMPORAL_FEATURE_COLUMNS + STATIC_FEATURE_COLUMNS
    return np.stack(feature_arrays, axis=-1), feature_names


def center_and_scale_features(
    feature_panel: np.ndarray,
    feature_names: list[str],
    train_date_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    panel = feature_panel.copy()
    center_indices = [
        idx
        for idx, name in enumerate(feature_names)
        if any(name.startswith(prefix) for prefix in CENTER_FEATURE_PREFIXES)
    ]
    if center_indices:
        train_slice = panel[train_date_mask][:, :, center_indices]
        sums = np.nansum(train_slice, axis=0)
        counts = np.isfinite(train_slice).sum(axis=0)
        node_means = sums / np.where(counts > 0, counts, 1.0)
        node_means = np.where(np.isfinite(node_means), node_means, 0.0)
        panel[:, :, center_indices] = panel[:, :, center_indices] - node_means[None, :, :]
    feature_mean = np.nanmean(panel[train_date_mask], axis=(0, 1))
    feature_std = np.nanstd(panel[train_date_mask], axis=(0, 1))
    feature_mean = np.where(np.isfinite(feature_mean), feature_mean, 0.0)
    feature_std = np.where(np.isfinite(feature_std) & (feature_std >= 1e-6), feature_std, 1.0)
    panel = (panel - feature_mean) / feature_std
    panel = np.nan_to_num(panel, nan=0.0, posinf=0.0, neginf=0.0)
    return panel, feature_mean.astype(np.float32), feature_std.astype(np.float32)


def build_edge_matrices(
    meta: pd.DataFrame,
    price_history: np.ndarray,
    config: GraphGatGruConfig,
) -> np.ndarray:
    coords = meta[["latitude", "longitude"]].to_numpy(dtype=np.float32)
    geo_adj = build_geo_adjacency(coords, config.k_neighbors)
    corr_adj = build_correlation_adjacency(price_history, config.corr_top_k)
    state_same = (meta["State"].to_numpy()[:, None] == meta["State"].to_numpy()[None, :]).astype(np.float32)
    cluster_same = (
        meta["cluster_id"].to_numpy()[:, None] == meta["cluster_id"].to_numpy()[None, :]
    ).astype(np.float32)
    np.fill_diagonal(state_same, 0.0)
    np.fill_diagonal(cluster_same, 0.0)
    full_adjacency = (
        config.corr_edge_weight * corr_adj
        + config.geo_edge_weight * geo_adj
        + config.state_edge_weight * state_same
        + config.cluster_edge_weight * cluster_same
    ).astype(np.float32)
    if config.graph_mode == "temporal_only":
        adjacency = np.eye(len(meta), dtype=np.float32)
    elif config.graph_mode == "geo_only":
        adjacency = geo_adj.astype(np.float32)
    elif config.graph_mode == "corr_only":
        adjacency = corr_adj.astype(np.float32)
    elif config.graph_mode == "shuffled_graph":
        perm = np.random.default_rng(config.random_state).permutation(len(meta))
        adjacency = full_adjacency[perm][:, perm]
    else:
        adjacency = full_adjacency
    return row_normalize(adjacency)


def make_windows(
    feature_panel: np.ndarray,
    target_delta: np.ndarray,
    anchor_panel: np.ndarray,
    observed_price_panel: np.ndarray,
    dates: pd.Index,
    validation_start: pd.Timestamp,
    input_window: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    train_samples = []
    val_samples = []
    for end_idx in range(input_window - 1, len(dates)):
        start_idx = end_idx - input_window + 1
        target = target_delta[end_idx]
        anchor = anchor_panel[end_idx]
        observed = observed_price_panel[end_idx]
        mask = np.isfinite(target) & np.isfinite(anchor) & np.isfinite(observed)
        if not mask.any():
            continue
        sample = (
            np.transpose(feature_panel[start_idx : end_idx + 1], (1, 0, 2)).astype(np.float32),
            np.nan_to_num(target, nan=0.0).astype(np.float32),
            mask.astype(np.float32),
            np.nan_to_num(anchor, nan=0.0).astype(np.float32),
        )
        if dates[end_idx] < validation_start:
            train_samples.append(sample)
        else:
            val_samples.append(sample)
    return train_samples, val_samples


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        x, y, mask, anchor = self.samples[index]
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask), torch.from_numpy(anchor)


class GraphAttentionLayer(nn.Module):
    def __init__(self, hidden_channels: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.key = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.value = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.out = nn.Linear(hidden_channels, hidden_channels)
        self.norm = nn.LayerNorm(hidden_channels)
        self.dropout = dropout
        self.edge_bias_scale = nn.Parameter(torch.tensor(2.0))
        self.scale = hidden_channels ** -0.5

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = torch.einsum("bnd,bmd->bnm", q, k) * self.scale
        scores = scores + self.edge_bias_scale * adjacency.unsqueeze(0)
        scores = scores.masked_fill(~edge_mask.unsqueeze(0), -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = torch.einsum("bnm,bmd->bnd", attn, v)
        out = self.out(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return self.norm(x + out)


class GraphGatGru(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_channels: int,
        gru_layers: int,
        dropout: float,
        correction_scale: float,
    ) -> None:
        super().__init__()
        self.feature_proj = nn.Linear(num_features, hidden_channels)
        self.gru = nn.GRU(
            input_size=hidden_channels,
            hidden_size=hidden_channels,
            num_layers=gru_layers,
            dropout=dropout if gru_layers > 1 else 0.0,
            batch_first=True,
        )
        self.gat1 = GraphAttentionLayer(hidden_channels, dropout)
        self.gat2 = GraphAttentionLayer(hidden_channels, dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.correction_scale = correction_scale

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, seq_len, num_features = x.shape
        x = self.feature_proj(x)
        x = x.reshape(batch_size * num_nodes, seq_len, -1)
        _, hidden = self.gru(x)
        node_repr = hidden[-1].reshape(batch_size, num_nodes, -1)
        node_repr = self.gat1(node_repr, adjacency, edge_mask)
        node_repr = self.gat2(node_repr, adjacency, edge_mask)
        return self.correction_scale * self.head(node_repr).squeeze(-1)


def masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    loss = loss * mask
    return loss.sum() / torch.clamp(mask.sum(), min=1.0)


def collect_validation_arrays(model, loader, adjacency, edge_mask, device):
    model.eval()
    pred_deltas, actual_deltas, anchors = [], [], []
    with torch.no_grad():
        for x, y, mask, anchor in loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            anchor = anchor.to(device)
            pred_delta = model(x, adjacency, edge_mask)
            pred_np = pred_delta.cpu().numpy()
            actual_np = y.cpu().numpy()
            anchor_np = anchor.cpu().numpy()
            mask_np = mask.cpu().numpy().astype(bool)
            pred_deltas.append(pred_np[mask_np])
            actual_deltas.append(actual_np[mask_np])
            anchors.append(anchor_np[mask_np])
    return np.concatenate(actual_deltas), np.concatenate(pred_deltas), np.concatenate(anchors)


def pick_best_alpha(
    actual_delta: np.ndarray,
    pred_delta: np.ndarray,
    anchor_log: np.ndarray,
    alpha_grid_step: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    actual = np.expm1(actual_delta + anchor_log)
    best_alpha = 0.0
    best_r2 = -math.inf
    best_pred = np.expm1(anchor_log)
    max_steps = max(int(round(1.0 / alpha_grid_step)), 1)
    for step in range(max_steps + 1):
        alpha = min(step * alpha_grid_step, 1.0)
        pred = np.expm1(anchor_log + alpha * pred_delta)
        score = r2_score(actual, pred)
        if score > best_r2:
            best_r2 = score
            best_alpha = alpha
            best_pred = pred
    return best_alpha, actual, best_pred


def train_one_crop(crop: str, config: GraphGatGruConfig) -> dict:
    print(
        f"\n=== Graph GAT-GRU {crop.upper()} horizon {config.horizon}d "
        f"mode={config.graph_mode} ==="
    )
    crop_path = config.data_dir / CROP_FILES[crop]
    frame = load_and_engineer_crop(
        crop_path,
        [config.horizon],
        dense_min_pct=config.dense_min_pct,
        observed_only=False,
    )
    max_date = frame["Date"].max()
    validation_start = max_date - pd.Timedelta(days=config.validation_days - 1)
    frame = filter_valid_series(frame, validation_start, config.min_series_observations)
    meta = build_metadata(frame, validation_start, config.cluster_count, config.random_state)
    frame = frame[frame["series_id"].isin(meta.index)].copy()
    frame["cluster_id"] = frame["series_id"].map(meta["cluster_id"]).astype(float)

    series_ids = meta.index.tolist()
    dates = pd.Index(sorted(frame["Date"].unique()))
    feature_panel_raw, feature_names = make_feature_panel(
        frame,
        meta,
        dates,
        series_ids,
        config.forward_fill_limit,
    )
    target_raw = pivot_target(frame, f"target_{config.horizon}d", dates, series_ids)
    current_raw = pivot_target(frame, "Modal_Price_CausalFilled", dates, series_ids)
    observed_price_raw = pivot_target(frame, "Modal_Price", dates, series_ids)
    if config.target_mode == "delta_current":
        anchor_raw = current_raw
    elif config.target_mode == "delta_roll7":
        anchor_raw = pivot_target(frame, "price_roll_mean_7", dates, series_ids)
    else:
        anchor_raw = pivot_target(frame, "price_roll_mean_28", dates, series_ids)
    target_delta = np.where(
        np.isfinite(target_raw) & np.isfinite(anchor_raw),
        np.log1p(np.clip(target_raw, 0.0, None)) - np.log1p(np.clip(anchor_raw, 0.0, None)),
        np.nan,
    )
    anchor_panel = np.where(np.isfinite(anchor_raw), np.log1p(np.clip(anchor_raw, 0.0, None)), np.nan)

    train_date_mask = np.asarray(dates < validation_start)
    feature_panel, feature_mean, feature_std = center_and_scale_features(
        feature_panel_raw,
        feature_names,
        train_date_mask,
    )

    price_history = pivot_temporal_feature(
        frame,
        "price_filled_log1p",
        dates,
        series_ids,
        config.forward_fill_limit,
    )[train_date_mask]
    adjacency = build_edge_matrices(meta, price_history, config)
    edge_mask = adjacency > 0.0

    train_samples, val_samples = make_windows(
        feature_panel,
        target_delta,
        anchor_panel,
        observed_price_raw,
        dates,
        validation_start,
        config.input_window,
    )
    if not train_samples or not val_samples:
        raise RuntimeError(f"Not enough samples for crop {crop}")

    device = torch.device(config.device)
    adjacency_t = torch.from_numpy(adjacency).to(device)
    edge_mask_t = torch.from_numpy(edge_mask).to(device)
    train_loader = torch.utils.data.DataLoader(WindowDataset(train_samples), batch_size=config.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(WindowDataset(val_samples), batch_size=config.batch_size, shuffle=False)
    model = GraphGatGru(
        num_features=len(feature_names),
        hidden_channels=config.hidden_channels,
        gru_layers=config.gru_layers,
        dropout=config.dropout,
        correction_scale=config.correction_scale,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_state = None
    best_val = math.inf
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        for x, y, mask, anchor in train_loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            pred = model(x, adjacency_t, edge_mask_t)
            loss = masked_smooth_l1(pred, y, mask)
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
                x = x.to(device)
                y = y.to(device)
                mask = mask.to(device)
                pred = model(x, adjacency_t, edge_mask_t)
                loss = masked_smooth_l1(pred, y, mask)
                val_loss_sum += float(loss.item())
                val_batches += 1
        train_loss = train_loss_sum / max(train_batches, 1)
        val_loss = val_loss_sum / max(val_batches, 1)
        print(f"  epoch {epoch:02d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError(f"Training failed for crop {crop}: no best state recorded")
    model.load_state_dict(best_state)
    actual_delta, pred_delta, anchor_log = collect_validation_arrays(
        model,
        val_loader,
        adjacency_t,
        edge_mask_t,
        device,
    )
    best_alpha, actual, pred = pick_best_alpha(
        actual_delta,
        pred_delta,
        anchor_log,
        config.alpha_grid_step,
    )
    metrics = {
        "crop": crop,
        "horizon_days": config.horizon,
        "target_mode": config.target_mode,
        "graph_mode": config.graph_mode,
        "correction_scale": config.correction_scale,
        "best_alpha": best_alpha,
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
        "feature_names": feature_names,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "adjacency": adjacency,
        "metadata": meta.reset_index().to_dict(orient="list"),
        "config": asdict(config),
        "metrics": metrics,
    }
    model_path = (
        crop_dir
        / f"{crop}_{config.target_mode}_{config.graph_mode}_graph_gat_gru_{config.horizon}d.joblib"
    )
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
