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
    build_crop_frame,
    filter_valid_series,
    pivot_feature,
    safe_wape,
)
from training.train_graph_wavenet_plus import (
    build_geo_adjacency,
    build_metadata,
    row_normalize,
)


FEATURE_COLUMNS = [
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
class SagmmConfig:
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
    experts: int
    dropout: float
    k_neighbors: int
    cluster_count: int
    random_state: int
    device: str


def parse_args() -> SagmmConfig:
    parser = argparse.ArgumentParser(
        description="Train a SAGMM-inspired spatiotemporal graph mixture model."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("final_data_hourly"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/sagmm_moe_15d"))
    parser.add_argument("--crops", type=str, default="onion,potato,tomato,wheat")
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--validation-days", type=int, default=90)
    parser.add_argument("--min-series-observations", type=int, default=30)
    parser.add_argument("--dense-min-pct", type=float, default=0.0)
    parser.add_argument("--input-window", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-channels", type=int, default=48)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--k-neighbors", type=int, default=12)
    parser.add_argument("--cluster-count", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    crops = [c.strip().lower() for c in args.crops.split(",") if c.strip()]
    invalid = [c for c in crops if c not in CROP_FILES]
    if invalid:
        raise ValueError(f"Unsupported crops: {invalid}. Valid crops: {sorted(CROP_FILES)}")
    return SagmmConfig(
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
        experts=args.experts,
        dropout=args.dropout,
        k_neighbors=args.k_neighbors,
        cluster_count=args.cluster_count,
        random_state=args.random_state,
        device=args.device,
    )


def make_windows(feature_panel, target_panel, anchor_panel, dates, validation_start, input_window):
    train_samples, val_samples = [], []
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


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        x, y, mask, anchor = self.samples[index]
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask), torch.from_numpy(anchor)


class SagmmMoE(nn.Module):
    def __init__(self, num_features: int, hidden_channels: int, experts: int, dropout: float) -> None:
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(num_features, hidden_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
        )
        self.dropout = dropout
        self.gate = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, experts),
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_channels * 2, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, 1),
                )
                for _ in range(experts)
            ]
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        h = self.temporal(x)[..., -1]  # [B, C, N]
        h = F.dropout(h, p=self.dropout, training=self.training)
        neigh = torch.einsum("nm,bcm->bcn", adjacency, h)
        node_repr = torch.cat([h, neigh], dim=1).permute(0, 2, 1)  # [B, N, 2C]
        gate_logits = self.gate(node_repr)
        gate_weights = F.softmax(gate_logits, dim=-1)
        expert_outs = torch.stack([expert(node_repr).squeeze(-1) for expert in self.experts], dim=-1)
        pred = (gate_weights * expert_outs).sum(dim=-1)
        return pred


def masked_mse(pred, target, mask):
    diff = ((pred - target) ** 2) * mask
    return diff.sum() / torch.clamp(mask.sum(), min=1.0)


def collect_predictions(model, loader, adjacency, device):
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for x, y, mask, anchor in loader:
            x, y, mask, anchor = x.to(device), y.to(device), mask.to(device), anchor.to(device)
            pred_delta = model(x, adjacency)
            pred = torch.expm1(pred_delta + anchor)
            actual = torch.expm1(y + anchor)
            pred_np = pred.cpu().numpy()
            actual_np = actual.cpu().numpy()
            mask_np = mask.cpu().numpy().astype(bool)
            preds.append(pred_np[mask_np])
            actuals.append(actual_np[mask_np])
    return np.concatenate(actuals), np.concatenate(preds)


def train_one_crop(crop: str, config: SagmmConfig) -> dict:
    print(f"\n=== SAGMM-MoE {crop.upper()} horizon {config.horizon}d ===")
    crop_path = config.data_dir / CROP_FILES[crop]
    frame = build_crop_frame(crop_path, config.horizon, config.dense_min_pct)
    max_date = frame["Date"].max()
    validation_start = max_date - pd.Timedelta(days=config.validation_days - 1)
    frame = filter_valid_series(frame, validation_start, config.min_series_observations)
    meta = build_metadata(frame, validation_start, config.cluster_count, config.random_state)
    frame = frame[frame["series_id"].isin(meta.index)].copy()
    series_ids = meta.index.tolist()
    dates = pd.Index(sorted(frame["Date"].unique()))

    feature_arrays = [pivot_feature(frame, feat, dates, series_ids) for feat in FEATURE_COLUMNS]
    feature_panel = np.stack(feature_arrays, axis=-1)
    target_raw = pivot_feature(frame, f"target_{config.horizon}d", dates, series_ids)
    current_raw = pivot_feature(frame, "Modal_Price_CausalFilled", dates, series_ids)
    target_log = np.where(
        np.isfinite(target_raw) & np.isfinite(current_raw),
        np.log1p(np.clip(target_raw, 0.0, None)) - np.log1p(np.clip(current_raw, 0.0, None)),
        np.nan,
    )
    anchor_panel = np.where(np.isfinite(current_raw), np.log1p(np.clip(current_raw, 0.0, None)), np.nan)

    train_mask = dates < validation_start
    dyn_idx = [0, 1]
    mean_by_node = np.nanmean(feature_panel[train_mask][:, :, dyn_idx], axis=0)
    feature_panel[:, :, dyn_idx] = feature_panel[:, :, dyn_idx] - mean_by_node[None, :, :]
    feat_mean = np.nanmean(feature_panel[train_mask], axis=(0, 1))
    feat_std = np.nanstd(feature_panel[train_mask], axis=(0, 1))
    feat_std = np.where(feat_std < 1e-6, 1.0, feat_std)
    feature_panel = (feature_panel - feat_mean) / feat_std
    feature_panel = np.nan_to_num(feature_panel, nan=0.0, posinf=0.0, neginf=0.0)

    geo_adj = build_geo_adjacency(meta[["latitude", "longitude"]].to_numpy(dtype=np.float32), config.k_neighbors)
    state_same = (meta["State"].to_numpy()[:, None] == meta["State"].to_numpy()[None, :]).astype(np.float32)
    np.fill_diagonal(state_same, 0.0)
    adjacency = row_normalize((0.7 * geo_adj + 0.3 * state_same).astype(np.float32))

    train_samples, val_samples = make_windows(feature_panel, target_log, anchor_panel, dates, validation_start, config.input_window)
    if not train_samples or not val_samples:
        raise RuntimeError(f"Not enough samples for crop {crop}")

    device = torch.device(config.device)
    adjacency_t = torch.from_numpy(adjacency).to(device)
    train_loader = torch.utils.data.DataLoader(WindowDataset(train_samples), batch_size=config.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(WindowDataset(val_samples), batch_size=config.batch_size, shuffle=False)
    model = SagmmMoE(len(FEATURE_COLUMNS), config.hidden_channels, config.experts, config.dropout).to(device)
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
            pred = model(x, adjacency_t)
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
                pred = model(x, adjacency_t)
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
    actual, pred = collect_predictions(model, val_loader, adjacency_t, device)
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
        "feature_columns": FEATURE_COLUMNS,
        "adjacency": adjacency,
        "metadata": meta.reset_index().to_dict(orient="list"),
        "config": asdict(config),
        "metrics": metrics,
    }
    model_path = crop_dir / f"{crop}_sagmm_moe_{config.horizon}d.joblib"
    joblib.dump(artifact, model_path)
    (crop_dir / f"{crop}_sagmm_metrics.json").write_text(json.dumps([metrics], indent=2), encoding="utf-8")
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
    summary_path = config.output_dir / "sagmm_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
