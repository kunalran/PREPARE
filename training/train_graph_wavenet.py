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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.train_global_price_model import (
    BASE_COLUMNS,
    WEATHER_HOURS,
    _causal_ratio_fill,
    filter_dense_market_years,
)


WEATHER_PREFIXES = ("t", "tp", "ssr", "r")
CROP_FILES = {
    "onion": "agmarknet_onion_data_final_hourly.csv",
    "potato": "agmarknet_potato_data_final_hourly.csv",
    "tomato": "agmarknet_tomato_data_final_hourly.csv",
    "wheat": "agmarknet_wheat_data_final_hourly.csv",
}
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


def weather_columns() -> list[str]:
    return [f"{prefix}{hour:02d}" for prefix in WEATHER_PREFIXES for hour in WEATHER_HOURS]


@dataclass(frozen=True)
class GraphConfig:
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
    random_state: int
    device: str


def parse_args() -> GraphConfig:
    parser = argparse.ArgumentParser(
        description="Train graph-based mandi forecasting models for a single horizon."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("final_data_hourly"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/graph_wavenet_15d"),
    )
    parser.add_argument(
        "--crops",
        type=str,
        default="onion,potato,tomato,wheat",
        help="Comma-separated crops to train.",
    )
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--validation-days", type=int, default=90)
    parser.add_argument("--min-series-observations", type=int, default=30)
    parser.add_argument("--dense-min-pct", type=float, default=0.0)
    parser.add_argument("--input-window", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--k-neighbors", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    crops = [c.strip().lower() for c in args.crops.split(",") if c.strip()]
    invalid = [c for c in crops if c not in CROP_FILES]
    if invalid:
        raise ValueError(f"Unsupported crops: {invalid}. Valid crops: {sorted(CROP_FILES)}")
    return GraphConfig(
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
        random_state=args.random_state,
        device=args.device,
    )


def build_crop_frame(path: Path, horizon: int, dense_min_pct: float | None) -> pd.DataFrame:
    usecols = BASE_COLUMNS + weather_columns()
    df = pd.read_csv(path, usecols=usecols, parse_dates=["Date"])
    if dense_min_pct is not None:
        rows_before = len(df)
        df = filter_dense_market_years(df, min_pct=dense_min_pct)
        print(
            f"  Dense market-year filter at {dense_min_pct:.0f}%: "
            f"{rows_before:,} -> {len(df):,} rows"
        )
    df["series_id"] = (
        df["Commodity"].astype(str)
        + "||"
        + df["State"].astype(str)
        + "||"
        + df["District"].astype(str)
        + "||"
        + df["Market"].astype(str)
    )
    df = df.sort_values(["series_id", "Date"]).reset_index(drop=True)

    temp_cols = [f"t{hour:02d}" for hour in WEATHER_HOURS]
    rain_cols = [f"tp{hour:02d}" for hour in WEATHER_HOURS]
    solar_cols = [f"ssr{hour:02d}" for hour in WEATHER_HOURS]
    humid_cols = [f"r{hour:02d}" for hour in WEATHER_HOURS]
    df["temp_mean"] = df[temp_cols].mean(axis=1)
    df["temp_min"] = df[temp_cols].min(axis=1)
    df["temp_max"] = df[temp_cols].max(axis=1)
    df["temp_range"] = df["temp_max"] - df["temp_min"]
    df["rain_sum"] = df[rain_cols].sum(axis=1)
    df["solar_sum"] = df[solar_cols].sum(axis=1)
    df["rh_mean"] = df[humid_cols].mean(axis=1)

    df["day_of_week_num"] = df["Date"].dt.dayofweek.astype(np.int16)
    df["day_of_year"] = df["Date"].dt.dayofyear.astype(np.int16)
    day_angle = 2.0 * np.pi * df["day_of_year"] / 366.0
    week_angle = 2.0 * np.pi * df["day_of_week_num"] / 7.0
    df["day_of_year_sin"] = np.sin(day_angle)
    df["day_of_year_cos"] = np.cos(day_angle)
    df["day_of_week_sin"] = np.sin(week_angle)
    df["day_of_week_cos"] = np.cos(week_angle)

    imputed_prices: list[pd.Series] = []
    imputed_arrivals: list[pd.Series] = []
    price_dow_ratios: list[pd.Series] = []
    arrival_dow_ratios: list[pd.Series] = []
    for _, group in df.groupby("series_id", sort=False):
        price_filled, price_ratio = _causal_ratio_fill(
            group["Modal_Price"],
            group["day_of_week_num"],
        )
        arrival_filled, arrival_ratio = _causal_ratio_fill(
            group["Arrival_Quantity"],
            group["day_of_week_num"],
        )
        imputed_prices.append(price_filled)
        imputed_arrivals.append(arrival_filled)
        price_dow_ratios.append(price_ratio)
        arrival_dow_ratios.append(arrival_ratio)

    df["Modal_Price_CausalFilled"] = pd.concat(imputed_prices).sort_index()
    df["Arrival_Quantity_CausalFilled"] = pd.concat(imputed_arrivals).sort_index()
    df["price_dow_ratio"] = pd.concat(price_dow_ratios).sort_index()
    df["arrival_dow_ratio"] = pd.concat(arrival_dow_ratios).sort_index()
    df["price_filled_log1p"] = np.log1p(df["Modal_Price_CausalFilled"].clip(lower=0.0))
    df["arrival_log1p"] = np.log1p(df["Arrival_Quantity_CausalFilled"].clip(lower=0.0))
    df[f"target_{horizon}d"] = df.groupby("series_id", sort=False)["Modal_Price"].shift(-horizon)
    return df


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


def build_knn_adjacency(coords: np.ndarray, k_neighbors: int) -> np.ndarray:
    n_nodes = len(coords)
    if n_nodes == 0:
        raise ValueError("No nodes available to build adjacency.")
    if n_nodes == 1:
        return np.eye(1, dtype=np.float32)

    coords = coords.astype(np.float32)
    diffs = coords[:, None, :] - coords[None, :, :]
    dists = np.sqrt((diffs**2).sum(axis=-1))
    np.fill_diagonal(dists, np.inf)
    k = min(k_neighbors, n_nodes - 1)
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for idx in range(n_nodes):
        neighbors = np.argpartition(dists[idx], k)[:k]
        weights = np.exp(-dists[idx, neighbors] / (np.nanmedian(dists[idx, neighbors]) + 1e-6))
        adjacency[idx, neighbors] = weights
    adjacency = np.maximum(adjacency, adjacency.T)
    adjacency += np.eye(n_nodes, dtype=np.float32)
    degree = adjacency.sum(axis=1, keepdims=True)
    adjacency = adjacency / np.clip(degree, 1e-6, None)
    return adjacency.astype(np.float32)


def pivot_feature(frame: pd.DataFrame, feature: str, index_dates: pd.Index, series_ids: list[str]) -> np.ndarray:
    pivot = frame.pivot(index="Date", columns="series_id", values=feature)
    pivot = pivot.reindex(index=index_dates, columns=series_ids)
    return pivot.to_numpy(dtype=np.float32)


def make_windows(
    feature_panel: np.ndarray,
    target_panel: np.ndarray,
    dates: pd.Index,
    validation_start: pd.Timestamp,
    input_window: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    train_samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    val_samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for end_idx in range(input_window - 1, len(dates)):
        start_idx = end_idx - input_window + 1
        target = target_panel[end_idx]
        mask = np.isfinite(target)
        if not mask.any():
            continue
        window = feature_panel[start_idx : end_idx + 1]  # [T, N, F]
        sample = (
            np.transpose(window, (2, 1, 0)).astype(np.float32),  # [F, N, T]
            np.nan_to_num(target, nan=0.0).astype(np.float32),
            mask.astype(np.float32),
        )
        if dates[end_idx] < validation_start:
            train_samples.append(sample)
        else:
            val_samples.append(sample)
    return train_samples, val_samples


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        x, y, mask = self.samples[index]
        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(mask),
        )


class DiffusionGraphConv(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Conv2d(channels * 3, channels, kernel_size=(1, 1))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, static_adj: torch.Tensor, adaptive_adj: torch.Tensor) -> torch.Tensor:
        x_static = torch.einsum("nm,bcmt->bcnt", static_adj, x)
        x_adapt = torch.einsum("nm,bcmt->bcnt", adaptive_adj, x)
        out = torch.cat([x, x_static, x_adapt], dim=1)
        out = self.proj(out)
        return F.dropout(out, p=self.dropout, training=self.training)


class GraphWaveNetBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.filter_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, 2),
            dilation=(1, dilation),
        )
        self.gate_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, 2),
            dilation=(1, dilation),
        )
        self.graph_conv = DiffusionGraphConv(channels, dropout)
        self.residual = nn.Conv2d(channels, channels, kernel_size=(1, 1))
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor, static_adj: torch.Tensor, adaptive_adj: torch.Tensor) -> torch.Tensor:
        residual = x
        left_pad = self.filter_conv.dilation[-1] * (self.filter_conv.kernel_size[-1] - 1)
        padded = F.pad(x, (left_pad, 0, 0, 0))
        gated = torch.tanh(self.filter_conv(padded)) * torch.sigmoid(self.gate_conv(padded))
        out = self.graph_conv(gated, static_adj, adaptive_adj)
        residual = residual[..., -out.size(-1) :]
        out = self.residual(out) + residual
        return self.norm(out)


class GraphWaveNetLite(nn.Module):
    def __init__(self, num_features: int, num_nodes: int, hidden_channels: int, dropout: float) -> None:
        super().__init__()
        self.input_proj = nn.Conv2d(num_features, hidden_channels, kernel_size=(1, 1))
        self.blocks = nn.ModuleList(
            [
                GraphWaveNetBlock(hidden_channels, dilation=1, dropout=dropout),
                GraphWaveNetBlock(hidden_channels, dilation=2, dropout=dropout),
                GraphWaveNetBlock(hidden_channels, dilation=4, dropout=dropout),
            ]
        )
        self.node_emb1 = nn.Parameter(torch.randn(num_nodes, 16))
        self.node_emb2 = nn.Parameter(torch.randn(16, num_nodes))
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=(1, 1)),
        )

    def adaptive_adj(self) -> torch.Tensor:
        scores = F.relu(self.node_emb1 @ self.node_emb2)
        return F.softmax(scores, dim=1)

    def forward(self, x: torch.Tensor, static_adj: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        adaptive_adj = self.adaptive_adj()
        for block in self.blocks:
            x = block(x, static_adj, adaptive_adj)
        out = self.head(x[..., -1:])
        return out.squeeze(1).squeeze(-1)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target) ** 2
    diff = diff * mask
    denom = torch.clamp(mask.sum(), min=1.0)
    return diff.sum() / denom


def safe_wape(actual: np.ndarray, pred: np.ndarray) -> float:
    denom = np.abs(actual).sum()
    if denom <= 1e-12:
        return float("nan")
    return float(np.abs(actual - pred).sum() / denom * 100.0)


def collect_masked_predictions(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    static_adj: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    actuals: list[np.ndarray] = []
    with torch.no_grad():
        for x, y, mask in loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            pred = torch.expm1(model(x, static_adj))
            y_true = torch.expm1(y)
            pred_np = pred.cpu().numpy()
            y_np = y_true.cpu().numpy()
            mask_np = mask.cpu().numpy().astype(bool)
            preds.append(pred_np[mask_np])
            actuals.append(y_np[mask_np])
    return np.concatenate(actuals), np.concatenate(preds)


def train_one_crop(crop: str, config: GraphConfig) -> dict:
    print(f"\n=== GraphWaveNet {crop.upper()} horizon {config.horizon}d ===")
    crop_path = config.data_dir / CROP_FILES[crop]
    frame = build_crop_frame(crop_path, config.horizon, config.dense_min_pct)
    max_date = frame["Date"].max()
    validation_start = max_date - pd.Timedelta(days=config.validation_days - 1)
    frame = filter_valid_series(frame, validation_start, config.min_series_observations)

    series_meta = (
        frame.groupby("series_id", sort=False)[["latitude", "longitude"]]
        .mean()
        .sort_index()
    )
    series_ids = series_meta.index.tolist()
    dates = pd.Index(sorted(frame["Date"].unique()))

    feature_arrays = [pivot_feature(frame, feature, dates, series_ids) for feature in FEATURE_COLUMNS]
    feature_panel = np.stack(feature_arrays, axis=-1)  # [T, N, F]
    target_panel = pivot_feature(frame, f"target_{config.horizon}d", dates, series_ids)

    train_date_mask = dates < validation_start
    feature_mean = np.nanmean(feature_panel[train_date_mask], axis=(0, 1))
    feature_std = np.nanstd(feature_panel[train_date_mask], axis=(0, 1))
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
    feature_panel = (feature_panel - feature_mean) / feature_std
    feature_panel = np.nan_to_num(feature_panel, nan=0.0, posinf=0.0, neginf=0.0)

    valid_target_mask = np.isfinite(target_panel) & (target_panel >= 0.0)
    target_panel_log = np.where(valid_target_mask, np.log1p(target_panel), np.nan)
    train_samples, val_samples = make_windows(
        feature_panel,
        target_panel_log,
        dates,
        validation_start,
        config.input_window,
    )
    if not train_samples or not val_samples:
        raise RuntimeError(f"Not enough graph samples for crop {crop}.")

    adjacency = build_knn_adjacency(series_meta.to_numpy(), config.k_neighbors)
    static_adj = torch.from_numpy(adjacency).to(torch.device(config.device))

    train_loader = torch.utils.data.DataLoader(
        WindowDataset(train_samples),
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        WindowDataset(val_samples),
        batch_size=config.batch_size,
        shuffle=False,
    )

    device = torch.device(config.device)
    model = GraphWaveNetLite(
        num_features=len(FEATURE_COLUMNS),
        num_nodes=len(series_ids),
        hidden_channels=config.hidden_channels,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_state = None
    best_val_loss = math.inf
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        for x, y, mask in train_loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            pred = model(x, static_adj)
            loss = masked_mse(pred, y, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss_sum += float(loss.item())
            train_batches += 1

        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for x, y, mask in val_loader:
                x = x.to(device)
                y = y.to(device)
                mask = mask.to(device)
                pred = model(x, static_adj)
                loss = masked_mse(pred, y, mask)
                val_loss_sum += float(loss.item())
                val_batches += 1
        train_loss = train_loss_sum / max(train_batches, 1)
        val_loss = val_loss_sum / max(val_batches, 1)
        print(f"  epoch {epoch:02d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError(f"Training failed for crop {crop}: no best state recorded.")
    model.load_state_dict(best_state)

    actual, pred = collect_masked_predictions(model, val_loader, static_adj, device)
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
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "adjacency": adjacency,
        "horizon_days": config.horizon,
        "input_window": config.input_window,
        "config": asdict(config),
        "metrics": metrics,
    }
    model_path = crop_dir / f"{crop}_graph_wavenet_{config.horizon}d.joblib"
    joblib.dump(artifact, model_path)
    metrics_path = crop_dir / f"{crop}_graph_metrics.json"
    metrics_path.write_text(json.dumps([metrics], indent=2), encoding="utf-8")
    print(
        f"Saved {model_path} | "
        f"MAE={metrics['mae']:.2f} RMSE={metrics['rmse']:.2f} "
        f"R2={metrics['r2']:.4f} WAPE={metrics['wape_pct']:.2f}%"
    )
    return metrics


def main() -> None:
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.random_state)
    np.random.seed(config.random_state)

    summary: dict[str, dict] = {}
    for crop in config.crops:
        summary[crop] = train_one_crop(crop, config)

    summary_path = config.output_dir / "graph_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
