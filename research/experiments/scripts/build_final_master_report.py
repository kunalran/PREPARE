from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "training") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "training"))

from training.train_global_price_model import safe_mape, safe_wape  # type: ignore  # noqa: E402

from prepare_experiments import (  # type: ignore  # noqa: E402
    ensure_venv,
    experiment_config,
    load_crop_frames,
    set_local_runtime_dirs,
    split_train_val,
)


REPORT_ROOT = REPO_ROOT / "newtests" / "reports"
REPORT_PATH = REPORT_ROOT / "final_master_report.md"
PERSISTENCE_PATH = REPORT_ROOT / "same_day_previous_price_baseline.csv"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def normalize_experiment_name(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "experiment_name" not in df.columns:
        df["experiment_name"] = np.nan
    for fallback in ("config_name", "display_name", "baseline_name"):
        if fallback in df.columns:
            df["experiment_name"] = df["experiment_name"].fillna(df[fallback])
    return df


def markdown_table(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    if columns is not None:
        df = df[columns].copy()
    if df.empty:
        return "_No rows._"
    headers = list(df.columns)
    rows: list[list[str]] = []
    for _, row in df.iterrows():
        rendered: list[str] = []
        for col in headers:
            value = row[col]
            if pd.isna(value):
                rendered.append("n/a")
            elif isinstance(value, float):
                rendered.append(f"{value:.4f}")
            else:
                rendered.append(str(value))
        rows.append(rendered)
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, sep_line] + body)


def r2_label(row: pd.Series) -> str:
    return f"{row['experiment_name']} ({row['r2']:.4f})"


def best_by_crop(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    subset = df[df["horizon_days"] == horizon].copy()
    return (
        subset.sort_values(["crop", "r2"], ascending=[True, False])
        .groupby("crop", as_index=False)
        .first()
    )


def best_row(df: pd.DataFrame, crop: str, horizon: int) -> pd.Series:
    subset = df[(df["crop"] == crop) & (df["horizon_days"] == horizon)].sort_values("r2", ascending=False)
    return subset.iloc[0]


def compute_same_day_persistence() -> pd.DataFrame:
    cfg = experiment_config()
    set_local_runtime_dirs(cfg.output_root)
    frames = load_crop_frames(cfg)
    rows: list[dict[str, object]] = []
    for crop, frame in frames.items():
        _, val_frame, _target_col, validation_start, validation_end = split_train_val(frame, 1, cfg.validation_days)
        subset = val_frame[["Date", "Modal_Price", "price_lag_1"]].dropna().copy()
        y_true = subset["Modal_Price"].to_numpy(dtype=float)
        preds = subset["price_lag_1"].to_numpy(dtype=float)
        rows.append(
            {
                "baseline_name": "same_day_previous_price",
                "crop": crop,
                "validation_rows": int(len(subset)),
                "validation_start": validation_start.strftime("%Y-%m-%d"),
                "validation_end": validation_end.strftime("%Y-%m-%d"),
                "mae": float(mean_absolute_error(y_true, preds)),
                "rmse": float(math.sqrt(mean_squared_error(y_true, preds))),
                "r2": float(r2_score(y_true, preds)),
                "mape_pct": float(safe_mape(y_true, preds)),
                "wape_pct": float(safe_wape(y_true, preds)),
                "prediction_definition": "Predict current-day Modal_Price using previous day's price_lag_1.",
            }
        )
    df = pd.DataFrame(rows).sort_values("crop").reset_index(drop=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    df.to_csv(PERSISTENCE_PATH, index=False)
    return df


def graph_extra_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    radius_ablation = read_csv(
        REPO_ROOT / "newtests" / "targeted_rebuild" / "results" / "gat_gru_radius_ablation" / "radius_ablation_summary.csv"
    )
    for row in radius_ablation.itertuples(index=False):
        rows.append(
            {
                "experiment_family": "graph_gat_gru_radius",
                "experiment_name": row.config_name,
                "crop": row.crop,
                "horizon_days": 15,
                "validation_rows": getattr(row, "validation_windows", np.nan),
                "validation_start": getattr(row, "validation_start", np.nan),
                "validation_end": getattr(row, "validation_end", np.nan),
                "mae": row.mae,
                "rmse": row.rmse,
                "r2": row.r2,
                "mape_pct": np.nan,
                "wape_pct": row.wape_pct,
            }
        )
    radius_refine = read_csv(
        REPO_ROOT / "newtests" / "targeted_rebuild" / "results" / "gat_gru_radius_refine" / "radius_refine_summary.csv"
    )
    for row in radius_refine.itertuples(index=False):
        rows.append(
            {
                "experiment_family": "graph_gat_gru_radius_refine",
                "experiment_name": row.config_name,
                "crop": row.crop,
                "horizon_days": 15,
                "validation_rows": getattr(row, "validation_windows", np.nan),
                "validation_start": getattr(row, "validation_start", np.nan),
                "validation_end": getattr(row, "validation_end", np.nan),
                "mae": row.mae,
                "rmse": row.rmse,
                "r2": row.r2,
                "mape_pct": np.nan,
                "wape_pct": row.wape_pct,
            }
        )
    ablation = read_csv(
        REPO_ROOT / "newtests" / "targeted_rebuild" / "results" / "gat_gru_ablations" / "ablation_summary.csv"
    )
    for row in ablation.itertuples(index=False):
        rows.append(
            {
                "experiment_family": "graph_gat_gru_ablation",
                "experiment_name": row.config_name,
                "crop": row.crop,
                "horizon_days": 15,
                "validation_rows": getattr(row, "validation_windows", np.nan),
                "validation_start": getattr(row, "validation_start", np.nan),
                "validation_end": getattr(row, "validation_end", np.nan),
                "mae": row.mae,
                "rmse": row.rmse,
                "r2": row.r2,
                "mape_pct": np.nan,
                "wape_pct": row.wape_pct,
            }
        )
    wheat_refine_summary = json.loads(
        (REPO_ROOT / "newtests" / "tomato_wheat_15d_followups" / "results" / "wheat_graph_refine_longwindow" / "graph_training_summary.json").read_text()
    )
    wheat = wheat_refine_summary["wheat"]
    rows.append(
        {
            "experiment_family": "graph_gat_gru_followup",
            "experiment_name": "wheat_graph_refine_longwindow",
            "crop": "wheat",
            "horizon_days": 15,
            "validation_rows": wheat["validation_windows"],
            "validation_start": wheat["validation_start"],
            "validation_end": wheat["validation_end"],
            "mae": wheat["mae"],
            "rmse": wheat["rmse"],
            "r2": wheat["r2"],
            "mape_pct": np.nan,
            "wape_pct": wheat["wape_pct"],
        }
    )
    return pd.DataFrame(rows)


def method_catalog() -> pd.DataFrame:
    rows = [
        {
            "method_family": "Explicit baselines",
            "what_it_is": "Naive forecast rules using lagged or smoothed price directly.",
            "tests_done": "previous_day_price, current_price, roll_mean_7, roll_mean_28 across 1..15 days for all 4 crops.",
            "primary_outputs": "targeted_rebuild/results/baseline_metrics/",
        },
        {
            "method_family": "Same-day persistence baseline",
            "what_it_is": "New report-only check: predict current-day observed price from previous day's price_lag_1.",
            "tests_done": "Computed on the same trailing 90-day validation windows used for the 1-day experiments.",
            "primary_outputs": "reports/same_day_previous_price_baseline.csv",
        },
        {
            "method_family": "Simple numeric HistGB",
            "what_it_is": "Single per-crop HistGradientBoosting regressor over numeric engineered features.",
            "tests_done": "1..15 days for all 4 crops.",
            "primary_outputs": "targeted_rebuild/results/simple_numeric_metrics/",
        },
        {
            "method_family": "Anchored HistGB",
            "what_it_is": "Predict a correction around an anchor such as current price or rolling mean, then invert back.",
            "tests_done": "1..15 days for all 4 crops; target encodings revisited later in GAT-GRU ablations.",
            "primary_outputs": "targeted_rebuild/results/anchored_histgb_metrics/",
        },
        {
            "method_family": "Cross-crop HistGB",
            "what_it_is": "Anchored HistGB with another crop's daily price features added based on national correlation pairing.",
            "tests_done": "1..15 days for all 4 crops plus pairing analysis.",
            "primary_outputs": "targeted_rebuild/results/cross_crop_histgb_metrics/",
        },
        {
            "method_family": "Classical linear models",
            "what_it_is": "LinearRegression, Ridge, and ElasticNet on the same tabular engineered feature space.",
            "tests_done": "Focused 1-day and 15-day runs for all 4 crops.",
            "primary_outputs": "targeted_rebuild/results/classical_ml_metrics/",
        },
        {
            "method_family": "Density / volume variants",
            "what_it_is": "HistGB variants that split markets by graph density or output-volume clusters.",
            "tests_done": "15-day specialized runs for all 4 crops.",
            "primary_outputs": "targeted_rebuild/results/density_variant_metrics/ and volume_variant_metrics/",
        },
        {
            "method_family": "Graph-style non-deep models",
            "what_it_is": "Neighbor-blend baseline and graph-augmented HistGB using threshold-based neighbor signals.",
            "tests_done": "15-day threshold sweeps at 75km / 150km / 300km for all 4 crops.",
            "primary_outputs": "targeted_rebuild/results/graph_neighbor_blend_metrics/ and graph_histgb_metrics/",
        },
        {
            "method_family": "Deep graph models",
            "what_it_is": "GraphWaveNet+ and GAT-GRU sequence models over mandi graphs.",
            "tests_done": "15-day runs for all 4 crops.",
            "primary_outputs": "targeted_rebuild/results/graph_wavenet_plus_mps/ and graph_gat_gru_mps/",
        },
        {
            "method_family": "GAT-GRU ablations",
            "what_it_is": "Target-mode, graph-mode, k-neighbor, edge-weight, and radius-threshold sweeps inside the deep graph model.",
            "tests_done": "15-day ablations plus radius sweeps 75/150/300km and onion/tomato low-radius refinement 10/25/50km.",
            "primary_outputs": "targeted_rebuild/results/gat_gru_ablations/, gat_gru_radius_ablation/, gat_gru_radius_refine/",
        },
        {
            "method_family": "Focused tuned tabular expansion",
            "what_it_is": "XGBoost, LightGBM, and ExtraTrees on 1-day and 15-day horizons with compact deterministic tuning.",
            "tests_done": "XGBoost 2 configs, LightGBM 2 configs, ExtraTrees 4 configs across all 4 crops at 1d and 15d.",
            "primary_outputs": "focused_1d_15d_execution/results/expanded_model_metrics/ and tuning_metrics/",
        },
        {
            "method_family": "Focused TCN expansion",
            "what_it_is": "Temporal convolutional network over windowed price / arrival / calendar sequences.",
            "tests_done": "Two tuned configs: 21x32 and 28x48 window/channel setups for all 4 crops at 1d and 15d.",
            "primary_outputs": "focused_1d_15d_execution/results/expanded_model_metrics/ and tuning_metrics/",
        },
        {
            "method_family": "Tomato / wheat 15d follow-ups",
            "what_it_is": "Targeted local-anchor, weighted-anchor, regime, long-window TCN, and wheat graph-refinement tests.",
            "tests_done": "Tomato local delta_roll14, delta_roll28, weighted 70/30 anchor, horizon-focused local; wheat regime delta_current, weighted 70/30, horizon-focused regime; TCN 42x64 and 56x96; wheat graph refine.",
            "primary_outputs": "tomato_wheat_15d_followups/results/",
        },
    ]
    return pd.DataFrame(rows)


def build_report() -> str:
    persistence = compute_same_day_persistence()

    targeted_root = REPO_ROOT / "newtests" / "targeted_rebuild" / "results"
    focused_root = REPO_ROOT / "newtests" / "focused_1d_15d_execution" / "results"
    followup_root = REPO_ROOT / "newtests" / "tomato_wheat_15d_followups" / "results"

    baseline = normalize_experiment_name(read_csv(targeted_root / "baseline_metrics" / "baseline_metrics.csv"))
    simple = normalize_experiment_name(read_csv(targeted_root / "simple_numeric_metrics" / "simple_numeric_metrics.csv"))
    anchored = normalize_experiment_name(read_csv(targeted_root / "anchored_histgb_metrics" / "anchored_histgb_metrics.csv"))
    cross_crop = normalize_experiment_name(read_csv(targeted_root / "cross_crop_histgb_metrics" / "cross_crop_histgb_metrics.csv"))
    classical = normalize_experiment_name(read_csv(targeted_root / "classical_ml_metrics" / "classical_ml_metrics.csv"))
    final15 = normalize_experiment_name(read_csv(targeted_root / "final_comparison_15d.csv"))
    focused = normalize_experiment_name(read_csv(focused_root / "expanded_model_metrics" / "expanded_model_metrics.csv"))
    focused_tuning = normalize_experiment_name(read_csv(focused_root / "tuning_metrics" / "tuning_metrics.csv"))
    followup = normalize_experiment_name(read_csv(followup_root / "followup_metrics" / "followup_metrics.csv"))
    followup_tcn = normalize_experiment_name(read_csv(followup_root / "followup_tcn_metrics" / "followup_tcn_metrics.csv"))
    graph_reference = normalize_experiment_name(read_csv(followup_root / "graph_reference_metrics" / "graph_reference_metrics.csv"))
    graph_extra = normalize_experiment_name(graph_extra_rows())

    one_day_existing = pd.concat([baseline, simple, anchored, cross_crop, classical], ignore_index=True)
    one_day_existing_best = best_by_crop(one_day_existing, 1)
    one_day_focused_best = best_by_crop(focused, 1)
    one_day_final = best_by_crop(pd.concat([one_day_existing, focused], ignore_index=True), 1)

    fifteen_existing = pd.concat([final15, classical, graph_extra], ignore_index=True)
    fifteen_existing_best = best_by_crop(fifteen_existing, 15)
    fifteen_focused_best = best_by_crop(focused, 15)
    fifteen_followup_best = best_by_crop(
        pd.concat([followup, followup_tcn, graph_reference, graph_extra], ignore_index=True),
        15,
    )
    fifteen_final = best_by_crop(
        pd.concat([fifteen_existing, focused, followup, followup_tcn, graph_reference], ignore_index=True),
        15,
    )

    previous_day_1d = (
        baseline[(baseline["experiment_name"] == "previous_day_price") & (baseline["horizon_days"] == 1)][
            ["crop", "r2", "wape_pct"]
        ]
        .rename(columns={"r2": "prev_day_to_tomorrow_r2", "wape_pct": "prev_day_to_tomorrow_wape"})
        .sort_values("crop")
    )
    previous_day_15d = (
        baseline[(baseline["experiment_name"] == "previous_day_price") & (baseline["horizon_days"] == 15)][
            ["crop", "r2", "wape_pct"]
        ]
        .rename(columns={"r2": "prev_day_to_day15_r2", "wape_pct": "prev_day_to_day15_wape"})
        .sort_values("crop")
    )
    current_day_compare = (
        persistence[["crop", "validation_rows", "r2", "wape_pct"]]
        .rename(columns={"r2": "prev_day_to_current_r2", "wape_pct": "prev_day_to_current_wape"})
        .merge(previous_day_1d, on="crop", how="left")
        .merge(previous_day_15d, on="crop", how="left")
        .sort_values("crop")
    )

    focused_pivot = (
        focused.pivot_table(index=["crop", "horizon_days"], columns="experiment_name", values="r2", aggfunc="first")
        .reset_index()
        .rename_axis(None, axis=1)
        .sort_values(["horizon_days", "crop"])
    )
    for col in ["xgboost_tuned", "lightgbm_tuned", "extratrees_tuned", "tcn_tuned"]:
        if col not in focused_pivot.columns:
            focused_pivot[col] = np.nan

    followup_all = pd.concat([followup, followup_tcn, graph_reference], ignore_index=True)
    followup_all = followup_all.sort_values(["crop", "r2"], ascending=[True, False]).reset_index(drop=True)

    graph_family_rows = pd.concat([final15, graph_extra, graph_reference], ignore_index=True)
    graph_family_rows = graph_family_rows[
        graph_family_rows["experiment_name"].isin(
            [
                "graph_wavenet_plus_mps",
                "graph_gat_gru_mps",
                "target_mode__delta_roll28",
                "full_graph_delta_current_default",
                "onion__radius_25km",
                "potato__radius_300km",
                "tomato_radius_10km_delta_roll28_existing",
                "wheat_graph_refine_longwindow",
            ]
        )
    ].copy()
    graph_family_rows = (
        graph_family_rows.sort_values(["crop", "r2"], ascending=[True, False])
        .drop_duplicates(["crop", "experiment_name"], keep="first")
        .reset_index(drop=True)
    )

    method_rows = method_catalog()

    final_1d_table: list[dict[str, object]] = []
    for crop in sorted(one_day_final["crop"].unique()):
        final_1d_table.append(
            {
                "crop": crop,
                "best_existing_before_expansion": r2_label(best_row(one_day_existing, crop, 1)),
                "best_focused_new_model": r2_label(best_row(focused, crop, 1)),
                "final_best_1d": r2_label(best_row(pd.concat([one_day_existing, focused], ignore_index=True), crop, 1)),
            }
        )
    final_15d_table: list[dict[str, object]] = []
    all_15 = pd.concat([fifteen_existing, focused, followup, followup_tcn, graph_reference], ignore_index=True)
    for crop in sorted(fifteen_final["crop"].unique()):
        followup_subset = pd.concat([followup, followup_tcn, graph_reference, graph_extra], ignore_index=True)
        followup_crop = followup_subset[(followup_subset["crop"] == crop) & (followup_subset["horizon_days"] == 15)]
        followup_label = "n/a"
        if not followup_crop.empty:
            followup_label = r2_label(followup_crop.sort_values("r2", ascending=False).iloc[0])
        final_15d_table.append(
            {
                "crop": crop,
                "best_existing_before_expansion": r2_label(best_row(fifteen_existing, crop, 15)),
                "best_focused_new_model": r2_label(best_row(focused, crop, 15)),
                "best_followup_model": followup_label,
                "final_best_15d": r2_label(best_row(all_15, crop, 15)),
            }
        )

    device_status = json.loads((focused_root / "environment_status" / "device_status.json").read_text())
    device_note = (
        f"Focused TCN runs used `{device_status.get('selected_device', 'cpu')}`. "
        f"MPS built={device_status.get('mps_built')}, available={device_status.get('mps_available')}."
    )

    lines: list[str] = []
    lines.append("# PREPARE Final Master Report")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "This report consolidates the full experiment history under `newtests/`: the original `targeted_rebuild` sweep, "
        "the later focused `1d` / `15d` tuned-model expansion, and the tomato / wheat 15-day follow-up tests."
    )
    lines.append("")
    lines.append("Primary outputs referenced here:")
    lines.append("")
    lines.append("- `newtests/targeted_rebuild/results/`")
    lines.append("- `newtests/focused_1d_15d_execution/results/`")
    lines.append("- `newtests/tomato_wheat_15d_followups/results/`")
    lines.append(f"- `newtests/reports/{PERSISTENCE_PATH.name}`")
    lines.append("")
    lines.append("## Method Catalog")
    lines.append("")
    lines.append(markdown_table(method_rows, ["method_family", "what_it_is", "tests_done", "primary_outputs"]))
    lines.append("")
    lines.append("## Baseline Definitions")
    lines.append("")
    lines.append("- `previous_day_price`: use `price_lag_1` to predict the future target at the selected horizon.")
    lines.append("- `current_price`: use today's `Modal_Price_CausalFilled` to predict the future target at the selected horizon.")
    lines.append("- `roll_mean_7` / `roll_mean_28`: use trailing rolling mean price anchors.")
    lines.append(
        "- `same_day_previous_price`: new in this report. Predict today's observed `Modal_Price` directly from yesterday's `price_lag_1` on the same trailing validation windows used for the 1-day experiments."
    )
    lines.append("")
    lines.append("### New Same-Day Persistence Baseline")
    lines.append("")
    lines.append(markdown_table(current_day_compare))
    lines.append("")
    lines.append(
        "Interpretation: `prev_day_to_current_r2` is the corrected baseline you asked for. "
        "It is not the same metric as the old horizon-specific `previous_day_price` baseline, which predicts tomorrow or day-15 from yesterday."
    )
    lines.append("")
    lines.append("## One-Day Results")
    lines.append("")
    lines.append(
        "One-day learned-model comparisons combine the original baselines / HistGB / cross-crop / classical ML runs "
        "with the later tuned `xgboost`, `lightgbm`, `extratrees`, and `tcn` expansion."
    )
    lines.append("")
    lines.append(markdown_table(pd.DataFrame(final_1d_table)))
    lines.append("")
    lines.append("Focused 1-day / 15-day tuned expansion R2 values:")
    lines.append("")
    lines.append(
        markdown_table(
            focused_pivot,
            ["crop", "horizon_days", "xgboost_tuned", "lightgbm_tuned", "extratrees_tuned", "tcn_tuned"],
        )
    )
    lines.append("")
    lines.append("## Fifteen-Day Results")
    lines.append("")
    lines.append(
        "Fifteen-day comparisons combine the original `targeted_rebuild` final comparison, classical ML, graph ablations, "
        "radius sweeps / refinements, the focused tuned-model expansion, and the tomato / wheat follow-up wave."
    )
    lines.append("")
    lines.append(markdown_table(pd.DataFrame(final_15d_table)))
    lines.append("")
    lines.append("Selected graph-family 15-day checkpoints:")
    lines.append("")
    lines.append(markdown_table(graph_family_rows[["crop", "experiment_name", "r2", "wape_pct"]]))
    lines.append("")
    lines.append("Tomato / wheat 15-day follow-up results:")
    lines.append("")
    lines.append(markdown_table(followup_all[["crop", "experiment_name", "r2", "wape_pct"]]))
    lines.append("")
    lines.append("## What Performed Best")
    lines.append("")
    best_1d = pd.DataFrame(final_1d_table)[["crop", "final_best_1d"]]
    best_15d = pd.DataFrame(final_15d_table)[["crop", "final_best_15d"]]
    lines.append("Final best completed `1d` result by crop:")
    lines.append("")
    lines.append(markdown_table(best_1d))
    lines.append("")
    lines.append("Final best completed `15d` result by crop:")
    lines.append("")
    lines.append(markdown_table(best_15d))
    lines.append("")
    lines.append("## Key Findings")
    lines.append("")
    lines.append("- `1d`: onion stayed baseline-dominated, potato favored classical linear / elastic-net style models, tomato favored the focused TCN, and wheat favored focused ExtraTrees.")
    lines.append("- `15d`: the graph family still won overall. Onion, potato, and tomato favored `delta_roll28` graph targets; wheat favored `delta_current` on the full graph.")
    lines.append("- Focused tuned tree models were useful, especially `extratrees_tuned`, but they did not displace the graph frontier at 15 days.")
    lines.append("- The tomato / wheat follow-up tabular and long-window TCN tests did not beat the existing 15-day graph winners.")
    lines.append(f"- Runtime note: {device_note}")
    lines.append("")
    lines.append("## Files Created By This Report")
    lines.append("")
    lines.append(f"- Report: `newtests/reports/{REPORT_PATH.name}`")
    lines.append(f"- Corrected baseline CSV: `newtests/reports/{PERSISTENCE_PATH.name}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ensure_venv()
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    report = build_report()
    REPORT_PATH.write_text(report)
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {PERSISTENCE_PATH}")


if __name__ == "__main__":
    main()
