from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / os.environ.get("NEWTESTS_RUN_DIR", "targeted_rebuild")
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def set_runtime() -> None:
    cache_dir = ROOT / ".cache" / "matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    sns.set_theme(style="whitegrid")


def read_csv(relative: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / relative)


def save_plot(name: str) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / name
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def plot_inventory(inventory: pd.DataFrame) -> Path:
    plt.figure(figsize=(8, 5))
    order = inventory.sort_values("mandis", ascending=False)["crop"]
    sns.barplot(data=inventory, x="crop", y="mandis", order=order, palette="crest")
    plt.title("Mandi Count by Crop")
    plt.xlabel("Crop")
    plt.ylabel("Mandis")
    return save_plot("mandi_count_by_crop.png")


def plot_horizon_curves(metrics: pd.DataFrame) -> Path:
    best = (
        metrics.sort_values(["crop", "horizon_days", "r2"], ascending=[True, True, False])
        .groupby(["crop", "horizon_days"], as_index=False)
        .first()
    )
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=best,
        x="horizon_days",
        y="r2",
        hue="experiment_name",
        style="crop",
        markers=False,
    )
    plt.title("Best Core Model Family by Horizon")
    plt.xlabel("Forecast Horizon (days)")
    plt.ylabel("Validation R2")
    return save_plot("core_model_horizon_curves.png")


def plot_specialized_15d(metrics_15d: pd.DataFrame) -> Path:
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=metrics_15d.sort_values(["crop", "r2"], ascending=[True, False]),
        x="crop",
        y="r2",
        hue="experiment_name",
        palette="viridis",
    )
    plt.title("15-Day Specialized and Graph Experiments")
    plt.xlabel("Crop")
    plt.ylabel("Validation R2")
    return save_plot("specialized_graph_15d.png")


def plot_graph_thresholds(graph_metrics: pd.DataFrame) -> Path:
    graph_metrics = graph_metrics.copy()
    graph_metrics["threshold_label"] = graph_metrics["threshold_km"].astype(int).astype(str) + "km"
    pivot = graph_metrics.pivot_table(
        index=["crop", "experiment_name"],
        columns="threshold_label",
        values="r2",
        aggfunc="max",
    )
    plt.figure(figsize=(9, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="mako")
    plt.title("Graph Threshold Sweep R2")
    plt.xlabel("Distance Threshold")
    plt.ylabel("Crop / Model")
    return save_plot("graph_threshold_heatmap.png")


def build_summary_tables() -> dict[str, pd.DataFrame]:
    baseline = read_csv("baseline_metrics/baseline_metrics.csv")
    simple = read_csv("simple_numeric_metrics/simple_numeric_metrics.csv")
    anchored = read_csv("anchored_histgb_metrics/anchored_histgb_metrics.csv")
    cross_crop = read_csv("cross_crop_histgb_metrics/cross_crop_histgb_metrics.csv")
    density = read_csv("density_variant_metrics/density_variant_metrics.csv")
    volume = read_csv("volume_variant_metrics/volume_variant_metrics.csv")
    graph_neighbor = read_csv("graph_neighbor_blend_metrics/graph_neighbor_blend_metrics.csv")
    graph_histgb = read_csv("graph_histgb_metrics/graph_histgb_metrics.csv")

    core = pd.concat([baseline, simple, anchored, cross_crop], ignore_index=True)
    specialized = pd.concat([density, volume, graph_neighbor, graph_histgb], ignore_index=True)

    best_horizon = (
        core.sort_values(["crop", "horizon_days", "r2"], ascending=[True, True, False])
        .groupby(["crop", "horizon_days"], as_index=False)
        .first()
    )
    best_15d = (
        pd.concat([core, specialized], ignore_index=True)
        .query("horizon_days == 15")
        .sort_values(["crop", "r2"], ascending=[True, False])
        .groupby("crop", as_index=False)
        .first()
    )

    REPORTS.mkdir(parents=True, exist_ok=True)
    best_horizon.to_csv(REPORTS / "best_model_by_horizon.csv", index=False)
    best_15d.to_csv(REPORTS / "best_model_15d.csv", index=False)
    return {
        "baseline": baseline,
        "simple": simple,
        "anchored": anchored,
        "cross_crop": cross_crop,
        "specialized": specialized,
        "core": core,
        "best_horizon": best_horizon,
        "best_15d": best_15d,
    }


def best_row(df: pd.DataFrame, crop: str, horizon: int) -> pd.Series:
    subset = df[(df["crop"] == crop) & (df["horizon_days"] == horizon)].sort_values("r2", ascending=False)
    return subset.iloc[0]


def report_text(tables: dict[str, pd.DataFrame], plot_paths: dict[str, Path]) -> str:
    inventory = read_csv("inventory/crop_inventory.csv")
    density = read_csv("inventory/graph_density_summary.csv")
    pairings = read_csv("cross_crop_pairings.csv")

    crops = sorted(inventory["crop"].unique())
    lines: list[str] = []
    lines.append("# PREPARE New Tests Report")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "This report summarizes a fresh experiment run on the imputed hourly crop data in "
        "`final_data_hourly_dow_imputed/`. All outputs were generated inside `newtests/` from "
        "a dedicated virtual environment."
    )
    lines.append("")
    lines.append("## Dataset Inventory")
    lines.append("")
    for row in inventory.itertuples(index=False):
        lines.append(
            f"- {row.crop}: {row.mandis} mandis, {row.rows:,} rows, date range "
            f"{row.date_min} to {row.date_max}, non-null prices {row.non_null_price_rows:,}"
        )
    lines.append("")
    lines.append(f"![Mandi count by crop]({plot_paths['inventory'].name})")
    lines.append("")
    lines.append("## Cross-Crop Pairing Rule")
    lines.append("")
    lines.append(
        "Cross-crop features were added by pairing each crop with the other crop that had the "
        "highest daily national-mean price correlation over overlapping dates."
    )
    lines.append("")
    for row in pairings.itertuples(index=False):
        lines.append(
            f"- {row.crop} paired with {row.paired_crop} "
            f"(daily national correlation {row.daily_national_price_corr:.3f})"
        )
    lines.append("")
    lines.append("## Horizon-Wise Core Results")
    lines.append("")
    lines.append(
        "Core comparisons include explicit baselines, a simple numeric HistGB model, an anchored "
        "per-crop HistGB variant, and the cross-crop augmented anchored HistGB model."
    )
    lines.append("")
    lines.append(f"![Core horizon curves]({plot_paths['core_curves'].name})")
    lines.append("")
    lines.append("Best model by crop at 15 days among the core families:")
    lines.append("")
    for crop in crops:
        row = best_row(tables["core"], crop, 15)
        lines.append(
            f"- {crop}: {row['experiment_name']} with R2={row['r2']:.4f}, "
            f"WAPE={row['wape_pct']:.2f}%"
        )
    lines.append("")
    lines.append("## Specialized 15-Day Tests")
    lines.append("")
    lines.append(
        "The 15-day focused tests add volume grouping, dense-vs-sparse mandi differentiation, "
        "and two graph-style threshold experiments."
    )
    lines.append("")
    lines.append(f"![15-day specialized results]({plot_paths['specialized'].name})")
    lines.append("")
    lines.append(f"![Graph threshold heatmap]({plot_paths['graph_heatmap'].name})")
    lines.append("")
    lines.append("Best 15-day overall result by crop:")
    lines.append("")
    for row in tables["best_15d"].itertuples(index=False):
        extra = ""
        if "threshold_km" in tables["best_15d"].columns and pd.notna(getattr(row, "threshold_km", None)):
            extra = f", threshold={int(row.threshold_km)}km"
        lines.append(
            f"- {row.crop}: {row.experiment_name} with R2={row.r2:.4f}, "
            f"WAPE={row.wape_pct:.2f}%{extra}"
        )
    lines.append("")
    lines.append("## Conclusions")
    lines.append("")

    anchored_15 = tables["anchored"].query("horizon_days == 15")
    cross_15 = tables["cross_crop"].query("horizon_days == 15")
    graph_15 = tables["specialized"].query("experiment_family == 'graph_model'")

    anchored_better = 0
    cross_better = 0
    graph_better = 0
    for crop in crops:
        base_best = best_row(tables["baseline"], crop, 15)["r2"]
        anchored_best = best_row(anchored_15, crop, 15)["r2"]
        cross_best = best_row(cross_15, crop, 15)["r2"]
        graph_best = best_row(graph_15, crop, 15)["r2"]
        if anchored_best > base_best:
            anchored_better += 1
        if cross_best > anchored_best:
            cross_better += 1
        if graph_best > anchored_best:
            graph_better += 1

    lines.append(
        f"- The anchored HistGB variant beat the best explicit baseline on {anchored_better} of "
        f"{len(crops)} crops at 15 days."
    )
    lines.append(
        f"- Cross-crop features improved on the anchored model for {cross_better} of {len(crops)} crops at 15 days."
    )
    lines.append(
        f"- The graph-style threshold models beat the anchored model for {graph_better} of {len(crops)} crops at 15 days."
    )

    densest = density.sort_values("mean_neighbors", ascending=False).iloc[0]
    sparsest = density.sort_values("mean_neighbors", ascending=True).iloc[0]
    lines.append(
        f"- The densest mandi graph setting observed was {densest.crop} at {int(densest.threshold_km)}km "
        f"with mean neighbors {densest.mean_neighbors:.1f}; the sparsest was {sparsest.crop} at "
        f"{int(sparsest.threshold_km)}km with mean neighbors {sparsest.mean_neighbors:.1f}."
    )
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- `newtests/results/`: raw experiment CSV and JSON outputs")
    lines.append("- `newtests/reports/best_model_by_horizon.csv`: best core model per crop and horizon")
    lines.append("- `newtests/reports/best_model_15d.csv`: best overall 15-day result per crop")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    set_runtime()
    REPORTS.mkdir(parents=True, exist_ok=True)
    inventory = read_csv("inventory/crop_inventory.csv")
    tables = build_summary_tables()

    plot_paths = {
        "inventory": plot_inventory(inventory),
        "core_curves": plot_horizon_curves(tables["core"]),
        "specialized": plot_specialized_15d(
            pd.concat([tables["specialized"], tables["core"].query("horizon_days == 15")], ignore_index=True)
        ),
        "graph_heatmap": plot_graph_thresholds(tables["specialized"].query("experiment_family == 'graph_model'")),
    }

    report_md = REPORTS / "prepare_newtests_report.md"
    report_md.write_text(report_text(tables, plot_paths), encoding="utf-8")
    summary_json = REPORTS / "report_artifacts.json"
    summary_json.write_text(
        json.dumps({key: str(value) for key, value in plot_paths.items()}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {report_md}")


if __name__ == "__main__":
    main()
