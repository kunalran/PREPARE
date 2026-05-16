from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = REPO_ROOT / "newtests" / os.environ.get("NEWTESTS_RUN_DIR", "focused_1d_15d_expansion")
RESULTS = RUN_ROOT / "results"
REFERENCE_ROOT = REPO_ROOT / "newtests" / "targeted_rebuild" / "results"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def ensure_columns(frame: pd.DataFrame, source_file: str) -> pd.DataFrame:
    copy = frame.copy()
    if "display_name" not in copy.columns:
        copy["display_name"] = copy["experiment_name"]
    if "graph_bucket" not in copy.columns:
        copy["graph_bucket"] = "non_graph"
    if "source_file" not in copy.columns:
        copy["source_file"] = source_file
    return copy


def load_reference_h1() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    paths = [
        REFERENCE_ROOT / "baseline_metrics" / "baseline_metrics.csv",
        REFERENCE_ROOT / "simple_numeric_metrics" / "simple_numeric_metrics.csv",
        REFERENCE_ROOT / "anchored_histgb_metrics" / "anchored_histgb_metrics.csv",
        REFERENCE_ROOT / "cross_crop_histgb_metrics" / "cross_crop_histgb_metrics.csv",
        REFERENCE_ROOT / "classical_ml_metrics" / "classical_ml_metrics.csv",
    ]
    for path in paths:
        if not path.exists():
            continue
        frames.append(ensure_columns(read_csv(path), str(path.relative_to(REFERENCE_ROOT))))
    merged = pd.concat(frames, ignore_index=True)
    return merged[merged["horizon_days"] == 1].copy()


def load_reference_h15() -> pd.DataFrame:
    path = REFERENCE_ROOT / "final_comparison_15d.csv"
    return ensure_columns(read_csv(path), str(path.relative_to(REFERENCE_ROOT)))


def load_new_models() -> pd.DataFrame:
    path = RESULTS / "expanded_model_metrics" / "expanded_model_metrics.csv"
    frame = ensure_columns(read_csv(path), "expanded_model_metrics/expanded_model_metrics.csv")
    return frame


def load_new_baselines() -> pd.DataFrame:
    path = RESULTS / "baseline_metrics" / "baseline_metrics.csv"
    return ensure_columns(read_csv(path), "baseline_metrics/baseline_metrics.csv")


def best_rows(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(["crop", "horizon_days", "r2"], ascending=[True, True, False])
    return ordered.groupby(["crop", "horizon_days"], as_index=False).first()


def save_frame(name: str, frame: pd.DataFrame) -> Path:
    out_dir = RESULTS / "final_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    frame.to_csv(path, index=False)
    return path


def build_summary(h1_frame: pd.DataFrame, h15_frame: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("# Focused 1-Day / 15-Day Model Expansion")
    lines.append("")

    def append_horizon_section(title: str, frame: pd.DataFrame, horizon: int) -> None:
        lines.append(f"## {title}")
        lines.append("")
        for crop in sorted(frame["crop"].unique()):
            subset = frame[frame["crop"] == crop].sort_values("r2", ascending=False)
            best_overall = subset.iloc[0]
            best_new = subset[subset["source_file"].str.contains("expanded_model_metrics", na=False)]
            best_ref = subset[~subset["source_file"].str.contains("expanded_model_metrics", na=False)]
            if best_new.empty or best_ref.empty:
                continue
            new_row = best_new.iloc[0]
            ref_row = best_ref.iloc[0]
            delta = float(new_row["r2"]) - float(ref_row["r2"])
            relation = "beat" if delta > 0 else "did not beat"
            lines.append(
                f"- {crop}: best new `{new_row['display_name']}` R2={new_row['r2']:.4f} "
                f"{relation} best existing `{ref_row['display_name']}` R2={ref_row['r2']:.4f} "
                f"(delta {delta:+.4f}); overall winner `{best_overall['display_name']}`"
            )
        lines.append("")

    append_horizon_section("One-Day Comparison", h1_frame, 1)
    append_horizon_section("Fifteen-Day Comparison", h15_frame, 15)
    lines.append("## Outputs")
    lines.append("")
    lines.append("- `results/final_comparison/horizon_1_full.csv`")
    lines.append("- `results/final_comparison/horizon_15_full.csv`")
    lines.append("- `results/final_comparison/horizon_1_best_by_crop.csv`")
    lines.append("- `results/final_comparison/horizon_15_best_by_crop.csv`")
    return "\n".join(lines)


def main() -> None:
    h1_reference = load_reference_h1()
    h15_reference = load_reference_h15()
    new_models = load_new_models()
    new_baselines = load_new_baselines()

    h1 = pd.concat(
        [
            h1_reference,
            new_baselines[new_baselines["horizon_days"] == 1],
            new_models[new_models["horizon_days"] == 1],
        ],
        ignore_index=True,
    )
    h15 = pd.concat(
        [
            h15_reference,
            new_baselines[new_baselines["horizon_days"] == 15],
            new_models[new_models["horizon_days"] == 15],
        ],
        ignore_index=True,
    )

    h1 = h1.sort_values(["crop", "r2"], ascending=[True, False]).reset_index(drop=True)
    h15 = h15.sort_values(["crop", "r2"], ascending=[True, False]).reset_index(drop=True)

    save_frame("horizon_1_full.csv", h1)
    save_frame("horizon_15_full.csv", h15)
    save_frame("horizon_1_best_by_crop.csv", best_rows(h1))
    save_frame("horizon_15_best_by_crop.csv", best_rows(h15))

    summary_path = RESULTS / "final_comparison" / "summary.md"
    summary_path.write_text(build_summary(h1, h15), encoding="utf-8")

    payload = {
        "run_root": str(RUN_ROOT),
        "reference_root": str(REFERENCE_ROOT),
        "summary_path": str(summary_path),
    }
    (RESULTS / "final_comparison" / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
