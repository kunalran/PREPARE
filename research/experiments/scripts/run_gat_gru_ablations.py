from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = REPO_ROOT / "training" / "train_graph_gat_gru.py"
PYTHON_BIN = REPO_ROOT / "newtests" / "venv" / "bin" / "python"
DATA_ALIAS = REPO_ROOT / "newtests" / "targeted_rebuild" / "data_alias"
OUTPUT_ROOT = REPO_ROOT / "newtests" / "targeted_rebuild" / "results" / "gat_gru_ablations"
LOG_DIR = REPO_ROOT / "newtests" / "targeted_rebuild" / "logs" / "gat_gru_ablations"
REFERENCE_SUMMARY = REPO_ROOT / "newtests" / "targeted_rebuild" / "results" / "graph_gat_gru_mps" / "graph_training_summary.json"


def base_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    env.setdefault("LOKY_MAX_CPU_COUNT", "8")
    return env


def base_args() -> list[str]:
    return [
        str(PYTHON_BIN),
        str(TRAIN_SCRIPT),
        "--data-dir",
        str(DATA_ALIAS),
        "--horizon",
        "15",
        "--epochs",
        "8",
        "--batch-size",
        "4",
        "--device",
        "mps",
    ]


def run_one(name: str, extra_args: list[str]) -> None:
    out_dir = OUTPUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{name}.log"
    cmd = base_args() + ["--output-dir", str(out_dir)] + extra_args
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("COMMAND:\n")
        log_file.write(" ".join(cmd))
        log_file.write("\n\n")
        log_file.flush()
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=base_env(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {proc.returncode}. See {log_path}")


def load_summary(path: Path, sweep: str, config_name: str) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for crop, metrics in data.items():
        row = dict(metrics)
        row["crop"] = crop
        row["sweep"] = sweep
        row["config_name"] = config_name
        rows.append(row)
    return rows


def write_aggregate() -> None:
    rows: list[dict[str, object]] = []
    if REFERENCE_SUMMARY.exists():
        rows.extend(load_summary(REFERENCE_SUMMARY, "reference", "full_graph_delta_current_default"))
    for summary_path in OUTPUT_ROOT.glob("*/graph_training_summary.json"):
        sweep = summary_path.parent.name.split("__", 1)[0]
        config_name = summary_path.parent.name
        rows.extend(load_summary(summary_path, sweep, config_name))
    if not rows:
        return
    df = pd.DataFrame(rows)
    df = df.sort_values(["crop", "sweep", "config_name"])
    df.to_csv(OUTPUT_ROOT / "ablation_summary.csv", index=False)


def main() -> None:
    if not PYTHON_BIN.exists():
        raise FileNotFoundError(f"Missing venv python: {PYTHON_BIN}")
    if not DATA_ALIAS.exists():
        raise FileNotFoundError(f"Missing data alias folder: {DATA_ALIAS}")

    jobs: list[tuple[str, list[str]]] = [
        ("graph_mode__geo_only", ["--graph-mode", "geo_only"]),
        ("graph_mode__corr_only", ["--graph-mode", "corr_only"]),
        ("graph_mode__temporal_only", ["--graph-mode", "temporal_only"]),
        ("graph_mode__shuffled_graph", ["--graph-mode", "shuffled_graph"]),
        ("target_mode__delta_roll7", ["--target-mode", "delta_roll7"]),
        ("target_mode__delta_roll28", ["--target-mode", "delta_roll28"]),
        ("k_neighbors__k6", ["--k-neighbors", "6"]),
        ("k_neighbors__k14", ["--k-neighbors", "14"]),
        ("k_neighbors__k18", ["--k-neighbors", "18"]),
        (
            "edge_weights__corr_heavy",
            [
                "--corr-edge-weight",
                "0.70",
                "--geo-edge-weight",
                "0.15",
                "--state-edge-weight",
                "0.05",
                "--cluster-edge-weight",
                "0.10",
            ],
        ),
        (
            "edge_weights__geo_heavy",
            [
                "--corr-edge-weight",
                "0.35",
                "--geo-edge-weight",
                "0.40",
                "--state-edge-weight",
                "0.10",
                "--cluster-edge-weight",
                "0.15",
            ],
        ),
        (
            "edge_weights__state_cluster_heavy",
            [
                "--corr-edge-weight",
                "0.35",
                "--geo-edge-weight",
                "0.20",
                "--state-edge-weight",
                "0.20",
                "--cluster-edge-weight",
                "0.25",
            ],
        ),
    ]

    for name, args in jobs:
        out_dir = OUTPUT_ROOT / name
        summary_path = out_dir / "graph_training_summary.json"
        if summary_path.exists():
            print(f"Skipping {name}: already complete")
            continue
        print(f"Running {name}")
        run_one(name, args)
        write_aggregate()

    write_aggregate()
    print(f"Wrote {OUTPUT_ROOT / 'ablation_summary.csv'}")


if __name__ == "__main__":
    main()
