from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_BIN = REPO_ROOT / "newtests" / "venv" / "bin" / "python"
TRAIN_SCRIPT = REPO_ROOT / "training" / "train_graph_gat_gru.py"
DATA_ALIAS = REPO_ROOT / "newtests" / "targeted_rebuild" / "data_alias"
OUTPUT_ROOT = REPO_ROOT / "newtests" / "targeted_rebuild" / "results" / "gat_gru_radius_ablation"
LOG_ROOT = REPO_ROOT / "newtests" / "targeted_rebuild" / "logs" / "gat_gru_radius_ablation"

RADIUS_KM = (75, 150, 300)
CROP_TARGET_MODE = {
    "onion": "delta_roll28",
    "potato": "delta_roll28",
    "tomato": "delta_roll28",
    "wheat": "delta_current",
}


def base_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    env.setdefault("LOKY_MAX_CPU_COUNT", "8")
    return env


def run_one(crop: str, radius_km: int) -> None:
    target_mode = CROP_TARGET_MODE[crop]
    name = f"{crop}__radius_{radius_km}km"
    out_dir = OUTPUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{name}.log"
    cmd = [
        str(PYTHON_BIN),
        str(TRAIN_SCRIPT),
        "--data-dir",
        str(DATA_ALIAS),
        "--output-dir",
        str(out_dir),
        "--crops",
        crop,
        "--horizon",
        "15",
        "--epochs",
        "8",
        "--batch-size",
        "4",
        "--device",
        "mps",
        "--target-mode",
        target_mode,
        "--geo-radius-km",
        str(radius_km),
    ]
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


def write_aggregate() -> None:
    rows: list[dict[str, object]] = []
    for crop, target_mode in CROP_TARGET_MODE.items():
        for radius_km in RADIUS_KM:
            summary_path = OUTPUT_ROOT / f"{crop}__radius_{radius_km}km" / "graph_training_summary.json"
            if not summary_path.exists():
                continue
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            row = dict(data[crop])
            row["crop"] = crop
            row["target_mode_expected"] = target_mode
            row["geo_radius_km"] = radius_km
            row["config_name"] = f"{crop}__radius_{radius_km}km"
            rows.append(row)
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values(["crop", "geo_radius_km"])
    df.to_csv(OUTPUT_ROOT / "radius_ablation_summary.csv", index=False)


def main() -> None:
    if not PYTHON_BIN.exists():
        raise FileNotFoundError(f"Missing venv python: {PYTHON_BIN}")
    if not DATA_ALIAS.exists():
        raise FileNotFoundError(f"Missing data alias folder: {DATA_ALIAS}")

    for crop in sorted(CROP_TARGET_MODE):
        for radius_km in RADIUS_KM:
            out_dir = OUTPUT_ROOT / f"{crop}__radius_{radius_km}km"
            summary_path = out_dir / "graph_training_summary.json"
            if summary_path.exists():
                print(f"Skipping {crop} radius {radius_km}km: already complete")
                continue
            print(f"Running {crop} radius {radius_km}km")
            run_one(crop, radius_km)
            write_aggregate()

    write_aggregate()
    print(f"Wrote {OUTPUT_ROOT / 'radius_ablation_summary.csv'}")


if __name__ == "__main__":
    main()
