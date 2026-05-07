from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CROPS = ("onion", "potato", "tomato", "wheat")


def load_metric(metric_path: Path, horizon: int) -> float | None:
    if not metric_path.exists():
        return None
    data = json.loads(metric_path.read_text(encoding="utf-8"))
    for row in data:
        if int(row.get("horizon_days", -1)) == horizon:
            value = row.get("r2")
            return None if value is None else float(value)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether all crop models meet a target R^2 at a given horizon."
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=Path("models/per_crop_histgb_targeted_v2"),
        help="Root folder containing per-crop metrics JSON files.",
    )
    parser.add_argument(
        "--target-r2",
        type=float,
        default=0.75,
        help="Required minimum R^2 threshold.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=15,
        help="Forecast horizon to inspect.",
    )
    parser.add_argument(
        "--crops",
        type=str,
        default="onion,potato,tomato,wheat",
        help="Comma-separated crops to inspect.",
    )
    args = parser.parse_args()

    crops = [crop.strip().lower() for crop in args.crops.split(",") if crop.strip()]
    failures: list[str] = []
    for crop in crops:
        metric_path = args.models_root / crop / f"{crop}_metrics.json"
        value = load_metric(metric_path, args.horizon)
        if value is None:
            failures.append(f"{crop}:missing")
            print(f"{crop}: missing")
            continue
        print(f"{crop}: {value:.4f}")
        if value < args.target_r2:
            failures.append(f"{crop}:{value:.4f}")

    if failures:
        print("status: not_met")
        return 1

    print("status: met")
    return 0


if __name__ == "__main__":
    sys.exit(main())
