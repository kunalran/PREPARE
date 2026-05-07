from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_CROPS = ["onion", "potato", "tomato", "wheat"]


PROMPT_TEMPLATE = """You are working in the PREPARE project.

Current crop focus: {crop}
Target:
- Improve the {crop} model until its 15-day R^2 is at least {target_r2}.

Rules:
- Focus on {crop} only for this run unless you need shared code changes that also affect other crops.
- Prefer editing existing training/evaluation scripts.
- Retrain only what is needed for {crop}.
- Keep the checked folder current: MODELS_ROOT is {models_root}. The final metrics for {crop} must end up in that folder so the checker can see them.
- Use safe, non-interactive commands only.
- If a model variant is worse, continue iterating rather than stopping.
- Prioritize techniques that can improve long-horizon performance: better target design, longer-memory features, alternative model classes, better validation-safe feature engineering, outlier handling, or crop-specific tuning.
- Use only the final hourly data in final_data_hourly.
- No shortcut branches, no crop-specific fallback baseline substitutions, no special 15d-only evaluation tricks.
- Keep 15d training methodologically aligned with the other horizons: same main trainer path, same validation_days rule, same honest time split.
- No future-aware filtering or feature construction that uses validation-period information to decide which rows/series are eligible.
- Do not use centered-window imputation of the target.
- If you introduce a justified alternative model class, train and evaluate it under the same strict holdout rules and replace the saved artifact only if it is genuinely better on the checked holdout.

Before ending:
- Report the current 15-day R^2 for {crop}.
- Leave the best known {crop} artifacts and metrics in {models_root}/{crop}.
"""


def run(cmd: list[str], cwd: Path, env: dict[str, str], timeout: int | None = None) -> int:
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, timeout=timeout)
    return proc.returncode


def check_crop(models_root: Path, crop: str, target_r2: float, horizon: int, cwd: Path) -> bool:
    cmd = [
        sys.executable,
        "check_r2_targets.py",
        "--models-root",
        str(models_root),
        "--target-r2",
        str(target_r2),
        "--horizon",
        str(horizon),
        "--crops",
        crop,
    ]
    return subprocess.run(cmd, cwd=str(cwd)).returncode == 0


def codex_iteration(
    cwd: Path,
    codex_bin: str,
    crop: str,
    target_r2: float,
    models_root: Path,
    timeout_seconds: int,
    log_path: Path,
) -> int:
    prompt = PROMPT_TEMPLATE.format(
        crop=crop,
        target_r2=target_r2,
        models_root=models_root,
    )
    env = os.environ.copy()
    env["TARGET_CROPS"] = crop
    env["TARGET_R2"] = str(target_r2)
    env["MODELS_ROOT"] = str(models_root)

    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} crop={crop} =====\n")
        fh.flush()
        proc = subprocess.run(
            [codex_bin, "exec", "--full-auto", prompt],
            cwd=str(cwd),
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
        )
        return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Round-robin Codex queue: work each crop up to a time budget until target R^2 is met."
    )
    parser.add_argument("--models-root", type=Path, default=Path("models/per_crop_histgb"))
    parser.add_argument("--target-r2", type=float, default=0.75)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--crops", type=str, default="onion,potato,tomato,wheat")
    parser.add_argument("--per-crop-seconds", type=int, default=7200)
    parser.add_argument("--sleep-seconds", type=int, default=10)
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--codex-bin", type=str, default="codex")
    parser.add_argument("--log-dir", type=Path, default=Path("autotune_logs"))
    parser.add_argument(
        "--no-target-stop",
        action="store_true",
        help="Do not stop when target is met; keep cycling through crops for max-rounds.",
    )
    args = parser.parse_args()

    cwd = Path.cwd()
    crops = [c.strip().lower() for c in args.crops.split(",") if c.strip()]
    if not crops:
        crops = DEFAULT_CROPS

    args.log_dir.mkdir(parents=True, exist_ok=True)

    round_num = 1
    while round_num <= args.max_rounds:
        print(f"=== Round {round_num} ===")
        all_met = True
        for crop in crops:
            crop_met = check_crop(args.models_root, crop, args.target_r2, args.horizon, cwd)
            if crop_met and not args.no_target_stop:
                print(f"{crop}: target already met, skipping")
                continue

            if not crop_met:
                all_met = False
            start = time.time()
            budget = args.per_crop_seconds
            status_msg = "target not met" if not crop_met else "target met but continuing"
            print(f"{crop}: {status_msg}, budget={budget}s")
            crop_log = args.log_dir / f"{crop}_round_{round_num}.log"

            try:
                codex_iteration(
                    cwd=cwd,
                    codex_bin=args.codex_bin,
                    crop=crop,
                    target_r2=args.target_r2,
                    models_root=args.models_root,
                    timeout_seconds=budget,
                    log_path=crop_log,
                )
            except subprocess.TimeoutExpired:
                with crop_log.open("a", encoding="utf-8") as fh:
                    fh.write(f"\nTimed out after {budget} seconds for crop={crop}\n")
                print(f"{crop}: timed out after {budget}s, moving to next crop")

            if check_crop(args.models_root, crop, args.target_r2, args.horizon, cwd):
                print(f"{crop}: target met after this slot")
            else:
                elapsed = int(time.time() - start)
                print(f"{crop}: target still not met after {elapsed}s, moving on")

            time.sleep(args.sleep_seconds)

        if all_met and not args.no_target_stop:
            print("All crop targets met. Stopping.")
            return 0
        round_num += 1

    print("Reached max rounds without all crops meeting target.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
