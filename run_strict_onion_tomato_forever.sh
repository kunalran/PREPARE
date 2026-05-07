#!/bin/zsh
set -euo pipefail

export TARGET_R2="${TARGET_R2:-999}"
export TARGET_CROPS="${TARGET_CROPS:-onion,tomato}"
export PER_CROP_SECONDS="${PER_CROP_SECONDS:-900}"
export MODELS_ROOT="${MODELS_ROOT:-models/per_crop_histgb_strict}"
export CODEX_BIN="${CODEX_BIN:-codex}"
export LOG_DIR="${LOG_DIR:-autotune_logs_strict}"
export MAX_ROUNDS="${MAX_ROUNDS:-999}"

exec python3 codex_crop_queue.py \
  --models-root "$MODELS_ROOT" \
  --target-r2 "$TARGET_R2" \
  --horizon 15 \
  --crops "$TARGET_CROPS" \
  --per-crop-seconds "$PER_CROP_SECONDS" \
  --sleep-seconds 10 \
  --max-rounds "$MAX_ROUNDS" \
  --codex-bin "$CODEX_BIN" \
  --log-dir "$LOG_DIR" \
  --no-target-stop
