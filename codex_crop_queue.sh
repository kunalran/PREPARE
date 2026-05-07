#!/bin/zsh
set -euo pipefail

TARGET_R2="${TARGET_R2:-0.75}"
HORIZON="${HORIZON:-15}"
TARGET_CROPS="${TARGET_CROPS:-onion,potato,tomato,wheat}"
PER_CROP_SECONDS="${PER_CROP_SECONDS:-7200}"
SLEEP_SECONDS="${SLEEP_SECONDS:-10}"
MAX_ROUNDS="${MAX_ROUNDS:-20}"
MODELS_ROOT="${MODELS_ROOT:-models/per_crop_histgb}"
CODEX_BIN="${CODEX_BIN:-codex}"
LOG_DIR="${LOG_DIR:-autotune_logs}"

exec python3 codex_crop_queue.py \
  --models-root "$MODELS_ROOT" \
  --target-r2 "$TARGET_R2" \
  --horizon "$HORIZON" \
  --crops "$TARGET_CROPS" \
  --per-crop-seconds "$PER_CROP_SECONDS" \
  --sleep-seconds "$SLEEP_SECONDS" \
  --max-rounds "$MAX_ROUNDS" \
  --codex-bin "$CODEX_BIN" \
  --log-dir "$LOG_DIR"
