#!/bin/zsh
set -euo pipefail

export TARGET_R2="${TARGET_R2:-0.75}"
export TARGET_CROPS="${TARGET_CROPS:-onion,tomato}"
export PER_CROP_SECONDS="${PER_CROP_SECONDS:-900}"
export MODELS_ROOT="${MODELS_ROOT:-models/per_crop_histgb_strict}"
export CODEX_BIN="${CODEX_BIN:-codex}"
export LOG_DIR="${LOG_DIR:-autotune_logs_strict}"

exec ./codex_crop_queue.sh
