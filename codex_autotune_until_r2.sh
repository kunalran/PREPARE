#!/bin/zsh
set -euo pipefail

TARGET_R2="${TARGET_R2:-0.75}"
HORIZON="${HORIZON:-15}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-25}"
MODELS_ROOT="${MODELS_ROOT:-models/per_crop_histgb}"
TARGET_CROPS="${TARGET_CROPS:-onion,potato,tomato,wheat}"
SLEEP_SECONDS="${SLEEP_SECONDS:-10}"
CODEX_BIN="${CODEX_BIN:-codex}"
LOG_DIR="${LOG_DIR:-autotune_logs}"
PROMPT_FILE="${PROMPT_FILE:-autotune_prompt.txt}"

mkdir -p "$LOG_DIR"

cat > "$PROMPT_FILE" <<'EOF'
You are working in the PREPARE project.

Goal:
- Keep improving the crop forecasting models until the requested 15-day R^2 target is met.
- Read the shell environment variables TARGET_CROPS and TARGET_R2 from the surrounding run context and optimize specifically for those crops and that threshold.

Rules:
- Operate autonomously in this workspace.
- Prefer editing existing training/evaluation scripts rather than creating throwaway notebooks.
- Retrain models as needed.
- Save improved per-crop model artifacts under models/per_crop_histgb or a clearly named replacement folder if you intentionally create a new version.
- Ensure per-crop metrics JSON files are updated and contain r2 for each horizon.
- After each iteration, leave the workspace in a state where check_r2_targets.py can evaluate the current best model folder.
- If a change makes results worse, continue iterating rather than stopping.
- Focus on improving 15-day R^2 for TARGET_CROPS.
- If TARGET_CROPS contains only onion, prioritize onion-only search, alternative targets, longer-memory features, and any safe model/feature changes that can improve onion 15d R^2.
- Keep the checked folder current: if you train into a versioned folder, copy or promote the best metrics and model artifacts back into MODELS_ROOT before ending the iteration.

When done with this iteration:
- Summarize the change you made.
- Report the current 15-day R^2 values for TARGET_CROPS.
EOF

attempt=1
while [[ "$attempt" -le "$MAX_ATTEMPTS" ]]; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Attempt $attempt: checking target in $MODELS_ROOT"
  if python3 check_r2_targets.py --models-root "$MODELS_ROOT" --target-r2 "$TARGET_R2" --horizon "$HORIZON" --crops "$TARGET_CROPS"; then
    echo "Target met. Stopping."
    exit 0
  fi

  ITER_LOG="$LOG_DIR/attempt_${attempt}.log"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Target not met. Launching Codex iteration. Log: $ITER_LOG"
  TARGET_CROPS="$TARGET_CROPS" TARGET_R2="$TARGET_R2" MODELS_ROOT="$MODELS_ROOT" "$CODEX_BIN" exec --full-auto "$(cat "$PROMPT_FILE")" | tee "$ITER_LOG"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Re-checking target after attempt $attempt"
  if python3 check_r2_targets.py --models-root "$MODELS_ROOT" --target-r2 "$TARGET_R2" --horizon "$HORIZON" --crops "$TARGET_CROPS"; then
    echo "Target met after attempt $attempt. Stopping."
    exit 0
  fi

  attempt=$((attempt + 1))
  sleep "$SLEEP_SECONDS"
done

echo "Reached MAX_ATTEMPTS=$MAX_ATTEMPTS without hitting target."
exit 1
