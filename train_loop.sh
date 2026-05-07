#!/bin/bash
TARGET_R2=0.75  # Set your target

MAX_ATTEMPTS=50  # Safety limit

attempt=0
while [ $attempt -lt $MAX_ATTEMPTS ]; do
  r2=$(codex exec --full-auto "evaluate current model R^2 score from metrics file; output ONLY the float value like 0.92" | grep -o '[0-9]\+\.[0-9]\+')
  
  if (( $(echo "$r2 >= $TARGET_R2" | bc -l) )); then
    echo "Target R² $r2 reached!"
    break
  fi
  
  echo "R²: $r2 at attempt $((attempt+1)). Continuing..."
  codex exec --full-auto "improve model based on current eval; train next iteration; save metrics"
  
  sleep 30  # Adjust pause between iterations
  attempt=$((attempt+1))
done