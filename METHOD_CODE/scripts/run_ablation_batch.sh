#!/usr/bin/env bash
# ============================================================
# Batch runner for ablation experiments.
#
# Usage:
#   cd /root/method/----/METHOD_CODE
#   bash scripts/run_ablation_batch.sh              # run ALL listed below
#   bash scripts/run_ablation_batch.sh margin rp8   # run only specified keywords
#
# To add a new scenario, just append its name to the SCENARIOS array.
#
# Logs are written to ./logs/batch_ablation_<timestamp>.log
# ============================================================

set -euo pipefail

# Force Python to flush stdout/stderr immediately (no buffering)
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/batch_ablation_${TIMESTAMP}.log"

# ── Scenario list (append new ones here) ────────────────────
SCENARIOS=(
  # ── new scenarios ──
  full_experiment_low_rank_delaykd_alpha03
  full_experiment_low_rank_delaykd_stable2
)

# ── Optional: filter by keyword arguments ───────────────────
if [ $# -gt 0 ]; then
  FILTERED=()
  for kw in "$@"; do
    for s in "${SCENARIOS[@]}"; do
      if [[ "$s" == *"$kw"* ]]; then
        FILTERED+=("$s")
      fi
    done
  done
  SCENARIOS=("${FILTERED[@]}")
fi

TOTAL=${#SCENARIOS[@]}
echo "========================================================" | tee -a "$LOG_FILE"
echo " Ablation batch: $TOTAL experiments  ($(date))"         | tee -a "$LOG_FILE"
echo " Log file: $LOG_FILE"                                    | tee -a "$LOG_FILE"
echo " Scenarios: ${SCENARIOS[*]}"                             | tee -a "$LOG_FILE"
echo "========================================================" | tee -a "$LOG_FILE"

FAILED=()
COUNT=0

for SCENARIO in "${SCENARIOS[@]}"; do
  COUNT=$((COUNT + 1))
  echo ""                                                      | tee -a "$LOG_FILE"
  echo ">>> [$COUNT/$TOTAL] Starting: $SCENARIO  ($(date))"   | tee -a "$LOG_FILE"
  echo "===================================================="  | tee -a "$LOG_FILE"

  if python scripts/launch_experiments.py --scenario "$SCENARIO" 2>&1 | tee -a "$LOG_FILE"; then
    echo "<<< [$COUNT/$TOTAL] DONE: $SCENARIO  ($(date))"     | tee -a "$LOG_FILE"
  else
    echo "!!! [$COUNT/$TOTAL] FAILED: $SCENARIO  (exit=$?)"   | tee -a "$LOG_FILE"
    FAILED+=("$SCENARIO")
  fi
done

echo ""                                                          | tee -a "$LOG_FILE"
echo "========================================================" | tee -a "$LOG_FILE"
echo " Batch finished: $(date)"                                 | tee -a "$LOG_FILE"

if [ ${#FAILED[@]} -eq 0 ]; then
  echo " All $TOTAL experiments completed successfully."       | tee -a "$LOG_FILE"
else
  echo " ${#FAILED[@]} experiment(s) FAILED:"                  | tee -a "$LOG_FILE"
  for f in "${FAILED[@]}"; do
    echo "   - $f"                                              | tee -a "$LOG_FILE"
  done
fi

echo " Full log: $LOG_FILE"                                    | tee -a "$LOG_FILE"
echo "========================================================" | tee -a "$LOG_FILE"
