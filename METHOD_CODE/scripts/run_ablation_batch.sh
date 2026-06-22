#!/usr/bin/env bash
# ============================================================
# Batch runner for ablation experiments #2 – #12
# Usage:
#   cd /root/method/----/METHOD_CODE
#   bash scripts/run_ablation_batch.sh
#
# Logs are written to  ./logs/batch_ablation_<timestamp>.log
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

# Experiments to run (scenario name → description)
declare -A EXPERIMENTS=(
  [2]="full_experiment_low_rank_margin"
  [3]="full_experiment_low_rank_delaykd"
  [4]="full_experiment_low_rank_lrboost"
  [5]="full_experiment_low_rank_stable2"
  [6]="full_experiment_low_rank_noprefix_diag"
  [7]="full_experiment_low_rank_rp8"
  [8]="full_experiment_low_rank_combo_hm"
  [9]="full_experiment_low_rank_combo_hd"
  [10]="full_experiment_low_rank_combo_hs"
  [11]="full_experiment_low_rank_combo_triple"
  [12]="full_experiment_low_rank_combo_quad"
)

# Ordered list
ORDER=(2 3 4 5 6 7 8 9 10 11 12)

echo "========================================================" | tee -a "$LOG_FILE"
echo " Ablation batch started: $(date)"                        | tee -a "$LOG_FILE"
echo " Log file: $LOG_FILE"                                    | tee -a "$LOG_FILE"
echo "========================================================" | tee -a "$LOG_FILE"

FAILED=()

for idx in "${ORDER[@]}"; do
  SCENARIO="${EXPERIMENTS[$idx]}"
  echo ""                                                      | tee -a "$LOG_FILE"
  echo ">>> [$idx/12] Starting: $SCENARIO  ($(date))"         | tee -a "$LOG_FILE"
  echo "===================================================="  | tee -a "$LOG_FILE"

  if python scripts/launch_experiments.py --scenario "$SCENARIO" 2>&1 | tee -a "$LOG_FILE"; then
    echo "<<< [$idx/12] DONE: $SCENARIO  ($(date))"           | tee -a "$LOG_FILE"
  else
    echo "!!! [$idx/12] FAILED: $SCENARIO  (exit=$?)"         | tee -a "$LOG_FILE"
    FAILED+=("[$idx] $SCENARIO")
  fi
done

echo ""                                                          | tee -a "$LOG_FILE"
echo "========================================================" | tee -a "$LOG_FILE"
echo " Batch finished: $(date)"                                 | tee -a "$LOG_FILE"

if [ ${#FAILED[@]} -eq 0 ]; then
  echo " All 11 experiments completed successfully."           | tee -a "$LOG_FILE"
else
  echo " ${#FAILED[@]} experiment(s) FAILED:"                  | tee -a "$LOG_FILE"
  for f in "${FAILED[@]}"; do
    echo "   - $f"                                              | tee -a "$LOG_FILE"
  done
fi

echo " Full log: $LOG_FILE"                                    | tee -a "$LOG_FILE"
echo "========================================================" | tee -a "$LOG_FILE"
