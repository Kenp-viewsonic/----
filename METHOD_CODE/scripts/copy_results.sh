#!/usr/bin/env bash
# ============================================================
# Copy experiment results excluding model checkpoints.
#
# Usage:
#   bash scripts/copy_results.sh <results_dir> <dest_dir>
#
# Example:
#   bash scripts/copy_results.sh \
#     outputs/full_experiment_low_rank_combo_hs_20260622_064155 \
#     /path/to/dest
# ============================================================

set -euo pipefail

if [ $# -ne 2 ]; then
  echo "Usage: $0 <results_dir> <dest_dir>"
  exit 1
fi

SRC="$1"
DEST="$2"

if [ ! -d "$SRC" ]; then
  echo "Error: results dir not found: $SRC"
  exit 1
fi

mkdir -p "$DEST"

rsync -av \
  --exclude='checkpoint-*/' \
  "$SRC/" "$DEST/"

echo ""
echo "Done. Results copied to: $DEST"
echo "Excluded: all checkpoint-* directories (model weights)"
