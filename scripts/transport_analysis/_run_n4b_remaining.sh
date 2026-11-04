#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
for model in random_forest extra_trees xgboost catboost shallow_mlp pretrained_element_embedding composition_attention; do
  echo "=== starting $model at $(date) ==="
  .venv/bin/python scripts/transport_analysis/n4b_nested_internal_validation.py --model "$model" \
    > "logs/n4b_${model}.log" 2>&1
  echo "=== finished $model at $(date) ==="
done
.venv/bin/python scripts/transport_analysis/n4b_nested_internal_validation.py --merge > logs/n4b_merge.log 2>&1
echo "=== N4b nested internal validation: ALL DONE at $(date) ==="
