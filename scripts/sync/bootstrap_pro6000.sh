#!/usr/bin/env bash
set -euo pipefail

project_root=/root/autodl-tmp/FieldScope
mkdir -p \
  "$project_root/FieldScope-internal" \
  "$project_root/datasets/raw" \
  "$project_root/datasets/prepared" \
  "$project_root/datasets/feature_cache" \
  "$project_root/checkpoints" \
  "$project_root/hf_home" \
  "$project_root/.incoming" \
  "$project_root/logs"

test "$(df -P "$project_root" | awk 'NR==2 {print $6}')" = "/root/autodl-tmp"
printf '%s\n' "$project_root"

