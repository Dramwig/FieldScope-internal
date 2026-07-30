#!/usr/bin/env bash
set -euo pipefail
: "${FIELDSCOPE_CHECKPOINTS_ROOT:?Set FIELDSCOPE_CHECKPOINTS_ROOT}"
: "${FIELDSCOPE_DATASETS_ROOT:?Set FIELDSCOPE_DATASETS_ROOT}"
python -m fieldscope.cli doctor --config configs/eval/auraflow_probe_smoke.yaml
python -m fieldscope.cli smoke --config configs/eval/auraflow_probe_smoke.yaml --steps 1

