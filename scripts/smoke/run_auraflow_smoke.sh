#!/usr/bin/env bash
set -euo pipefail
python_bin="${FIELDSCOPE_PYTHON:-python}"
: "${FIELDSCOPE_CHECKPOINTS_ROOT:?Set FIELDSCOPE_CHECKPOINTS_ROOT}"
: "${FIELDSCOPE_DATASETS_ROOT:?Set FIELDSCOPE_DATASETS_ROOT}"
"$python_bin" -m fieldscope.cli doctor --config configs/eval/auraflow_contract_smoke.yaml
"$python_bin" -m fieldscope.cli smoke --config configs/eval/auraflow_contract_smoke.yaml --steps 1
