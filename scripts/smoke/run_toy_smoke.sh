#!/usr/bin/env bash
set -euo pipefail
python_bin="${FIELDSCOPE_PYTHON:-python}"
"$python_bin" -m fieldscope.cli smoke --config configs/eval/toy_smoke.yaml --steps 2
