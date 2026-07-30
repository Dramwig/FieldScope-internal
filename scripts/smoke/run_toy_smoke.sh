#!/usr/bin/env bash
set -euo pipefail
python -m fieldscope.cli smoke --config configs/eval/toy_smoke.yaml --steps 2

