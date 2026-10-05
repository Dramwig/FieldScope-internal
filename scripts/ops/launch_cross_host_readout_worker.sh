#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 8 ]]; then
  echo "usage: $0 ROOT REPOSITORY DATASET GPU LOCK PIDFILE LOG WATCHDOG_STATE" >&2
  exit 2
fi

root="$1"
repository="$2"
dataset="$3"
gpu="$4"
lock_file="$5"
pid_file="$6"
log_file="$7"
watchdog_state="$8"
python_bin="$root/.venv/bin/python"
expected_revision="020c1de567edd88e0eda245fd085335ffe678f47"
config="$root/recovery/fixed-revision-readout/auraflow_v03_readout_0g_0be38fa17ba13c1a9e0b248642c85832cd8cf7e63afe2b799fe89f0f0ea98ed1.yaml"
profile="$root/recovery/fixed-revision-readout/readout_runtime_profile_020c1de567edd88e0eda245fd085335ffe678f47_0be38fa17ba13c1a9e0b248642c85832cd8cf7e63afe2b799fe89f0f0ea98ed1_30c6073ce73f6de74803eb29f497ceedf1ee3cafc488340cf236395824ea42e3.json"
state_helper="${FIELDSCOPE_CROSS_HOST_STATE_HELPER:-$root/FieldScope-internal/scripts/ops/fieldscope_cross_host_monitor_state.py}"

case "$dataset" in
  nyuv2) train_script="train_nyuv2_readout.sh" ;;
  voc2012) train_script="train_voc2012_readout.sh" ;;
  *) echo "unsupported delegated dataset: $dataset" >&2; exit 2 ;;
esac
if ! [[ "$gpu" =~ ^[01]$ ]]; then
  echo "A800 physical GPU must be 0 or 1: $gpu" >&2
  exit 2
fi
for path in "$python_bin" "$config" "$profile" "$state_helper"; do
  [[ -f "$path" ]] || { echo "missing required file: $path" >&2; exit 3; }
done
[[ "$(git -C "$repository" rev-parse HEAD)" == "$expected_revision" ]] || exit 4
[[ -z "$(git -C "$repository" status --porcelain)" ]] || exit 4
"$python_bin" "$state_helper" \
  --state "$watchdog_state" \
  --expected-revision "$expected_revision" \
  --max-age-seconds 1200 >/dev/null

mkdir -p "$(dirname "$lock_file")" "$(dirname "$pid_file")" "$(dirname "$log_file")"
exec 9>"$lock_file"
if ! flock -n 9; then
  echo "dataset ownership lock is held: $lock_file" >&2
  exit 13
fi
printf '%s\n' "$$" >"${pid_file}.tmp.$$"
mv "${pid_file}.tmp.$$" "$pid_file"
trap 'rm -f "$pid_file"' EXIT
cd "$repository"
exec env \
  CUDA_VISIBLE_DEVICES="$gpu" \
  PYTHONPATH="$repository/src:$root/recovery/orchestration/020c1de/sitecompat:$repository/src" \
  FIELDSCOPE_ROOT="$root" \
  FIELDSCOPE_DATASETS_ROOT="$root/datasets" \
  FIELDSCOPE_CHECKPOINTS_ROOT="$root/checkpoints" \
  FIELDSCOPE_REPOSITORY="$repository" \
  FIELDSCOPE_CONFIG="$config" \
  FIELDSCOPE_READOUT_CONFIG="$config" \
  FIELDSCOPE_READOUT_RUNTIME_PROFILE="$profile" \
  FIELDSCOPE_CACHE_TAG=auraflow_v03 \
  FIELDSCOPE_PYTHON="$python_bin" \
  FIELDSCOPE_EXPECTED_REVISION="$expected_revision" \
  FIELDSCOPE_GPU_INDEX="$gpu" \
  FIELDSCOPE_STORAGE_POLICY=readout_sparse \
  bash "scripts/train/$train_script" >>"$log_file" 2>&1
