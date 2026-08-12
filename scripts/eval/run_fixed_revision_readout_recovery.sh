#!/usr/bin/env bash
set -euo pipefail

: "${FIELDSCOPE_ROOT:?Set FIELDSCOPE_ROOT}"
: "${FIELDSCOPE_DATASETS_ROOT:?Set FIELDSCOPE_DATASETS_ROOT}"
: "${FIELDSCOPE_CHECKPOINTS_ROOT:?Set FIELDSCOPE_CHECKPOINTS_ROOT}"
: "${FIELDSCOPE_EXPECTED_REVISION:?Set FIELDSCOPE_EXPECTED_REVISION}"

repository="${FIELDSCOPE_REPOSITORY:-$FIELDSCOPE_ROOT/FieldScope-internal}"
repository="$(cd -- "$repository" && pwd -P)"
python_bin="${FIELDSCOPE_PYTHON:-$FIELDSCOPE_ROOT/.venv/bin/python}"
cache_tag="${FIELDSCOPE_CACHE_TAG:-auraflow_v03}"
tracked_config="${FIELDSCOPE_TRACKED_CONFIG:-configs/model/auraflow_v03.yaml}"
external_root="${FIELDSCOPE_RECOVERY_EXTERNAL_ROOT:-$FIELDSCOPE_ROOT/recovery/fixed-revision-readout}"
output_root="${FIELDSCOPE_RECOVERY_OUTPUT_ROOT:-$repository/outputs/full_validation/$cache_tag}"
runtime_profile="${FIELDSCOPE_RUNTIME_PROFILE:-$repository/outputs/runtime_gate/auraflow_runtime_profile_${FIELDSCOPE_EXPECTED_REVISION}.json}"
readout_cache_gib="${FIELDSCOPE_RECOVERY_READOUT_CACHE_GIB:-0}"
gpu_free_checks_required="${FIELDSCOPE_RECOVERY_GPU_FREE_CHECKS:-5}"
gpu_poll_seconds="${FIELDSCOPE_RECOVERY_GPU_POLL_SECONDS:-60}"
readout_contract_sha256="4422430edf4eb8cdb20a99db8cdf7bac53c4cb934f0b09ba8dd3c21aee3facb8"
source_tree_sha256="6ef1305effef91b29d432c9d2c1505b4726645531adad72f60299168d7119c4d"
config_sha256="0be38fa17ba13c1a9e0b248642c85832cd8cf7e63afe2b799fe89f0f0ea98ed1"
recovery_tag="${FIELDSCOPE_RECOVERY_TAG:-${FIELDSCOPE_EXPECTED_REVISION:0:7}-readout0g-v1}"
log_root="${FIELDSCOPE_LOG_ROOT:-$FIELDSCOPE_ROOT/logs}"
recovery_log="$log_root/full_validation_recovery_${recovery_tag}.log"
lock_file="$log_root/full_validation_recovery_${recovery_tag}.lock"
pid_file="$log_root/full_validation_recovery_${recovery_tag}.pid"
state_file="$log_root/full_validation_recovery_${recovery_tag}.state.json"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
config_path="$external_root/auraflow_v03_readout_${readout_cache_gib}g_${config_sha256}.yaml"
readout_runtime_profile="$external_root/readout_runtime_profile_${FIELDSCOPE_EXPECTED_REVISION}_${config_sha256}.json"
manifest_registry="$external_root/recovery_registry_${FIELDSCOPE_EXPECTED_REVISION}_${config_sha256}.json"

# Normalize paths once.  The recovery script is intentionally runnable from
# outside the checkout (for the fixed old revision), so no default output or
# wrapper path may depend on the caller's current working directory.
if [[ "$output_root" != /* ]]; then
  output_root="$repository/$output_root"
fi
if [[ "$runtime_profile" != /* ]]; then
  runtime_profile="$repository/$runtime_profile"
fi
if [[ "$external_root" != /* ]]; then
  external_root="$FIELDSCOPE_ROOT/$external_root"
fi
if [[ "$log_root" != /* ]]; then
  log_root="$FIELDSCOPE_ROOT/$log_root"
fi

config_path="$external_root/auraflow_v03_readout_${readout_cache_gib}g_${config_sha256}.yaml"
readout_runtime_profile="$external_root/readout_runtime_profile_${FIELDSCOPE_EXPECTED_REVISION}_${config_sha256}.json"
manifest_registry="$external_root/recovery_registry_${FIELDSCOPE_EXPECTED_REVISION}_${config_sha256}.json"
recovery_log="$log_root/full_validation_recovery_${recovery_tag}.log"
lock_file="$log_root/full_validation_recovery_${recovery_tag}.lock"
pid_file="$log_root/full_validation_recovery_${recovery_tag}.pid"
state_file="$log_root/full_validation_recovery_${recovery_tag}.state.json"

resolve_repository_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$repository" "$1" ;;
  esac
}

tracked_config_path="$(resolve_repository_path "$tracked_config")"
final_wrapper="${FIELDSCOPE_RECOVERY_FINAL_WRAPPER:-$script_dir/run_fixed_revision_final_recovery.sh}"
wrapper_root="${FIELDSCOPE_RECOVERY_WRAPPER_ROOT:-$script_dir}"

if [[ -n "${FIELDSCOPE_IMAGENET1K_ROOT:-}" && "$FIELDSCOPE_IMAGENET1K_ROOT" != /* ]]; then
  FIELDSCOPE_IMAGENET1K_ROOT="$FIELDSCOPE_ROOT/$FIELDSCOPE_IMAGENET1K_ROOT"
fi

if ! [[ "$readout_cache_gib" =~ ^0([.]0+)?$ ]]; then
  echo "fixed-revision recovery requires exactly 0 GiB shared readout cache" >&2
  exit 2
fi
if ! [[ "$gpu_free_checks_required" =~ ^[1-9][0-9]*$ ]]; then
  echo "FIELDSCOPE_RECOVERY_GPU_FREE_CHECKS must be a positive integer" >&2
  exit 2
fi
if ! [[ "$gpu_poll_seconds" =~ ^[1-9][0-9]*$ ]]; then
  echo "FIELDSCOPE_RECOVERY_GPU_POLL_SECONDS must be a positive integer" >&2
  exit 2
fi

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

verify_repository() {
  local actual_revision actual_tree branch
  actual_revision="$(git -C "$repository" rev-parse HEAD)"
  if [[ "$actual_revision" != "$FIELDSCOPE_EXPECTED_REVISION" ]]; then
    echo "revision mismatch: expected=$FIELDSCOPE_EXPECTED_REVISION actual=$actual_revision" >&2
    exit 3
  fi
  if branch="$(git -C "$repository" symbolic-ref -q --short HEAD)"; then
    echo "fixed-revision recovery requires detached HEAD; found branch=$branch" >&2
    exit 4
  fi
  if [[ -n "$(git -C "$repository" status --porcelain)" ]]; then
    echo "fixed-revision recovery requires a clean worktree" >&2
    exit 5
  fi
  actual_tree="$({
    cd "$repository"
    "$python_bin" - <<'PY'
from fieldscope.experiments import code_provenance
print(code_provenance()["code_tree_sha256"])
PY
  })"
  if [[ "$actual_tree" != "$source_tree_sha256" ]]; then
    echo "code tree mismatch: expected=$source_tree_sha256 actual=$actual_tree" >&2
    exit 6
  fi
}

verify_no_formal_worker() {
  local process_dir pid cwd allow_recovery_pid
  local -a argv
  allow_recovery_pid="${1:-}"
  for process_dir in /proc/[0-9]*; do
    pid="${process_dir##*/}"
    [[ "$pid" != "$$" ]] || continue
    [[ -z "$allow_recovery_pid" || "$pid" != "$allow_recovery_pid" ]] || continue
    cwd="$(readlink -f "$process_dir/cwd" 2>/dev/null || true)"
    [[ "$cwd" == "$repository" ]] || continue
    argv=()
    mapfile -d '' -t argv <"$process_dir/cmdline" 2>/dev/null || continue
    # The formal error-analysis waiter is intentionally allowed. It is read-only
    # until the final decision appears and lives in the independent analyzer cwd.
    if [[ ${#argv[@]} -ge 3 && "${argv[0]}" == "$python_bin" && \
          "${argv[1]}" == "-m" && "${argv[2]}" == "fieldscope.cli" ]]; then
      echo "refusing duplicate formal worker pid=$pid argv=${argv[*]}" >&2
      exit 7
    fi
    if [[ ${#argv[@]} -ge 2 && "${argv[0]}" == "bash" ]]; then
      case "${argv[1]}" in
        scripts/eval/run_validation_supervisor.sh | \
        scripts/eval/run_full_validation_when_ready.sh | \
        scripts/eval/run_full_validation_after_signal_gate.sh | \
        scripts/eval/run_final_conclusion_after_main.sh | \
        scripts/eval/run_causal_validation_after_main.sh | \
        scripts/eval/run_extension_after_main.sh | \
        scripts/eval/run_fixed_revision_readout_recovery.sh | \
        scripts/train/*.sh | scripts/data/*.sh)
          echo "refusing duplicate formal worker pid=$pid argv=${argv[*]}" >&2
          exit 7
          ;;
      esac
    fi
  done
}

verify_signal_decision() {
  "$python_bin" - \
    "$repository/outputs/signal_gate/promotion_decision.json" \
    "$FIELDSCOPE_EXPECTED_REVISION" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
revision = sys.argv[2]
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("code_revision") != revision or payload.get("code_dirty") is not False:
    raise SystemExit("signal-gate provenance mismatch")
if payload.get("verdict") not in {"proceed", "stop_or_redesign"}:
    raise SystemExit("signal-gate verdict is incomplete or unexpected")
if payload.get("verdict") == "stop_or_redesign" and payload.get("paper_evidence") is not False:
    raise SystemExit("negative signal-gate result was not preserved")
PY
}

verify_required_artifacts() {
  local path
  for path in \
    "$runtime_profile" \
    "$output_root/preflight/auraflow_backbone_asset.json" \
    "$output_root/preflight/imagenet100_split_audit.json" \
    "$output_root/preflight/voc2012_split_audit.json" \
    "$output_root/preflight/ade20k_split_audit.json" \
    "$output_root/preflight/nyuv2_split_audit.json" \
    "$output_root/voc2012_unsupervised.json"; do
    if [[ ! -f "$path" ]]; then
      echo "missing same-revision prerequisite artifact: $path" >&2
      exit 8
    fi
  done
  "$python_bin" - \
    "$FIELDSCOPE_EXPECTED_REVISION" \
    "$source_tree_sha256" \
    "$runtime_profile" \
    "$output_root/preflight/auraflow_backbone_asset.json" \
    "$output_root/preflight/imagenet100_split_audit.json" \
    "$output_root/preflight/voc2012_split_audit.json" \
    "$output_root/preflight/ade20k_split_audit.json" \
    "$output_root/preflight/nyuv2_split_audit.json" \
    "$output_root/voc2012_unsupervised.json" <<'PY'
import json
import pathlib
import sys

revision = sys.argv[1]
tree = sys.argv[2]
for raw_path in sys.argv[3:]:
    path = pathlib.Path(raw_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("code_revision") != revision or payload.get("code_tree_sha256") != tree:
        raise SystemExit(f"same-revision prerequisite provenance mismatch: {path}")
    if payload.get("code_dirty") is not False:
        raise SystemExit(f"same-revision prerequisite came from a dirty worktree: {path}")
    if payload.get("status") not in {"passed", None}:
        raise SystemExit(f"same-revision prerequisite is not passed: {path}")
PY
}

materialize_readout_config() {
  local tracked_path temporary actual_sha
  tracked_path="$tracked_config_path"
  if [[ ! -f "$tracked_path" ]]; then
    echo "missing tracked extraction config: $tracked_path" >&2
    exit 8
  fi
  if ! grep -Fxq '  readout_memory_cache_gib: 160' "$tracked_path"; then
    echo "tracked fixed-revision config no longer has the registered 160 GiB value" >&2
    exit 9
  fi
  mkdir -p "$external_root"
  temporary="$config_path.tmp.$$"
  sed 's/^  readout_memory_cache_gib: 160$/  readout_memory_cache_gib: 0/' \
    "$tracked_path" >"$temporary"
  actual_sha="$(sha256_file "$temporary")"
  if [[ "$actual_sha" != "$config_sha256" ]]; then
    rm -f "$temporary"
    echo "readout config SHA-256 mismatch: expected=$config_sha256 actual=$actual_sha" >&2
    exit 10
  fi
  if [[ -f "$config_path" && "$(sha256_file "$config_path")" != "$config_sha256" ]]; then
    rm -f "$temporary"
    echo "refusing to overwrite mismatched content-addressed config: $config_path" >&2
    exit 11
  fi
  if [[ -f "$config_path" ]]; then
    rm -f "$temporary"
  else
    mv "$temporary" "$config_path"
  fi
  chmod 0444 "$config_path"
}

verify_cache_manifests() {
  local scope="${1:-all}"
  "$python_bin" - \
    "$FIELDSCOPE_DATASETS_ROOT/feature_cache/$cache_tag" \
    "$FIELDSCOPE_EXPECTED_REVISION" \
    "$source_tree_sha256" \
    "$manifest_registry" \
    "$scope" <<'PY'
import hashlib
import json
import pathlib
import sys

cache_root = pathlib.Path(sys.argv[1]).resolve()
revision = sys.argv[2]
tree = sys.argv[3]
output = pathlib.Path(sys.argv[4])
scope = sys.argv[5]
expected = {
    "imagenet100": {"train": 116455, "val": 12940, "test": 5000},
    "voc2012": {"train": 1318, "val": 146, "test": 1449},
    "ade20k": {"train": 18189, "val": 2021, "test": 2000},
    "nyuv2": {"train": 715, "val": 80, "test": 654},
}
if scope != "all":
    if scope not in expected:
        raise SystemExit(f"unknown cache-manifest audit scope: {scope}")
    expected = {scope: expected[scope]}
registry = {}
for dataset, splits in expected.items():
    registry[dataset] = {}
    for split, count in splits.items():
        cache_dir = cache_root / f"{dataset}_{split}"
        manifest_path = cache_dir / "dataset_manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("complete") is not True:
            raise SystemExit(f"incomplete cache manifest: {manifest_path}")
        if payload.get("dataset") != dataset or payload.get("split") != split:
            raise SystemExit(f"cache identity mismatch: {manifest_path}")
        if int(payload.get("num_samples", -1)) != count:
            raise SystemExit(f"cache sample count mismatch: {manifest_path}")
        if payload.get("storage_policy") != "readout_sparse":
            raise SystemExit(f"cache storage policy mismatch: {manifest_path}")
        if payload.get("code_revision") != revision or payload.get("code_tree_sha256") != tree:
            raise SystemExit(f"cache provenance mismatch: {manifest_path}")
        if payload.get("code_dirty") is not False:
            raise SystemExit(f"cache came from a dirty worktree: {manifest_path}")
        if payload.get("config", {}).get("runtime", {}).get("readout_memory_cache_gib") != 160:
            raise SystemExit(f"cache config identity changed: {manifest_path}")
        shards = payload.get("shards")
        if not isinstance(shards, list) or not shards:
            raise SystemExit(f"cache shard registry missing: {manifest_path}")
        ranges = []
        names = set()
        total = 0
        for item in shards:
            name = str(item.get("path", ""))
            if not name or name in names:
                raise SystemExit(f"duplicate cache shard path: {manifest_path}")
            names.add(name)
            start = int(item.get("start", -1))
            end = int(item.get("end", -1))
            samples = int(item.get("num_samples", -1))
            shard_path = cache_dir / name
            if end - start != samples or samples < 1:
                raise SystemExit(f"cache shard range mismatch: {shard_path}")
            if not shard_path.is_file() or shard_path.stat().st_size != int(item.get("bytes", -1)):
                raise SystemExit(f"cache shard missing or truncated: {shard_path}")
            digest_state = hashlib.sha256()
            with shard_path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest_state.update(block)
            digest = digest_state.hexdigest()
            if digest != item.get("sha256"):
                raise SystemExit(f"cache shard SHA-256 mismatch: {shard_path}")
            ranges.append((start, end))
            total += samples
        ranges.sort()
        if ranges[0][0] != 0 or ranges[-1][1] != count:
            raise SystemExit(f"cache shard coverage mismatch: {manifest_path}")
        for previous, current in zip(ranges, ranges[1:]):
            if previous[1] != current[0]:
                raise SystemExit(f"cache shard gap or overlap: {manifest_path}")
        if total != count:
            raise SystemExit(f"cache shard sample total mismatch: {manifest_path}")
        temporary = [
            path.name
            for path in cache_dir.iterdir()
            if path.name.startswith(".") or path.suffix in {".tmp", ".partial"}
        ]
        if temporary:
            raise SystemExit(f"temporary cache files present: {cache_dir}: {temporary}")
        raw = manifest_path.read_bytes()
        registry[dataset][split] = {
            "path": str(manifest_path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "num_samples": count,
            "shards": len(shards),
        }
existing = {}
if output.is_file():
    existing = json.loads(output.read_text(encoding="utf-8")).get("caches", {})
existing.update(registry)
output.parent.mkdir(parents=True, exist_ok=True)
temporary = output.with_name(f".{output.name}.tmp")
temporary.write_text(
    json.dumps(
        {
            "schema_version": 1,
            "status": "passed",
            "code_revision": revision,
            "code_tree_sha256": tree,
            "caches": existing,
        },
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)
temporary.replace(output)
PY
}

verify_readout_contract() {
  local actual_contract
  actual_contract="$({
    cd "$repository"
    "$python_bin" - "$config_path" <<'PY'
import sys
from fieldscope.config import load_config
from fieldscope.readout_runtime_gate import readout_execution_contract_sha256
print(readout_execution_contract_sha256(load_config(sys.argv[1])))
PY
  })"
  if [[ "$actual_contract" != "$readout_contract_sha256" ]]; then
    echo "readout execution contract mismatch: expected=$readout_contract_sha256 actual=$actual_contract" >&2
    exit 12
  fi
}

verify_readout_profile() {
  [[ -f "$readout_runtime_profile" ]] || return 1
  (
    cd "$repository"
    env -u FIELDSCOPE_RUNTIME_PROFILE "$python_bin" -m fieldscope.cli readout-runtime-gate \
      --config "$config_path" \
      --train-cache-dir "$FIELDSCOPE_DATASETS_ROOT/feature_cache/$cache_tag/imagenet100_train" \
      --val-cache-dir "$FIELDSCOPE_DATASETS_ROOT/feature_cache/$cache_tag/imagenet100_val" \
      --test-cache-dir "$FIELDSCOPE_DATASETS_ROOT/feature_cache/$cache_tag/imagenet100_test" \
      --output "$readout_runtime_profile" \
      >/dev/null
  )
}

generate_readout_profile() {
  local temporary="${readout_runtime_profile}.tmp.$$"
  rm -f "$temporary"
  (
    cd "$repository"
    env -u FIELDSCOPE_RUNTIME_PROFILE "$python_bin" -m fieldscope.cli readout-runtime-gate \
      --config "$config_path" \
      --train-cache-dir "$FIELDSCOPE_DATASETS_ROOT/feature_cache/$cache_tag/imagenet100_train" \
      --val-cache-dir "$FIELDSCOPE_DATASETS_ROOT/feature_cache/$cache_tag/imagenet100_val" \
      --test-cache-dir "$FIELDSCOPE_DATASETS_ROOT/feature_cache/$cache_tag/imagenet100_test" \
      --output "$temporary"
  )
  mv -f "$temporary" "$readout_runtime_profile"
}

wait_for_free_gpu() {
  local free_checks=0
  while (( free_checks < gpu_free_checks_required )); do
    verify_repository
    verify_no_formal_worker "$$"
    if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits |
      grep -Eq '^[[:space:]]*[0-9]+'; then
      free_checks=0
      echo "$(date --iso-8601=seconds) fixed-revision recovery waiting for genuinely free GPU"
    else
      free_checks=$((free_checks + 1))
      echo "$(date --iso-8601=seconds) fixed-revision recovery GPU free check $free_checks/$gpu_free_checks_required"
    fi
    if (( free_checks < gpu_free_checks_required )); then
      sleep "$gpu_poll_seconds" 9>&-
    fi
  done
}

write_state() {
  local stage="$1"
  "$python_bin" - \
    "$state_file" "$stage" "$FIELDSCOPE_EXPECTED_REVISION" "$source_tree_sha256" \
    "$config_path" "$config_sha256" "$readout_runtime_profile" "$manifest_registry" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "stage": sys.argv[2],
    "code_revision": sys.argv[3],
    "code_tree_sha256": sys.argv[4],
    "readout_config": {"path": sys.argv[5], "sha256": sys.argv[6]},
    "readout_runtime_profile": sys.argv[7],
    "cache_manifest_registry": sys.argv[8],
}
temporary = path.with_name(f".{path.name}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
PY
}

run_formal_recovery() {
  local dataset
  # Use absolute paths because the old 020c1de wrappers resolve FIELDSCOPE_CONFIG
  # relative to their invocation cwd.
  export FIELDSCOPE_CONFIG="$tracked_config_path"
  export FIELDSCOPE_EXTRACTION_CONFIG="$tracked_config_path"
  export FIELDSCOPE_READOUT_CONFIG="$config_path"
  export FIELDSCOPE_RUNTIME_PROFILE="$runtime_profile"
  export FIELDSCOPE_READOUT_RUNTIME_PROFILE="$readout_runtime_profile"
  export FIELDSCOPE_CACHE_TAG="$cache_tag"
  export FIELDSCOPE_STORAGE_POLICY="readout_sparse"

  for dataset in imagenet100 voc2012 ade20k nyuv2; do
    verify_repository
    verify_no_formal_worker "$$"
    echo "$(date --iso-8601=seconds) verifying/reusing tracked-config cache dataset=$dataset"
    env FIELDSCOPE_CONFIG="$tracked_config_path" \
      FIELDSCOPE_EXTRACTION_CONFIG="$tracked_config_path" \
      bash "scripts/eval/run_${dataset}_extract.sh"
    verify_cache_manifests "$dataset"
    echo "$(date --iso-8601=seconds) training recovered zero-shared-cache readout matrix dataset=$dataset"
    env FIELDSCOPE_CONFIG="$config_path" \
      FIELDSCOPE_READOUT_CONFIG="$config_path" \
      bash "scripts/train/train_${dataset}_readout.sh"
  done
  verify_cache_manifests all

  "$python_bin" -m fieldscope.cli audit-full-evidence \
    --imagenet100-matrix "$output_root/imagenet100/matrix_report.json" \
    --voc2012-matrix "$output_root/voc2012/matrix_report.json" \
    --ade20k-matrix "$output_root/ade20k/matrix_report.json" \
    --nyuv2-matrix "$output_root/nyuv2/matrix_report.json" \
    --voc-unsupervised "$output_root/voc2012_unsupervised.json" \
    --backbone-asset "$output_root/preflight/auraflow_backbone_asset.json" \
    --runtime-profile "$runtime_profile" \
    --readout-runtime-profile "$readout_runtime_profile" \
    --imagenet100-split-audit "$output_root/preflight/imagenet100_split_audit.json" \
    --voc2012-split-audit "$output_root/preflight/voc2012_split_audit.json" \
    --ade20k-split-audit "$output_root/preflight/ade20k_split_audit.json" \
    --nyuv2-split-audit "$output_root/preflight/nyuv2_split_audit.json" \
    --output "$output_root/evidence_decision.json"

  if [[ ! -f "$final_wrapper" ]]; then
    echo "missing fixed-revision final recovery wrapper: $final_wrapper" >&2
    exit 14
  fi
  exec env \
    FIELDSCOPE_ROOT="$FIELDSCOPE_ROOT" \
    FIELDSCOPE_DATASETS_ROOT="$FIELDSCOPE_DATASETS_ROOT" \
    FIELDSCOPE_CHECKPOINTS_ROOT="$FIELDSCOPE_CHECKPOINTS_ROOT" \
    FIELDSCOPE_REPOSITORY="$repository" \
    FIELDSCOPE_TRACKED_CONFIG="$tracked_config_path" \
    FIELDSCOPE_READOUT_CONFIG="$config_path" \
    FIELDSCOPE_RUNTIME_PROFILE="$runtime_profile" \
    FIELDSCOPE_READOUT_RUNTIME_PROFILE="$readout_runtime_profile" \
    FIELDSCOPE_CACHE_TAG="$cache_tag" \
    FIELDSCOPE_PYTHON="$python_bin" \
    FIELDSCOPE_EXPECTED_REVISION="$FIELDSCOPE_EXPECTED_REVISION" \
    FIELDSCOPE_IMAGENET1K_ROOT="${FIELDSCOPE_IMAGENET1K_ROOT:-}" \
    FIELDSCOPE_RECOVERY_WRAPPER_ROOT="$wrapper_root" \
    bash "$final_wrapper"
}

mkdir -p "$log_root" "$external_root"
exec 9>"$lock_file"
if ! flock -n 9; then
  echo "another fixed-revision recovery holds $lock_file" >&2
  exit 13
fi
printf '%s\n' "$$" >"${pid_file}.tmp.$$"
mv -f "${pid_file}.tmp.$$" "$pid_file"
trap 'rm -f "$pid_file"' EXIT
exec > >(tee -a "$recovery_log") 2>&1

verify_repository
verify_no_formal_worker "$$"
verify_signal_decision
verify_required_artifacts
materialize_readout_config
verify_readout_contract
# ImageNet-100 is the authoritative already-complete cache needed by the
# readout runtime gate. Other task caches are audited immediately after their
# tracked-config extraction/resume step instead of being required up front.
verify_cache_manifests imagenet100
write_state preflight_passed
wait_for_free_gpu
verify_repository
verify_no_formal_worker "$$"
if ! verify_readout_profile; then
  echo "$(date --iso-8601=seconds) generating fresh schema-2 readout runtime profile"
  generate_readout_profile
fi
verify_readout_profile
write_state runtime_profile_passed
verify_repository
verify_no_formal_worker "$$"
write_state running
cd "$repository"
run_formal_recovery
