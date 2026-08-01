#!/usr/bin/env bash
set -euo pipefail

: "${FIELDSCOPE_ROOT:?Set FIELDSCOPE_ROOT}"
: "${FIELDSCOPE_DATASETS_ROOT:?Set FIELDSCOPE_DATASETS_ROOT}"

python_bin="${FIELDSCOPE_PYTHON:-python}"
archive="${FIELDSCOPE_IMAGENET100_ARCHIVE:-$FIELDSCOPE_DATASETS_ROOT/raw/imagenet100/imagenet100_full.tar}"
prepared_root="${FIELDSCOPE_IMAGENET100_PREPARED_ROOT:-$FIELDSCOPE_DATASETS_ROOT/prepared/imagenet100}"
report_root="$FIELDSCOPE_ROOT/FieldScope-internal/outputs/asset_verification"
expected_bytes=17319391232
expected_members=134600
expected_sha256="c5b57e1f6b6994d709ba5842952e7c669d0dd9b6d5ab2390e603e5fc2ffb0e6d"

if [[ ! -f "$archive" ]]; then
  echo "missing ImageNet-100 archive: $archive" >&2
  exit 2
fi
actual_bytes="$(stat -c '%s' "$archive")"
actual_mtime_ns="$(
  "$python_bin" - "$archive" <<'PY'
import sys
from pathlib import Path

print(Path(sys.argv[1]).stat().st_mtime_ns)
PY
)"
if [[ "$actual_bytes" != "$expected_bytes" ]]; then
  echo "archive size mismatch: expected=$expected_bytes actual=$actual_bytes" >&2
  exit 3
fi
actual_sha256="$(sha256sum "$archive" | awk '{print $1}')"
if [[ "$actual_sha256" != "$expected_sha256" ]]; then
  echo "archive SHA-256 mismatch: expected=$expected_sha256 actual=$actual_sha256" >&2
  exit 4
fi
actual_members="$(tar -tf "$archive" | wc -l)"
if [[ "$actual_members" != "$expected_members" ]]; then
  echo "archive member mismatch: expected=$expected_members actual=$actual_members" >&2
  exit 5
fi

"$python_bin" - "$archive" "$prepared_root" <<'PY'
import sys
import tarfile
from pathlib import Path

archive = Path(sys.argv[1])
destination = Path(sys.argv[2])
destination.mkdir(parents=True, exist_ok=True)
root = destination.resolve()
with tarfile.open(archive, "r:") as bundle:
    for member in bundle.getmembers():
        target = (destination / member.name).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"unsafe archive member: {member.name}")
        if member.issym() or member.islnk():
            raise ValueError(f"links are not accepted: {member.name}")
    bundle.extractall(destination, filter="data")
PY

read -r class_count train_count validation_count < <(
  "$python_bin" - "$prepared_root" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
classes = [path for path in (root / "train").iterdir() if path.is_dir()]
train = sum(1 for path in (root / "train").rglob("*") if path.is_file())
validation = sum(1 for path in (root / "val").rglob("*") if path.is_file())
print(len(classes), train, validation)
PY
)
if [[ "$class_count" != 100 || "$train_count" != 129395 || "$validation_count" != 5000 ]]; then
  echo \
    "prepared counts mismatch: classes=$class_count train=$train_count val=$validation_count" \
    >&2
  exit 6
fi

mkdir -p "$report_root"
"$python_bin" -m fieldscope.cli audit-dataset-splits \
  --dataset imagenet100 \
  --root "$prepared_root" \
  --image-size 512 \
  --classes-file "$prepared_root/classes.txt" \
  --expected-split train=116455 \
  --expected-split val=12940 \
  --expected-split test=5000 \
  --output "$report_root/imagenet100_remote_split_audit.json"

"$python_bin" - "$report_root/imagenet100_remote_asset_verification.json" <<PY
import json
import sys

payload = {
    "status": "passed",
    "archive": "$archive",
    "archive_bytes": $actual_bytes,
    "archive_mtime_ns": $actual_mtime_ns,
    "archive_sha256": "$actual_sha256",
    "archive_members": $actual_members,
    "prepared_root": "$prepared_root",
    "classes": $class_count,
    "train_images": $train_count,
    "official_validation_images": $validation_count,
}
open(sys.argv[1], "w", encoding="utf-8").write(
    json.dumps(payload, indent=2) + "\n"
)
print(json.dumps(payload, indent=2))
PY
