# DSW-M3Call A800 Cross-Host Readiness

Status: infrastructure readiness `passed`; no formal worker was started or
migrated, and no scientific effect is inferred.

At `2026-08-24T08:56:00Z`, `ssh dsw-M3Call` exposed two
`NVIDIA A800-SXM4-80GB` devices and 128 logical CPUs. Both fixed worktrees were
clean at `020c1de567edd88e0eda245fd085335ffe678f47`; the project `.venv` loaded
PyTorch `2.7.1+cu126` with CUDA 12.6. All 12 registered feature-cache manifests
for ImageNet-100, VOC2012, NYUv2, and ADE20K were complete and passed with the
same content hashes used by the H200 execution.

The H200 and A800 nodes mount the same CephFS at `/mnt/omni_ssd`. A formal VOC
report had the same inode, size, and mtime from both nodes. A bounded lock test
held a recovery lock on H200 and observed return code 1 from a nonblocking A800
`flock`, establishing that the shared filesystem can enforce a cross-host
single-writer lock.

The candidate partition is ImageNet-100 and ADE20K on H200 GPUs 6 and 7, with
NYUv2 and VOC2012 on A800 GPUs 0 and 1. This may reduce total wall time by
removing CPU, I/O, and GPU scheduling competition on H200, even though an A800
is not expected to beat an H200 for the same isolated task. The current
conservative estimate is a 1.2--1.5 total throughput multiplier; it is a
planning estimate, not measured formal throughput.

No migration was executed. A safe launch still requires an atomic cell or
terminal boundary, confirmed exit of the old task writer, a task-specific
cross-host lock, an independent A800 process group and PID record, and recovery
logic that cannot relaunch the same task on H200. The machine-readable report
is `artifacts/reports/dsw_m3call_a800_cross_host_readiness_20260824.json`.
