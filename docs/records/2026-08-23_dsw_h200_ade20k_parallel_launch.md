# H200 ADE20K cross-task parallel launch

- Launched and audited at `2026-08-23T10:37:12.076154Z` on `dsw-h200` at fixed revision `020c1de567edd88e0eda245fd085335ffe678f47`; formal and analysis worktrees were clean.
- Existing primary recovery, ImageNet-100 GPU-6 worker, and VOC2012 GPU-7 worker were not restarted or modified.
- New worker: ADE20K segmentation, physical GPU 7, parent/PGID `2969678`, CLI PID `2969687`, safety-monitor PID `2969870`; 80 epochs, batch 2, 150 segmentation classes, 20 registered representations and seeds `4121/7319/104729`.
- Parallelism remains cross-task only. The ADE20K output root is disjoint from ImageNet-100 and VOC2012, and an independent lock/PID file prevents a duplicate ADE20K writer. The registered runtime contract remains `selected_profile.seed_workers=1`, formal cache batch 2, and readout memory cache 0 GiB.
- ADE20K train/val/test manifests remain `passed`, complete, clean, and bound to the fixed revision for `18189/2021/2000` samples.
- GPU-7 resource-guard pause remains present. The ADE20K-only process group was verified to contain parent, task script, matrix script, and CLI only. If primary recovery PID `41610` exits, the sidecar sends `TERM` to the full group, waits 15 seconds, then escalates to `KILL` if the CLI remains.
- A post-launch read-only `tail` check failed because a PowerShell-to-SSH pipeline retained a trailing carriage return in the log path. The worker, lock, and monitor had already launched successfully; training did not fail or restart. This negative operational artifact is preserved separately.
- After launch, the three FieldScope workers consumed approximately 96 CPU cores in aggregate inside a 100-core cgroup. Therefore NYUv2 was not started immediately despite ample GPU memory; an additional worker is gated on observed aggregate throughput and CPU headroom.
- This launch is operational evidence only: `execution_complete=false`, `method_effectiveness_conclusion=null`, `changes_scientific_verdict=false`.

Authoritative launch audit: `artifacts/reports/ade20k_gpu7_parallel_launch_audit_20260823.json` (SHA-256 `e5a82171b85833351447023dfe2429b3fa7eda855a78d6d03f13dd2be882970b`).

Preserved non-authoritative failed check: `artifacts/reports/ade20k_gpu7_parallel_launch_non_authoritative_failed_20260823.json` (SHA-256 `315b9ff1b248da8046b049d0609b2b80d1ebaec0bce91fa74ac29a9b2313d5fe`).

## Epoch-1 failure

At `2026-08-23T11:11:58Z`, the first ADE20K cell
`random_feature_local / seed 4121` exited before committing an atomic epoch,
training report, checkpoint, or matrix entry. The immutable worker log records
`Non-finite training loss task=segmentation representation=random_feature_local
seed=4121 epoch=1`, followed at `loss.backward()` by
`CUBLAS_STATUS_EXECUTION_FAILED` in `cublasSgemmStridedBatched`. The log SHA-256
is `519edea4a325e5fb24ccd5d3939cd538b283d065303d159f5eaf9428490d029f`.

The parent/CLI/monitor PIDs `2969678/2969687/2969870` were all dead at audit,
the worker PID file had been removed by its exit trap, the independent lock was
free, and the ADE output root contained no files. Consequently ADE20K remains
at committed epoch `0`, and no formal optimizer step or sample exposure is
credited. Both fixed-revision worktrees remained clean at
`020c1de567edd88e0eda245fd085335ffe678f47`; train/val/test manifests remained
passed and complete for `18189/2021/2000` samples with their original hashes.

This was not an OOM or disk-full event: all cgroup OOM counters were zero and
the project filesystem retained `43121211604992` available bytes. GPU 7 showed
zero volatile/aggregate ECC errors, no pending row remap, and no requested
recovery action. A later kernel `Xid 43` at `11:14:27Z` reported transient PID
`1067227`, not any failed ADE PID; its causal relation is therefore
`not_inferred`, not silently attributed either to hardware or to ADE.

The failed condition and all negative evidence are preserved. No learning
rate, batch size, seed, representation, cache, or scientific contract was
changed. Recovery is limited to fixed-revision diagnostics and an exclusive
same-condition retry. This remains operational failure evidence rather than a
method-effectiveness result: `execution_complete=false`,
`method_effectiveness_conclusion=null`, `changes_scientific_verdict=false`.

Machine-readable failure audit:
`artifacts/reports/ade20k_gpu7_epoch1_failure_audit_20260823.json`.

## Same-condition retry 1

After the immutable failure audit was synchronized, GPU 7 passed a fixed-UUID
deterministic CUBLAS health check: one visible H200, 12 synchronized 4096×4096
matrix iterations, and all checked scalar values finite. ECC counts remained
zero and the driver requested no recovery action. This is operational health
evidence only and does not erase or reinterpret the failed first attempt.

At `2026-08-23T11:23:37Z`, ADE20K retry 1 started with parent/PGID `3232226`,
CLI PID `3232250`, and safety-monitor PID `3232368`. It uses the same 80 epochs,
batch 2, LR `0.001`, weight decay `0.0001`, 150 classes, representation/seed
order, cache manifests, fixed revision, `seed_workers=1`, and zero-GiB readout
memory cache. `strict_host_sync` was not added. No ADE output file existed at
restart, the old failure log was preserved separately, the independent lock is
held, and the verified process group contains only the ADE parent, task script,
matrix script, and CLI. The existing recovery-exit `TERM`→15 seconds→`KILL`
contract remains active.

Several first read-only diagnostics failed because the PyTorch UUID object was
not normalized/JSON-serializable and because a PowerShell→SSH tail command
retained a carriage return. They did not start a second worker, restart retry 1,
or alter formal outputs and are preserved in the non-authoritative failed-check
artifact. Retry 1 is running but has no atomic epoch yet; formal ADE progress
therefore remains 0. `execution_complete=false`,
`method_effectiveness_conclusion=null`, `changes_scientific_verdict=false`.

Machine-readable launch audit:
`artifacts/reports/ade20k_gpu7_retry1_launch_audit_20260823.json` (SHA-256
`cd0849e3c7d2d7f78f32d900f58ffcbd7bb0c617465f0a0e7e9ac73cdf46a27c`).

Preserved non-authoritative failed checks:
`artifacts/reports/ade20k_gpu7_retry1_launch_non_authoritative_failed_20260823.json`
(SHA-256 `dec80bdaf0af3bbb17db0782bb726d87dfb7bf4f7f0866a90b4886aa7b0dfa03`).

## Deterministic all-ignore root cause and recovery gate

A complete low-priority read-only scan of all 18,189 train targets found 16
samples with zero valid pixels and no invalid non-ignore label. Replaying the
registered epoch-1 sampler exactly showed one all-ignore batch: zero-based batch
1697 contains cache indices `5336/5328`, sample IDs
`ADE_train_00005913/ADE_train_00005905`. With mean reduction, PyTorch 2.7.1
cross-entropy over only ignore label 255 returns NaN. The first failure is
therefore deterministic data-edge behavior rather than evidence of OOM, cache
corruption, or method effectiveness. Full-scan SHA-256:
`ef4d7b5a0a9474fbe8692de8ca3c87191a80b8335155ce292719cb28fcf83788`.

Retry 1 was stopped after this proof, before it reached the guaranteed failing
batch. It produced no formal output or atomic progress. Its final log contains
only `Terminated` and has SHA-256
`b4a6c06672677cf0e25ec72bec83beae33305cee4b8087f168962014373b02bb`;
exit-evidence SHA-256 is
`aad83fc4aae8b5c5d67c5e0fffe9576ee07d04f28955a2712953ea417a779703`.
Stopping this provably redundant attempt does not erase the original failure.

The recovery protocol was locked before retry 2. It retains every sample and
keeps batch size, sampler, seeds, LR, weight decay, epochs, representation
order, optimizer steps, and sample exposures unchanged. A content-addressed
ADE-only `sitecustomize.py` returns differentiable `logits.sum() * 0.0` only
when the entire segmentation batch has no valid target pixels; the registered
optimizer step still executes. Every batch with at least one valid pixel calls
the original PyTorch cross-entropy unchanged. Non-finite logits remain
non-finite and are still rejected by the fixed finite-value gate.

The final shim SHA-256 is
`f95bbb0ed70ebbc127e6e08cd8d47c7bb18e6d394c2233a5677dc9cdfbf7bac4`.
Ruff and `py_compile` passed. A GPU-7 self-test proved exact loss and gradient
equality for valid and mixed batches, zero finite loss/gradient for all-ignore,
unchanged non-mean behavior, and continued rejection of non-finite logits;
self-test SHA-256 is
`1aa00970165abc2ace956ba60259df0d5cc1b845acfe42f9a678d0162c291cc8`.
Three earlier content-addressed import-order candidates are retained but were
never used for formal training.

Protocol:
`docs/experiment_plans/2026-08-23_ade20k_all_ignore_recovery_protocol.md`.
Machine-readable recovery gate:
`artifacts/reports/ade20k_all_ignore_recovery_gate_20260823.json`.
This is an implementation-completeness repair only:
`execution_complete=false`, `method_effectiveness_conclusion=null`,
`changes_scientific_verdict=false`.

## Retry 2 launch

At `2026-08-23T11:49:51Z`, retry 2 started on physical GPU 7 with parent/PGID
`3634680`, CLI PID `3634699`, safety-monitor PID `3635008`, and low-priority
exit-evidence monitor PID `3639896`. Before launch, the command reverified the
two clean fixed-revision worktrees, primary recovery PID `41610`, GPU-7 guard
pause, empty ADE output root, free exclusive lock, absence of another ADE
writer, and exact SHA-256 values for the recovery gate, full label scan,
self-test, and final shim.

The CLI environment directly confirms `CUDA_VISIBLE_DEVICES=7`, the final shim
in `PYTHONPATH`, and its explicit SHA identity. The ADE process group contains
only parent/task/matrix/CLI and holds the independent lock. The same 80 epochs,
batch 2, LR, weight decay, 150 classes, 20 representations × 3 seeds,
`seed_workers=1`, and zero-GiB readout cache remain active. At launch audit it
had no atomic epoch or formal output, while ImageNet-100/VOC2012/NYUv2 had
continued independently to `epoch 32 / epoch 37 / epoch 8` for their active
cells.

Machine-readable retry-2 launch audit:
`artifacts/reports/ade20k_gpu7_retry2_launch_audit_20260823.json`.
The decisive next ADE evidence is an atomic epoch-1 report that crosses the
registered all-ignore batch; process liveness alone remains non-evidence.

The full fixed sampler schedule was also enumerated for every seed and all 80
epochs. It contains 22 all-ignore training batches: `7/10/5` for seeds
`4121/7319/104729`, respectively, out of 43,656,000 registered optimizer steps.
Thus the shim is required across the whole ADE matrix, not only the first
epoch. Validation contains one all-ignore sample but zero all-ignore batches at
batch 2; test contains none. Schedule SHA-256:
`ace13a3ba06dd20f28e45a29a9e692e30ff657a7c91e2a84412f1a50de0571fd`;
val/test scan SHA-256:
`e60d7cbd259baff88986b17cbae2118e89510674e80101d7b08e7c847e9c0e78`.
Combined coverage audit:
`artifacts/reports/ade20k_all_ignore_coverage_audit_20260823.json`.

A one-shot read-only epoch-1 monitor is active as PID `3678237` (`nice=19`,
60-second poll). Script SHA-256:
`d56bd39ac8bb06d701da731c8d547418e247df506d31f20697ea50b0032b1c6a`.
When the first atomic report appears, it runs the standard strong cell audit
and additionally binds the live retry-2 CLI environment, shim SHA, recovery
gate, full label scan, fixed worktrees, and registered all-ignore batch into a
single machine-readable audit. If retry 2 exits first, it records that failure
instead. Monitor launch audit:
`artifacts/reports/ade20k_retry2_epoch1_bound_audit_monitor_launch_20260823.json`.

At `2026-08-23T12:11:17Z`, retry 2 and its one-shot epoch-1 monitor remained
alive, with no ADE report or checkpoint yet. The CLI continued to hold the
exclusive ADE lock on physical GPU 7; no second ADE writer was started. The
same snapshot showed an external M3Call worker on each authorized GPU, using
about `61808/61522 MiB` on GPUs `6/7`, while approximately `77407/79025 MiB`
remained free. These external processes were not terminated. All four
FieldScope task workers remained alive and the registered per-task
`seed_workers=1` contract was unchanged. This overlap is preserved in
`artifacts/reports/gpu67_m3call_overlap_20260823.json`; its scientific effect is
`not_inferred`. ADE progress remains exactly zero until the epoch-1 atomic
report and shim-bound audit pass.

## Retry 2 epoch-1 acceptance

At `2026-08-23T14:17:38Z`, retry 2 produced its first atomic report and passed
both the standard external strong audit and the ADE-specific shim-bound audit.
The standard audit SHA-256 is
`cf40806701b71423e4a27ce751784f1580f63e460614ac75df9ed73b6813f4f1`;
the bound audit SHA-256 is
`d99c553ac9aa0b66fc32bff4ac734d873cffcfdcd796f0e5a80d6e73424e4782`.
Both report `status=passed` and `problems=[]`.

Epoch 1 contains exactly `9095` optimizer steps and `18189` sample exposures,
with finite train loss `3.269691086987271`. The standard audit closes the
history, optimizer, scheduler `1/80`, four RNG states, report/checkpoint,
cache/control/config, fixed revision, and both clean worktrees. The enhanced
audit additionally binds the live GPU-7 CLI environment to final shim SHA-256
`f95bbb0ed70ebbc127e6e08cd8d47c7bb18e6d394c2233a5677dc9cdfbf7bac4`,
recovery-gate SHA-256
`fe59df4ce274fb809dd12acb3f143872cdade242b102a7ce7ee2c0b8d703625d`,
label-scan SHA-256
`ef4d7b5a0a9474fbe8692de8ca3c87191a80b8335155ce292719cb28fcf83788`,
and registered all-ignore batch 1697 with cache indices `5336/5328`.

Train/validation times were `7209.330173842609 / 1636.0241944994777`
seconds. Validation mean-IoU/pixel accuracy were
`0.0033956333063542843 / 0.1695859432220459`; this low intermediate result is
preserved without reinterpretation. The original failure and stopped retry-1
evidence remain authoritative provenance. ADE formal progress is now one
committed epoch but zero terminal cells; this does not trigger any main,
causal, extension, replay, or final verdict gate.

At `2026-08-23T14:22:12Z`, the independent automatic watcher also audited the
same committed ADE epoch 1 and returned `status=passed`, `problems=[]`; artifact
SHA-256 is `10ab8c6e2ae619a941e16ebf9e9407fd16fe9c39c51dab5acc49129937d494c5`.
This is additional duplicate evidence for the already credited `9095` optimizer
steps and `18189` exposures, so it is preserved without incrementing the global
ledger a second time. Retry 2 continued into epoch 2.
