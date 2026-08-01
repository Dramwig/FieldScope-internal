# Extension combined-budget checkpoint gate

Date: 2026-08-02  
Evidence scope: resource planning and execution gating only; no method-effectiveness
claim.

## Pre-result correction

Before any real FieldScope signal, main-task, causal, or extension metric existed,
an audit of the conditional ImageNet-1k runbook found that its first sparse-cache
budget included the registered 8 GiB allowance for 120 best/last readout
checkpoints, but the final combined ImageNet-1k plus 15-ablation disk gate carried
forward only the projected sparse-cache bytes. The first report's `fits` value was
not itself an execution gate, so the combined gate could understate required free
space by 8 GiB.

`scripts/eval/run_extension_after_main.sh` now adds both
`imagenet_projected_bytes` and `imagenet_checkpoint_budget_bytes` to the combined
gate. The existing 1.15 safety factor and 10 GiB reserve remain unchanged. A failed
combined gate still exits `resource_blocked` and does not reduce images, pixels,
classes, representations, seeds, epochs, or ablation variants.

## Independent capacity preview

Using the retained real AuraFlow 512 px, three-time, R=8 resource-gate feature and
the current cache serializer, a one-sample classification
`readout_sparse` payload measured 584,146 bytes. This deliberately conservative
single-sample measurement includes more serialization overhead per sample than a
64-sample formal shard. Applying the registered 1.15 safety factor to the
1,331,167 ImageNet-1k images alone projects far beyond the currently free space on
the fixed 1 TB data volume. The available multi-terabyte public mounts are
read-only. Therefore a positive main-task result will require additional writable
capacity or an explicitly authorized cleanup before the conditional extension can
start; the runbook must report `resource_blocked` rather than silently shrink the
protocol.

This preview is a resource fact, not a task metric or evidence that the core
hypothesis is positive or negative. The formal runbook will repeat its budget from
completed same-revision measurement caches and live free-space values if the
conditional branch is reached.
