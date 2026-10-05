# Cross-lane ownership handback preflight

Status: operational `action_required`; formal scientific validation remains
active with `execution_complete=false`, `method_effectiveness_conclusion=null`,
and `changes_scientific_verdict=false`.

At `2026-08-25T06:56:00Z`, a read-only audit confirmed a future ownership
boundary that is not covered by the existing H200 and A800 side-worker
monitors. The active primary recovery PID `41610` is executing the fixed
revision H200 recovery wrapper with SHA-256
`98fc2baea7203d3c225a890deb6eb26307f4164deb8d94f1b73997b7b662f3e9`.
Its loaded `run_formal_recovery` loop enters datasets in the order
ImageNet-100, VOC2012, ADE20K, and NYUv2. The wrapper does not acquire the
registered per-task locks held by the side-worker launchers.

The local parallel monitor SHA-256
`e0677f25e01dccb4ccaf723a7f188efecd5cc760aa423877bce5a843161b8960`
checks fixed checkout and authoritative-watchdog liveness, but not the primary
recovery's current dataset. The A800 supervisor SHA-256
`5d39bfe899459e688962898ab9354696ccb526fb83a39bb64d9d3c85c488fe66`
exits when its own matrix reaches 60 terminal runs, but otherwise may restart a
missing worker while the shared H200 heartbeat remains active. Therefore, if a
side-owned matrix is incomplete when the primary ImageNet-100 worker exits,
the primary recovery can enter the same dataset without an atomic ownership
handback.

No overlap exists now. The authoritative `2026-08-25T06:46:25Z` ledger has
ImageNet-100 `13/60`, NYUv2 `13/60`, VOC2012 `12/60`, and ADE20K `0/60`, for
`38/240` terminal cells. All four current writers remain unique. This audit
sent no signal, restarted no process, changed no fixed worktree, and wrote no
formal output.

Before ImageNet-100 reaches `59/60` terminal cells, a content-addressed,
tested, fail-closed handback gate must be deployed. It must freeze the primary
recovery parent before the ImageNet-100 worker exits, stop side-worker
supervisors before releasing side-worker ownership, preserve the last atomic
checkpoints and all negative evidence, require acknowledgements from the H200
ADE lane and both A800 lanes, and resume the primary recovery only after those
acknowledgements pass. Deployment must not restart the current formal workers.

Machine-readable preflight:
`artifacts/reports/cross_lane_ownership_handback_preflight_20260825.json`.
This is an operational safety finding, not method-effectiveness evidence and
not a downstream trigger.
