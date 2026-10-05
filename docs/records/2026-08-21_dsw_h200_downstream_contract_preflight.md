# DSW H200 downstream-contract preflight

At `2026-08-21T08:34:34Z`, the fixed-revision orchestration chain was audited read-only while the formal main matrix continued on physical GPU 6. This audit covers execution wiring only and does not constitute a method-effectiveness result.

The source and analyzer worktrees are both clean and fixed at `020c1de567edd88e0eda245fd085335ffe678f47`. The active H200 readout-recovery wrapper and watchdog match the SHA-256 identities registered in the readiness record.

The downstream chain is complete and fail-closed:

- The main matrix is exactly four tasks × 20 registered representations × three seeds, or 240 cells.
- After all four matrices, `audit-full-evidence` writes the main machine-readable decision.
- All four registered causal/conditional controls run after either a complete positive or complete negative main result: random flow, spatially shuffled probe, neutral prompt, and unrelated prompt.
- The ImageNet-1k and 15-variant VOC extension is entered only when the main verdict is exactly `main_tasks_supported_pending_causal_audits`. Negative main evidence skips it. Resource shortage causes waiting without shrinking the registered scope.
- Final evidence always requires causal evidence; extension evidence is required only on the positive-main path.
- The already-running supervised-error waiter accepts only a complete same-revision final decision. It then runs four tasks with 33 checkpoint reports per task, giving the registered 132 per-sample replays, and requires 12 registered comparisons per task.
- Each replay requires aligned unique sample IDs, sample-ID SHA-256 agreement, finite per-sample values, and formal checkpoint/cache provenance. The generated registry must be `passed` and must not change the main verdict.
- The terminal completion audit is accepted only when `status=passed`, `execution_complete=true`, and `changes_scientific_verdict=false`.

The waiter is alive as PID `41355`, bound to the formal source worktree, the independent analysis worktree, and the registered analysis-output directory. No final decision, replay registry, or completion audit exists yet, so `method_effectiveness_conclusion=null` and the goal remains active.

Machine-readable details and all current wrapper/module SHA-256 identities are in `artifacts/reports/2026-08-21_dsw_h200_downstream_contract_preflight.json`.
