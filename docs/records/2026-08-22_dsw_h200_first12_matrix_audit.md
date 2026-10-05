# H200 ImageNet-100 first-12-cell matrix audit

- Observed at: `2026-08-21T16:16:20Z` (`2026-08-22T00:16:20+08:00`).
- Host/device: `dsw-h200`, physical GPU 6.
- Fixed revision: `020c1de567edd88e0eda245fd085335ffe678f47`; formal and analysis worktrees were clean.
- Scope: the first 12 completed ImageNet-100 main-matrix cells only: `random_feature_local`, `z0`, `zt`, and `trajectory`, each with seeds `4121`, `7319`, and `104729`, in the registered representation-outer/seed-inner order.
- Corrected audit status: `passed`; `problems=[]`.
- Matrix remains `status=running`; matrix SHA-256 at audit was `d6dfa5f2e50b8e96f8205c4a01162078113e2562afec41fb9fa312e243a350aa`.
- The 12 cells contain `982800` completed optimizer steps and `125771400` completed training-sample exposures.
- Every cell has a terminal 90-epoch training report with contiguous history, `81900` optimizer steps and `10480950` exposures.
- Every terminal last-checkpoint SHA, best-checkpoint SHA, held-out test report, test checkpoint identity, test metric copied into the matrix, batch/revision contract, and stable report read passed.
- Full per-cell report/checkpoint/test SHA-256 values and held-out metrics are retained in the machine-readable audit.
- This is intermediate matrix evidence only. It does not establish the four-task main result or method effectiveness; `execution_complete=false`, `method_effectiveness_conclusion=null`, and `changes_scientific_verdict=false`.

The first bulk-audit operation was non-authoritative and failed only because the audit code resolved the expected checkpoint path to an absolute path before comparing it with the registered relative path held identically by the matrix, training report, and test report. That operation is preserved with SHA-256 `ef319fa527bc5874a95cf6e5998b266a869838ff3862ce65330bf7ae900e0b0d`. The corrected audit uses the fixed-revision `_require_report_contract` raw-path semantics.

Machine-readable corrected audit: `artifacts/reports/imagenet100_first12_completed_cells_strong_audit_20260821.json` (SHA-256 `15a4292524a94738c0e867ec5a3cca6389e1ac17a30af1a2b1ff053a3e2afe77`). Preserved non-authoritative failure: `artifacts/reports/imagenet100_first12_completed_cells_strong_audit_non_authoritative_failed_20260821.json`.
