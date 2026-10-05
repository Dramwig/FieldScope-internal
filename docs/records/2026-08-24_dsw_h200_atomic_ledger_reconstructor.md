# DSW-H200 Atomic Ledger Reconstructor

At `2026-08-24T10:06:35.984704Z`, a read-only atomic-ledger reconstructor was
validated against the live H200 watcher state. It derives terminal cells only
from passed terminal audits, selects the numerically latest passed intermediate
audit for each non-terminal cell, and excludes a cell from the active subtotal
as soon as its exact dataset/representation/seed terminal key is accepted.
This prevents terminal/active double counting during the transition between a
completed cell and its successor.

The script is `scripts/ops/fieldscope_reconstruct_atomic_ledger.py`, SHA-256
`6c4fd17e10a5f080a6c2903a32a14e3ad1eafbf0b272d0704e092046484983d8`.
Targeted tests passed (`2 passed`) and Ruff passed. The live reconstruction
contained 20 terminal cells and four unique active cells, with terminal
subtotals of 1,225,340 optimizer steps and 136,893,950 exposures and active
subtotals of 107,014 steps and 5,038,918 exposures. The global totals were
1,332,354 steps and 141,932,868 exposures, exactly matching the independently
calculated values.

The generated machine-readable snapshot is
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824.json`,
SHA-256
`e54578b4798683b06ec7816b8ce3c3bfd0ed6ccc2aa35b971c35b612fea7b71b`.
It reports `status=passed`, `problems=[]`, `execution_complete=false`, and
`changes_scientific_verdict=false`. Completion is fail-closed: it requires all
240 terminal cells, no active cells, no watcher failures, and no ledger
problems. The reconstructor does not modify training outputs or infer a
scientific conclusion.
