# FieldScope controlled shutdown

- Time (UTC): 2026-08-26T14:29:55.738556+00:00
- Fixed revision: `020c1de567edd88e0eda245fd085335ffe678f47`
- Hosts: `dsw-h200`, `dsw-M3Call`
- Main-matrix terminal cells at stop: `74/240` (`{'imagenet100': 13, 'nyuv2': 34, 'voc2012': 27}`)
- Intermediate audits: `897`; classified transient races: `25`; unresolved failures: `0`
- All registered FieldScope process groups exited; all formal locks were verified free. No `SIGKILL` was needed.
- Stable reports/checkpoints and all negative evidence were retained.
- Scientific downstream artifacts remain absent; `execution_complete=false`, `changes_scientific_verdict=false`.
- Machine-readable audit: `/mnt/omni_ssd/user_workspace/wangzixi/FieldScope/FieldScope-internal/artifacts/reports/controlled_shutdown_cross_host_20260826T142955Z.json` (SHA-256 `7bbe699f3fd838879916258c8db62783e62fcfc1c8bd368d671e58010b2c4ca2`)
- Host audits: /mnt/omni_ssd/user_workspace/wangzixi/FieldScope/FieldScope-internal/artifacts/reports/controlled_shutdown_20260826T142749Z_h200.json, /mnt/omni_ssd/user_workspace/wangzixi/FieldScope/FieldScope-internal/artifacts/reports/controlled_shutdown_20260826T142751Z_a800.json

Resume must use the fixed worktree, existing locks, and the recovery/watchdog chain; the two 80/80 reports not yet included in the terminal-audit count must be reconciled before advancing the ledger.
