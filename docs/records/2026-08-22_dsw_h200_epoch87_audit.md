# H200 formal readout epoch-87 audit

- Observed at: `2026-08-21T16:05:10Z` (`2026-08-22T00:05:10+08:00`).
- Host: `dsw-h200`; formal device remains physical GPU 6.
- Fixed revision: `020c1de567edd88e0eda245fd085335ffe678f47`.
- Cell: ImageNet-100 / `velocity` / seed `4121`.
- Atomic report commit: `2026-08-21T16:01:44Z`; committed epoch 87/90; report status remains `running`.
- Corrected strong-audit status: `passed`; `problems=[]`.
- Cumulative optimizer steps: `79170`; cumulative training sample exposures: `10131585`.
- Epoch-87 train/validation seconds: `3931.5707562919706` / `362.44979706406593`.
- Epoch-87 validation top-1/top-5: `0.13786707882534777` / `0.37812982998454403`.
- Best remains epoch 75 with validation top-1 `0.14049459041731066`. This is an intermediate cell result, not a scientific verdict.
- Report SHA-256: `d5f9a85860e484c4c9936832cc18c681d398ed792277d60f72de570bcb573aef`.
- Last-checkpoint SHA-256: `1649020101981dfcd1629630ac92a8bb7928fd72c1139c4db8d18f47045df706`.
- Best-checkpoint SHA-256: `ea5c9d87d458f29db3effbd9ddfbb1d577b1849b66f56406e836739e9406e4f3`.
- Stable double-read, contiguous history, AdamW step, scheduler, RNG, report/checkpoint history, cache/config/control/coverage, batch/runtime and clean-worktree checks all passed.
- No external M3Call compute process was observed at commit/audit. Earlier overlap effects remain `not_inferred`; `changes_scientific_verdict=false`.

The first epoch-87 audit operation was non-authoritative and failed because it incorrectly required the running report's registered `last_checkpoint_sha256=null` field to equal an independently computed checkpoint digest. The failed operation is preserved with SHA-256 `4012254ae886931a6e81468a70b23ad7ead82fe9a91621ea4c299b0c0d1a5730`; it changed no scientific artifact or verdict. The corrected audit follows the registered schema and independently records the actual checkpoint digest.

Machine-readable corrected audit: `artifacts/reports/imagenet100_velocity_seed4121_epoch87_strong_audit_20260821.json`; preserved non-authoritative failure: `artifacts/reports/imagenet100_velocity_seed4121_epoch87_strong_audit_non_authoritative_failed_20260821.json`. Remote copies are under `/mnt/omni_ssd/user_workspace/wangzixi/FieldScope/logs/`.
