# H200 formal readout epoch-88 audit

- Observed at: `2026-08-21T17:05:06Z` (`2026-08-22T01:05:06+08:00`).
- Host/device: `dsw-h200`, physical GPU 6; fixed revision `020c1de567edd88e0eda245fd085335ffe678f47`.
- Cell: ImageNet-100 / `velocity` / seed `4121`; atomic report commit `2026-08-21T17:03:49Z`; epoch 88/90; report remains `running`.
- Strong-audit status: `passed`; `problems=[]`.
- Cumulative optimizer steps/exposures: `80080` / `10248040`; AdamW state step `80080`; scheduler `last_epoch=88`, `T_max=90`.
- Epoch train/validation seconds: `3352.447165149264` / `372.19575950317085`.
- Validation top-1/top-5: `0.13717156105100464` / `0.37534775888717153`; best remains epoch 75 with top-1 `0.14049459041731066`.
- Report SHA-256: `235a917551adf799d9277b9935dc6872a1220d1ddc709eff9ad88da3de9ee95e`.
- Last/best checkpoint SHA-256: `880867e19b75ac5f789fa7e7122b5958c21c710c1e228a62436304915e0377c6` / `ea5c9d87d458f29db3effbd9ddfbb1d577b1849b66f56406e836739e9406e4f3`.
- Stable double-read, contiguous report/checkpoint history, RNG, cache/config/control/coverage, batch/runtime profile, and both clean worktrees passed.
- No external M3Call compute was observed. Earlier overlap effects remain `not_inferred`; this intermediate cell result leaves `changes_scientific_verdict=false`.

Machine-readable audit: `artifacts/reports/imagenet100_velocity_seed4121_epoch88_strong_audit_20260821.json`; remote copy: `/mnt/omni_ssd/user_workspace/wangzixi/FieldScope/logs/imagenet100_velocity_seed4121_epoch88_strong_audit_20260821.json`.
