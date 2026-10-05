# H200 ImageNet-100 velocity seed-7319 epoch-10 audit

- Observed at: `2026-08-22T05:28:48.993843Z` (`2026-08-22T13:28:48.993843+08:00`).
- Cell: ImageNet-100 / `velocity` / seed `7319`; epoch 10/90 committed at `2026-08-22T05:01:34.887283Z`; status remains `running`.
- Host/device/revision: `dsw-h200`, physical GPU 6, `020c1de567edd88e0eda245fd085335ffe678f47`.
- Strong audit: `passed`; `problems=[]`; contiguous history `1..10`.
- Cumulative optimizer steps/exposures: `9100 / 1164550`; increment from the last recorded epoch-7 audit exactly `2730 / 349365`; AdamW step `9100`; scheduler `last_epoch=10/T_max=90`.
- Epoch-10 train/validation seconds: `2916.47917692177 / 315.47849005740136`; validation top-1/top-5 remained `0.010046367851622875 / 0.05023183925811438`; best remains epoch 1.
- Report SHA-256: `6bbc3d2092a943d59d244dcc4e532a9bb84d74ddc86063262b939f169be462e3`.
- Last/best checkpoint SHA-256: `15101ebeae8c8a8c2fac667432d5aa28472f5a75eb4ee78e7bf6c1cdfbbc7aed` / `ee2c47e02764fa58d2c63493584b2780056d456b96583f6a7e0f03adfa404a98`.
- Report/checkpoint history under JSON tuple normalization, RNG, cache/config/control/coverage, registered runtime profile `selected_profile.seed_workers=1`, readout batch `128`, cache batch `2`, memory cache `0 GiB`, stable double-read and both clean fixed-revision worktrees all passed. The running report correctly retains `last_checkpoint_sha256=null`.
- GPU 6 contained the FieldScope worker, paused resource guard and an external `m3call_baselines` process using about `61806 MiB` at audit time. No external process was terminated and no FieldScope batch/cache/runtime/scientific contract was changed; resource-overlap scientific effect remains `not_inferred`.
- The first audit attempt read `seed_workers` from an incorrect JSON path and therefore produced a non-authoritative `failed` audit. It is preserved at `artifacts/reports/imagenet100_velocity_seed7319_epoch10_strong_audit_non_authoritative_failed_20260822.json` with SHA-256 `c35e374c6da05a5173f4c7ad0c4682f4ae57c96b607b19bff9338a92309115c1`. This was an audit-reader failure, not a training or configuration failure.
- The unchanged negative intermediate metric is preserved without interpretation. This remains single-cell intermediate evidence only; `execution_complete=false`, `method_effectiveness_conclusion=null`, `changes_scientific_verdict=false`.

Machine-readable authoritative audit: `artifacts/reports/imagenet100_velocity_seed7319_epoch10_strong_audit_20260822.json` (SHA-256 `25c0e48395af190f71b429f1fbb70b50d42911194c548e5284ec6d6b11f48387`).
