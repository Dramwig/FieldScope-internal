# DSW H200 epoch-83 audit record

At `2026-08-21T11:36:51.158318Z`, ImageNet-100 `velocity` / seed `4121` passed the semantic strong audit at fixed revision `020c1de567edd88e0eda245fd085335ffe678f47`.

- Epoch `82 -> 83`: `910` optimizer steps and `116455` training-sample exposures.
- Cumulative totals: `75530` optimizer steps and `9665765` exposures.
- History `1..83` is contiguous; AdamW step is `75530`; scheduler is `last_epoch=83`, `T_max=90`.
- RNG keys are `numpy`, `python`, `torch_cpu`, and `torch_cuda`.
- Registered runtime/cache contracts passed: `seed_workers=1`, readout batch `128`, cache batch `2`, and readout memory cache `0 GiB`.
- Validation top-1/top-5 is `0.1375579598145286 / 0.37812982998454403`; best epoch remains `75`, with best primary metric `0.14049459041731066`.
- Report/last/best checkpoint SHA-256 values are `ca93063502ffc13783c990be78d20bfe2cc8c6b2eb028fdb348a258535e708c4`, `78f43805fdc6893944d21bc0962d48e4460e677e36298286c59cbf1575bf3521`, and `ea5c9d87d458f29db3effbd9ddfbb1d577b1849b66f56406e836739e9406e4f3`.
- Audit-log SHA-256 is `d8b5ced33119fe64a82d54e690bac7fface6d6241a8286d43bb9cb4e75d51446`.

The report was atomically committed at `2026-08-21T11:35:56.810681427Z`. At that time, external M3Call/OmniCall worker PID `3336415` was still using physical GPU 6 while completing the resumed step `256 -> 512` workload. Step 512 was observed in its metrics immediately after the FieldScope commit, and the external process was still alive during the audit.

The audit verified stable double reads, report/checkpoint history equality under JSON tuple normalization, AdamW/scheduler/RNG state, registered cache/control/coverage contracts, the selected `seed_workers=1` runtime profile, and clean formal and analysis worktrees at the fixed revision. `problems=[]`. This consistency does not prove that the concurrent external workload had no timing or scientific effect; the overlap remains separate provenance.

This remains intermediate cell evidence only: `execution_complete=false`, `method_effectiveness_conclusion=null`, and `changes_scientific_verdict=false`. The worker continued toward epoch 84.
