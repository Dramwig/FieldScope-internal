# DSW H200 epoch-81 audit record

At `2026-08-21T09:46:09.212285Z`, ImageNet-100 `velocity` / seed `4121` passed the semantic strong audit at fixed revision `020c1de567edd88e0eda245fd085335ffe678f47`.

- Epoch `80 -> 81`: `910` optimizer steps and `116455` training-sample exposures.
- Cumulative totals: `73710` optimizer steps and `9432855` exposures.
- History `1..81` is contiguous; AdamW step is `73710`; scheduler is `last_epoch=81`, `T_max=90`.
- RNG keys are `numpy`, `python`, `torch_cpu`, and `torch_cuda`.
- Registered runtime/cache contracts passed: `seed_workers=1`, readout batch `128`, cache batch `2`, and readout memory cache `0 GiB`.
- Validation top-1/top-5 is `0.13809891808346214 / 0.37302936630602784`; best epoch remains `75`, with best primary metric `0.14049459041731066`.
- Report/last/best checkpoint SHA-256 values are `60e38f05385df22fa9a01cdb73824d0c6c928fe30e22896c7c1e99a4b8ddfb9c`, `ff315f38c69d89137a8711da5220d87097da36d567e25cde3b2109a8e8202966`, and `ea5c9d87d458f29db3effbd9ddfbb1d577b1849b66f56406e836739e9406e4f3`.
- Audit-log SHA-256 is `4789f9bc702da3cede2fbc926867ccfdcbe95eed16829a1011e161b82aa99a73`.

The report was atomically committed at `2026-08-21T09:42:09.246936811Z`. At that time, an external M3Call/OmniCall worker PID `1817990`, launched at `2026-08-21T09:21:57Z` with `--stop-after-step 256`, was also using physical GPU 6 and held approximately `62716 MiB`. The FieldScope operator did not terminate that process, restart the formal worker, or modify any runtime or scientific contract. The overlap is retained as provenance; no absence or presence of a scientific effect is inferred from the resource observation.

The audit verified stable double reads, report/checkpoint history equality under JSON tuple normalization, AdamW/scheduler/RNG state, registered cache/control/coverage contracts, the selected `seed_workers=1` runtime profile, and clean formal and analysis worktrees at the fixed revision. `problems=[]`.

This remains intermediate cell evidence only: `execution_complete=false`, `method_effectiveness_conclusion=null`, and `changes_scientific_verdict=false`. The worker continued toward epoch 82.
