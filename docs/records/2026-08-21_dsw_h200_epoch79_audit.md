# DSW H200 epoch-79 audit record

At `2026-08-21T08:28:14.338779Z`, ImageNet-100 `velocity` / seed `4121` passed the semantic strong audit at fixed revision `020c1de567edd88e0eda245fd085335ffe678f47`.

- Epoch `78 -> 79`: `910` optimizer steps and `116455` training-sample exposures.
- Cumulative totals: `71890` optimizer steps and `9199945` exposures.
- History `1..79` is contiguous; AdamW step is `71890`; scheduler is `last_epoch=79`, `T_max=90`.
- RNG keys are `numpy`, `python`, `torch_cpu`, and `torch_cuda`.
- Registered runtime/cache contracts passed: `seed_workers=1`, readout batch `128`, cache batch `2`, and readout memory cache `0 GiB`.
- Validation top-1/top-5 is `0.1394899536321484 / 0.3829211746522411`; best epoch remains `75`, with best primary metric `0.14049459041731066`.
- Report/last/best checkpoint SHA-256 values are `f08f92952a25bcbf0be489cccf7c130905088e32988423d809659ff2e998f0a8`, `9412369343af5170efc50c0034163ef7b6b9d66a4be5dc5d741a9c29266f4d30`, and `ea5c9d87d458f29db3effbd9ddfbb1d577b1849b66f56406e836739e9406e4f3`.
- Audit-log SHA-256 is `1dc624d0fa11df7af8e2823682dfac11589af8a0abcd98b4779dccb5da761702`.

The audit verified stable double reads, report/checkpoint history equality under JSON tuple normalization, the registered cache/control/coverage contracts, and a clean formal worktree at the fixed revision. `problems=[]`.

During the surrounding epoch, the kernel repeatedly emitted `mpt3sas` controller `log_info` events (`0x310f0400`, `0x3003011d`, and `0x30030109`). At the `2026-08-21T08:30:28Z` observation there were no explicit kernel `I/O error` lines, no NVIDIA Xid lines, and no worker exit. The FieldScope worker remained active on physical GPU 6 at 100% utilization and completed the epoch-79 atomic commit. This is retained as nonfatal resource provenance; no restart or scientific-artifact modification was performed.

This remains intermediate cell evidence only: `execution_complete=false`, `method_effectiveness_conclusion=null`, and `changes_scientific_verdict=false`. The worker continued toward epoch 80.
