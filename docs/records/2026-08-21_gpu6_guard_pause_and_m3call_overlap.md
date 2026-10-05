# GPU 6 guard pause and external overlap provenance

Observed on `dsw-h200` at 2026-08-21T13:54:42Z while the fixed-revision formal worker was running ImageNet-100 / `velocity` / seed 4121 at committed epoch 85/90.

The resource guard's process-level samples attributed approximately 88--99% SM to guard PID 1813 while the FieldScope worker PID 102910 remained a small cached-readout workload. The guard's own unoccupied probes measured only 0--10%, below its 25% automatic-yield threshold, and therefore repeatedly restarted the large-matmul occupier. Following the guard's documented per-device control interface, `gpu.6.pause` was created for the current runtime. The guard monitor remains alive, GPU 6 reports `state=paused` with `pause_scope=device`, and GPUs 0--5 were not changed. No FieldScope batch, cache, seed, runtime, code, or revision contract changed.

After release, an external M3Call `torchrun --nproc_per_node=2` job was directly observed on GPUs 6 and 7. Its GPU-6 child PID 1841812 held approximately 61064 MiB and sampled at roughly 8--16% SM. FieldScope did not terminate or mutate that external job. The FieldScope worker, watchdog, recovery wrapper, and analyzer remained alive; worker `rchar` continued to grow and the report remained atomically committed at epoch 85.

At 2026-08-21T13:58:19Z the M3Call launcher and both children were no longer present; they ended naturally. GPU 6 then contained only the paused guard context and the FieldScope context. A subsequent 20-second `pmon` window rounded FieldScope SM usage to 0% while `rchar` still increased by about 1.05 GB, identifying this portion of the cached-readout epoch as predominantly cache/CPU-path work rather than a dead worker.

This is resource-overlap provenance only. Its scientific effect is `not_inferred`; it does not establish either absence or presence of an effect on the final method comparison. `changes_scientific_verdict=false`.

Machine-readable record: `artifacts/reports/2026-08-21_gpu6_guard_pause_and_m3call_overlap.json`.
