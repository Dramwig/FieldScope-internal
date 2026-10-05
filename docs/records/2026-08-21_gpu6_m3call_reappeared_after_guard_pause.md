# GPU 6 external M3Call overlap reappeared after guard pause

At 2026-08-21T14:04:51Z, while the fixed-revision FieldScope worker was continuing ImageNet-100 / `velocity` / seed 4121 from committed epoch 85/90, a second external M3Call GPU 6/7 run was observed. Its children PID 2057281 and 2057282 started at 14:01:15Z. The GPU-6 child held approximately 61010 MiB and sampled at 15--21% SM; the FieldScope worker sampled at 0--1% SM in the same short window while its cache-read counter continued to grow.

The documented GPU-6-only pause for resource-guard PID 1813 remained effective (`state=paused`, `pause_scope=device`), so this utilization was not attributed to the guard. FieldScope did not terminate or mutate the external job and did not alter its revision, batch, seed, cache, runtime, or recovery contracts.

This is a distinct resource-overlap provenance event. Scientific effect remains `not_inferred`; neither no-effect nor effect is claimed. `changes_scientific_verdict=false`.

At 2026-08-21T14:21:18Z the launcher and both GPU children were no longer present; the second overlap ended naturally. FieldScope did not terminate them.

Machine-readable record: `artifacts/reports/2026-08-21_gpu6_m3call_reappeared_after_guard_pause.json`.
