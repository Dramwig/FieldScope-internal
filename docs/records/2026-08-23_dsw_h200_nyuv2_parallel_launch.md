# H200 NYUv2 cross-task parallel launch

- Launched on `dsw-h200` after ADE20K remained stable without OOM or output-lock conflict. Fixed revision: `020c1de567edd88e0eda245fd085335ffe678f47`; formal and analysis worktrees were clean.
- New worker: NYUv2 depth, physical GPU 6, parent/PGID `2992173`, CLI PID `2992183`, safety-monitor PID `2992347`; 80 epochs, batch 4, 20 registered representations and seeds `4121/7319/104729`.
- This made all four registered tasks concurrent: GPU 6 hosts ImageNet-100 and NYUv2; GPU 7 hosts VOC2012 and ADE20K. Existing workers and the primary recovery/watchdog/analyzer chain were not restarted or modified.
- Parallelism remains cross-task only. NYUv2 has an independent output root, lock, PID file, process group, and recovery-exit monitor. Per-task `selected_profile.seed_workers=1`, formal cache batch 2, and readout memory cache 0 GiB remain unchanged.
- NYUv2 train/val/test manifests remain `passed`, complete, clean, and fixed-revision-bound for `715/80/654` samples. GPU-6 resource-guard pause remains present.
- The verified NYUv2-only group contains parent, task script, matrix script, and CLI only. If primary recovery PID `41610` exits, the monitor sends group `TERM`, waits 15 seconds, then escalates to `KILL` if the CLI remains.
- CPU oversubscription risk is explicitly recorded: before the fourth launch, three workers already used nearly the full 100-core cgroup over a wall-clock sample. The user explicitly authorized continued cross-task parallelism while GPU memory remained sufficient. Aggregate epoch throughput will decide whether four-way concurrency is retained.
- The authoritative audit was successfully written before the SSH transport executed an isolated trailing carriage-return command. That transport failure did not modify or restart training and is preserved as a non-authoritative failed artifact.
- This launch is operational evidence only: `execution_complete=false`, `method_effectiveness_conclusion=null`, `changes_scientific_verdict=false`.

Authoritative launch audit: `artifacts/reports/nyuv2_gpu6_parallel_launch_audit_20260823.json` (SHA-256 `25fe195222381d8fab7a8672a68246de85e62f56deeae59d39b32b0fd25706f3`).

Preserved non-authoritative transport failure: `artifacts/reports/nyuv2_gpu6_parallel_launch_non_authoritative_failed_20260823.json` (SHA-256 `401e5b6e46d77e1df1a1b4a3950ed6dda0b2c331ed014a68882803d9c6bcd61e`).

## First atomic recovery point

NYUv2 `random_feature_local / seed 4121` subsequently committed epoch 1 without restart: `179` optimizer steps, `715` sample exposures, train/validation time `499.51651001535356 / 59.95765916723758` seconds, and current best primary metric `0.36730616837739943`. External strong audit verified continuous history, last/best checkpoint identity, optimizer/scheduler state, all four RNG states, cache/control/coverage contracts, fixed revision/tree, clean formal and analysis worktrees, and `seed_workers=1`; `status=passed`, `problems=[]`. Artifact: `artifacts/reports/nyuv2_random_feature_local_seed4121_epoch1_strong_audit_20260823.json` (SHA-256 `2644a003a8988fcae2a0a9bc6a8775e0b904cb28e1ec2c3ef6b009916111320c`). This is an intermediate execution result only and does not change the scientific verdict.
