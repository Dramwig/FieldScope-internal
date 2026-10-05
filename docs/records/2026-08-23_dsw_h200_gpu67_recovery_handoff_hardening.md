# H200 GPU 6/7 recovery-handoff hardening

Status: `superseded_failed`. The initial operational hardening described below
was accepted at `2026-08-23T14:50:08Z`, but a later live failure showed that
the first replacement monitor was not valid. Formal scientific validation
remains `active` and incomplete.

At `2026-08-23T15:04:17Z`, all three replacement monitors falsely classified
the authoritative watchdog as absent. The watchdog was still PID `41096` with
argv `bash watchdog_h200.sh`; the monitor incorrectly resolved that relative
script path against its own cwd instead of `/proc/41096/cwd`. The preserved
exit artifacts show `terminated_watchdog_absent` for VOC2012, NYUv2, and
ADE20K. They terminated the three registered side-worker groups. No accepted
atomic report or checkpoint was changed, but each task lost its uncommitted
current-epoch work. The last accepted states remained VOC epoch 48, NYUv2
epoch 28, and ADE20K epoch 1.

The workers were restored from those accepted checkpoints at
`2026-08-23T15:09:34Z`. A corrected content-addressed monitor resolves relative
watchdog argv against `/proc/<watchdog_pid>/cwd`. Its SHA-256 is
`e0677f25e01dccb4ccaf723a7f188efecd5cc760aa423877bce5a843161b8960`;
the new registry SHA-256 is
`486c21301c656fedf2ea662a0240e2c14fe43708e195a2dd178ca6ded0aea189`.
Syntax validation, exact self-tests for all three datasets, short live timeout
tests, and a full live poll interval passed before monitor PIDs
`1615506/1615510/1615515` were accepted. The corrective machine-readable audit
is `artifacts/reports/gpu67_parallel_monitor_false_watchdog_recovery_20260823.json`.

The remainder of this record is retained verbatim as the provenance of the
initial, subsequently falsified hardening decision.

At `2026-08-23T14:50:08Z`, the three registered cross-task workers remained
active as VOC2012 PID/PGID/start-ticks `3925626/3921096/766870704`, NYUv2
`2992183/2992173/777122866`, and ADE20K
`3634699/3634680/777535163`. The fixed formal and analysis worktrees were clean
at revision `020c1de567edd88e0eda245fd085335ffe678f47`.

The previous side-worker monitors were bound to the current primary recovery
PID `41610` and would terminate their worker groups whenever that PID exited.
That behavior could discard the current uncommitted epoch during a normal
future recovery handoff. Their exact command identities were verified before
stopping only monitor PIDs `2925950/2992347/3635008`; none of the training
workers was signalled or restarted.

Three content-addressed replacement monitors now bind each worker's PID, PGID,
process start tick, cwd, CLI, dataset cache and output root to a read-only
registry. They continue across a current-primary-PID change while the
authoritative watchdog and fixed clean checkout remain valid. They terminate a
registered group only if the checkout becomes invalid or the authoritative
watchdog remains absent beyond a 900-second grace period. Monitor PIDs are
`1330465/1330468/1330475`; script SHA-256 is
`9317f0e108c769c1d7533e56a80c1e97c92d3aa8cc94ef57dc4d96885d417b3f`,
and registry SHA-256 is
`50bc7d697ae4fc152431d6f9eecd1c634d271285e64a9f9b7c6be49936d2e02d`.

The external H200 recovery wrapper was also updated for future watchdog
restarts. Its old SHA-256
`fef17407a2b261335183ae17f9dd23dedf4446aa36eb41f32f3776a22d21c251`
is preserved as a read-only backup; the installed SHA-256 is
`98fc2baea7203d3c225a890deb6eb26307f4164deb8d94f1b73997b7b662f3e9`.
The only execution-contract addition is a fail-closed ADE20K branch that
requires and injects the already audited all-ignore shim SHA-256
`f95bbb0ed70ebbc127e6e08cd8d47c7bb18e6d394c2233a5677dc9cdfbf7bac4`.
The current recovery PID was not restarted and had already loaded the prior
wrapper, so the replacement changes no current cell.

This work changed no fixed-revision source, dataset, cache, sample, sampler,
batch size, seed, representation, optimizer step, exposure, result, or
scientific threshold. It produced no main, causal, extension, replay, final,
registry, or completion artifact. Machine-readable audit:
`artifacts/reports/gpu67_recovery_handoff_hardening_20260823.json`.
`execution_complete=false`, `method_effectiveness_conclusion=null`, and
`changes_scientific_verdict=false`.
