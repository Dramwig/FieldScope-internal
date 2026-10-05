# H200 parallel-monitor false-watchdog recovery

Status: operational recovery `passed`; formal scientific validation remains
`active`, `execution_complete=false`, and no method-effectiveness conclusion
has been made.

At `2026-08-23T15:04:17Z`, the first content-addressed side-worker monitor
(SHA-256 `9317f0e108c769c1d7533e56a80c1e97c92d3aa8cc94ef57dc4d96885d417b3f`)
falsely terminated the registered VOC2012, NYUv2, and ADE20K process groups.
It saw watchdog argv `bash watchdog_h200.sh` but resolved the relative script
against the monitor cwd instead of `/proc/<watchdog_pid>/cwd`. All three
`terminated_watchdog_absent` exit artifacts and the old registry SHA-256
`50bc7d697ae4fc152431d6f9eecd1c634d271285e64a9f9b7c6be49936d2e02d`
are retained.

Accepted atomic reports/checkpoints were unchanged. The last accepted states
were VOC2012 `random_feature_local / seed 104729 / epoch 48`, NYUv2
`random_feature_local / seed 4121 / epoch 28`, and ADE20K
`random_feature_local / seed 4121 / epoch 1`. Only uncommitted current-epoch
work was lost; this negative operational result is not hidden or rewritten.

At `2026-08-23T15:09:34Z`, the three tasks were relaunched from those accepted
checkpoints with one seed worker per task and no duplicate dataset/cell writer:
VOC PID/PGID/start-ticks `1571769/1571730/778733472` on physical GPU 7, NYUv2
`1571768/1571732/778733472` on GPU 6, and ADE20K
`1571772/1571745/778733476` on GPU 7. ADE retained the audited all-ignore shim
SHA-256 `f95bbb0ed70ebbc127e6e08cd8d47c7bb18e6d394c2233a5677dc9cdfbf7bac4`.

The corrected monitor SHA-256 is
`e0677f25e01dccb4ccaf723a7f188efecd5cc760aa423877bce5a843161b8960`,
and the current content-addressed registry SHA-256 is
`486c21301c656fedf2ea662a0240e2c14fe43708e195a2dd178ca6ded0aea189`.
All three exact self-tests returned `parallel monitor self-test passed`; three
short live timeout tests and a full 30-second poll interval also passed. The
accepted monitor PIDs are `1615506/1615510/1615515`, with no new exit artifact.

Machine-readable evidence is in
`artifacts/reports/gpu67_parallel_monitor_false_watchdog_recovery_20260823.json`.
This correction changes no accepted training output, registered scientific
condition, downstream trigger, or scientific verdict.
