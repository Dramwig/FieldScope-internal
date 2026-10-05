# DSW-M3Call A800 cross-host formal-worker handoff

Status: operational handoff `passed`; formal scientific validation remains
`active`, `execution_complete=false`, and no method-effectiveness conclusion is
made.

At `2026-08-24T10:40:45Z`, the registered H200 NYUv2 and VOC2012 side workers
exited after exact PID, PGID, start-tick, cwd and CLI checks. Their shared task
locks were observed free from the A800 host before replacement launch. The last
accepted H200 strong-audit boundaries were NYUv2
`random_feature_local / seed 104729 / epoch 5` and VOC2012
`z0 / seed 104729 / epoch 24`. Work after those commits may have been lost;
accepted checkpoints, reports and earlier negative evidence were not changed.

The same fixed-revision commands resumed on `dsw-M3Call`: NYUv2 uses physical
GPU 0 and VOC2012 uses physical GPU 1. The A800 node shares the same CephFS,
formal output roots and caches. Each launcher holds the pre-existing task lock
for its full process lifetime. The new registry binds hostname, PID, PGID,
start ticks, cwd, complete CLI identity, cache/output roots, GPU environment and
lock. Independent monitors additionally require a clean fixed worktree and a
fresh shared H200 watchdog heartbeat.

The first registration attempt failed closed with
`expected one nyuv2 worker, found 0`: the registrar resolved the configured
Python symlink but compared it with the unresolved argv path. It wrote no
registry and did not stop or modify either worker. The failure is preserved in
`artifacts/reports/dsw_m3call_a800_registration_attempt1_non_authoritative_failed_20260824.json`.
After comparing resolved paths symmetrically, registration and both exact
monitor self-tests passed.

The first live cross-host monitor was then hardened without restarting either
worker. Its loop had treated any failed identity recheck as a normal worker
exit. The content-addressed replacement instead checks PID start ticks
independently and terminates a still-live group if registry, argv, GPU or lock
identity becomes invalid. Replacement SHA-256
`a59ebefcec01a67db6b56051a9f105983168cb6152273985956e04558768c306`
completed full live polls as PIDs `55616/55621`; only superseded monitor PIDs
`54412/54419` were then terminated. Worker PIDs and PGIDs were unchanged.

The H200 atomic watcher subsequently accepted A800-produced strong audits for
NYUv2 epoch 7 and VOC2012 epoch 26. Both audits verify contiguous histories,
optimizer/scheduler counters, RNG state, cache and control contracts, fixed
revision and clean worktrees. The reconstructed global ledger at
`2026-08-24T10:47:30.723705Z` has 261 accepted intermediate audits, 20 of 240
terminal cells, no unresolved failures and `status=passed`; it remains
incomplete.

Initial A800 epoch timings were about 177 seconds for NYUv2 training plus
validation and 200 seconds for VOC2012, versus roughly 600 seconds and 4--5
minutes respectively under H200 four-task CPU contention. These are operational
throughput observations, not scientific effectiveness evidence.

Post-handoff verification passed full Ruff over `src tests scripts`, all 180
tests in 957.50 seconds, and the two-step toy smoke with
`cache_reload_equal=true`. The toy smoke used the CPU toy backend and did not
write any formal matrix output.

After that verification load completed, the `2026-08-24T11:17:58.888727Z`
ledger snapshot still passed with 267 accepted intermediate audits and no
unresolved failures. A800 evidence had advanced to NYUv2 epoch 17 and VOC2012
epoch 35 without changing the 20 terminal-cell count. Immutable snapshot
SHA-256: `2f5e5a16d64a41c2216f6e75a3462bab5fadac7b7f1dadec984abbb8f7401dd2`.

Machine-readable handoff audit:
`artifacts/reports/dsw_m3call_a800_cross_host_handoff_20260824.json`.

## Supervisor deployment and measured throughput

The handoff audit above remains immutable at SHA-256
`88eca19e88a75a5a1431ba8739681fbedf330f7647b68f069e3457dc17849bcb`.
A separate machine-readable deployment audit records the later recovery
supervisors at
`artifacts/reports/dsw_m3call_a800_supervisor_deployment_audit_20260824.json`.

Two non-authoritative deployment failures are retained in that audit. One
launch wrote the literal shell token `$!` to the supervisor PID files, and the
first background-persistence command left no persistent supervisor process.
Neither attempt restarted a formal worker or modified a formal output. Initial
supervisors `60215/60207` were subsequently superseded, after live self-tests,
by content-addressed supervisor SHA-256
`5d39bfe899459e688962898ab9354696ccb526fb83a39bb64d9d3c85c488fe66`
as NYUv2/VOC2012 PIDs `61255/61253`. Formal worker PIDs `53492/53491` and
monitor PIDs `55616/55621` remained unchanged. Both task-specific live
self-tests, Bash syntax, the fixed-revision clean-worktree gate, shared H200
heartbeat gate, task-lock exclusion and a complete 60-second live poll passed.
Automatic restart was not destructively exercised against a live formal
worker.

At NYUv2 epoch 27 and VOC2012 epoch 44, report histories gave the following
operational timing comparison while preserving batch, sampler, learning rate,
step and exposure contracts. NYUv2 epochs 1--5 on H200 averaged 489.33 seconds
for training plus validation; epochs 6--27 on A800 averaged 177.00 seconds, a
2.76x throughput improvement. VOC2012 epochs 1--25 on H200 averaged 218.40
seconds; epochs 26--44 on A800 averaged 203.90 seconds, a 1.07x improvement.
The primary benefit is four independent task lanes across the two hosts. These
timings are operational evidence only and do not change the scientific verdict.

The next immutable reconstruction at `2026-08-24T11:48:33.965717Z` accepted
NYUv2 epoch 27 and VOC2012 epoch 44. It contains 273 intermediate audits,
20/240 terminal cells and no unresolved failures, with `status=passed` and
`execution_complete=false`. Snapshot SHA-256:
`e1c1b12c3debe3779ebb7c4a40d2530d603d647228f8f3aca7273122a97006db`.

After the final supervisor and launcher edits, Ruff passed over
`src tests scripts`, all 180 tests passed in 2852.31 seconds, and the two-step
CPU toy smoke passed with `cache_reload_equal=true` and the unchanged cache
fingerprint. The slower test wall clock reflects concurrent formal workload on
the A800 node; it did not change formal worker PIDs or write formal outputs.

The post-verification immutable reconstruction at
`2026-08-24T12:39:27.532713Z` contains 284 accepted intermediate audits,
20/240 terminal cells and no unresolved failures. Active strong-audit
boundaries are ImageNet-100 epoch 44, ADE20K epoch 8, NYUv2 epoch 33 and
VOC2012 epoch 51. It reports `status=passed`, `execution_complete=false` and
`changes_scientific_verdict=false`. Snapshot SHA-256:
`07402d0b9bd130140999b27f7855f96fc37e7df932c6db7a1723e7996c66991a`.

At `2026-08-24T12:49:37.348711Z`, the next immutable reconstruction passed
with 286 accepted intermediate audits, 20/240 terminal cells, no unresolved
failures and active boundaries ImageNet-100 epoch 44, ADE20K epoch 8, NYUv2
epoch 35 and VOC2012 epoch 52. The cross-host downstream pretrigger audit
confirmed one exact writer per task, both fixed worktrees clean, all recovery,
watcher, monitor, supervisor and replay-waiter processes alive, and all six
premature downstream artifacts absent. Its decision is
`wait_for_all_240_main_matrix_cells`; no scientific verdict changed.

Machine-readable pretrigger audit:
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T124937Z.json`.

## Atomic-watcher rollover and current ledger

At `2026-08-24T14:00:22.190036Z`, the content-addressed atomic watcher with
SHA-256 `00a04df6d6eae0ddcd13a4a57f74f8222842d1954b4a3400a8a9819fa44253c7`
replaced only the prior watcher process. H200 recovery and ImageNet-100/ADE20K
worker PIDs `41610/102910/1571772`, A800 NYUv2/VOC2012 worker PIDs
`53492/53491`, monitors `55616/55621`, and supervisors `61255/61253` remained
unchanged. Both fixed-revision worktrees remained clean.

The rollover followed full Ruff, `4 passed in 3.81s` targeted tests,
`182 passed in 2854.28s` full tests, and a two-step CPU toy smoke with
`cache_reload_equal=true`. The prior unresolved VOC2012 `z0 / seed 104729 /
epoch 55` artifact remains byte-identical at SHA-256
`09bce9d05511805f18ae959db99014795cb31984553bef9709c035edc0f37d76`.
Its stable report was at epoch 55 while the stable checkpoint had advanced to
epoch 56; the passed epoch-58 strong audit, SHA-256
`7d7f2a06dde816ea438b92e8ea15411335b5f49c005e3d60ce788feca316c210`,
covers that checkpoint boundary with the same identity and counters. It is now
classified as
`checkpoint_ahead_of_stable_report_confirmed_by_later_strong_audit`, retained
as negative operational evidence, and is not an unresolved scientific failure.

The post-rollover state has 299 accepted intermediate audits, 20 terminal
audits, five preserved transient audit races and zero unresolved failures.
The immutable ledger reconstruction at the same boundary has active strong
audits ImageNet-100 epoch 45, ADE20K epoch 8, NYUv2 epoch 45 and VOC2012 epoch
63. It reports `status=passed`, `problems=[]`, `execution_complete=false` and
`changes_scientific_verdict=false`; all premature downstream artifacts remain
absent. The scientific goal therefore remains active.

Machine-readable rollover audit:
`artifacts/reports/atomic_audit_watcher_checkpoint_ahead_rollover_20260824.json`.
Immutable ledger SHA-256:
`abc1d3e53e4fc354850c810a773d9948707780667c2244570a616f5b17827484`.

At `2026-08-24T14:10:28.984337Z`, the next content-addressed watcher cycle
accepted NYUv2 epoch 48 and VOC2012 epoch 66 without adding a failure or race.
The immutable reconstruction contains 301 accepted intermediate audits, 20/240
terminal cells, five preserved transient audit races and zero unresolved
failures. Active accepted boundaries are ImageNet-100 epoch 45, ADE20K epoch 8,
NYUv2 epoch 48 and VOC2012 epoch 66. Its SHA-256 is
`957d9fa690559f7f562572d2bc1bd36ca6da2366c084d1d57157854a25deb9f9`.

The corresponding cross-host pretrigger audit rechecked exact worker registry
identity, all four worker PIDs, both A800 monitors and supervisors, fixed clean
worktrees, and absence of all six downstream/final artifacts. Its decision
remains `wait_for_all_240_main_matrix_cells`; it reports `status=passed`,
`execution_complete=false`, `method_effectiveness_conclusion=null`, and
`changes_scientific_verdict=false`.

Machine-readable pretrigger audit:
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T141035Z.json`.

At `2026-08-24T14:20:35.874705Z`, the next watcher cycle accepted NYUv2 epoch
52 and VOC2012 epoch 69. The immutable ledger has 303 accepted intermediate
audits, 20/240 terminal cells, five preserved transient races, zero unresolved
failures and active boundaries ImageNet-100 45, ADE20K 8, NYUv2 52 and VOC2012
69. Ledger SHA-256:
`665351f155e107f44fd48df945eb493cc38a2c961354732300271d67e9557278`.

The incremental pretrigger recheck at `14:23:01Z` found both fixed worktrees
clean, every registered worker/orchestrator/monitor/supervisor PID alive, the
preserved epoch-55 failed artifact byte-identical, and all six downstream/final
artifacts absent. No terminal cell was added and downstream remains closed.
Machine-readable incremental audit:
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T142042Z.json`.

At `2026-08-24T14:30:42.690710Z`, the watcher accepted NYUv2 epoch 55 and
VOC2012 epoch 72. The immutable ledger has 305 accepted intermediate audits,
20/240 terminal cells, five preserved races and zero unresolved failures;
ImageNet-100/ADE20K remain at accepted epochs 45/8. Ledger SHA-256:
`a53ffa28ffade6a8729d6b00bc648728ff2eff2148f77b87a45d35213c03f61f`.

The `14:32:44Z` incremental pretrigger recheck again found clean fixed
worktrees, all registered processes alive and all downstream/final artifacts
absent. No terminal evidence was available, so the gate remained closed.
Machine-readable audit:
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T143050Z.json`.

At `2026-08-24T14:40:50.144708Z`, the watcher accepted ADE20K epoch 9, NYUv2
epoch 59 and VOC2012 epoch 75; ImageNet-100 remained at epoch 45. The ledger
has 308 accepted intermediate audits, 20/240 terminal cells, five preserved
races and zero unresolved failures. Ledger SHA-256:
`c26af0c58e02cf832cc41c2969dc18b48e1a63e3f94cc951c9dd6df5b90d91ef`.

The `14:43:07Z` pretrigger recheck found all registered processes alive, both
fixed worktrees clean, the known failed artifact unchanged, and all downstream
and final artifacts absent. The gate remains
`wait_for_all_240_main_matrix_cells`.

At 2026-08-24T14:51:00.094717Z, the content-addressed watcher accepted the
latest cross-host state without any new unresolved failure: 310 intermediate
audits, 20/240 terminal cells, five classified transient races, and zero
unresolved failures. Active accepted boundaries are ImageNet-100 epoch 45,
ADE20K epoch 9, NYUv2 epoch 63, and VOC2012 epoch 78. The immutable ledger
SHA-256 is eef69d202d730315fe3cc768dee116ccdf5cc60179833f9214b4a762198959b5
and the ledger remains status=passed, execution_complete=false, and
changes_scientific_verdict=false.

The matching pretrigger audit is
artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T145107Z.json
with SHA-256 9fccd1dcd847f1266a13c5158dd9dd4627833b5093837ea0eab4bd244c97b373.
It reconfirmed clean fixed worktrees, one writer and seed_workers=1 per task,
all registered H200/A800 worker identities alive, preservation of the known
VOC epoch-55 race artifact, and absence of main, causal, extension, final,
registry.json, and completion_audit.json artifacts. Downstream remains closed
until all 240 main cells are terminal.

At 2026-08-24T15:01:07.729705Z, the watcher state advanced NYUv2 to epoch 65.
The read-only reconstruction accepted 311 intermediate audits while retaining
20/240 terminal cells, five classified transient races, and zero unresolved
failures. Active boundaries are ImageNet-100 45, ADE20K 9, NYUv2 65, and
VOC2012 78. Ledger SHA-256:
44e8ed2cf2c705907ffc00b3f30f96eeb488b37590541ea8e35f35a8cf41ed25.
The matching pretrigger audit is
artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T150107Z.json
with SHA-256 60da1d580e501a11efe22ec31b0931067013c76417fcfc15ff554bcfee1d8d6.
Its process, clean-worktree, absence, and preserved-negative-evidence checks
remain passed; downstream remains closed and the scientific goal remains active.

At 2026-08-24T15:11:11.130709Z, the watcher/reconstructor accepted 314
intermediate audits and 21/240 terminal cells. VOC2012 added its sixth terminal
cell; five transient races remain classified and unresolved failures remain zero.
Active accepted boundaries are ImageNet-100 epoch 46, ADE20K epoch 9, NYUv2
epoch 68, and VOC2012 zt/seed-4121 epoch 2. Ledger SHA-256:
0fda8f7fa3dc15e2d2d79478f19a9786496ad04eb7b5999b82ae20a7c6623426.
The matching pretrigger audit is
artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T151111Z.json
with SHA-256 a0e3a7f97a38e3e05a3278fd24491dd93705579773bf6017bc9e542ea7e7a0c7.
It passed clean-worktree, process-identity, downstream-absence, and negative-evidence
checks; downstream remains closed.

At 2026-08-24T15:21:26.971723Z, the watcher/reconstructor accepted 316
intermediate audits with 21/240 terminal cells, five classified transient races,
and zero unresolved failures. Active accepted boundaries are ImageNet-100 epoch
46, ADE20K epoch 9, NYUv2 epoch 72, and VOC2012 zt/seed-4121 epoch 6. Ledger
SHA-256: 31fe7b51081fb79a7b8d27d2e91a778901d4911bb49c57272c144f3cdd24e4ee.
The matching pretrigger audit is
artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T152126Z.json
with SHA-256 76e901288deb1794e5694ab997a8b684aeebfe37139db37c44df5375ea12bd6f.
All worker, worktree, downstream-absence, and preserved-negative-evidence checks
passed; the downstream gate remains closed.

At 2026-08-24T15:31:34.441717Z, the watcher/reconstructor accepted 318
intermediate audits with 21/240 terminal cells, five classified transient races,
and zero unresolved failures. Active boundaries are ImageNet-100 epoch 46,
ADE20K epoch 9, NYUv2 epoch 75, and VOC2012 zt/seed-4121 epoch 9. Ledger
SHA-256: 8cdc397be10aa262dd40065aef9fa9f1ddc791fd7ee8bac9da2c2a2a473d69a4.
The matching pretrigger audit is
artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T153134Z.json
with SHA-256 07d5059dee5b2a9055a9b05e4eb451ff61773432d00f475416ec3137faf08f1c.
Worker identity, clean-worktree, downstream-absence, and negative-evidence checks
passed; downstream remains closed.

At 2026-08-24T15:41:41.319704Z, the watcher/reconstructor accepted 320
intermediate audits with 21/240 terminal cells, five classified transient races,
and zero unresolved failures. Active boundaries are ImageNet-100 epoch 46,
ADE20K epoch 9, NYUv2 epoch 79, and VOC2012 zt/seed-4121 epoch 12. Ledger
SHA-256: 5fe709c9341d33cf7a8ca6a068db5f61677b6e2d0484bd52cf0e9adc982075e5.
The matching pretrigger audit is
artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T154141Z.json
with SHA-256 e1ffadb0fe5ae76c90349a01b0e04c768f29f79dccbf8344b5d2155ff93d095c.
All worker, clean-worktree, downstream-absence, and preserved-negative-evidence
checks passed; downstream remains closed.

At 2026-08-24T16:02:03.818706Z, a fresh read-only reconstruction accepted 324
intermediate audits and 22/240 terminal cells. NYUv2 added its third terminal
cell; five transient races remain classified and unresolved failures remain
zero. Active accepted boundaries are ImageNet-100 epoch 46, ADE20K epoch 9,
NYUv2 z0/seed-4121 epoch 15, and VOC2012 zt/seed-4121 epoch 24. The new ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T160500Z.json`
has SHA-256
`93fc7382da587996031644227481dc73cbbe3b705b2f8cc9280ed1cab23451be`.
The matching pretrigger audit is
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T160203Z.json`
with SHA-256
`b158948afa73550d3793b34027a1b4c9496faa53732a5c0dc1e3050fd8482796`.
Worker identities, fixed worktrees, downstream absence, and preserved negative
evidence all passed; the gate remains
`wait_for_all_240_main_matrix_cells`, with `execution_complete=false` and
`changes_scientific_verdict=false`.

At 2026-08-24T16:12:10.983707Z, the watcher accepted two further active-cell
audits, bringing the accepted intermediate count to 326. Terminal cells remain
22/240, with five classified transient races and zero unresolved failures.
Active accepted boundaries are ImageNet-100 epoch 46, ADE20K epoch 9, NYUv2
z0/seed-4121 epoch 25, and VOC2012 zt/seed-4121 epoch 31. Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T161900Z.json`
has SHA-256
`a04f8c048d308cb0cd8e918e24c96a6338155a6a2bcc77bf29f5bed3e88b2183`.
The matching pretrigger audit is
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T161210Z.json`
with SHA-256
`a7d885e3efd8ea00c17a4f246df52b079093e476a816cae62dd7a626291883f4`.
All identity, worktree, downstream-absence, and preserved-negative-evidence
checks passed; the gate remains closed.

At 2026-08-24T17:13:11.804717Z, an independent reconstruction accepted 338
intermediate audits and 23/240 terminal cells. NYUv2 added its fourth terminal
cell; active accepted boundaries are ADE20K epoch 10, ImageNet-100 epoch 47,
and VOC2012 zt/seed-4121 epoch 69. Six transient races remain classified,
including the preserved VOC2012 epoch-56 checkpoint/report mismatch, and
unresolved failures remain zero. Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T171719Z.json`
has SHA-256
`56e7531000f605c6322a89b8c1390239415054939ba22443891c0388dba1fd88`.

The first live-identity pretrigger recheck at 17:21:33Z is preserved as an
operational false-negative (`downstream_gate_cross_host_pretrigger_audit_20260824T172133Z.json`,
SHA-256 `e66315e629bb4a89625d4b2f8af7baec61f46f7de21ac526dc07801657fa2f98`):
the H200-side attempt to SSH back to the `dsw-M3Call` alias failed because that
alias is only configured on the operator machine, while both hosts' direct PID
checks were alive. A corrected cross-host pretrigger audit using direct checks
is `downstream_gate_cross_host_pretrigger_audit_20260824T172245Z.json`, SHA-256
`a0ac7f8d9b0cdbd6aa68ccdb5548cd5a10363b404396cb556881614c6afd3201`.
It reports `status=passed`, `all_required_processes_alive=true`, no problems,
all downstream artifacts absent, and the gate
`wait_for_all_240_main_matrix_cells`; `execution_complete=false` and
`changes_scientific_verdict=false` remain unchanged.

At 2026-08-24T19:26:26Z, the cross-host reconstruction accepted 365
intermediate audits while terminal cells remained 25/240. Active boundaries
advanced to ADE20K epoch 11, ImageNet-100 epoch 48, NYUv2 z0/seed-104729
epoch 39, and VOC2012 zt/seed-7319 epoch 65. Seven transient races remain
classified and unresolved failures remain zero. Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T192600Z.json`
has SHA-256
`720105ac401d9456b4dec45ae624cf3f99b1a3584da095e11becd2abadc4cff5`.
The matching pretrigger gate
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T192626Z.json`
has SHA-256
`f20b1214cb56788972e2fc61eb1caebe0973bf1e12ef714353b5fc7490942424`;
downstream remains closed and no scientific verdict changed.

At 2026-08-24T19:45:34Z, the H200 watcher accepted ImageNet-100
velocity/seed-7319 epoch 49, NYUv2 z0/seed-104729 epoch 57, and VOC2012
zt/seed-7319 epoch 77 strong audits from the A800/H200 lanes. The ledger now
contains 370 accepted intermediate audits and 25/240 terminal cells; ADE20K
remains active at epoch 11. Seven transient races remain classified and
unresolved failures remain zero. Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T201000Z.json`
has SHA-256
`0f130327da2ec01e9df86122df189f6d4dfe245b0bae68cf76c861131ca4e602`.
The matching pretrigger gate
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T201000Z.json`
has SHA-256
`e28b03746590366deccc1e14ec3b16697cbc8852525dc13596946be59b1b6e59`;
downstream remains closed and no scientific verdict changed.

At 2026-08-24T19:35:27Z, the H200-side watcher accepted the A800 VOC2012
zt/seed-7319 epoch-71 and NYUv2 z0/seed-104729 epoch-48 strong audits.
The reconstructed ledger contains 367 accepted intermediate audits and
25/240 terminal cells; ADE20K and ImageNet-100 remain active at epochs 11
and 48. Seven transient races remain classified, unresolved failures remain
zero, and the one-writer/one-seed-worker cross-host registration remains
valid. Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T193600Z.json`
has SHA-256
`577d83961a11ec48005cbeba37466ff54cc59b7fe0f2092638e824ed4c43d440`.
The matching pretrigger gate
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T193600Z.json`
has SHA-256
`77f939dab4ce13247125e1af7e84f14b492e13d9659193e06ddd6a05d4b8ac2c`;
downstream remains closed and no scientific verdict changed.

At 2026-08-24T19:18:47Z, the cross-host reconstruction accepted 363
intermediate audits while terminal cells remained 25/240. Active boundaries
advanced to ADE20K epoch 11, ImageNet-100 epoch 48, NYUv2 z0/seed-104729
epoch 30, and VOC2012 zt/seed-7319 epoch 59. Seven transient races remain
classified and unresolved failures remain zero. Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T191800Z.json`
has SHA-256
`d1f101be89043a1949e873854206790531b3f472014fc486df9af3916ea6a36f`.
The matching pretrigger gate
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T191847Z.json`
has SHA-256
`384ef71bf69a50cf33753e4d1f3b64c1462ce6e9ea0063e59142d67fe9b33dd7`;
downstream remains closed and no scientific verdict changed.

At 2026-08-24T19:07:06Z, the cross-host reconstruction accepted 361
intermediate audits while terminal cells remained 25/240. Active boundaries
advanced to ADE20K epoch 11, ImageNet-100 epoch 48, NYUv2 z0/seed-104729
epoch 21, and VOC2012 zt/seed-7319 epoch 53. Seven transient races remain
classified and unresolved failures remain zero. Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T190600Z.json`
has SHA-256
`e383a1e3fda95e3271f54c79c75a084078c7e4d05b37e4e81a420dbfcc1905de`.
The matching pretrigger gate
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T190706Z.json`
has SHA-256
`cbab6433d6defd50363bc956df810d9f1131c1ba31b581690373f60802049a1c`;
downstream remains closed and no scientific verdict changed.

At 2026-08-24T18:56:09Z, the cross-host reconstruction accepted 359
intermediate audits while terminal cells remained 25/240. Active boundaries
advanced to ADE20K epoch 11, ImageNet-100 epoch 48, NYUv2 z0/seed-104729
epoch 12, and VOC2012 zt/seed-7319 epoch 47. Seven transient races remain
classified and unresolved failures remain zero. Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T185500Z.json`
has SHA-256
`5579c0420107f075ad7d9a9b02ab6000de3609f9c9776d4b8359cf8eda2418d5`.
The matching pretrigger gate
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T185609Z.json`
has SHA-256
`fbec18519751ec3ba7e2e71c0c5c53bbe11bbe5a69eb6a51317c180380e99ee4`;
downstream remains closed and no scientific verdict changed.

At 2026-08-24T16:52:50.031719Z, the watcher accepted 335 intermediate audits.
Terminal cells remain 22/240. NYUv2 reached accepted epoch 63 and ADE20K
accepted epoch 10; ImageNet-100 remains epoch 47 and VOC2012 remains at the
last valid accepted epoch 50. The VOC2012 epoch-56 strong audit was explicitly
preserved as a failed checkpoint/report race (SHA-256
`8f2d1063ba6c471b4938c4eb88004257e8c3d9378e728d6d73f04f3fbbe25a23`) and
classified as the sixth transient race; unresolved failures remain zero.
Ledger `artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T165300Z.json`
has SHA-256
`23c2312848fe79d4a7508129d42877372bc97ee967f7632e1887e8e06d236a91`.
The matching pretrigger audit is
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T165250Z.json`
with SHA-256
`33c79a8c36c5f91a08a7d7ba8ef05e1efcfa4bafe6f983db0ff5d70999e60e3d`.
The gate remains closed and no scientific verdict was changed.

At 2026-08-24T17:03:03.384715Z, the watcher accepted 337 intermediate audits.
Terminal cells remain 22/240. Active accepted boundaries are ImageNet-100
epoch 47, ADE20K epoch 10, NYUv2 z0/seed-4121 epoch 73, and VOC2012
zt/seed-4121 epoch 63. Six transient races and zero unresolved failures remain;
the failed VOC2012 epoch-56 audit remains preserved. Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T170320Z.json`
has SHA-256
`841a1def635a8f1df6ba33bdabb55869293f02436addac653d2cdc1074a154ed`.
The matching pretrigger audit is
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T170303Z.json`
with SHA-256
`167ef35748c465f357750d88be3cf47d162e1ff67146cae43df6bf29540e957d`.
The gate remains closed and no scientific verdict was changed.

At 2026-08-24T16:42:37.370706Z, the watcher accepted 333 intermediate audits.
Terminal cells remain 22/240, with five classified transient races and zero
unresolved failures. Active accepted boundaries are ImageNet-100 epoch 47,
ADE20K epoch 9, NYUv2 z0/seed-4121 epoch 54, and VOC2012 zt/seed-4121 epoch 50.
Ledger `artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T164300Z.json`
has SHA-256
`98822cc6a5d7f8b9ccffed0cb11fede3a9a0c21c701327a7f98c53a54f1050ca`.
The matching pretrigger audit is
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T164237Z.json`
with SHA-256
`522c254d6a437766ef706cc65052a79b8c839afd7fb9691d9ae29827029c1c78`.
All identity, worktree, downstream-absence, and preserved-negative-evidence
checks passed; the gate remains closed.

At 2026-08-24T16:22:19.740711Z, the watcher accepted 328 intermediate audits.
Terminal cells remain 22/240, with five classified transient races and zero
unresolved failures. Active accepted boundaries are ImageNet-100 epoch 46,
ADE20K epoch 9, NYUv2 z0/seed-4121 epoch 34, and VOC2012 zt/seed-4121 epoch 37.
Ledger `artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T162300Z.json`
has SHA-256
`53526279519e3ab0dff0dbc920ad0e995125ba86da9a42d565455c5593a4dc72`.
The matching pretrigger audit is
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T162219Z.json`
with SHA-256
`38e7e77bb85ee41e4d360263224a3b2fb05690f3d704d6cc588e9bdf90bc58af`.
The identity, worktree, downstream-absence, and preserved-negative-evidence
checks passed; the gate remains closed.

At 2026-08-24T16:32:29.181718Z, the watcher accepted 330 intermediate audits.
Terminal cells remain 22/240, with five classified transient races and zero
unresolved failures. Active accepted boundaries are ImageNet-100 epoch 46,
ADE20K epoch 9, NYUv2 z0/seed-4121 epoch 44, and VOC2012 zt/seed-4121 epoch 43.
Ledger `artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T163400Z.json`
has SHA-256
`9b350ce8454d35bc4e75999d40f82a08d5d46bf33ee293f55648ecee36d09ddd`.
The matching pretrigger audit is
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T163229Z.json`
with SHA-256
`bc2deb5c8d519f19a4f85ae562af427a334b12f8ac0b0b93666384516de0abd7`.
All identity, worktree, downstream-absence, and preserved-negative-evidence
checks passed; the gate remains closed.

At 2026-08-24T17:23:27.488721Z, the watcher/reconstructor accepted 340
intermediate audits while terminal cells remained 23/240. Active accepted
boundaries advanced to ADE20K epoch 10, ImageNet-100 epoch 47, NYUv2
z0/seed-7319 epoch 11, and VOC2012 zt/seed-4121 epoch 75. The ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T173129Z.json`
has SHA-256
`d28a2042e67bff10eac84468bfe98cd57f8ba6d2e623f6cf56340aab2848536b`.
The matching gate
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T173157Z.json`
has SHA-256
`18cb1cf4ea22e4f6828567461b56f2185654536ea5f52abec8f26dfe16ee1d9e` and
passes direct cross-host identity/worktree checks. Six transient races remain
classified, unresolved failures remain zero, downstream artifacts remain
absent, and the gate remains `wait_for_all_240_main_matrix_cells` with
`execution_complete=false` and `changes_scientific_verdict=false`.

At 2026-08-24T17:43:39.667722Z, the watcher/reconstructor accepted 343
intermediate audits and advanced the main matrix to 24/240 terminal cells.
NYUv2 active boundary is z0/seed-7319 epoch 27; VOC2012 has advanced to the
new zt/seed-7319 cell at epoch 5; ADE20K remains epoch 10 and ImageNet-100
epoch 47. The ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T174730Z.json`
has SHA-256
`14678cf57ad60c9f629ac4f2491df34093d0a5866607d7f65e6377d5180cdc99`.
The matching gate
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T174800Z.json`
has SHA-256
`3038ac3b6823147a9d6f1182b19105702e717e5111091e20ccf47d744bc4429a`;
downstream remains closed with no scientific verdict change.

At 2026-08-24T17:33:35.427708Z, a later read-only reconstruction at 17:42:15Z
was byte-identical in scientific content: 341 accepted audits, 23/240
terminal cells, and the same four active boundaries. The duplicate ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T174215Z.json`
retains SHA-256
`6b3b3bc38c1004bff5c3e6b78d97aa4e79fc5d9f7aa4dfd7e3cc394db4c38a70`.
Its matching gate
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T174251Z.json`
has SHA-256
`5d0d1613bfb0e040ec44579054637db7ee99a3deeaa785888446b349b0bc2060`;
the report/checkpoint writes observed on A800 had not yet become accepted
terminal audits, so downstream remains closed.

At 2026-08-24T17:33:35.427708Z, the independent reconstruction accepted 341
intermediate audits; terminal cells remain 23/240. Active boundaries are
ADE20K epoch 10, ImageNet-100 epoch 47, NYUv2 z0/seed-7319 epoch 19, and
VOC2012 zt/seed-4121 epoch 75. Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T173616Z.json`
has SHA-256
`6b3b3bc38c1004bff5c3e6b78d97aa4e79fc5d9f7aa4dfd7e3cc394db4c38a70`.
The matching pretrigger audit
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T173941Z.json`
has SHA-256
`999ae9b8112853444168da41426f4798c8f03019ac2680ccf3069f8e41ff01a6`;
direct worker/worktree checks passed, downstream remains absent, and the gate
is still `wait_for_all_240_main_matrix_cells` with no scientific verdict change.

At 2026-08-24T17:50:23Z, a subsequent reconstruction remained identical to
the 17:43 accepted ledger: 343 audits and 24/240 terminal cells. The retained
duplicate ledger is
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T175023Z.json`
with SHA-256
`14678cf57ad60c9f629ac4f2491df34093d0a5866607d7f65e6377d5180cdc99`; its
matching gate is
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T175054Z.json`
with SHA-256
`4e87cba1fd50d2aaaabec26fac68121f8f603a514a3feee4536eac88d8836ade`.
Report writes had not become new accepted terminal audits, so downstream stays
closed and no scientific verdict changed.

At 2026-08-24T17:53:54.424712Z, the watcher/reconstructor accepted 345
intermediate audits. Terminal cells remain 24/240; active boundaries advanced
to NYUv2 z0/seed-7319 epoch 36 and VOC2012 zt/seed-7319 epoch 11, while ADE20K
and ImageNet-100 remain at epochs 10 and 47. Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T175442Z.json`
has SHA-256
`26aca22cd338d82d253152b5b1bf85f46c8bd735d3afa026b28c290be52aa5cb`.
The matching gate
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T175510Z.json`
has SHA-256
`cb23aa03bf1b99f80fc2ccee845a64804975d030165ac0a2104747361e83e353` and keeps
the downstream gate closed.

At 2026-08-24T18:04:03.118707Z, the watcher/reconstructor accepted 347
intermediate audits and retained 24/240 terminal cells. Active boundaries are
NYUv2 z0/seed-7319 epoch 46, VOC2012 zt/seed-7319 epoch 17, ADE20K epoch 10,
and ImageNet-100 epoch 47. Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T180452Z.json`
has SHA-256
`2e6e9019811bf30cc6e0643ffc2fddea58ead26a5bf61f0cafcf2bc84c7afba9`.
The matching gate
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T180526Z.json`
has SHA-256
`1effe439c04202e7e7987bfdbd585aacff5c2135dd835f2bc043c96e39986eb2`;
downstream remains closed and no scientific verdict changed.

At 2026-08-24T18:36:51Z, the cross-host reconstruction accepted 354
intermediate audits while terminal cells remained 24/240. NYUv2 advanced to
z0/seed-7319 epoch 74 and VOC2012 to zt/seed-7319 epoch 35; ADE20K and
ImageNet-100 remain at epochs 10 and 48. Seven transient races remain
classified and unresolved failures remain zero. Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T183700Z.json`
has SHA-256
`e938ca3bfbcbd2aec3ab76fb303ed0d036a7cad85ab176aa94aaa09b46658957`.
The matching pretrigger gate
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T183651Z.json`
has SHA-256
`6236d2347b7e01d4a671165d773f75896e1eda28e92fd95fc7936aff657d8548`;
downstream remains closed and no scientific verdict changed.

At 2026-08-24T18:18:27Z, the H200-side independent reconstruction after the
A800 report writes remained unchanged at 350 accepted intermediate audits and
24/240 terminal cells. The A800 lanes remain registered as one worker per
task on physical GPUs 0 (NYUv2) and 1 (VOC2012), with no overlapping output
writers. The retained ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260825T021827Z.json`
has SHA-256
`4cadfffa0e395bc8389019edbf3fd8a75d59a6e502619753cdc0907877b4aec1`.
This was a no-change audit: downstream remains closed,
`execution_complete=false`, and `changes_scientific_verdict=false`; all
classified transient races and prior negative signal-gate evidence remain
preserved.

At 2026-08-24T18:30:48Z, the H200-side reconstruction accepted 352
intermediate audits while terminal cells remained 24/240. NYUv2 advanced to
z0/seed-7319 epoch 65 and VOC2012 to zt/seed-7319 epoch 29; ADE20K and
ImageNet-100 remain at epochs 10 and 48. Seven transient races are classified
and unresolved failures remain zero. The ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260825T0226Z.json`
has SHA-256
`d35db363e9bc42cce0c0f02815131fe57eacdf18a34049f93d538a70b7b83c2f`.
The matching pretrigger gate
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T183048Z.json`
has SHA-256
`8667441ee0d91db2d83f7ea17a2eac7edbf58bd39a26c1f404743cfd84b52e1d`;
downstream remains closed and no scientific verdict changed.

At 2026-08-24T18:46:22Z, the cross-host reconstruction accepted 356
intermediate audits and advanced the main matrix to 25/240 terminal cells.
NYUv2 completed its fifth terminal cell and moved to z0/seed-104729 epoch 2;
VOC2012 is at zt/seed-7319 epoch 41, while ADE20K and ImageNet-100 remain at
epochs 10 and 48. Seven transient races remain classified and unresolved
failures remain zero. Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T184500Z.json`
has SHA-256
`479d886f76c5c3641093f1d694662074e087087967c8d1c9f2f635300645d86d`.
The matching pretrigger gate
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T184622Z.json`
has SHA-256
`83ed10773f2dc0a723d31c09b85d3901e706f75476620b1bb884398bd8d2d8b7`;
downstream remains closed and no scientific verdict changed.

At 2026-08-24T19:55:44Z, the H200 watcher accepted the VOC2012
zt/seed-7319 terminal cell and new NYUv2 z0/seed-104729 epoch-64 and VOC2012
zt/seed-104729 epoch-1 strong audits. The ledger now contains 372 accepted
intermediate audits and 26/240 terminal cells; ADE20K and ImageNet-100 remain
active at epochs 11 and 49. Seven transient races remain classified and
unresolved failures remain zero. Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T195700Z.json`
has SHA-256
`25a7649618537a5703fcde97f9af06bc93aabd52acebc895bad45d46ef062713`.
The matching pretrigger gate
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T195700Z.json`
has SHA-256
`6b890c7d473fea3c4a631c92975bfc1e9946fe041a0f6ce830c945fdd1ab5a18`;
downstream remains closed and no scientific verdict changed.

At 2026-08-24T20:05:56Z, the H200 watcher accepted NYUv2
z0/seed-104729 epoch 73 and VOC2012 zt/seed-104729 epoch 7 strong audits.
The ledger now contains 374 accepted intermediate audits and 26/240 terminal
cells; the A800 one-writer and one-seed-worker registrations remain valid.
Seven transient races remain classified and unresolved failures remain zero.
Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T200700Z.json`
has SHA-256
`4700179ad4d07c4386a62208c661bd48cedefe8967885f26b263877d6200118b`.
The matching pretrigger gate
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T200700Z.json`
has SHA-256
`1043424a9e52cbf9dcc18e42ecda84be096857e44b2f3e70c7298d9d8c9a210c`;
downstream remains closed and no scientific verdict changed.

At 2026-08-24T20:16:03Z, the H200 watcher accepted the NYUv2
z0/seed-104729 terminal cell plus NYUv2 zt/seed-4121 epoch 2 and VOC2012
zt/seed-104729 epoch 13 strong audits. The ledger now contains 376 accepted
intermediate audits and 27/240 terminal cells. Seven transient races remain
classified and unresolved failures remain zero; the cross-host one-writer and
one-seed-worker registrations remain valid. Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T201700Z.json`
has SHA-256
`955546ddfc7a7c5afb4996f199f1c012206f769374a8e2a874e4888048f7b582`.
The matching pretrigger gate
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T201700Z.json`
has SHA-256
`4326c1138b71211d2c1c9e83be8025f3f235c6e2e5050a769526b6062f949668`;
downstream remains closed and no scientific verdict changed.

At 2026-08-24T20:26:15Z, the H200 watcher accepted NYUv2
zt/seed-4121 epoch 11 and VOC2012 zt/seed-104729 epoch 19 strong audits.
The ledger now contains 378 accepted intermediate audits and 27/240 terminal
cells. Seven transient races remain classified and unresolved failures remain
zero; cross-host one-writer and one-seed-worker registrations remain valid.
Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T202700Z.json`
has SHA-256
`51a3e221d70fa495a3110ac7f9144cc416920557263bdd1798a06b368e167a02`.
The matching pretrigger gate
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T202700Z.json`
has SHA-256
`04f80fa6bb97ea0abdd1b48c80eed8fccb062d0b33a13fa36412a2f65e2724e0`;
downstream remains closed and no scientific verdict changed.

At 2026-08-24T20:36:22Z, the H200 watcher accepted NYUv2
zt/seed-4121 epoch 21 and VOC2012 zt/seed-104729 epoch 25 strong audits.
The ledger now contains 380 accepted intermediate audits and 27/240 terminal
cells. Seven transient races remain classified and unresolved failures remain
zero; cross-host one-writer and one-seed-worker registrations remain valid.
Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T203700Z.json`
has SHA-256
`d57ddd7e7558e572c18a81f65e4862c2243bfa73b0acba6b0842d478caba82a4`.
The matching pretrigger gate
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T203700Z.json`
has SHA-256
`898fd0979d7dfc0d53fe61fcbcd86f2c81c8e142c67e7ed0c3af135c7ecfea9c`;
downstream remains closed and no scientific verdict changed.

At 2026-08-24T20:46:28Z, the H200 watcher accepted NYUv2
zt/seed-4121 epoch 30 and VOC2012 zt/seed-104729 epoch 31 strong audits.
The ledger now contains 382 accepted intermediate audits and 27/240 terminal
cells. Seven transient races remain classified and unresolved failures remain
zero; cross-host one-writer and one-seed-worker registrations remain valid.
Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T204700Z.json`
has SHA-256
`f9a680d031457665e578f33ea16f77ab7aba28b1c21e9cba932bb62b6975a7a4`.
The matching pretrigger gate
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T204700Z.json`
has SHA-256
`d40c3a944f306ecfbfe6e6351fcc892225c55df7690754a3d728a181b6274527`;
downstream remains closed and no scientific verdict changed.

At 2026-08-24T20:56:36Z, the H200 watcher accepted NYUv2
zt/seed-4121 epoch 40 and VOC2012 zt/seed-104729 epoch 37 strong audits.
The ledger now contains 384 accepted intermediate audits and 27/240 terminal
cells. Seven transient races remain classified and unresolved failures remain
zero; cross-host one-writer and one-seed-worker registrations remain valid.
Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T205700Z.json`
has SHA-256
`d23febd4041c785838e8bcdbf5b8394f36d1dbb2c28babdd6bb13c122ea2f32a`.
The matching pretrigger gate
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T205700Z.json`
has SHA-256
`555f27f09b4b074955335d9e202770f936835fe5297178e286046775fbf15c13`;
downstream remains closed and no scientific verdict changed.

At 2026-08-24T21:06:43Z, the watcher preserved a new failed intermediate
audit for the A800 NYUv2 lane, zt/seed-4121 epoch 49, due to
`stable_checkpoint_during_hash_and_load`. This is not yet classified as a
transient race; `unresolved_scientific_failure=false`, while the independent
ledger reports `status=failed` and one unresolved watcher failure. The failed
artifact and state are retained, no passed gate was generated, and downstream
remains closed. Failed ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T210700Z.json`
has SHA-256
`568a2a3d900cda1dae3b4667e1567a27406bf415f928968a6b6d76537d632528`.

At 2026-08-24T21:16:55Z, later strong audits were present for NYUv2
zt/seed-4121 epoch 59, VOC2012 zt/seed-104729 epoch 50, and ImageNet-100
velocity/seed-7319 epoch 50. The prior NYUv2 epoch-49 failure remained in
the watcher failure list pending the next migration pass, so the refreshed
ledger reports 389 accepted audits, `status=failed`, and one unresolved
watcher failure. No passed gate was generated and downstream remains closed.
Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T211800Z.json`
has SHA-256
`61e30f3ffc71783156fac5b81dc495f84f3527ff1ecef5260eb481f11dacac8e`.

At 2026-08-24T21:27:07Z, the watcher classified the prior A800 NYUv2
zt/seed-4121 epoch-49 failure as an unstable observation confirmed by the
later epoch-59 strong audit, preserving the failed artifact. A new unresolved
NYUv2 zt/seed-4121 epoch-68 failure is now pending classification. The
refreshed ledger reports 390 accepted audits, `status=failed`, one unresolved
failure, and eight classified transient races; downstream remains closed.
Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T212800Z.json`
has SHA-256
`a40bc90c70311cca7e0b88c88026b37174b3c670fdd3bff2e620ffddacb867d5`.

At 2026-08-24T21:37:16Z, the cross-host reconstruction retained one
unresolved A800-lane NYUv2 zt/seed-4121 epoch-68 failure
(`stable_checkpoint_during_hash_and_load`). The later epoch-77 strong audit
passed, but watcher migration had not yet occurred. The ledger reports 392
accepted intermediate audits, 27/240 terminal cells, eight classified transient
races, `status=failed`, and one unresolved failure; terminal counts are
ImageNet-100 13, NYUv2 6, VOC2012 8, and ADE20K 0. All failed/race artifacts
remain preserved, downstream stays closed, and
`execution_complete=false`/`changes_scientific_verdict=false` remain unchanged.
Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T213900Z.json`
has SHA-256
`b0b87600ad1828bff95f65d21a231899bbfee118676849f3a8c4dd9aa6014d21`.

At 2026-08-24T21:47:25Z, the watcher classified the A800-lane NYUv2
zt/seed-4121 epoch-68 stability failure from later strong evidence. Its state
now has `failures=[]` and nine classified transient races. The independent
cross-host ledger accepted 394 intermediate audits and 28/240 terminal cells:
ImageNet-100 13, NYUv2 7, VOC2012 8, ADE20K 0. Active boundaries are ADE20K
random_feature_local/4121 epoch 12, ImageNet-100 velocity/7319 epoch 50,
NYUv2 zt/7319 epoch 6, and VOC2012 zt/104729 epoch 68. Ledger status is
`passed`, while `execution_complete=false` and
`changes_scientific_verdict=false`; the downstream gate remains closed until
all 240 main cells are terminal. Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T214800Z.json`
has SHA-256
`4a1857ae5d6d982ed5d8502ad5a3027c5b72677aa24ec42b87a6cf4f4927d233`.
The matching pretrigger gate
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T214800Z.json`
has SHA-256
`b259531caef74957b322c8fa317fa1a1c099102497647fa175f4670a4a4818fd`; it
confirmed live H200/A800 identities, one writer per task, and absence of all
premature downstream/final artifacts. The initial malformed ledger filename
ending in `reconstruction_.json` is retained as non-authoritative operation
provenance; the explicit timestamped ledger above is authoritative.

At 2026-08-24T21:57:40Z, the A800 NYUv2/VOC2012 lanes produced later strong
audits through NYUv2 zt/seed-7319 epoch 15 and VOC2012 zt/seed-104729 epoch
74. The watcher has 396 accepted intermediate audits, nine classified races,
and zero unresolved failures. The cross-host ledger remains `status=passed` at
28/240 terminal cells (ImageNet-100 13, NYUv2 7, VOC2012 8, ADE20K 0), with
four active cells and `execution_complete=false`/
`changes_scientific_verdict=false`. The pretrigger gate remains closed with
decision `wait_for_all_240_main_matrix_cells`; no causal, extension, replay, or
final artifact was generated. Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T215800Z.json`
has SHA-256
`77cc24c7408440b00891300908d30790181fcc833cab4a1a27b2fc939b9b533f`; matching
gate `artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T215800Z.json`
has SHA-256
`2820e624486f695f6cada7bfb91daeb57a2545ddf0de10b17b306cecfd4c15b3`.

At 2026-08-24T22:07:48Z, the A800 NYUv2 lane produced a passed strong audit
for zt/seed-7319 epoch 23. The watcher has 397 accepted intermediate audits,
nine classified races, and zero unresolved failures. The cross-host ledger
remains `status=passed` at 28/240 terminal cells (ImageNet-100 13, NYUv2 7,
VOC2012 8, ADE20K 0); active NYUv2 is now epoch 23. The downstream gate remains
closed with `execution_complete=false` and
`changes_scientific_verdict=false`. Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T220800Z.json`
has SHA-256
`2c8d6ac5ac0602f7742ff65a7b4a1d5eb720fa041c8457b75e67eb89c5765fbb`; matching
pretrigger gate
`artifacts/reports/downstream_gate_cross_host_pretrigger_audit_20260824T220800Z.json`
has SHA-256
`7c2db15ce0aec251855669e8835dc2c3fe062461b9a973bfe3f8f86b8cbbf04e`.

At 2026-08-24T22:17:52Z, the watcher accepted two additional intermediate
audits. The independent reconstruction has `399` accepted intermediate audits
and `29/240` terminal cells: ImageNet-100 13, NYUv2 7, VOC2012 9, and ADE20K
0. VOC2012 completed `zt/seed-104729` and advanced to
`trajectory/seed-4121/epoch-4`; NYUv2 advanced to
`zt/seed-7319/epoch-31`. The active ADE20K and ImageNet-100 boundaries remain
epochs 12 and 50. There are nine classified transient races and zero
unresolved failures. The ledger remains `status=passed`, while
`execution_complete=false` and `changes_scientific_verdict=false`; all
downstream artifacts remain absent and the gate decision remains
`wait_for_all_240_main_matrix_cells`. Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T222000Z.json`
has SHA-256
`301475d5c6d46e12042121ac9189c50d620430ae3d9c1a3f12708e00187ea274`.

The 20260824T222300Z reconstruction was byte-equivalent in scientific state
(`399` accepted audits and `29/240` terminal cells), so no new downstream
trigger was permitted. The new ledger is retained as an explicit timestamped
audit; the older malformed filename remains non-authoritative provenance.

At 2026-08-24T22:28:07Z, the watcher accepted two further intermediate
audits. NYUv2 advanced to `zt/seed-7319/epoch-40` and VOC2012 advanced to
`trajectory/seed-4121/epoch-10`. The independent reconstruction contains 401
accepted intermediate audits and remains at `29/240` terminal cells (ImageNet-
100 13, NYUv2 7, VOC2012 9, ADE20K 0). There are nine classified transient
races and zero unresolved failures. The ledger remains `status=passed`, with
`execution_complete=false` and `changes_scientific_verdict=false`; no
downstream artifact was generated and the gate remains
`wait_for_all_240_main_matrix_cells`. Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T223000Z.json`
has SHA-256
`57d1e642b2279b26a818a047be3757ce0e4df4fa117c984e0e9559bdb6d046d0`.

At 2026-08-24T22:38:17Z, the watcher accepted two further intermediate
audits. NYUv2 advanced to `zt/seed-7319/epoch-50` and VOC2012 advanced to
`trajectory/seed-4121/epoch-16`. The independent reconstruction contains 403
accepted intermediate audits and remains at `29/240` terminal cells (ImageNet-
100 13, NYUv2 7, VOC2012 9, ADE20K 0). There are nine classified transient
races and zero unresolved failures. The ledger remains `status=passed`, with
`execution_complete=false` and `changes_scientific_verdict=false`; no
downstream artifact was generated and the gate remains
`wait_for_all_240_main_matrix_cells`. Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T224000Z.json`
has SHA-256
`98f806a578b93416146111cabf0c93786fbcb5b44db90ce1f3ecea00ec702458`.

At 2026-08-24T22:48:26Z, the watcher accepted two further intermediate
audits. NYUv2 advanced to `zt/seed-7319/epoch-59` and VOC2012 advanced to
`trajectory/seed-4121/epoch-22`. The independent reconstruction contains 405
accepted intermediate audits and remains at `29/240` terminal cells (ImageNet-
100 13, NYUv2 7, VOC2012 9, ADE20K 0). There are nine classified transient
races and zero unresolved failures. The ledger remains `status=passed`, with
`execution_complete=false` and `changes_scientific_verdict=false`; no
downstream artifact was generated and the gate remains
`wait_for_all_240_main_matrix_cells`. Ledger
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T225000Z.json`
has SHA-256
`e3b6a2d6ed7b8ff7f037934f6513e98a1392beaf7c1968d1f15c228d7b088e93`.

At 2026-08-24T22:58:36Z, the watcher accepted two new intermediate audits:
ImageNet-100 `velocity/seed-7319/epoch-51` and VOC2012
`trajectory/seed-4121/epoch-28`. It also produced one unresolved watcher
failure for NYUv2 `zt/seed-7319/epoch-68`; the failed audit reports
`stable_report_double_read` plus history, report/checkpoint, optimizer-state,
and scheduler consistency problems. No later strong audit has yet classified
this observation as a transient race, so the independent reconstruction is
intentionally `status=failed` with `unresolved_failures=1`, while terminal
cells remain `29/240` and `execution_complete=false`. The failed artifact and
state are preserved byte-for-byte; the downstream gate remains closed and no
scientific verdict was changed. The failed ledger is retained separately as
`artifacts/reports/four_task_atomic_ledger_reconstruction_20260824T230000Z_failed_unresolved_watcher.json`.
