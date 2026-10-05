# H200 atomic-audit watcher epoch-race hardening

Status: operational hardening `passed`; formal scientific validation remains
`active` and incomplete.

At `2026-08-23T15:33:41Z`, the watcher selected NYUv2
`random_feature_local / seed 4121 / epoch 30`, then the training worker
atomically advanced report and checkpoint to epoch 31 before the strong-audit
helper read them. The resulting epoch-30 audit correctly failed its requested
epoch/step/exposure/scheduler checks, but it did not indicate a training or
scientific failure. Its SHA-256
`e360b37cb1ffc612215e4c554032dc570aa87999449e1d0a815266b5e82ead60`
is preserved as a non-authoritative failed race artifact.

Stable epoch-31 re-audits independently passed twice. The manual audit SHA-256
is `2b5300348ee1d48e1e694abcad37a0408c1cded8be79a60e978ab3bab98f0d43`;
the restarted watcher's standard audit SHA-256 is
`17c026af12f0993a3dc9dec623924c53603b7cfc47e0454b99fd4607c83cd388`.
Both bind report SHA-256
`49e5b90afe27a514889289bd26e416be4f3985e8e8a3f8cadacc956f4d9b1ca6`,
epoch 31, `5549` optimizer steps, and `22165` sample exposures.

The external watcher was hardened without changing fixed-revision source or
training. Old PID/SHA `3062994/ae99087a7fb34519ed8938df02b8f65150908b198e76342c95e47c6576317648`
was replaced by PID/SHA
`1870499/e02012a02729ebbba6f0a3c2b85f81bb0770a68df251219e8aa1c442a0057c55`.
A race classification is accepted only when the report has advanced beyond
the requested epoch, all reported problems belong to the expected
epoch/step/exposure/scheduler mismatch set, report and checkpoint reads were
stable, and their histories matched. The failed artifact remains in
`transient_audit_races`; it is not erased. Current `failures=[]`.

Machine-readable evidence is
`artifacts/reports/atomic_audit_watcher_epoch_race_hardening_20260823.json`.
No training output, condition, downstream gate, or scientific verdict changed.
