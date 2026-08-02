# Structurally undefined graph-metric audit recovery

Date locked: 2026-08-02

Evidence status: post-observation implementation-recovery protocol, locked before
the replacement formal revision is created and before any of the four formal
downstream 20-representation by three-seed matrices produce results.

## Why a replacement formal revision is required

The complete 1,449-image VOC2012 diagnostic produced by formal revision
`10224a14a618f54609de43e5ad80e2336dbc0a65` revealed a mismatch between the
diagnostic metric contract and the evidence auditor. `diagnose-segmentation`
correctly records an undefined AUROC/AP/F1 value when a target contains no
positive/negative pair or no local boundary, and excludes that undefined value
from its finite aggregate. The formal evidence auditor instead rejects any
single undefined per-sample value as a numerical failure.

A read-only replay of the existing source report showed that this makes the
registered main audit deterministically incomplete: it produced zero
comparisons and 13 problems. This is an execution blocker, not an observed
scientific effect. The source report is retained at:

- path:
  `/root/autodl-tmp/FieldScope/FieldScope-internal/outputs/full_validation/auraflow_v03/voc2012_unsupervised.json`;
- SHA-256:
  `27d0d5b6b78e003216623415d1dc2dff74ab3db21fe3fe1d4b85b96d8a62498f`.

The old report was inspected before this recovery rule was locked. Therefore
this protocol must not be described as preregistered before all VOC results. It
is a transparent correction of a metric-domain bug. It does not change the
candidate representations, controls, metric directions, bootstrap seed,
resample count, downstream training budget, causal variants, or final decision
logic.

## Fixed structural-eligibility rule

For every graph metric and sample, inspect the value for the exact registered
11-representation set:

```text
response, response_shuffled, state, dit_attention, mismatch, velocity,
zt, trajectory, z0, endpoint, dit_hidden
```

The audit classifies the sample independently for each metric:

1. If all 11 values are finite, the sample is eligible for that metric.
2. If all 11 values are non-finite, the metric is structurally undefined for
   that target and the sample is excluded for that metric.
3. If some but not all values are finite, the report has a
   representation-specific numerical failure and the audit is incomplete.
4. At least 95% of all 1,449 registered images must be eligible for every
   decision metric. The threshold is fixed at 95%; the observed old-run counts
   are not used as acceptance thresholds.
5. Every candidate-control comparison for a metric must use the exact same
   eligible sample IDs. No representation-specific or direction-dependent
   filtering is allowed.

For causal/condition reports, the all-finite/all-non-finite classification and
eligible sample IDs must also agree exactly across `empty_prompt`,
`random_flow`, `spatially_shuffled_probe`, `neutral_prompt`, and
`unrelated_prompt`. A disagreement is incomplete even if each source retains
more than 95% of samples.

The audit output must record, per metric:

- total, eligible, and structurally undefined sample counts;
- eligible and excluded sample-ID SHA-256 values;
- retained fraction and the fixed minimum fraction;
- whether the all-or-none finite-value contract passed.

## Observed old-run structure used to diagnose the bug

These counts are disclosed because the recovery follows observation. They are
diagnostic facts, not new thresholds:

| metric | all 11 finite | all 11 non-finite | mixed |
| --- | ---: | ---: | ---: |
| pairwise AUROC | 1,433 | 16 | 0 |
| boundary AUROC | 1,432 | 17 | 0 |
| boundary average precision | 1,432 | 17 | 0 |
| boundary best F1 | 1,432 | 17 | 0 |
| foreground spectral IoU | 1,449 | 0 | 0 |

Because every undefined value is shared by all 11 representations, the pattern
is consistent with target-domain degeneracy rather than a representation output
failure. The implementation must nevertheless derive and verify this structure
from each formal report; it must not hard-code these counts or sample IDs.

## Replacement-run and provenance rules

- The recovery is implemented in a new clean revision after this protocol
  commit.
- Formal caches, reports, checkpoints, and decisions from `10224a1` are never
  silently relabeled as outputs of the replacement revision.
- Before launching the replacement revision, the current formal process tree is
  stopped by exact PID/PGID verification and its material outputs are moved to a
  recoverable, revision-labelled archive on the same data filesystem.
- Prepared datasets, immutable model weights, and read-only source assets remain
  unchanged.
- The replacement run starts from its own clean cache/output roots and repeats
  every provenance-bound signal, diagnostic, main, causal, conditional, and
  final gate required by the existing protocol.
- The old negative VOC diagnostic remains a historical result, but it is not
  substituted for the replacement revision's formal evidence.

## Implementation acceptance gate

Before replacing the remote formal run, all of the following must pass:

1. A synthetic structurally undefined sample is excluded symmetrically and the
   paired comparison uses the remaining exact sample IDs.
2. A mixed finite/non-finite sample makes the audit incomplete.
3. Retaining less than 95% makes the audit incomplete.
4. Causal sources with different eligibility masks make the audit incomplete.
5. Existing all-finite evidence tests remain unchanged and pass.
6. Full Ruff, full pytest, two-step toy smoke, and `git diff --check` pass.
7. A read-only replay against the archived 1,449-image VOC report produces all
   registered comparisons, reports the structural exclusions, and has no
   non-finite-value audit problem.

Passing this recovery gate proves only that the registered computation is
well-defined and auditable. It does not make the old or replacement scientific
result positive and does not relax any final claim boundary.
