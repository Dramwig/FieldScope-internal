# Structurally undefined metric recovery gate

Date: 2026-08-02

Evidence scope: implementation-recovery validation only. This record does not
change the completed VOC diagnostic result and is not evidence that the full
FieldScope method is effective.

## Failure reproduced on the old formal revision

The complete 1,449-image VOC2012 diagnostic at formal revision
`10224a14a618f54609de43e5ad80e2336dbc0a65` contains structurally undefined
per-sample graph metrics for targets with no valid positive/negative pair or no
local boundary. A direct read-only invocation of the old formal
`_unsupervised_checks` returned:

- `passed=false`;
- zero registered comparisons;
- 13 audit problems;
- 12 non-finite-value problems plus one runtime-profile mismatch in the
  deliberately profile-free diagnostic invocation.

Supplying the formal runtime identity removes the profile mismatch but cannot
remove the old auditor's non-finite-value failures. Therefore the old revision
cannot produce a complete main evidence decision after the four matrices finish.
Waiting for the matrices would only defer a deterministic final blocker.

The source report used to reproduce the issue was unchanged:

- path:
  `/root/autodl-tmp/FieldScope/FieldScope-internal/outputs/full_validation/auraflow_v03/voc2012_unsupervised.json`;
- bytes: `4,879,572`;
- SHA-256:
  `27d0d5b6b78e003216623415d1dc2dff74ab3db21fe3fe1d4b85b96d8a62498f`;
- formal code-tree SHA-256:
  `f1f25da1e34e12c54f57fe9849e17dfa6631c5f0a2b1d2bc762554de050d8f13`.

## Temporal record

The recovery explicitly follows observation of the old VOC result:

1. recovery protocol commit:
   `13cd4df` (`research(protocol): lock undefined-metric recovery`);
2. implementation commit:
   `adc86b7` (`fix(evidence): audit structurally undefined metrics`).

The protocol fixes a representation-independent all-or-none finite-value rule,
requires exact eligibility alignment across causal sources, and retains at least
95% of all registered samples. It does not alter metric directions, candidates,
controls, seeds, bootstrap settings, or scientific decision thresholds.

## Local validation of implementation `adc86b7`

- focused recovery and regression tests: 5 passed;
- complete evidence test file: 17 passed;
- Ruff over `src`, `tests`, and `scripts`: passed;
- full pytest: 173 passed in 237.22 seconds;
- two-step toy smoke: passed;
- toy losses: `3.2219009399414062 -> 1.9724923372268677`;
- toy cache reload exact equality: true;
- `git diff --check`: passed.

## Real read-only replay against the old 1,449-image report

The clean isolated analyzer worktree at `adc86b7` replayed the unchanged old
source report and its dense-cache manifest. The replay asserted the source
SHA-256 and returned:

- registered comparisons: 12/12;
- audit problems: 0;
- total samples: 1,449;
- pairwise AUROC eligible: 1,433;
- pairwise AUROC structurally undefined: 16;
- pairwise AUROC eligible sample-ID SHA-256:
  `f10901e2019026e7cc168679cf8a3e5efff4a4420c22a713a737bbb1799b95e3`;
- pairwise AUROC excluded sample-ID SHA-256:
  `976d55b46b17e00cd5ae906d82ff2ef130e339553fa9dbb937f9bf4b281a830c`;
- boundary AP eligible: 1,432;
- boundary AP structurally undefined: 17;
- boundary AP eligible sample-ID SHA-256:
  `81eaadde2465cd5decdf1ff6576887c8c88a9c588f40bf417866d81881c03def`;
- boundary AP excluded sample-ID SHA-256:
  `5cc352a90cc803dc4cf340ec4077f452698ff6cb110e5d1cbeddb411525a9433`;
- mixed finite/non-finite samples: zero for both decision metrics;
- retained fractions: `0.988958` and `0.988268`, both above the fixed 95%
  minimum.

The replay preserved `scientific_gate_passed=false`. Thus the implementation
repairs audit completeness without converting the observed negative VOC result
into a positive result.

## Operational consequence

The old formal run must be replaced before it spends weeks producing matrices
that its own auditor cannot consume. Its source artifacts will be moved, not
deleted, into a revision-labelled archive on the same data filesystem. Prepared
datasets, immutable model weights, and source assets remain unchanged. The
replacement formal run must start from a clean fixed revision containing the
locked recovery and must regenerate every provenance-bound cache, report,
checkpoint, main decision, causal decision, conditional extension if triggered,
and final decision.

The complete method-effectiveness objective remains open until the replacement
run and the independent completion audit both finish.
