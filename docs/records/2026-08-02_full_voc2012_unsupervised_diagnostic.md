# Full VOC2012 unsupervised diagnostic milestone

Date: 2026-08-02
Evidence scope: completed full unsupervised segmentation diagnostic only. This is
not a main-task, causal, extension, or final method-effectiveness conclusion.

## Provenance and completion

The formal run on `pro6000` produced the complete PASCAL VOC 2012 test diagnostic
at clean revision
`10224a14a618f54609de43e5ad80e2336dbc0a65` (`10224a1`) with code-tree SHA-256
`f1f25da1e34e12c54f57fe9849e17dfa6631c5f0a2b1d2bc762554de050d8f13`.

- source output:
  `/root/autodl-tmp/FieldScope/FieldScope-internal/outputs/full_validation/auraflow_v03/voc2012_unsupervised.json`;
- source output: 4,879,572 bytes, SHA-256
  `27d0d5b6b78e003216623415d1dc2dff74ab3db21fe3fe1d4b85b96d8a62498f`;
- source dense-cache manifest:
  `/root/autodl-tmp/FieldScope/datasets/feature_cache/auraflow_v03_dense/voc2012_test/dataset_manifest.json`,
  SHA-256
  `c946a2d5c293fe94b18aba8e3b5b10770dc165133222f7e16020557dbd7d2c47`;
- runtime profile SHA-256:
  `8480d5f6c348ef094c16e13dfe307116dd59df7849539a6807d642f1450c5c3c`;
- cache manifest: `status=passed`, `complete=true`, 1,449 samples, 23
  shards, dense storage, and sample-ID SHA-256
  `62239fdc4c3d52c5d73e63496b71e22619596f8f40f5b7115df2c299b7cb1096`;
- diagnostic: `status=passed`, 1,449 per-sample records, 11
  representations, five finite aggregate metrics per representation, and 2,000
  bootstrap resamples per metric.

Here, `status=passed` means the registered computation and provenance checks
completed. It does not mean the research hypothesis passed.

## Registered primary metrics

| representation | boundary AP | pairwise AUROC |
| --- | ---: | ---: |
| response | 0.067567 | 0.555525 |
| response shuffled | 0.071644 | 0.535121 |
| mismatch | 0.100171 | 0.526670 |
| state | 0.176682 | 0.651174 |
| z0 | 0.178140 | 0.641206 |
| zt | 0.178098 | 0.641198 |
| trajectory | 0.178118 | 0.641200 |
| velocity | 0.196577 | 0.645991 |
| endpoint | 0.194834 | 0.647501 |
| DiT attention | 0.185377 | 0.642168 |
| DiT hidden | 0.203166 | 0.662464 |

The response graph exceeded its shuffled control on pairwise AUROC by
`+0.020403`, but its boundary AP was lower by `-0.004077`. The response graph was
also below every registered static, trajectory, velocity, endpoint, and hidden
control on both registered primary metrics. Therefore this completed diagnostic
does not support the specific unsupervised boundary-recovery sub-hypothesis.

## Exploratory paired check

The source report contains per-representation bootstrap intervals, not paired
intervals for representation differences. A read-only, post-result exploratory
check used the 1,449 retained per-sample records, seed 4121, and 2,000 paired
bootstrap resamples. It is not part of the registered evidence gate.

For response minus response-shuffled:

- pairwise AUROC: `+0.020403`, 95% CI `[+0.017740, +0.022957]`;
- boundary AUROC: `-0.014876`, 95% CI `[-0.019750, -0.010058]`;
- boundary AP: `-0.004077`, 95% CI `[-0.005131, -0.002912]`;
- boundary best F1: `-0.000874`, 95% CI `[-0.002450, +0.000516]`;
- foreground spectral IoU: `+0.018344`, 95% CI
  `[+0.014218, +0.022380]`.

The same exploratory paired check placed response below state, DiT hidden, DiT
attention, velocity, and endpoint on both registered primary metrics, with all
reported 95% intervals remaining below zero. The compact values and exact source
identities are retained in
`artifacts/reports/2026-08-02_voc2012_full_unsupervised_summary.json`.

## Exploratory failure-mode analysis

A second read-only, post-result analysis aligned all 1,449 diagnostic records
with the dense-cache targets and used the actual 16 by 16 diagnostic grid. This
analysis was not preregistered and does not alter the registered gate.

Across samples, response minus response-shuffled was positive for 66.43% of
finite pairwise-AUROC comparisons, with a mean delta of `+0.020403`. It was
positive for only 38.34% of boundary-AP comparisons, with a mean delta of
`-0.004077`, while foreground spectral IoU was positive for 52.93% of samples,
with a mean delta of `+0.018344`.

The pairwise-AUROC delta increased across foreground-fraction quartiles:
`+0.00304`, `+0.01717`, `+0.03186`, and `+0.02953`; the corresponding fractions
of positive samples were 51.7%, 63.1%, 76.8%, and 74.1%. By contrast, the
boundary-AP delta became most negative in the highest boundary-density quartile:
`-0.00198`, `-0.00317`, `-0.00283`, and `-0.00831`, with only 31.6% positive in
the last quartile. Foreground spectral-IoU deltas increased across the same
boundary-density quartiles from `+0.00919` to `+0.03186`.

Multi-object composition showed the same split. From one-class to two-class and
three-or-more-class images, the mean boundary-AP deltas were `-0.00227`,
`-0.00753`, and `-0.00898`; mean pairwise-AUROC deltas were `+0.02173`,
`+0.01952`, and `+0.01177`. Boundary density correlated negatively with the
boundary-AP delta (`r=-0.13318`), and class count did likewise (`r=-0.12726`).
Foreground fraction correlated positively with the pairwise-AUROC delta
(`r=+0.16984`).

These post-hoc strata suggest that the response representation may contain a
coarse regional-coupling signal while failing to calibrate precise boundaries,
especially for dense boundaries and multi-object scenes. This is an exploratory
interpretation to guide later causal and downstream analyses, not a registered
method conclusion.

## Research boundary and continuation

This is one completed subtest. It neither proves nor disproves the full FieldScope
method on its own. The formal run continues through ImageNet-100, VOC2012,
ADE20K, and NYUv2 with the registered 20-representation by three-seed readout
matrices, followed by the causal audits and machine-readable final verdict. No
paper claim should be updated from this diagnostic alone.
