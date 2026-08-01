# Same-revision ImageNet-100 asset resume gate

Date: 2026-08-02
Evidence scope: orchestration correctness and release validation only. This is not
method-effect evidence.

## Change

The full-validation waiter now reuses an existing ImageNet-100 asset report only
when all of the following remain exact:

- report status and registered archive path, bytes, SHA-256, member count, class
  count, train count, and official-validation count;
- current archive nanosecond modification time, recorded when the report is
  produced;
- prepared-root identity;
- clean code revision of the split audit;
- exact `116455/12940/5000` FieldScope split counts and zero pairwise overlap.

Missing legacy fields, a revision change, archive identity change, dirty provenance,
wrong counts, or overlap causes the existing report to be rejected and the full
verification/extraction path to run. The gate therefore avoids repeated extraction
of the same 17,319,391,232-byte archive after a same-revision waiter restart without
turning a prior report into an unconditional waiver.

The conditional extension evidence path also gained a positive-path unit test that
requires all 15 registered one-factor variants and verifies the supported verdict
only after the ImageNet-1k candidate/control checks and all variant comparisons are
present.

## Validation

- `ruff check src tests scripts`: passed.
- `python -m pytest`: 144 tests passed.
- toy smoke with two optimization steps: passed.
- `git diff --check`: passed.
- the changed asset verifier and full waiter passed `bash -n` on `pro6000`.

No real FieldScope signal, task, causal, or extension metric was produced while
this gate was changed or validated.
