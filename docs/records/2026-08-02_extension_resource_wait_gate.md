# Conditional-extension persistent resource wait gate

Date: 2026-08-02  
Evidence scope: orchestration and resource-state handling only; no real task metric
or method-effectiveness conclusion.

## Pre-result correction

Before any real FieldScope signal, main-task, causal, or extension result existed,
the conditional extension failure path was audited against the revision-bound
supervisor. A positive main-task verdict with insufficient extension storage would
make `run_extension_after_main.sh` exit. The supervisor could then replay the
completed main chain until its restart limit was exhausted, even though no code or
scientific input had changed.

The extension runbook now remains alive on a complete `resource_blocked` combined
budget. It rewrites that machine-readable budget from live free space every 600
seconds by default, verifies the clean fixed revision between attempts, and
continues automatically only after `fits=true`. The poll interval may be changed
only to another positive integer through
`FIELDSCOPE_EXTENSION_RESOURCE_POLL_SECONDS`; it does not alter images, pixels,
classes, representations, seeds, epochs, or the 15 registered ablation variants.

This converts an eventual restart-limit failure into an auditable wait for a real
capacity change. It does not claim that current storage is sufficient and does not
turn a resource limitation into negative scientific evidence.
