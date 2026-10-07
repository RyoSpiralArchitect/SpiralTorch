# Two-adapter native-runtime replay

Both full and retained-short history pass exact old/new replay and old-checkpoint
to new-runtime continuation on local pretrained GPT-2. See the
[protocol, limits and offline commands](../../../docs/fractional_multi_adapter_replay.md).

- Six successful processes execute 20 auxiliary updates (eight unique trajectory
  updates, twelve replay-only updates); this is not a new primary quality study.
- Loss tensor bytes, all eight parameter gradients, adapter parameters, named
  Adam state and RNG state match. The frozen model remains unchanged.
- The later native input VJP executes and is nonzero after the identity-initial
  update. Removing it changes the earlier adapter's gradients in a regression
  test, without changing that pass's forward loss or later adapter gradients.
- No text, token values, weights, native packages or private raw logs are published.
  Their needed checkpoint/source hashes and every numeric outcome are retained.

## Files

| File | Content |
| --- | --- |
| `summary.json` | Counts, scope, all six state/checkpoint/report receipts |
| `full-comparison.json`, `short-comparison.json` | Executed comparisons and full per-update numeric records |
| `run-reports.json.gz` | Six byte-preserved JSON reports, including recipes, schedules and receipts |
| `runtime-sha256.json` | Both complete frozen package manifests |
| `validation.json` | Validation record and preserved pre-training admission failure |
| `SHA256SUMS` | Closed public-file inventory and digests |

Public-only tests verify the archive's internal consistency, hashes, coverage
and declared boundaries. They cannot reexecute private checkpoint tensor
comparisons; that proof comes from the recorded offline runs and retained states.

The archive has no speed measurements and no heldout or generation outcome.
The small full/short loss differences must not be presented as a quality result.
