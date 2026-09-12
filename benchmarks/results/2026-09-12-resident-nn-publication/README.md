# Resident NN Publication Checks

This companion to [code PR #2101](https://github.com/RyoSpiralArchitect/SpiralTorch/pull/2101)
preserves the original runtime/experiment history while keeping the reviewed
code tree identical to the code branch. The evidence PR adds no further runtime
changes. Earlier compressed raw archives and unsuccessful experiments are retained.

The new wider benchmark publishes all results, validation records, hashes and
rerun instructions in `../2026-09-12-resident-direct-vjp-wide`. Its full original
tensor payloads remain local by explicit owner choice; they are not Git blobs.

`wide-results-verification.json` records an independent check of the result
manifest, all 54 saved conditions, 4,320 retained intervals, and all 61 local
measurement files (10,937,941,609 bytes). This publication-time check hashes
original bytes and rechecks aggregation; it does not rerun numerical validation
or GPU computations. The six original numerical validation reports are included
in the wide result directory.

`wide-publication-screen.json` records bounded credential-pattern screening of
the new result package. It is not a full security audit or a guarantee that no
sensitive content exists. Machine paths and build identities are deliberately
preserved as provenance.

The code review fixes cover public Python exports, WGPU pointwise type stubs,
real-GPU NN CI opt-in, and fixed-width guard-test decoding. Exact test outcomes,
failed attempts, source identities and limits are recorded in the wide package's
`review` directory. GitHub CI/review completion and merge state belong to the PRs,
not these earlier frozen benchmark receipts.
