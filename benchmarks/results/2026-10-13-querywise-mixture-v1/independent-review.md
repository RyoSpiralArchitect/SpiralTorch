# Independent Core Review

A separate read-only reviewer inspected the five querywise core/backend files
relative to `1a027c262afa31d3679bb0afef1d1ad038ea986a`:

- `crates/st-kernel-contracts/src/mixture.rs` and `mixture/querywise.rs`.
- `crates/st-backend-wgpu/src/resident_tensor/mixture.rs`,
  `mixture/querywise.rs` and `mixture/tests.rs`.

No actionable findings were reported. Static inspection confirmed the
`[B,H,Q,S,K]` grouping, key-only router reductions, checked shape products,
forward shape/context/device preflight before materialization, strided view
handling, retained tape ownership and propagation through the existing guard
paths. The reviewer made no changes and performed no build, GPU/Torch, browser
or network work.

Shared shader, layout, guard and grid-helper internals were outside this
review's boundary and were not independently re-audited. Measurement scripts
and runtime evidence were not part of this core review. No input-conditioned
LM behavior, quality or performance claim follows from this static review.

## Independent Measurement Review And Repair

A second read-only reviewer inspected the CPU snapshot probe, Torch fixture
generator, shared native/browser harness and associated recipe. It identified
three P2 gaps: browser snapshots could be mislabeled native; absolute tolerances
could hide a skipped tiny router update in four cases; and two guarded siblings
did not distinguish whole-update rejection from independent per-parameter
rejection. The measured original native input was complete and native; these
findings did not establish incorrect recorded GPU updates.

After repair, the same reviewer reported no remaining P1/P2 findings in a
focused source recheck. It accepted stored native-full128 metadata and rejected
partial-37, browser and Torch reports; seven isolated stdlib gate tests passed.
A scalar mirror accepted all 54 reference updates and rejected all 54 frozen
controls. It confirmed the healthy independent sibling and positive control,
and that original fixture/v1 report hashes remain unchanged.

No additional mathematical finding was reported in the exploratory cancellation,
row-centering, cosine or query-to-shared gradient decomposition. The reviewer
used CPU standard-library inspection only: no code edits, builds, Torch imports,
GPU/browser/network execution. The v2 GPU remeasurement is separate evidence,
not independently re-executed or attested by this review. Input-conditioned LM
learning remains unimplemented and no quality advantage follows.
