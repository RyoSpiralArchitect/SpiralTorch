# AdamW Parent-PR Review Integration

Base: PR #2270 head `d487e367c1cc97112564a56cea2d9fccf6bce527`.
This backport contains only AdamW review repairs and their regression coverage,
not the later study recipes, checkpoint-selection API, Python recovery package,
WASM MLOps package or shared Attention numerical changes.

Eleven implementation/test/fixture files were restored from reviewed descendant
`e9054eee521e7aa380ca610f76e28c46e6a28939` and checked byte-for-byte. Their only
intervening changes originated in `f27ca0aae` and `e9054eee5`. The parent-specific
documentation was edited separately. An isolated 51-MiB sparse worktree avoided
changing the main development checkout or its 8,234 intentional tracked deletions.

## Repairs

- Compute AdamW bias correction by composing complements on the exact integer
  accepted-update clock, rather than subtracting a rounded near-one power.
  Rust and WGSL retain the same recurrence and validation policy.
- Accept rank-zero `shape=[]` state as one scalar, consistent with `NdLayout`.
  Missing/extra moments, zero dimensions, wrong rank and invalid clocks reject.
- Use explicitly guarded divide-by-zero for the frozen invalid-gradient control,
  instead of assuming that an out-of-range multiplication produces infinity.

The scalar test checks restored shape, weight/moment bits and clock before an
update and after three updates, against uninterrupted execution. It also checks
that a retained initial snapshot remains unchanged.

## Exact-Candidate Validation

- Shared CPU AdamW contracts: 5 passed.
- Native parameter/AdamW suite with real GPU opt-in: 8 passed, including three
  fixed-gradient 1,024-update trajectories and eight restored-clock probes.
- Native byte decoder/study/serving suite with GPU opt-in: 111 passed, including
  joint geometry learning, freeze, rejection and full checkpoint resume.
- Python study protocol suite: 65 tests, 59 passed and 6 optional/reference skips.
- CPU Torch 2.12.1 fixture regeneration: byte-identical to the committed fixture.
- Native and actual browser numerical reports: passed. Browser controls include
  the scalar checks and twelve reference updates. Physical browser hardware
  execution remains unverified; browser success is not an adapter attestation.
- Strict backend Clippy on native and WASM, scoped pinned rustfmt and staged
  whitespace checks passed. Vendored dependency warnings were not suppressed.

The three long optimizer trajectories use CPU Torch and Rust CPU float32 as
required references, with scalar f64 diagnostics. The eight restored-clock probes
instead use analytic f64 references constructed in Rust; they are not Torch
large-clock qualification. Full synthetic reports are retained in this folder;
`validation.json` records source, fixture, WASM and private-log hashes.

A fresh read-only local reviewer inspected this exact backport against its older
parent, found no actionable code/dependency defect, and corrected one P3
documentation attribution issue distinguishing the two kinds of reference above.
The reviewer did not execute tests; runtime validation was performed separately.

## Reproduce

Use Rust 1.99.0. To reuse an existing build cache, pass `--target-dir` explicitly;
all tests must still compile the integration source, not run an older binary.

```sh
cargo +1.99.0 test --locked -p st-kernel-contracts --lib adamw
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo +1.99.0 test --locked \
  -p st-backend-wgpu --lib resident_training::parameters -- --test-threads=1
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo +1.99.0 test --locked \
  -p st-nn --no-default-features --features wgpu --lib resident::byte_ -- --test-threads=1
python3 -I -S -B -m unittest discover -s tools -p 'test_byte_corpus*.py'
python3 -I -B tools/generate_resident_adamw_numerics_fixture.py > /tmp/adamw-numerics.json
cmp /tmp/adamw-numerics.json crates/st-backend-wgpu/tests/fixtures/resident_adamw_numerics.json
cargo +1.99.0 run --locked -p st-backend-wgpu --example resident_adamw_numerics
cargo +1.99.0 build --locked -p st-backend-wgpu --release \
  --target wasm32-unknown-unknown --example resident_parameters_browser
```

Generate the web bindings with wasm-bindgen 0.2.129 and serve only those generated
files plus `adamw_numerics_browser.html` over loopback. Do not serve the private
validation root. The page offers a complete result download, including failures.

These checks do not establish full-language numerical equivalence or language
quality. The separately recorded long-language Key-bias discrepancy remains
unresolved; no gradient masks or relaxed tolerances were introduced here. This
repair is raw-coordinate AdamW, not Riemannian momentum transport. Local results
do not substitute for successful remote CI and review at the exact merge head.
