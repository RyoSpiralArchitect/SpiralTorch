# Complex Projective QKV Qualification

**Status: experimental, NOT fully qualified.** Native WGPU and browser WebGPU
each passed 4/6 fixed cases. All six completed 16 SGD updates and exact
same-runtime resume, but two numerical gates remain failed in both runtimes.
No tolerance, seed, fixture, or failing criterion was relaxed.

The new family is complex projective **chord squared**, not Fubini-Study
geodesic distance or quantum computation. It joins Poincare, scaled Euclidean
and categorical Fisher-Rao metrics in the Rust-owned byte decoder, including
four-way shared/query mixing and complete coordinate/parameter gradients.
The existing `PauliSpinor` helper also calls the backend-neutral contract.
See [the API and exact definition](../../../docs/complex_projective_geometry.md).

## Final results

| Route | Blocks | Trainable scalars | Native | Browser |
| --- | ---: | ---: | --- | --- |
| Projective only | 1 | 2516 | PASS | PASS |
| Projective only | 2 | 2678 | FAIL | FAIL |
| Four-way shared | 1 | 2524 | PASS | PASS |
| Four-way shared | 2 | 2694 | PASS | PASS |
| Four-way query | 1 | 2564 | FAIL | FAIL |
| Four-way query | 2 | 2774 | PASS | PASS |

One-block cases zero Q/K to isolate geometry. Two-block cases retain ordinary
QKV, external biases and a Topos gate. Each uses two four-byte input windows,
256-way next-byte CE, SGD rate 0.125 and 16 updates. Seeds are 1761 and 1863.
These are tiny synthetic correctness checks, not a corpus experiment or LLM FT.
Parameter counts differ; neither equal-capacity nor equal-compute superiority
is claimed. No speed benchmark was performed for this increment.

| Unresolved gate | Native relative L2 | Browser relative L2 | Maximum allowed |
| --- | ---: | ---: | ---: |
| Projective-only, 2 blocks, step 12 `geometry.raw_phase` gradient | 0.00333251 | 0.00337040 | 0.002 |
| Four-way query, 1 block, detached-router pullback contrast | 0.00279526 | 0.00277772 | 0.002 |

Replaying the failed phase-gradient calculation at the **same native weights**
reduces the discrepancy to 0.0000243991 against Torch CPU f32 and 0.0000428906
against the f64 diagnostic. This suggests sensitivity to the accumulated
parameter trajectory; it neither proves a completely correct derivative nor
waives the frozen-trajectory failure. The diagnostic retains the existing wave
constants and is not a newly qualified full-f64 training oracle. The router
contrast failure has not been resolved by this diagnosis.

## Evidence

- [Final scalar results](reviewed/results.json) include every case's 16 losses,
  numerical errors, failure records and independently recomputed exact-resume
  checks. [Final validation](reviewed/validation.json) binds source, fixture,
  binary, raw-result and log hashes.
- [Initial results](results.json) and [initial validation](validation.json)
  preserve the earlier source snapshot and its failures. The initial strict
  native invocation exited 1 at the phase-gradient gate; the subsequent
  observation mode retained the complete failed trajectory instead of dropping it.
- [Intermediate native result](intermediate-native.json) records the native-only
  rerun after overlap compensation, before the base-endpoint/control-preflight
  fixes. It also passed 4/6. No browser result is claimed for this snapshot.
- [Review and fixes](review.md) records the independent reviews. The final
  bounded follow-up found no actionable issue in the latest two fixes by
  inspection; reviewers did not independently rerun the workloads.

Final local tests: 9 pure projective, 16 native GPU pair, 1 spinor, 28 byte
decoder, 4 harness, and 3 independent Torch tests passed (61 total). Native and
WASM release builds succeeded. The native **qualification** still exited 1;
build success is not qualification success. Both runtimes passed eight boundary
cases, including five analytic cancellation regressions. The publication
checker also rejected six deliberately corrupted reports and retained an
injected finite numerical failure without promoting it to a pass.

Raw tensors, checkpoints, binaries and screenshots remain local. Public results
contain scalar projections, validation receipts, source hashes and reproduction
instructions only. Earlier contextual-v1 failures, v8 corpus outcomes and the
rejected guard-batching performance candidate are not superseded.

## Reproduce

Reference environment: Torch 2.12.1, CPU float32, one thread. Actual native
adapter: Apple M4 / Metal. Use a private fresh `RAW` directory and a reusable
`CARGO_TARGET_DIR`. Disable automatic Python site hooks when generating the
independent reference:

```sh
export CARGO_INCREMENTAL=0
export SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1
python3 -I -S -B -c 'import sys,sysconfig,runpy; sys.path.append(sysconfig.get_path("purelib")); sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name="__main__")' \
  tools/generate_resident_byte_projective_torch_fixture.py "$RAW/reference.json"
shasum -a 256 "$RAW/reference.json"
python3 -I -S -B -c 'import sys,sysconfig,unittest; sys.path.extend([sysconfig.get_path("purelib"),sys.argv[1]]); r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName("test_projective_reference")); raise SystemExit(not r.wasSuccessful())' \
  "$PWD/tools"
cargo test --locked -p st-kernel-contracts projective:: --lib
cargo test --locked -p st-backend-wgpu resident_tensor::poincare::tests --lib -- --test-threads=1
cargo test --locked -p st-core --no-default-features projective_chord_matches_bloch --lib
cargo test --locked -p st-nn --no-default-features --features wgpu \
  resident::byte_decoder:: --lib -- --test-threads=1
cargo test --locked -p st-nn --no-default-features --features wgpu \
  --example resident_byte_flat_metric -- --test-threads=1
cargo build --locked --release -p st-nn --no-default-features --features wgpu \
  --example resident_byte_flat_metric
"$CARGO_TARGET_DIR/release/examples/resident_byte_flat_metric" \
  "$RAW/reference.json" --projective > "$RAW/native.json" 2> "$RAW/native.log"
```

Frozen fixture SHA-256:
`07b87441cd15ff7d2478cf87dd460c21d58b1118ee78509992ae231014795358`.
The failed qualification's nonzero exit is expected; keep its report and do
not treat a wrapper's successful exit as a passed study.

Build the same source for the browser, using wasm-bindgen CLI matching the
locked crate version (0.2.129 for this run):

```sh
cargo build --locked --release --target wasm32-unknown-unknown \
  -p st-nn --no-default-features --features wgpu --example resident_byte_decoder_browser
wasm-bindgen --target web --out-dir "$RAW/web" \
  "$CARGO_TARGET_DIR/wasm32-unknown-unknown/release/examples/resident_byte_decoder_browser.wasm"
```

Use a fresh private web root containing only the test HTML and the generated
assets. Serve the HTML at `crates/st-nn/tests/byte_projective_browser.html`, and
the JS, WASM and **same frozen** `reference.json` at
`target/resident-byte-projective-web/`. Bind the server to loopback, open the
HTML, run all six cases once, and retain its downloaded JSON locally. Do not
serve the entire private raw/checkpoint directory or commit the downloaded file.

## Next boundary

Close the two numerical gates before expanding this new family into the large
corpus/combination sweep. Separate same-state derivative accuracy from
accumulated training drift. Then compare each family, shared mixtures and
query-conditioned mixtures with explicit parameter/compute budgets and held-out
loss. Query routing currently reads shared causal wave coordinates, not each
block's hidden state. Any richer attention mechanism is a new experiment, not
an explanation retrofitted to these failed gates.
