# Paired Query-Coordinate Pullback

**A selective learning control, not promotion of the projective family.**
Native WGPU and browser WebGPU each passed both paired router cases. Their
entire original six-case reports, including outputs, VJPs, update trajectories
and checkpoint bytes, exactly reproduced their own archived baseline. That
qualification remains **4/6**, with both original failed gates preserved.

The new Rust API stops only the query router's input adjoint at shared causal
coordinates. Router parameters, metric derivatives and ordinary QKV remain
trainable. Full backward remains the default. See
[API, update and resume semantics](../../../docs/query_coordinate_pullback.md).

## Results

| Paired router contrast | Native relative L2 | Browser relative L2 | Limit |
| --- | ---: | ---: | ---: |
| Four-way query, 1 block | 0.0002983405 | 0.0002988002 | 0.002 |
| Four-way query, 2 blocks | 0.0000495528 | 0.0000495528 | 0.002 |

Both modes share the same forward, initial parameters and cotangent. The
comparison is `(GPU Full - GPU Stop)` versus `(CPU Full - CPU Stop)`, with
f32 subtraction in each runtime. The frozen threshold and nonzero-reference
floor were not relaxed. Zero and wrong-sign controls are rejected. All 26/43
stopped parameter tensors and embeddings are also checked against an independent
Torch autograd graph with `coordinates.detach()` at the query input. All
geometry tensors additionally pass the `.002` relative check. Unrelated
parameter VJPs (20/37 slots) are bit-exact Full versus Stop; repeated Full,
captured logits and model weights are unchanged.

The old one-block router comparison still fails at native `0.002795259` and
browser `0.002777722`. It subtracts **CPU Stop from GPU Full**, so rounding
differences in the full gradient contaminate the small router contrast. Even
substituting CPU f64 Full in that mixed calculation fails (`0.005465475`),
while the paired CPU f32 versus paired CPU f64 contrast is `0.000471579`.
This diagnoses an attribution problem, not a newly improved forward kernel.
The two-block diagnostic is recorded too; no case was discarded.

## Phase Failure Stays Open

The projective-only two-block step-12 phase-gradient gate is still failed:

| Comparison | Native | Browser |
| --- | ---: | ---: |
| Frozen learning trajectory | 0.003332510 | 0.003370401 |
| CPU f32 at the same runtime weights | 0.0000243991 | 0.0000719465 |
| CPU f64 at the same runtime weights | 0.0000428906 | 0.0000294703 |

These are relative L2 errors. The diagnostic covers all 16 states in both
runtimes, and first reproduces all six initial CPU-f32 references bit-exactly.
It separates same-state derivative accuracy from accumulated trajectory drift;
it does not prove long-horizon stability or waive the old gate. The f64 graph
retains existing wave constants and is not a fully qualified f64 oracle.

## Evidence And Limits

- [Results](results.json) include every stopped parameter's scalar errors,
  paired contrasts, raw input hashes and exact legacy report comparisons.
- [Same-state diagnosis](same-state.json) includes all six initial cases and
  both 16-step phase traces, including unfavorable comparisons.
- [Frozen paired protocol](protocol.json) predates the new native/browser
  measurement. It defines a separate paired test, not replacement criteria.
- [Validation](validation.json) and [review](review.md) record exact verification
  scope, source hashes, binary/log hashes and independent-review limitations.

Tests include stopped CE updates with exact native checkpoint continuation when
the caller reapplies the mode. Browser evidence does **not** claim stopped-mode
training/resume, only paired VJPs and unchanged Full training/resume. The first
new API test used overly symmetric weak weights and could not detect the tiny
router contribution; its failure is retained in the local log. The fixture was
made asymmetric without changing the library or test threshold.

These remain tiny synthetic four-way cases, not a corpus experiment, matched
capacity/compute comparison, LLM FT benefit, throughput improvement, CUDA
validation or universal geometry qualification. No large combination sweep is
promoted here. Raw tensors, weights, checkpoints, binaries and screenshots stay
local; public files contain scalars, provenance and reproduction instructions.
No previous qualification or rejected optimization evidence was rewritten.

## Reproduce

Use a fresh private `RAW` directory. `BASELINE` holds `reference.json`,
`native-observed.json` and `browser.json` from the final run of
`fa6113be2db7562746997875adfde95f114bc4fe`. The
[previous recipe](../2026-10-13-complex-projective-v1/README.md#reproduce)
recreates those inputs on the same runtime. The reference hash remains
`07b87441cd15ff7d2478cf87dd460c21d58b1118ee78509992ae231014795358`.
Native adapter: Apple M4 / Metal; Torch 2.12.1 CPU f32/f64, one thread.

```sh
export CARGO_INCREMENTAL=0
export SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1
cargo test --locked -p st-nn --no-default-features --features wgpu \
  resident::byte_decoder:: --lib -- --test-threads=1
cargo test --locked -p st-nn --no-default-features --features wgpu \
  --example resident_byte_flat_metric -- --test-threads=1
python3 -I -S -B -c 'import sys,unittest;sys.path.insert(0,"tools");r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName("test_validate_resident_query_pullback"));raise SystemExit(not r.wasSuccessful())'
python3 -I -S -B -c 'import sys,sysconfig,runpy;sys.path.append(sysconfig.get_path("purelib"));sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name="__main__")' \
  tools/diagnose_resident_byte_projective.py "$BASELINE" "$RAW/diagnostic"
cargo build --locked --release -p st-nn --no-default-features --features wgpu \
  --example resident_byte_flat_metric
"$CARGO_TARGET_DIR/release/examples/resident_byte_flat_metric" \
  "$BASELINE/reference.json" --query-pullback > "$RAW/native-paired.json"
"$CARGO_TARGET_DIR/release/examples/resident_byte_flat_metric" \
  "$BASELINE/reference.json" --projective > "$RAW/native-legacy.json"
```

The last command intentionally exits 1 for the unchanged old qualification.
Keep its output; do not interpret a successful surrounding script as a pass.
Build `resident_byte_decoder_browser` for `wasm32-unknown-unknown` with the same
features and `wasm-bindgen 0.2.129 --target web`. Serve only the generated JS/WASM
and frozen `reference.json` under `target/resident-byte-query-pullback-web/`,
plus `crates/st-nn/tests/byte_query_pullback_browser.html`, on loopback. Run once
and retain the downloaded bundle as `$RAW/browser.json`. Do not serve the whole
private artifact directory.

```sh
python3 -I -S -B tools/validate_resident_query_pullback.py \
  "$BASELINE" "$RAW" "$RAW/scalar-results.json"
```

This rejects changed legacy reports rather than reclassifying them, verifies
the independent stopped VJPs, and emits only scalar results and hashes. Use a
fresh output path: neither the diagnostic nor publisher overwrites evidence.
