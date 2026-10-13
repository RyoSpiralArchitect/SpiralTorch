# Matched-Graph Geometric QKV Learning, V2

This is an explicitly versioned native/WASM correctness qualification, not a
promotion of the original v1 result and not a language-quality benchmark.
Production model arithmetic and numerical thresholds are unchanged.

## What Is Compared

The same six fixed synthetic byte-model cases cover a projective-only metric,
a shared four-way mixture, and a query-position-conditioned four-way mixture,
each with one or two blocks. The mixture contains Poincare squared distance,
Euclidean squared chord, categorical Fisher-Rao squared distance, and complex
projective squared chord. Learned geometry projection, causal wave decay/phase,
metric gains and routing parameters receive gradients through the QKV model.
The two-block cases also exercise Topos and external biases.

The contextual router uses geometry coordinates at the query position; it does
not directly consume the transformer's projected Q tensor. The one-block cases
start with Q/K zero to isolate the geometric score route. Each model is tiny:
two documents, four input bytes per document, width four and two heads. All 16
updates repeatedly use those synthetic windows. This is not a corpus study.

V2 changes two comparison definitions, both frozen before measurement:

1. CPU Torch evaluates `x + (attention @ W + bias)`, matching Rust's completed
   output-projection module before the residual add. The generator's default
   still evaluates the original v1 association.
2. Query-coordinate pullbacks compare GPU `(Full - StopCoordinates)` with CPU
   `(Full - StopCoordinates)`, rather than subtracting CPU Stop from GPU Full.
   All stopped parameter VJPs are independently checked against CPU, and
   unrelated routes must remain exact. The old mixed-runtime formula remains
   diagnostic, not a renamed historical result.

The initial weights, inputs, targets, seeds 1761/1863, geometry scales, learning
rate 0.125, 16-step budget, absolute tolerance `3e-6 + 5e-5*abs(reference)` and
relative geometry threshold 0.002 remain fixed. Schema and execution-contract
mismatches are rejected; v2 cannot silently enter the v1 reader. The
[protocol](protocol.json) and [fixture identity](fixture-identity.json) are
separate from measured outcomes.

## Measured Result

Fresh native Metal and actual browser WASM/WebGPU runs each qualify **6/6** under
v2: all 96 updates, 720 learning-step geometry VJPs, both contextual controls and
all six exact checkpoint continuations pass. The contextual cases independently
check 26 and 43 stopped parameter VJPs in each runtime. Scalar results are in
[results.json](results.json), including every condition and learning step.

| Geometry / routing | Blocks | Final synthetic CE | Native max geometry relative error | Browser max geometry relative error |
| --- | ---: | ---: | ---: | ---: |
| Projective | 1 | 3.942208 | 0.000030914 | 0.000021443 |
| Projective | 2 | 3.927249 | 0.001835437 | 0.001871310 |
| Four-way shared | 1 | 3.942498 | 0.000008551 | 0.000011651 |
| Four-way shared | 2 | 3.915614 | 0.000300096 | 0.000244222 |
| Four-way contextual | 1 | 3.942504 | 0.000014528 | 0.000014830 |
| Four-way contextual | 2 | 3.915651 | 0.000090339 | 0.000186536 |

Reported first/final losses agree between these two runs. Initial CE is about
5.52-5.55. The one-/two-block comparison is not architecture-matched, and these
repeated synthetic windows do not rank language-model quality. The paired query
relative errors are about `0.000299` and `0.0000496`, within the fixed 0.002 gate.
No claim of a faster or more accurate production kernel follows from aligning
the reference graph.

## Evidence Boundaries

The original v1 fixture regenerates byte-identically with SHA-256
`07b87441cd15ff7d2478cf87dd460c21d58b1118ee78509992ae231014795358`. A fresh native
v1 run with the final code exactly matches the entire retained report, including
its failed gates and exit status 1. The earlier native/browser v1 result remains
**4/6**, not rewritten as a pass. The archived browser v1 outcome is retained,
not claimed as a new v1 browser run in this increment.

For v2, required checks include logits, every parameter/embedding/external-bias
VJP, active geometry and Topos routes, all losses/parameters/geometry VJPs and
mixture snapshots, causality, tape ownership, atomic rejection, projective
boundaries, exact checkpoint continuation and both paired contextual controls.
The scalar publisher independently recomputes numerical errors and exact resume
values. Private causality witnesses are checked against the original tolerance
and omitted from public output; checkpoints are represented only by hashes.

Finite stopped-VJP and unchanged-route mismatches fail qualification while
retaining observations. Malformed or non-finite data still aborts. An intentional
stopped-oracle corruption verifies a failing exit with all six cases, 96 updates
and both paired observations retained; this is a control, not a model result.
The earlier JavaScript-serialized corruption stopped at a separate checkpoint
identity guard before reaching that control and is preserved in the attempt log.

Thirty Python tests pass normally and under `-O`; eight Rust example tests cover
version separation, malformed oracles, failure retention and existing guards.
The [independent review](review.md) is source-only; the parent performed runtime
and artifact verification. See [validation](validation.json) for attempts,
source/binary/raw hashes and terminal exit codes, and [checksums](SHA256SUMS).

Raw weights, model vectors, checkpoints, binaries and screenshots stay local.
There is no general-equivalence, larger-mixture, long-context, language-quality,
speed, CUDA, PR, CI, merge or release claim. The next learning study should fix
multiple seeds and held-out byte sequences before comparing geometry mixtures;
synthetic correctness and loss reduction alone cannot pick a winning geometry.

## Reproduce

Use Rust 1.97.0, wasm-bindgen 0.2.129, CPU Torch 2.12.1 and Python 3.12. Use a
fresh private `RAW` directory and serialize builds/GPU runs. The generator
refuses existing outputs. Isolated Python avoids user site customization/MPS
monkey patches; the generator sets one CPU thread and deterministic algorithms.

```sh
python3 -I -S -B -c 'import sys,sysconfig,runpy;sys.path.append(sysconfig.get_path("purelib"));sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name="__main__")' \
  tools/generate_resident_byte_projective_torch_fixture.py "$RAW/reference.json" --module-residual
python3 -I -S -B -c 'import sys,sysconfig,runpy;sys.path.append(sysconfig.get_path("purelib"));sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name="__main__")' \
  tools/generate_resident_byte_projective_torch_fixture.py "$RAW/legacy-reference.json"
export CARGO_INCREMENTAL=0
export SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1
cargo test --locked -p st-nn --no-default-features --features wgpu \
  --example resident_byte_flat_metric -- --test-threads=1
cargo build --locked --release -p st-nn --no-default-features --features wgpu \
  --example resident_byte_flat_metric
"$CARGO_TARGET_DIR/release/examples/resident_byte_flat_metric" \
  "$RAW/reference.json" --module-projective > "$RAW/native.json"
"$CARGO_TARGET_DIR/release/examples/resident_byte_flat_metric" \
  "$RAW/legacy-reference.json" --projective > "$RAW/legacy-native.json"
```

The legacy command intentionally exits 1. Preserve that exit/report; do not
treat it as the v2 outcome. Set `CARGO_TARGET_DIR` explicitly or use the normal
`target` path. Verify the generated v2 SHA-256 against `fixture-identity.json`
here, then place that identity file next to the private fixture without changing
it. A differing reference hash is a different experiment, not this reproduction.

```sh
cargo build --locked --release -p st-nn --no-default-features --features wgpu \
  --target wasm32-unknown-unknown --example resident_byte_decoder_browser
wasm-bindgen "$CARGO_TARGET_DIR/wasm32-unknown-unknown/release/examples/resident_byte_decoder_browser.wasm" \
  --target web --out-dir "$RAW/web"
```

On loopback only, serve `crates/st-nn/tests/byte_module_projective_browser.html`.
Map its generated JS/WASM, `reference.json` and `fixture-identity.json` to
`target/resident-byte-module-projective-web/`. Whitelist those exact files;
do not expose the entire private artifact directory. Run all six cases once,
inspect the page and retain its download as `$RAW/browser.json`.

```sh
python3 -I -S -B -c 'import sys,runpy;sys.path.insert(0,"tools");sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name="__main__")' \
  tools/validate_module_projective_results.py "$RAW" "$RAW/scalar-results.json"
python3 -I -S -B -c 'import sys,sysconfig,unittest;sys.path.append(sysconfig.get_path("purelib"));sys.path.insert(0,"tools");r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(["test_module_residual_geometry_reference","test_residual_association_reference","test_validate_resident_query_pullback","test_validate_module_projective_results"]));raise SystemExit(not r.wasSuccessful())'
```

Repeat the tests with `-O`. The publisher accepts complete verified v2 passes,
not subsets or raw `passed` flags alone; failures must remain failed raw evidence.
The previous [trajectory diagnosis](../2026-10-13-phase-trajectory-origin-v1/README.md)
and [original qualification](../2026-10-13-wave-batch-adjoint-v1/README.md) remain
available with their original criteria and limitations.
