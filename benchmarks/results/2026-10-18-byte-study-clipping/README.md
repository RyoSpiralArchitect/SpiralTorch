# Request-Bound Language Clipping

Source-only follow-up to the [resident clipping primitive](../2026-10-18-resident-language-clipping/README.md).
The same Rust mechanism now runs through a v10 study recipe, the installed Python
wheel and the real packaged WASM client. Clipping is an explicit part of exact
request/checkpoint identity, independent of the eight existing geometry designs
and either SGD or AdamW. Legacy v1-v9 stay unclipped; v10 requires a declared
disabled or active-global-L2 policy. See the [API guide](../../../docs/byte_corpus_optimizers.md).

## Fixed Probe

These are **synthetic two-update mechanism checks**, not a corpus quality study.
The fixture uses seed 11, B=1, T=4, width=4, hidden=8, two attention heads and one
block. Ordinary, learned Poincare and frozen Poincare arms share backbone bits
and exact byte windows. Rate is 0.001. The geometric arms share initialization.

Four v10 conditions combine SGD/AdamW with clipping disabled/active at 0.01.
AdamW declares beta1=0.9, beta2=0.999, epsilon=1e-5, weight decay=0.01 and geometric
decay=0. Both v9 controls additionally run in the browser. No seed, clipping limit,
reference, data or tolerance was changed to obtain passing comparisons.

## Results

- Installed Python: all four runs survive an intentional SIGKILL after the first
  durable seal. Fresh-process resume matches uninterrupted report/checkpoint
  bytes, retains explicit parent lineage and leaves the source run unchanged.
  Restored artifact logits and four-byte greedy generation also match.
  [Recovery receipts](results/python-recovery.json)
- Packaged WASM: both disabled recipes reproduce their v9 case results exactly;
  active clipping changes actual updates. Learned geometry changes, frozen
  geometry stays fixed, and changing clipping policy rejects resume. Separate
  fresh pages reproduce complete report/checkpoint bytes for both active recipes.
  [Browser](results/browser.json), [resume](results/browser-resume.json)
- Independent CPU PyTorch: **three of four conditions pass** on both clients.
  SGD-off, AdamW-off and AdamW-active pass existing parameter, geometry-delta,
  loss and moment/clock criteria. **SGD-active remains unqualified**, not waived.
  `geometry.raw_decay` and `geometry.raw_gain.0` have zero net change in both
  implementations; the reference minimum learning delta of 1e-8 is not met.
  All five learned geometry tensors match the reference in this short native
  run, but equality without a measurable update is not proof of learning.
  [Native comparison](results/native-comparison.json), [browser comparison](results/browser-comparison.json)
- Python also runs the v10 AdamW policy through the 25-arm v8 contextual-mixture
  design, including frozen routers/query projections. Two-step full versus
  split resume is exact; all declared frozen parameter bits, moments and clocks
  stay frozen. This is execution coverage, not a 25-arm numerical/quality study.

Small global clipping limits can suppress a small geometry update below parameter
precision. The failure above is retained as a useful negative control. It is not
the separate unresolved 1,024-update Key-bias discrepancy, which this work neither
fixes nor masks. No default is selected and no geometric quality/speed gain is
established. Native adapter: Apple M4 / Metal / IntegratedGpu. Browser reports
BrowserWebGpu / Other; physical hardware execution is not independently verified.
Same-client exact resume does not promise general cross-device bitwise equality.

Validation: Rust clipping tests 4/4 with real GPU opt-in; full Rust study filter
85/85 with GPU bodies disabled (not 85 runtime tests); Python binding 15 passed
and one CPU-wheel-only skip, plus the contextual GPU test above; protocol suite
68 passed/10 optional-Torch skips; resume suite 22 passed; independent clipping
oracle 6/6 with Torch. Rebuilt `nn` and `webgpu` WASM packages pass 25/35 Node
checks and their export matrices. Minimal WASM check and scoped rustfmt pass.
Standard st-nn clippy passes with 24 warnings in unchanged code; a strict
`-D warnings` invocation fails on those diagnostics and is **not** reported green.
The wheel remains a private source candidate at 0.4.31, not a PyPI upload.

An independent read-only local review found no actionable issues in the v10
semantic core/reference/getter patch. It exercised 11 selected Python probes,
184 corrupted report/resume cases and 17 scalar boundaries; Rust was source-read
only. Client runtime verification above was performed by the implementing agent.
The user-deleted files were preserved. CI must still qualify the pushed head
before merge; local results do not bypass it.

## Reproduce

Recorded tools: Rust 1.99.0, rustfmt nightly-2026-04-15, Python 3.12.6,
PyTorch 2.12.1, wasm-pack 0.14.0 and wasm-bindgen 0.2.129. Create an isolated
candidate-wheel environment. `PRIVATE` is a new directory outside the repository;
never serve the private root directly. Build the WASM package there:

```sh
RUSTUP_TOOLCHAIN=1.99.0 maturin build --locked --release \
  -m bindings/st-py/Cargo.toml --out "$PRIVATE/wheels"
RUSTUP_TOOLCHAIN=1.99.0 wasm-pack build bindings/st-wasm --target web \
  --out-dir "$PRIVATE/webgpu-package" --release -- --locked --features webgpu
python -I -B benchmarks/results/2026-10-18-byte-study-clipping/prepare.py \
  "$PRIVATE/browser" --package "$PRIVATE/webgpu-package"
# Use the isolated interpreter where this exact wheel is installed.
python -I -B benchmarks/results/2026-10-18-byte-study-clipping/compare.py native \
  "$PRIVATE/browser" "$PRIVATE/native" --wheel "$WHEEL"
# Exit 1 is expected for the retained SGD-active minimum-delta failure.
python -I -S -B benchmarks/results/2026-10-18-packaged-language-mlops/qualify.py serve \
  --repo "$PRIVATE/browser/site" --package "$PRIVATE/webgpu-package" \
  --output "$PRIVATE/browser" --port 8828
```

Open `http://127.0.0.1:8828/`, wait for passed, then open a fresh page at
`http://127.0.0.1:8828/?phase=resume`. The reused frozen-asset server exposes only
allowlisted synthetic fixtures/package files and writes results exclusively.
Close it after validation. Do not run native and browser GPU checks concurrently.

```sh
python -I -B benchmarks/results/2026-10-18-byte-study-clipping/compare.py browser \
  "$PRIVATE/browser" "$PRIVATE/browser-comparison"
python -I -B tools/test_byte_corpus_gradient_clip.py
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo +1.99.0 test --locked -p st-nn \
  --no-default-features --features wgpu --lib resident::byte_study::clipping \
  -- --test-threads=1
```

`publish.py PRIVATE NEW_OUTPUT` rechecks source/package/input identities, every
comparison disposition and retained recovery hashes before exporting only
metrics, checks and hashes. It also expects `node-nn.json`/`node-webgpu.json` from
the shared `bindings/st-wasm/tests/byte_lm_node.mjs` transport checks (using a
complete packaged-MLOps fixture as described in the preceding qualification).
Raw requests, parameters, moments, checkpoints and generation remain private.
[Inputs and hashes](results/inputs.json) bind evidence, not independent attestation.
