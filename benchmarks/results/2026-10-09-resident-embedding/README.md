# Resident embedding correctness, 2026-10-09

This is an input-layer correctness and resident training-path record, **not** a
language-model result, tokenizerless quality result, or throughput benchmark.
Native WGPU and actual browser WebGPU ran the same Rust probe against one frozen
independent CPU-f32 PyTorch 2.12.1 fixture, with `atol=3e-6, rtol=5e-5` fixed before
measurement. The browser did not use a Python or JavaScript arithmetic replica.

## Observations

- Both routes passed 12 forward/table-VJP conditions: six cases in contiguous
  and strided layouts, including repeated IDs, multiple workgroups, scalar IDs,
  empty sample axes, an empty table and a zero-width table. Maximum absolute
  output/VJP differences were zero for these dyadic-value fixtures; this does
  not establish bitwise equality for arbitrary floating-point inputs.
- A non-contextual embedding lookup classifier ran all 16 CE/SGD steps before
  any activation readback. Every saved logit, table gradient, loss and final
  parameter was compared with Torch. Final parameter maximum absolute error was
  `1.4901161193847656e-8` on both native and browser. Synthetic CE decreased from
  `1.0999140739440918` to `0.8106073141098022` at the measured pre-update steps.
- Ten invalid shape/ID cases, seven guard cases and two cross-device cases were
  rejected. Broadcast values, signed-zero gather bits, stable accumulation order
  and subnormal contributions were also checked.
- Overflowing duplicate-ID gradients rejected both zero-rate and nonzero-rate
  updates across a two-parameter owner. Every parameter and element remained
  bitwise unchanged; replaying stale gradients was rejected.
  Four negative controls prevent success, invalid readback, wrong-stage rejection
  or unrelated flag bits from being accepted as this numerical rejection.
- The resident tensor regression suite passed 96 tests; one explicit timestamp
  experiment remained intentionally ignored. Shared kernel contracts passed 55
  tests. Existing strict-WGPU row-indexing/autograd/tied-embedding tests passed
  all seven cases, including IDs above float's exact integer range.

[native.json](native.json) and [browser.json](browser.json) contain the observed
reports. [validation.json](validation.json) pins source, fixture, executable/WASM
and raw-log hashes. Native adapter was Apple M4 / Metal / IntegratedGpu; browser
reported BrowserWebGpu / Other with an empty adapter name. Do not infer its
hardware identity from the native observation. Host was macOS 26.4.1 arm64,
Rust 1.97.0; wasm-bindgen CLI was 0.2.129.

## Implementation and review

Exact integer IDs are prepared/grouped on CPU once and uploaded as immutable u32
transport. Lookup values, duplicate-ID table VJPs, loss seeds and SGD remain on
the owning GPU queue until an explicit snapshot. Stable grouping is now shared
with the existing host Tensor indexing code. There is no dense one-hot matrix,
implicit gradient average or floating-point atomic scatter. Input preparation
is not a GPU-native index-production API.

The independent read-only review found a missing check for a custom runtime with
zero uniform bindings. The fix rejects it before lazy pipeline creation. Pure
preflight regressions cover missing binding count, insufficient uniform size,
empty/nonempty results and two-dimensional dispatch boundaries. Native and
browser probes were rerun after the fix; initial records remain preserved.
The follow-up review also tightened update-rejection assertions to require the
typed stage-zero inherited-tensor failure, not merely any error. Both runtimes
were rerun with those stronger assertions and their negative controls.

The local default Python startup has unrelated runtime patches. Fixture
generation deliberately used `python3 -I -S`, adding only the standard package
directory before running the independent generator. The generator explicitly
requests CPU tensors and one thread, imports no SpiralTorch code, and refuses
to overwrite an existing fixture. Do not regenerate it to fix a failed test.

## Reproduction and boundaries

See [resident embedding documentation](../../../docs/resident_embedding.md) for
the Rust API, native command and browser build/serve commands. The checked-in
CI runs the native probe and builds the WASM example; a build alone is not the
browser runtime observation recorded here. Raw build/test logs and binaries
are retained locally, not published. Only derived reports, hashes and source
are included; unrelated human-owned artifact deletions are not part of this work.

The next layer is a single-owner causal byte decoder: token/position tables,
residual blocks and output head, with shifted targets and prefix invariance.
A causal trainable Z-Space encoder and its parameter VJP remain separate work.
The existing whole-text language-wave DFT is not a safe drop-in causal feature
producer. No Python/WASM model facade, release bump or measured speed claim is
added by this primitive.
