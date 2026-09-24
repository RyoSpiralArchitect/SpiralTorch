# Reuse the resident LayerNorm centered tape in forward

Exploratory native Apple M4 / Metal comparison. Baseline source is
`0eb238fc848504b729de3eb4fd64e1a866d163d5`; candidate source is
`97e497bb4f04c8f9fd74469109dab02e4d80956f`. Both release examples were
built with Rust 1.98.0 on the same host. Retained baseline binary SHA-256 is
`cdb10e44db3f7905b3298d9e966ea64b5ae4549e32a4b65065adafbe0cd62d42`;
candidate is
`67641159ea25e2bf12325489721563c7bd954eb7774d9adf0b3daf0862e04fbc`.
The candidate stores the extended-range centered value while computing row
variance, then reuses it for affine forward instead of recomputing it. It
changes no public ABI, CPU path, default dispatch, or backward shader.

Three balanced pairs ran baseline/candidate, candidate/baseline, then
baseline/candidate. Each report has six shapes, three warmups and nine
measured intervals per route and stage, with 32 training steps per route
interval. Each cell below is the median of the three run medians in
milliseconds. These host-observed times include queue and allocation costs,
not just device kernel execution.

| Shape | Forward baseline | Candidate | All-VJP baseline | Candidate | Affine-only baseline | Candidate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2x3 | 0.670 | 0.620 | 28.450 | 28.156 | 26.377 | 26.182 |
| 8x257 | 0.614 | 0.617 | 43.063 | 43.353 | 26.029 | 25.965 |
| 32x256 | 0.706 | 0.706 | 51.083 | 51.046 | 30.839 | 30.435 |
| 64x768 | 2.172 | 1.913 | 193.011 | 191.034 | 101.635 | 99.612 |
| 128x1025 | 5.175 | 5.191 | 490.499 | 482.778 | 230.284 | 223.627 |
| 256x256 | 3.267 | 3.269 | 269.298 | 267.233 | 132.547 | 130.554 |

At 64x768, the forward probe fell about 12%, while the 32-step all-VJP
route fell about 1% and affine-only about 2%. The 128x1025 and 256x256
32-step routes were shorter in all three pairs, but their separate forward
probe medians were flat; do not infer a device-only speedup from those totals.
Smaller shapes showed no material end-to-end change. Stage probes each include
one terminal snapshot and are not additive components of the training route.
The preloaded training routes include one terminal readback. Affine-only
omits dx and is not work-matched to all-VJP.

All six primary native reports passed the LayerNorm Rust/PyTorch numerical
gate: matched f32 input/target digests, decreasing loss, 32-step outputs, and
strict scaled tolerance. Serialized final outputs were identical across both
sources and all three runs at every shape. The six `validated-fresh-*.json`
files publish every route/stage median, validation result, and SHA-256 of its
retained raw report. The three preexisting PyTorch CPU/MPS controls were
reused for **numerical validation only**, not a new speed comparison.

An earlier three-pair exploratory comparison used the prior baseline binary
from source `10fe5b0ae4a6464684f5e3fd9dabc012327d1442`, SHA-256
`6c55c0c0034335e649c08233006ed2bcaa2a82b4dabfa893c83dd074cfc5730b`.
Native crate sources and `Cargo.lock` were unchanged between that commit and
the fresh baseline, but the old pairs showed large shared-host variation at
128x1025, so they are not used for the main table. Their six validation
records (`validated-baseline-run*.json` and `validated-candidate-run*.json`)
are also published rather than hidden.

The same candidate source was compiled for `wasm32-unknown-unknown` and bound
with `wasm-bindgen-cli 0.2.104`. Chrome 153 on a non-fallback Apple WebGPU
adapter passed the `layer-norm-resident` fixture: seven cases, 400 learning
steps, statistics-only VJP, guard checks, and batched readback, with zero page
or console errors. Its values were identical to the earlier published browser
fixture. `browser-candidate.json` includes browser/page/WASM hashes and the
separate adapter probe. This is a **browser correctness** check, not a browser
performance or same-device attestation.

Verify published files in this directory with `shasum -a 256 -c SHA256SUMS`.
The raw native reports, both release binaries, and generated browser assets
remain under `~/Library/Logs/SpiralTorch/layernorm-forward-tape-reuse-20260924/`.
Replay the two source commits in separate clean checkouts. Build the release
example with the command in `benchmarks/layer-norm-training/README.md` (use
`cargo build` instead of `cargo run`), hash each binary, and run them
sequentially in the balanced order above with `SPIRALTORCH_STRICT_GPU=1`.
Validate each raw report with `benchmarks/layer-norm-training/compare.py`
against the matching fixed-input PyTorch control, requiring the default
`sequential` update mode. Do not run GPU routes concurrently. These
observations do not establish a universal GPU or NN training advantage.
