# Paired byte-corpus learning

The [resident byte decoder](resident_byte_decoder.md) can train on document
windows, not only frozen synthetic examples. `resident_byte_learning` and
`resident_byte_learning_browser` use the **same Rust runner and model**.
Python prepares explicit data/initial weights and provides an independent
CPU-f32 PyTorch comparison; it never implements SpiralTorch's execution route.
This is a bounded pilot runner, not a new public checkpoint or tokenizer API.

The [2026-10-09 pilot record](../benchmarks/results/2026-10-09-byte-corpus-matched-learning/README.md)
publishes every arm's scalar trajectory, the fixed acceptance criteria and
paired seed differences. Raw initial/final weights remain local with hashes.
The [pair-cache replay](../benchmarks/results/2026-10-09-byte-corpus-pair-cache-replay/README.md)
reuses those exact frozen inputs after the metric optimization; both native and
browser raw outputs remain byte-identical to their own original pilot outputs.

## What is held constant

Every seed has an ordinary decoder and a causal-geometry decoder. The ordinary
parameters must match bit for bit before either model is compiled. Geometry
adds its learned projection, wave decay/phase and per-block/head gains under
the existing single model owner. Both arms receive exactly the same ordered
batches, learning rate, update count and held-out windows. No new optimizer,
activation readback, Python gradient injection or quality-dependent gate is
introduced.

Train and validation documents are separate. Windows contain `T+1` bytes;
inputs and targets are shifted within that one document. There is no implicit
concatenation, EOS, normalization, padding or subword vocabulary. Candidate
windows have nonoverlapping target spans. Training shuffles them with a fixed
data seed and cycles only after exhausting an epoch. Validation selects evenly
spaced distinct windows before running either model. Exact duplicate documents
across splits are rejected; near-duplicate prose is not deduplicated.

The initial pilot uses repository prose pinned to an existing commit, with
three training documents and two different validation documents. This is not
a large language benchmark, a pretrained-model FT run or a test on independent
domains. Different files can share terminology, code and boilerplate.

## Prepare And Run

Run from the repository root with an installed CPU PyTorch for the reference.
Choose a **new** output directory; creation refuses to overwrite frozen inputs.

```sh
python3 -I -S -B tools/byte_corpus_study.py prepare target/byte-corpus-pilot \
  --revision 01e12ac8cc9bb10a1b151a8d7b1fda6b157fea64 \
  --train-doc docs/reference/native-api.md \
  --train-doc docs/reference/learning.md \
  --train-doc docs/geometric_learning_bridge.md \
  --validation-doc docs/reference/geometry.md \
  --validation-doc docs/spiraltorch_manifesto.md \
  --seed 11 --seed 23 --seed 37

python3 -I -B tools/byte_corpus_study.py reference \
  target/byte-corpus-pilot/request.json target/byte-corpus-pilot/torch.json

cargo build --locked --release -p st-nn --no-default-features --features wgpu \
  --example resident_byte_learning
target/release/examples/resident_byte_learning target/byte-corpus-pilot/request.json \
  > target/byte-corpus-pilot/native.json

python3 -I -S -B tools/byte_corpus_study.py compare \
  target/byte-corpus-pilot/request.json target/byte-corpus-pilot/torch.json \
  target/byte-corpus-pilot/native.json target/byte-corpus-pilot/native-comparison.json
```

Defaults: batch 2, context 32 bytes, width 16, MLP width 32, two attention heads,
one ordinary transformer block, geometry dimension 4, curvature -0.75, SGD 0.05,
128 updates and validation at revisions 0/64/128. Each arm sees 8,192 training
target bytes and 2,048 held-out target bytes. Geometry adds 74 scalars to the
11,216-scalar ordinary model. The comparison is **not parameter- or
compute-matched**, and does not measure speed. Topos is supported by the Rust
request's block flags but is off in this first study, isolating the causal
wave/metric addition rather than changing two mechanisms together.

The preparation JSON records source revision, file hashes, shared-initial-weight
hashes, selection rules, request hash and numerical criteria. Seed labels are
limited to nonnegative, exact browser-JSON integers; the saved learning rate
must remain positive finite float32. Native input reads are bounded to 64 MiB,
including protection against a file growing after metadata inspection.

## Browser Execution

Build `resident_byte_learning_browser` for `wasm32-unknown-unknown` with WGPU,
then run the wasm-bindgen CLI version matching `Cargo.lock` (0.2.129 for this
record). Clear host-only compiler/linker environment overrides for the WASM
build, as in the [byte decoder recipe](resident_byte_decoder.md).

```sh
env -u CARGO_ENCODED_RUSTFLAGS -u CARGO_BUILD_RUSTFLAGS \
  -u CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS -u LIBRARY_PATH \
  -u PKG_CONFIG_PATH RUSTFLAGS= \
  cargo build --locked -p st-nn --target wasm32-unknown-unknown --release \
    --no-default-features --features wgpu --example resident_byte_learning_browser
wasm-bindgen --target web --out-dir target/resident-byte-learning-web \
  target/wasm32-unknown-unknown/release/examples/resident_byte_learning_browser.wasm
cp target/byte-corpus-pilot/request.json target/resident-byte-learning-web/request.json
python3 -I -S -m http.server 8771 --bind 127.0.0.1
```

Open `http://127.0.0.1:8771/crates/st-nn/tests/byte_learning_browser.html`, download
the result, and pass that JSON to the same `compare` command instead of
`native.json`. The page reports **completion**, not a numerical pass: the
independent comparison is still required. No page-side model mathematics is
used. Do not expose a local server containing private inputs to other hosts.

## Acceptance And Limits

Every training CE, held-out batch CE and final parameter is compared against
independent PyTorch with the predeclared `3e-6 + 5e-5 * abs(reference)` gate.
Every geometry parameter must actually change; the final change-vector relative
L2 error must be at most 0.002, with reference norm greater than 1e-8. Merely
passing a scalar loss check cannot promote a disconnected small derivative.
All revisions, parameter shapes and held-out coverage must be present. The
request hash and distinct reference/measured engine identities are mandatory;
these detect accidental record mixups, not cryptographic runtime attestation.

Updates are queued in bounded resident bursts. At checkpoints, the runner reads
only update acceptance plus scalar losses. It reads parameter arrays initially
to verify construction and finally for the independent comparison. Full raw
outputs and weights need not be committed; retain them locally with hashes.

Finite updates and falling held-out CE prove that this tiny model can learn this
sampled task. They do not prove that geometry helps, scales to LLMs, or improves
prose. Report each paired seed difference, including regressions; do not select
a favorable seed, tune on this held-out set, or loosen the numerical gate after
seeing the result. Larger independent corpora, repeated runs, capacity/compute
controls, generation evaluation and complete-model checkpoint/resume remain
separate work.
