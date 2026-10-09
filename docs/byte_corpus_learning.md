# Paired byte-corpus learning

The [resident byte decoder](resident_byte_decoder.md) can train on document
windows, not only frozen synthetic examples. `resident_byte_learning` and
`resident_byte_learning_browser` use the **same Rust runner and model**.
Python prepares explicit data/initial weights and provides an independent
CPU-f32 PyTorch comparison; it never implements SpiralTorch's execution route.
The runner is the public Rust `st_nn::resident::ByteCorpusStudy` API (feature
`wgpu`). It is a bounded paired-study protocol, not a general training scheduler
or a tokenizer API. Its request-bound checkpoints include complete model values,
the common data cursor and scalar histories for all cases.

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

## Checkpoint And Resume

Pause at an absolute update count **per case**, even between scheduled
evaluations, then resume in a new process with the exact original request:

```sh
target/release/examples/resident_byte_learning target/byte-corpus-pilot/request.json \
  --stop-after 37 --checkpoint-out target/byte-corpus-pilot/checkpoint-37.json \
  > target/byte-corpus-pilot/partial-37.json
target/release/examples/resident_byte_learning target/byte-corpus-pilot/request.json \
  --resume target/byte-corpus-pilot/checkpoint-37.json \
  --checkpoint-out target/byte-corpus-pilot/checkpoint-128.json \
  > target/byte-corpus-pilot/resumed.json
```

Output checkpoint paths must be new. The CLI writes a same-directory temporary
file, syncs its contents, then persists without replacing an existing file.
This is not a full power-loss durability guarantee for the directory entry.
Stop/resume flags require `--checkpoint-out`; unknown or duplicate flags fail.
Input and checkpoint reads reject non-regular files and are bounded to 64 MiB.
Resumable segments additionally preflight a conservative worst-case checkpoint
size through the **final** update, including scalar histories and growth of
float text. `checkpoint_size_bound()` exposes that estimate. A request that
fits the input limit may still be too large for resumable execution; it fails
before requesting a GPU, rather than after learning. The original uninterrupted
`run` path retains its existing request limits.

Rust clients use `ByteCorpusStudy::from_json`, `checkpoint_from_json`,
`validate_segment`, and `advance(runtime, checkpoint.as_ref(), stop_after)`.
These CPU preflight methods require no live device; the module is WGPU-gated.
`run(runtime)` retains the original uninterrupted result format without taking
a checkpoint. `advance` returns `ByteCorpusStudySegment { report, checkpoint }`;
`checkpoint.to_json()` is portable model/state data, not live GPU buffers.

The checkpoint schema is `spiraltorch.byte_corpus.checkpoint.v1`. Its SHA-256
binds the **exact request bytes**, including whitespace, initial weights, case
order, seeds, documents, explicit batch selections, fixed SGD rate, geometry and
evaluation schedule. Keep that request unchanged alongside the checkpoint.
There is no omitted RNG cursor: all selections are already in the request, and
the model uses stateless SGD. The model checkpoint also fixes window-local
position/geometry reset semantics. An exact hash detects accidental mixing;
it is not a signature or proof that the recorded history was computed honestly.

All cases must have the same cursor, continuous training histories, exactly the
scheduled evaluation history, finite losses, matching model topology and matching
attempted revisions. Every update in a segment must be accepted before the new
checkpoint is returned. Failure or cancellation drops that segment's local
models and leaves the caller's previous checkpoint unchanged. Partial reports
use `spiraltorch.byte_corpus.partial.v1`, never the completed-result schema.
A pause at 37 preserves evaluations at 0/64/128: it does **not** introduce an
extra validation at 37. A completed checkpoint can be loaded without retraining.

The browser export `advance_resident_byte_learning(input, checkpoint, stop)`
calls the same Rust API. Open `byte_learning_resume_browser.html?stop=37` under
the same local server, download its checkpoint and report, then close the page.
Put that checkpoint at `target/resident-byte-learning-web/checkpoint.json` and
open a **fresh page** at `byte_learning_resume_browser.html?resume=1`. It resumes
to the request's total unless a `stop` query parameter is supplied. Checkpoint
JSON is passed as an opaque string, preserving model float32 values and signed
zero; the page does not parse/reconstruct model weights. Both report downloads
also preserve the original Rust JSON string; a separately parsed copy is only
used for UI validation/display. JavaScript `JSON.stringify` would erase signed
zero and is therefore not used to produce the downloaded report. The segment
export returns `report_json` and `checkpoint_json` strings. Neither browser page
stores data remotely or implements a second learning loop.

Compare resumed and uninterrupted trajectories/final weights within each
runtime, then run the same independent Torch gate. Native and browser remain
separate evidence scopes; this contract does not promise bitwise agreement
across hardware, kernel options or driver versions.

```sh
python3 -I -S -B tools/verify_byte_corpus_resume.py \
  target/byte-corpus-pilot/request.json target/byte-corpus-pilot/native.json \
  target/byte-corpus-pilot/partial-37.json target/byte-corpus-pilot/resumed.json \
  target/byte-corpus-pilot/checkpoint-37.json target/byte-corpus-pilot/resume-check.json
```

The verifier requires type-aware, signed-zero-sensitive equality of the entire
resumed/uninterrupted report and checks paused prefixes, stored float32 histories,
and every saved tensor's shape and float32 bits against the paused readback.
Full graph/operation validity remains the Rust preflight's responsibility.
It writes a new
small hash/result record, not raw weights. Use the corresponding browser files
for a separate browser record. Retain raw reports and checkpoints locally.

## Acceptance And Limits

Every training CE, held-out batch CE and final parameter is compared against
independent PyTorch with the predeclared `3e-6 + 5e-5 * abs(reference)` gate.
Every geometry parameter must actually change; the final change-vector relative
L2 error must be at most 0.002, with reference norm greater than 1e-8. Merely
passing a scalar loss check cannot promote a disconnected small derivative.
The change vector starts from the effective float32 initial weights, not their
unrounded JSON spelling. Decimal-to-float32 conversion alone is not learning.
The independent reference explicitly constructs float32 parameter tensors,
including values written as JSON integers. The
[float32 review record](../benchmarks/results/2026-10-09-byte-corpus-f32-review/README.md)
preserves the failing controls and unchanged frozen-pilot comparisons.
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
controls, generation evaluation and general streamed/long-running training
remain separate work. The resumable fixed study does not supply an Adam state,
changing learning-rate schedule, continuous recurrent context or live sampler.
