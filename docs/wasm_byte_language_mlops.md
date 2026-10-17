# Geometric Language MLOps In WASM

Source-build API; this change is not an npm or PyPI release.

The packaged `spiraltorch-wasm` client now connects Rust-owned language studies,
exact resume, immutable model artifacts, inference and measured serving switches.
Geometry is part of the model being operated, not a separate dashboard: artifacts
retain ordinary weights, geometric encoders and learned head/query routers.
Serving gates measure language CE, causal geometry RMS and router mass/entropy
on explicitly named cohorts. Each cohort must pass its own declared limits.

This is a bounded, fixed-window byte-LM stack. It is **not** an HF adapter format,
distributed deployment service, streaming decoder or KV cache. The owner reuses
the resident training forward, including retained tapes. No throughput or model
quality improvement follows from these bindings.

## Build And Import

```sh
RUSTUP_TOOLCHAIN=1.99.0 wasm-pack build bindings/st-wasm --target web \
  --out-dir ../../target/language-package --release -- --locked --features webgpu
```

Import the resulting `spiraltorch_wasm.js` and initialize its WASM module.
`webgpu` includes `nn`. With `nn` alone, study/checkpoint/artifact/gate transport
and saved-checkpoint selection are available without a device; GPU methods and
resident owner classes are absent. The bundled handwritten TypeScript declarations describe the feature
superset; generated declarations describe the actual build. No CPU fallback is
used for the resident path. All GPU clients reuse the package's installed runtime.

## Train And Retain Geometry

Use the same bounded request JSON as [Python](python/byte-corpus-studies.md),
including the [v9 optimizer recipe](byte_corpus_optimizers.md) when appropriate.
Keep exact request bytes, not a JS-parsed/re-serialized reconstruction.

```js
import init, {ByteCorpusStudy} from './spiraltorch_wasm.js';
await init();
const study = ByteCorpusStudy.fromJson(requestJson);
const segment = await study.advanceWebGpu(5); // Absolute accepted-update boundary.
const saved = segment.checkpoint;
const checkpointJson = saved.toJson();       // Persist privately, unchanged.
segment.free();

// A fresh page/process can read requestJson and checkpointJson in the same way.
const restored = study.checkpointFromJson(checkpointJson);
const final = await study.resumeWebGpu(restored, study.totalUpdates);
const finalCheckpoint = final.checkpoint;
const artifact = study.caseArtifact(finalCheckpoint, caseName, 'fixed_window_frozen.v1');
study.validateArtifact(finalCheckpoint, artifact);
const artifactJson = artifact.toJson();
const artifactId = artifact.artifactId;
// Store the trusted selection's ID separately from the downloadable artifact.
artifact.free();
restored.free(); saved.free(); finalCheckpoint.free(); final.free(); study.free();
```

Checkpoint getters return independently owned handles. Resume validation checks
every case and its original request, optimizer history and frozen geometry
policy before acquiring a device. Calling `advanceWebGpu` again starts fresh;
only `resumeWebGpu` continues a checkpoint. Partial artifacts are allowed but not
automatically selected or promoted. Geometry calibration/preparation remains in
Rust/Python and standalone qualification APIs; it is not an additional packaged
method in this slice.

## Select Actual Saved Checkpoints

Retain evaluation-boundary checkpoints before inspecting their losses. The
packaged selector calls the same Rust contract as
[Python/native selection](byte_corpus_checkpoint_selection.md): minimum
development CE, earliest supplied cursor on an exact tie, and no unsaved minimum.

```js
const candidates = study.checkpointCandidates();
for (const json of savedCheckpointJsonStrings) {
  candidates.appendJson(json); // Import the exact stored string, without JS reconstruction.
}
const selected = candidates.selectCheckpoint(caseName, 'fixed_window_frozen.v1');
const decision = JSON.parse(selected.reportJson); // Metrics and identities, not weights.
const selectedArtifact = selected.artifact;
selected.free(); candidates.free();
// selectedArtifact remains valid, with the original geometry and optimizer history.
const inference = await selectedArtifact.restoreWebGpu(); // Optional; requires webgpu.
selectedArtifact.free();
```

The collection owns its study and accepts at most 32 opaque checkpoint strings,
without converting an arbitrary JS array/iterator. Imports validate request and
checkpoint state, and enforce the combined canonical byte budget; selection validates
1..32 increasing evaluation cursors, exact request identity, every case's
history prefixes and a 64 MiB combined canonical JSON budget in Rust. Empty,
foreign, duplicate, reversed or forked sequences reject. A failed append leaves
the collection unchanged. None of these new selection methods borrows two
exported Rust handles in one WASM ABI call, so a disposed input cannot poison another owner's generated
borrow. Selection does not request an adapter, even in a
`webgpu` build, and is not early stopping or serving admission.

Persist original request/checkpoint strings privately. The returned artifact is
byte-identical to direct extraction of the winning saved model. Each artifact
getter returns an independent handle; release it with `free()` when finished.

## Run And Switch Real Models

```js
import {ByteCorpusModelArtifact, ByteCorpusServingGate} from './spiraltorch_wasm.js';
const initial = ByteCorpusModelArtifact.fromJson(artifactJson, trustedArtifactId);
const gate = ByteCorpusServingGate.fromJson(gateJson);
const slot = await gate.bootstrapWebGpu(initial);
const output = JSON.parse(await slot.forward(windows)); // Uint8Array, B*(T+1) bytes.

const next = ByteCorpusModelArtifact.fromJson(candidateJson, trustedCandidateId);
const permit = await slot.prepareCandidate(next);       // Real same-window measurements.
const assessment = JSON.parse(permit.toJson());
if (permit.accepted) {
  const transition = JSON.parse(slot.activate(permit));
  const generated = JSON.parse(await slot.generateGreedy(prompts, 16, slot.generation));
  // prompts is Uint8Array of exact B*T bytes, no padding/truncation/tokenizer.
  const rollback = JSON.parse(slot.rollback(slot.generation)); // One previous owner.
}
permit.free(); next.free(); slot.free(); gate.free(); initial.free();
```

See [serving gates](python/byte-serving-gates.md) for both v1 and named-cohort v2
schemas and all six required limits. Threshold choice and data provenance remain
the caller's responsibility. Successful permissive mechanics limits are not a
production endorsement. Candidate JSON is evidence only, never an activation
permit. Foreign, stale, already-consumed and rejected permits cannot switch the
live owner. Rollback restores the exact prior owner rather than retraining it.

`artifact.restoreWebGpu()` also returns `ResidentByteInference`, without serving
admission, updates or external score biases. `forward` returns a JSON string
with artifact ID, decimal revision, flat logits and scalar mean CE. A serving
forward additionally carries the decimal generation (`null` for bare inference).
`generateGreedy` uses Rust's full-window byte selection and sliding-window policy;
generated bytes may be non-UTF-8 or private. Prompt hashes are not anonymization.

## Boundary And Lifetime Rules

- Opaque JSON is bounded to 64 MiB in UTF-8 **before the Rust copy**. Only primitive
  strings are accepted; lone UTF-16 surrogates fail rather than changing identity.
- Counts require primitive finite integer numbers. Generation identities require
  canonical decimal `u64` strings, not rounded JS numbers.
- Inputs require genuine, exact-size `Uint8Array` views, including cross-realm
  views. Arrays, coercible objects, other typed arrays, shared and detached
  storage are rejected. Ordinary inputs are snapshotted before returning a promise.
- A pending owner operation holds an exclusive lease. Overlapping work and live
  state getters fail with `operation in progress`; immutable `windowShape` remains
  available. Retry after awaiting the original promise. A busy activation leaves
  the permit unconsumed; a core admission attempt consumes it even if rejected.
- Calling `free()` releases that JS handle; it does **not** cancel work already
  started. Pending promises own their inputs/state. Success and ordinary rejection
  restore the owner lease. A WASM trap/device loss is not a cancellation guarantee.

The [package qualification](../benchmarks/results/2026-10-18-packaged-language-mlops/README.md)
uses actual trained artifacts, AdamW pause/resume, real browser WebGPU and negative
ownership/ingress tests. Integrity, runtime correctness and language-quality
evidence are deliberately separate. CPU transport is not GPU execution evidence.
