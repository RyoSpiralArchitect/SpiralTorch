# Resident Loss Feedback: Wiring, Restart And An Open Cross-Runtime Gate

The existing Rust loss-feedback gate now participates in resident classifier
training through the same Rust/Python/WASM trainer. Proposals remain the five
prescribed reports from the earlier control fixture; the gate, unlike those
proposals, responds to actual accepted-step cross-entropy. This is synthetic
wiring evidence, not a real-image quality, stability or performance result.

## Results

Both constant and warmup/cosine cases use the same 24-tensor / 456-value model
and 100 attempts, including 90 accepted and ten rejected updates. Each case
compares uninterrupted execution against a 37/63 restart.

| Check | Constant | Cosine |
| --- | --- | --- |
| Native fresh-process records/checkpoint | Exact | Exact |
| Browser fresh-document records/checkpoint | Exact | Exact |
| Actual rates changed versus ungated proposals | 72 / 100 | 72 / 100 |
| Browser observations replayed through native Rust gate | Exact at all 100 attempts | Exact at all 100 attempts |
| Strict Python-to-browser feedback replay | **Failed** | **Failed** |

The native adapter reports Apple M4 / Metal. The browser reports
BrowserWebGpu / Other with an empty name; its physical GPU model is not inferred.
Rust tests independently replay the core gate and compare every updated
parameter with a plain resident-SGD owner supplied those explicit rates.
The new build also reproduces the prior external-control-only v2 fixture
byte-for-byte and roundtrips/continues earlier v1 checkpoints.

## Retained Failure

Python-to-browser continuation restores the checkpoint exactly but observes
slightly different GPU losses afterward. The first differing feedback state
is at revision 45 for constant rate and 39 for cosine. Maximum loss difference
is `5.960464477539063e-8`: up to two f32 ULPs for constant and one for cosine.
Loss EMA differences persist, so 56 / 62 continuation records differ.

Inputs, actual rate bits, acceptance decisions, discrete gate decisions and
all 456 final parameter values remain identical in these two probes. This is
diagnostic evidence, **not** a substitute for the failed full-state check.
Near a gate threshold, a small observation difference could also change a
decision. No threshold tolerance, observation quantization or fixture
comparison was relaxed. The exact shader operation causing the loss difference
has not been isolated.

Replaying the browser's recorded losses through the native Rust core matches
every feedback state exactly. Thus these observations expose a numerical-input
boundary, not a demonstrated split between Python and WASM control formulas.
Full cross-runtime feedback-state equivalence remains open.

## Source And Reproduction

Implementation source is captured by
`113dc84dcc28959bca3400c9ae7d489c954e8b94`. Fresh native wheel and WebGPU WASM
build receipts, ten raw fixture/phase receipts (including both failures), and
served browser asset hashes are in `summary.json`. Originals and build outputs
are retained locally; only results and validation records are published.

Validation completed with Rust 1.98.0: `st-nn` 739 tests, CPU-only `st-vision`
78 passed, native WGPU `st-vision` 112 passed / one ignored, scoped strict Clippy,
nightly formatting, and release WASM compilation. These are separate test
configurations, not a count of unique tests. The public Python feedback process test passes, as does
native-core replay of the browser observations. Generated and shipped
TypeScript contracts pass. Eight actual browser phases were exercised:
six passed, and both strict cross-runtime probes failed as described above.
`SHA256SUMS` covers this result note and its summary; verify it with
`shasum -a 256 -c SHA256SUMS` from this directory.

Build and run the fixtures using [the contract guide](../../../docs/resident_vision_feedback.md).
For the independent core replay, with the retained browser receipt directory:

```bash
SPIRALTORCH_VISION_FEEDBACK_BROWSER_DIR="$BROWSER_RECEIPTS" \
python -I bindings/st-py/tests/test_vision_trainer_feedback.py \
  Surface.test_browser_observations_replay_exactly_through_native_core -v
```

The loss gate moves an external intervention back toward nominal SGD; it does
not guarantee baseline stability. External proposal-producer latent state,
resident geometric updates, real-image policy benefit, throughput and memory
remain separate unfinished work.
