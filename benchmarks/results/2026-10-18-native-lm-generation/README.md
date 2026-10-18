# Native learned-geometry generation qualification

Functional source: `6b9779d0be85d915faae896314635e6710ce68d1`.
This is a development candidate after PyPI 0.4.32, not a release announcement.

The native generation session executes the frozen GPT-2 decoder, attached learned
geometry, ordinary or Z-Space token selection, EOS and seeded continuation in
Rust. Python supplies local files/IDs, not a Torch generation loop.

## Fixed real-model exercise

The input is the same pinned local GPT-2 snapshot and native update-20 checkpoint
from [corpus training](../2026-10-18-native-corpus-training/README.md). Only that
exercise's already admitted training chapter is used. Three eight-token prefixes
at offsets 0, 128 and 512 are frozen before generation; no new corpus, development
or held-out material is read.

Two models (base / learned geometry), three prompts and three policies (ordinary
greedy / ordinary sampling / Z-Space sampling) give **18 cases, 144 new tokens**.
Sampling uses one fixed seed, top-k 40 and top-p 0.9. The Z-Space case has explicit
curvature, temperature, repetition and n-gram configuration. There is no KV cache,
padding, context cropping, Torch fallback or claimed throughput comparison.

Results in [`comparison.json`](comparison.json):

- All 18 cases resumed in a separate native process from token 3 with identical remaining step reports and final checkpoint bytes.
- Ordinary paths matched the optional Torch decoder/probability reference over 12 cases / 96 steps. Maximum selected-probability absolute error was `8.820909040530012e-6`; maximum entropy error was `3.293801335324531e-5`, within the sealed gates. Greedy IDs and retained candidate counts matched exactly.
- The Z-Space control changed the top candidate on 8 steps in aggregate. This is a measured control effect, not independent validation of the Z-Space formula or a quality improvement.
- All nine base-versus-learned eight-token continuations were identical at the selected seed/settings. No improved generation text is claimed for this small learned update. Their recorded distributions need not be identical.

The reference uses the same f32 Torch GPT-2 weights/geometry and f64 softmax on
the final logits. Sampled reference probabilities are evaluated on native token
histories; Torch's RNG is **not** compared. Geometry helpers are interoperability
references, not an independent mathematical oracle for every native mechanism.
The existing Rust Z-Space transformation is tested directly at the selection
boundary, not relabeled a Torch ablation.

## Checks and limits

Local checks: 53 Rust tests (6 selection, 7 existing control, 40 causal LM),
55 no-foreign-ML Python tests (including 9 generation/example tests), 45 optional
Torch generation comparisons, four qualification-comparator tests, WASM compile,
Clippy, targeted rustfmt/Ruff and Python-3.8 grammar checks. The actual local
Python interpreter is 3.12, not 3.8. Clippy retains pre-existing crate/vendor
warnings. WASM compilation is not a browser generation execution claim.
Release-workflow/smoke checks added another 27 passes. The final rebuilt
development wheel also replayed all 18 original cases with identical reports
and checkpoints. [`verification.json`](verification.json) records its distinct
binary hash, validation counts and retained raw-log hashes; neither development
wheel was published.

A read-only independent local review found no actionable findings in the Rust
selection/session and Python transport scope. It did not run tests or establish
cross-platform replay. Raw model weights, token histories, checkpoint bytes and
logs remain private; public results retain hashes, all conditions and replay code.

Two test-authoring failures are retained: an exact-equality assertion ignored a
one-ULP normalization difference, and the example test initially forgot to retain
its weight fixture. The tolerance-only assertion and test fixture were corrected;
no production arithmetic was altered to pass a numerical comparison. Real-model
qualification passed its initially sealed gates without tuning.

The API is intentionally bounded: full-prefix f32 CPU GPT-2 only, not arbitrary
HF/LoRA/full-model FT, KV-cached serving, performance parity, general quality uplift
or cryptographic proof of generated history. Training-checkpoint handoff verifies
model/geometry/optimizer ownership but not corpus identity or training completion.

## Replay

Use the pinned model/tokenizer, encoded `data.json` and
`native-full/checkpoint-000020.json` from the corpus exercise. Build the functional
source into a private development wheel and install into an isolated environment.
Never upload a development wheel with the already published `0.4.32` filename.

```bash
python -I benchmarks/results/2026-10-18-native-lm-generation/qualify.py prepare \
  --cohort /new/private/cohort --model-dir /pinned/gpt2 \
  --data /prior/real-corpus/data.json \
  --checkpoint /prior/real-corpus/native-full/checkpoint-000020.json
```

Run `native`, then `resume`, with the same arguments in **separate** no-Torch
processes. Run `reference` in the optional Torch/Transformers environment with the
same native build. Finally run `report` with the same arguments and
`--output /new/private/comparison.json`. Inputs, configurations, gates and source
hashes must remain unchanged after `prepare`. Each output is create-only. Any
failed attempt is retained, not overwritten or silently retried.

The qualification script needs Python 3.11+ for `hashlib.file_digest`; the public
[generation example](../../../bindings/st-py/examples/native_geometry_generation.py)
and native smoke test retain the package's Python-3.8 syntax floor.
