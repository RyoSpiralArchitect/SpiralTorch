# Byte-corpus checkpoint handoff between clients

This record extends the [within-runtime resume checks](../2026-10-09-byte-corpus-resume/README.md)
to two actual execution-client handoffs on the same Apple M4 host:

- Native Metal checkpoint at update 37, then a fresh browser WebGPU page to 128.
- Browser WebGPU checkpoint at update 37, then a fresh native Metal process to 128.

Each direction resumes all six fixed cases: ordinary/causal-geometry pairs for
seeds 11, 23 and 37. The original request, reference and source checkpoints are
unchanged. Existing validated native/WASM binaries were reused; no rebuild,
new initialization, data selection, optimizer or tolerance change was needed.

## Results

Both directions pass the independent CPU-f32 Torch comparison for every loss,
held-out evaluation and final weight. All geometry tensors retain qualified
nonzero learning deltas. The comparator includes the separately reviewed
[float32-baseline repair](../2026-10-09-byte-corpus-f32-review/README.md).

| Direction | Retained history | Saved tensor bits vs readback | Maximum final-weight error vs Torch |
| --- | --- | --- | ---: |
| Native to browser | Exact through 37 | Exact at 37 and 128 | 2.384185791015625e-7 |
| Browser to native | Exact through 37 | Exact at 37 and 128 | 2.384185791015625e-7 |

Evaluation revisions remain 0/64/128, with no extra evaluation at the handoff.
The existing acceptance limits remain `3e-6 + 5e-5 * abs(reference)` and
geometry change-vector relative L2 at most 0.002 with reference norm above
`1e-8`. The starting point for those deltas is the effective float32 initial
parameter value, not its original decimal JSON spelling.

Neither handoff report is bitwise equal to its target client's uninterrupted
report. The largest final-parameter difference is 2.384185791015625e-7 in both
directions. This diagnostic does not introduce another acceptance threshold:
the unchanged independent Torch criteria remain the gate. Cross-client
numerical agreement is not cross-device bitwise determinism.

## Reproduction

Use the original request and update-37 checkpoints from the linked resume
record. Do not reconstruct their weights or rewrite the request JSON.

For browser-to-native, pass the browser checkpoint to the native example's
`--resume` option, together with a new `--checkpoint-out` path. Preserve stdout
as `browser-to-native.json` and the new checkpoint as
`browser-to-native-checkpoint-128.json`.

For native-to-browser, copy the native checkpoint to
`target/resident-byte-learning-web/checkpoint.json`, retain the original
`request.json`, serve only on localhost, and open a fresh
`crates/st-nn/tests/byte_learning_resume_browser.html?resume=1` page. Download
both opaque Rust JSON artifacts as `native-to-browser.json` and
`native-to-browser-checkpoint-128.json`. Do not reserialize them in JavaScript.

Run the normal `tools/byte_corpus_study.py compare` command for each output.
Then run this record's [verify.py](verify.py):

```sh
python3 -I -S -B benchmarks/results/2026-10-09-byte-corpus-cross-client-resume/verify.py \
  /path/to/frozen/request.json /path/to/torch-reference.json \
  /path/to/within-runtime-resume-artifacts /path/to/cross-client-artifacts \
  /new/path/to/verification.json
```

The script verifies source hashes against the earlier immutable inventory,
checks checkpoint tensor bits and histories at both endpoints, requires exact
retained prefixes, and invokes the same Torch comparator. Eight in-memory
negative controls must reject altered prefixes, weights, cursors and request
identity. It never changes the saved inputs. Full model validity is still
owned by Rust preflight, not a second Python model implementation.

[verification.json](verification.json) contains the results, all numerical
comparison summaries and raw input hashes. [validation.json](validation.json)
records source/binary identities and the execution procedure. Full raw weights,
checkpoints, outputs and logs remain local; historical records are unchanged.

## Limits

This is one small recipe on one host, not cross-hardware qualification, a
bitwise reproducibility promise, live-state migration, runtime attestation,
pretrained-model FT, a speedup or a geometry-quality win. Window-local geometry
state resets and stateless SGD semantics are unchanged. Source adapter strings
are recorded metadata, not cryptographic execution proof. No cache, model,
experiment record or user-deleted archive was removed for this experiment.
