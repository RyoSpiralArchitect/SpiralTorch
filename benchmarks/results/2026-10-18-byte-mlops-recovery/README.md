# Installed Python Byte-LM Recovery

The source-candidate Python package now exposes a synchronous
[durable segment primitive](../../../docs/python/byte-study-runs.md), rather
than keeping checkpoint execution in a standalone repository script. Rust owns
the study/checkpoint semantics; Python stores exact bytes and explicit ancestry.

## Actual Execution

On an Apple M4 / Metal integrated GPU, an isolated source wheel ran nine models:
seeds **11, 29, 47**, each with ordinary attention, learned geometric attention,
and frozen geometric attention. The v9 request uses AdamW, B=2, T=8, width=8,
hidden=16, two heads, one ordinary transformer block, geometry width=4 and
curvature=-0.75. This is a small native byte-LM fixture, not HF-scale FT.

1. Run 12 updates continuously through the installed `spiral-byte-corpus-run`.
2. Run a fresh process to update 5; inject **SIGKILL** immediately after the
   segment seal is durably published, before the Python loop records completion.
3. Load that exact seal and run updates 5 -> 12 through the installed CLI in
   another process/new directory, pinning the selected receipt hash.
4. Compare full report and checkpoint bytes with the continuous run. Both are
   **exactly equal**, including every case, scalar history and AdamW state.
5. Restore all nine model artifacts from both outputs. Artifact identities,
   full forward-logit hashes and four-byte greedy generation hashes agree for
   every case. Each generated first byte matches the measured next-byte argmax.

The interrupted process exited `-9` and left neither `failed.json` nor
`completed.json`. Its files remained unchanged by recovery; child ancestry
matched the selected source claim, seal and checkpoint. The source wheel,
installed extension and installed Python helper were byte-checked against one
another before execution. See [qualification.json](qualification.json).

The [same retained PyTorch reference](../2026-10-18-packaged-language-mlops/README.md)
also passed the existing numerical gate without relaxation:

- Maximum final-parameter absolute error: **2.044705553316817e-6**.
- Existing tolerances: `atol=3e-6`, `rtol=5e-5`; geometric-delta and optimizer-
  moment relative L2 limits both `0.002`.
- Frozen geometry bits, full parameter comparison, losses and AdamW-state
  comparisons passed. See [torch-comparison.json](torch-comparison.json).

**No geometric quality win is established.** Both geometric arms have slightly
worse final validation BPB than the ordinary arm for all three seeds in this
12-update fixture. Learned geometry is slightly better than frozen for seeds
11/29 and slightly worse for 47. The older 1024-update numerical failure remains
unresolved; this short qualification does not supersede it.

## Validation And Limits

The store has 14 source-only tests covering private modes, exclusive concurrent
claims, immutable recovery, bad cursors/requests/pins, publication failures at
six boundaries on both sides of the write, atomic no-replace visibility,
corruption, symlinks, file bounds/FIFO rejection, Rust-validator delegation and
failure-record errors. Two compatibility-wrapper tests pass. The rebuilt
native study suite passes 11 tests with five feature/opt-in skips. The new API
exports and console entry point were exercised from the installed wheel.
Three qualifier metadata/gate regressions, three native export tests and four
native serving-contract tests also pass; four serving GPU opt-ins were skipped.
Actual native inference/generation is covered by the qualification above.

This tests **process interruption**, not power loss, distributed execution,
Windows/network storage, authenticating another writer, generation quality, or
large-model recovery. Digests provide integrity/provenance references, not
training attestation or promotion. The storage protocol requires trusted local
POSIX parents, hard links and file/directory fsync.

Source baseline: `ae64915e9537dc70bba9c0c7469fe621ebd1f974`, plus this commit's
Python execution primitive/tests/docs. Rust source is unchanged by this patch.
The wheel was rebuilt with Rust 1.99.0, maturin `--release --locked --features
logic,kdsl`, default WGPU route, and macOS deployment target 11.0. Its metadata
still says `0.4.31`: this is a **private source candidate**, not a PyPI upload.
Exact wheel/extension/helper/qualifier hashes are in `qualification.json`.

Request SHA-256:
`208975bab0905a1ee968d5940975259491163680ce7bd2c0132bce23a637e523`.
Retained PyTorch-reference SHA-256:
`cec0bde4d5a6f8f57e5a9a5d5d3960e692f4fdf6c6c15672a5f57851d703385c`.
Comparator `tools/byte_corpus_study.py` SHA-256:
`3f55ab2b2f901e0f6f095374035487eab2dc0213165bc6fdb46c1c9df3e2bfaa`.

## Reproduce

Keep requests, weights and logs private. Build/install the source-candidate
wheel into an isolated environment using the
[build guide](../../../docs/python/build-and-validation.md), then run:

```bash
SOURCE_VENV/bin/python -I -B tools/qualify_byte_corpus_runs.py run \
  PRIVATE_REQUEST.json NEW_PRIVATE_OUTPUT --stop 5 --wheel SOURCE_WHEEL.whl
python3 -I -B tools/byte_corpus_study.py compare \
  PRIVATE_REQUEST.json RETAINED_TORCH_REFERENCE.json \
  NEW_PRIVATE_OUTPUT/resumed/report-12.json NEW_COMPARISON.json
python3 -I -S -B bindings/st-py/tests/test_byte_corpus_runs.py
python3 -I -S -B tools/test_run_byte_corpus_segments.py
python3 -I -S -B tools/test_qualify_byte_corpus_runs.py
```

The exact request/reference, wheel, build log, source and recovered weights,
raw generation contexts and process logs remain in private local validation
storage. Only selected results, verification records, hashes and instructions
are published here. The qualifier does not independently establish provenance
of the original corpus/reference.
