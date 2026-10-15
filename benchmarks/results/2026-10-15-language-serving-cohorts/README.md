# Named Language/Geometry Serving Cohorts

This qualification exercises the Rust-owned v2 serving gate through installed
Python and standalone WASM clients. It is a bounded model-selection primitive,
not an HF deployment service or evidence that a geometry improves an LLM.

The prior v1 gate merged every evaluation batch into one policy. v2 pins up to
32 named cohorts, each with all six explicit language/geometry/router limits.
Every cohort must pass; no aggregate mean can rescue a failing group. v1 request
and report shapes remain supported with an implicit `default` cohort.
See the [API guide](../../../docs/python/byte-serving-gates.md).

## Evaluation Design

`verify_cohorts.py prepare` freezes data selection and permissive **mechanics**
limits before inference. Source revision:
`01e12ac8cc9bb10a1b151a8d7b1fda6b157fea64`.

| Caller-declared group | New source | Selection |
|---|---|---|
| autograd-prose | `docs/autograd_contract.md` | Outside fenced blocks |
| geometry-prose | `docs/causal_zspace_wave.md` | Outside fenced blocks |
| rust-code | `docs/ecosystem_integration.md` | Inside Rust fences |

Each group has 8 batches of 2 rows, 32 next-byte targets per row: 512 targets,
1,536 total per model measurement. Each selected input **and** target 32-byte
sequence is absent from every byte of all five historical train/validation
documents. Windows do not overlap within their original source file. Near
duplicates, shared boilerplate at shorter lengths and vocabulary/domain overlap
are **not** excluded. These are repository-derived groups, not independent
external domains or languages; the names are not detected model attributes.

Reuse the historical v8 128-update native-trained and browser-trained archives.
At each of seeds 11/23/37, compare the ordinary model with the Poincare/chord
query-router train and frozen controls. This gives 12 paired replay conditions,
not 12 independent seeds. Both archive origins are first measured through native
WGPU. A separate actual browser probe covers the seed-11 browser-origin train
candidate. There are **zero new optimizer updates**, no sweep, adaptive threshold
selection, winning-arm selection or production promotion. Historical models are
tiny fixed-window byte models, not HF models; no generation-quality claim follows.

## Reproduce Privately

Build/install the source candidate into an isolated environment using the
[build guide](../../../docs/python/build-and-validation.md). Published PyPI
0.4.31 does not contain this API. Never upload a private candidate wheel using
that already-published version number.

```bash
# ARCHIVE is the retained 2026-10-13-byte-corpus-query-controls-v8 directory.
# PRIVATE is a private directory outside this checkout; prepared/replay must be new.
python -I -S -B benchmarks/results/2026-10-15-language-serving-cohorts/verify_cohorts.py \
  prepare . "$ARCHIVE" "$PRIVATE/prepared"
python -I -B benchmarks/results/2026-10-15-language-serving-cohorts/verify_cohorts.py \
  run "$ARCHIVE" "$PRIVATE/prepared" "$PRIVATE/replay"
```

The original request and both checkpoint digests are pinned in `selection.json`
before inference and checked again afterward. Exact source files, offsets,
window hashes, gate identity and every selected comparison are recorded. Source
companions are validated by Rust before artifact restoration. The probe checks
measured-vs-live output hashes and exact rollback; failures leave partial output
for inspection rather than silently selecting another candidate or threshold.

Compile the standalone `resident_byte_learning_browser` example and its matching
wasm-bindgen glue using the [browser recipe](../../../docs/byte_corpus_learning.md#browser-execution).
Serve `crates/st-nn/tests/byte_serving_browser.html`, the generated JS/WASM and
private `replay/serving-input.json` on a **loopback-only allowlisted server**.
The input is served at `/target/resident-byte-learning-web/serving-input.json`.
Do not serve the whole private archive or publish input/model bytes. The page
compares every v2 cohort with an isolated v1 gate and tests one-cohort rejection,
typed permit ownership, stale/reused/ABA handles and exact-output rollback.

Public receipts contain measurements, selection metadata, hashes and validation
records only. Raw documents, model payloads, wheels and browser input remain
private. A successful permissive gate is not a production threshold endorsement.
