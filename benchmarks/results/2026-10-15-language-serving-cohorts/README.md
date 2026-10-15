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

## Observed Results

All 12 native-WGPU replay conditions passed the fixed permissive gate; all
activated output hashes matched their measured candidates and all rollbacks
were bit-exact. Both source checkpoint digests and the source request stayed
unchanged. The post-review replay reproduced the earlier receipt exactly.

Ranges below cover both checkpoint origins, three seeds and train/frozen query
controls. CE is natural-log next-byte cross entropy; negative delta favors the
candidate. These are descriptive ranges, not confidence intervals.

| Cohort | Mean CE delta | Worst paired batch CE delta |
|---|---:|---:|
| autograd-prose | -0.00333104 to -0.00232825 | +0.00093365 to +0.00197864 |
| geometry-prose | -0.00047028 to -0.00007579 | +0.00314522 to +0.00748587 |
| rust-code | -0.01210749 to -0.00105086 | +0.00318789 to +0.00696993 |

Every cohort mean improved slightly, yet every condition also contained a
regressing batch in each cohort. The permissive +2.0 regression limits knowingly
allow this; the experiment does **not** establish a production-safe threshold.
This is why mean and worst-batch checks remain separate.

Candidate centered RMS ranged from about 0.2444 to 0.4118 across all cohort,
block and head observations. Minimum normalized router entropy exceeded
0.9999999848, essentially uniform. The largest train-vs-frozen query mean-CE
difference was only 8.94e-8. These results do not establish a query-router
learning benefit, routing specialization, external generalization or generation
quality. The useful operational result is separate, source-bound observations
and admission decisions for each declared group.

## Validation And Limits

The [full paired receipt](qualification.json), [actual BrowserWebGpu receipt](browser-receipt.json)
and [source/binary/test record](validation.json) retain all conditions. Local
validation passed 68 test executions across eight suites, with eight explicit
feature skips, including CPU-only and WGPU installed wheels. These counts are
not independent samples. Installed native binaries and type stubs matched their
wheels; stubs also matched source. Browser v2 cohorts exactly matched isolated
v1 measurements, and all 16,384 logits returned to their original hash on rollback.

Independent read-only review fixed typed-deserialization memory amplification,
optimized-Python evidence checks, independent geometry RMS coverage, isolated
cohort-count limits, legacy v1 positional input and batch/row wording. The first
browser attempt failed in its expected-JSON key-order comparison; that verifier
was corrected and the browser rerun passed. The failed attempt is retained.

An additional strict `st-nn --lib -D warnings` probe under local Clippy 0.1.97
failed on 23 pre-existing lints in source files verified unchanged from the base;
there were no serving-file diagnostics. `st-nn` is outside CI's strict subset.
The dependency-inclusive probe also encountered seven existing unknown lint
names. These diagnostics are recorded as **failed**, not counted as test passes
or silently suppressed. No claim of workspace-wide lint cleanliness is made.
