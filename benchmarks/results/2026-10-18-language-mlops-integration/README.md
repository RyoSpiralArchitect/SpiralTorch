# Integrated Language MLOps Candidate

This qualifies the combined source candidate after incorporating the repaired
AdamW parent and the reviewed Python selection boundary. It does not publish
PyPI/npm packages, change optimizer/geometry policies, loosen numerical limits,
or establish a language-quality gain.

Production source: `8a4b003704f0f308b1e5147b4ba87bb16d56373b`.
The [manifest](results/manifest.json) records source, wheel, WASM and log hashes.
Generated receipts are consistency evidence, not independent attestation.

The subsequent [CI follow-up](CI_FOLLOW_UP.md) repairs native-only example
compilation on WASM and adds public synthetic transport checks to CI. It does
not replace the original lifecycle qualification below.

## Results

- **Installed Python:** rebuilt release-equivalent WGPU wheel, nine AdamW cases,
  seeds 11/29/47, ordinary/learned/frozen geometry, 12 updates. SIGKILL after the
  step-5 seal and fresh-process resume reproduce the complete report and
  checkpoint byte-for-byte. All nine restored inference/generation identities
  match uninterrupted execution. Source files remain unchanged. See
  [recovery](results/python-recovery.json).
- **Saved-model selection:** independently run the same request with retained
  steps 0/4/8/12. Final output remains byte-identical to uninterrupted execution.
  Rust selects among these real saved models for every case; selected artifacts
  equal direct extraction from their source checkpoints and survive opaque
  import unchanged. All nine select step 12 in this fixture. This is development
  selection, not an early-stopping or quality win. See
  [selection](results/python-selection.json).
- **Packaged browser:** rebuilt WebGPU package passes 101 ownership, boundary,
  inference, admission, rejection and rollback checks. Fresh-page 5-to-12 resume
  reproduces the full report/checkpoint exactly. Both rebuilt `nn` and `webgpu`
  packages pass CPU Node transport/feature checks, 20 and 30 respectively. See
  [browser](results/browser.json), [resume](results/browser-resume.json), and
  [verification](results/browser-verification.json).
- **Unchanged short Torch gate:** all nine cases pass for native and browser.
  Maximum parameter absolute error is 2.0447056e-6 native and 2.5760787e-6 browser,
  with `atol=3e-6`, `rtol=5e-5`, geometry-delta and optimizer-moment relative L2
  limits 0.002. Full metrics remain in [native](results/python-torch.json) and
  [browser](results/browser-torch.json) comparisons.

Rust 1.99.0 release `resident::byte_study::` tests: **81 passed**, with
`SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1` and serialized execution. Installed Python
portable study tests: **11 passed, 5 feature/runtime skips**; source-only store
tests: **14 passed**. Publisher guards: **3 passed**. Local Markdown targets,
Python/stub syntax and scoped rustfmt/diff checks passed. Vendor WGPU warnings
and the existing `block` future-incompatibility warning remain unsuppressed.

Native execution reports Apple M4 / Metal / IntegratedGpu. The browser reports
BrowserWebGpu / Other, so physical browser hardware remains **unverified**.
These are not speed measurements. Both geometric arms still have worse final
validation BPB than the ordinary arm at all three seeds in the short request.
The earlier [1024-update Key-bias numerical failure](../2026-10-18-wide-rounding/README.md)
is unchanged and is not superseded by this qualification.

## Review Follow-Up

Two independent, read-only reviewers covered Rust optimizer/selection plus WASM
lifecycle, and Python durable storage plus binding ingress. The first found no
actionable bug. The second found that PyO3's eager `Vec` extraction could reserve
a Python sequence's claimed length before the Rust 32-candidate bound.

The installed pre-fix wheel reproduced `PanicException: capacity overflow` for
`select_checkpoint(range(sys.maxsize), ...)`. The fixed binding takes at most
33 typed candidates without trusting a length hint, then invokes Rust's existing
validator. Regression cases cover huge declared lengths, infinite iterators,
exactly 33 consumed elements, normal iterator parity and the original trigger,
which now raises `TypeError`. Independent re-review found no further issue;
the rebuilt installed-wheel suite and lifecycle qualification above ran after
the fix. The public stub now describes `Iterable` candidates.

The reviewers did not run GPU tests and their code review is not runtime
evidence. Backend numeric fixes retain their separate isolation/review records.
Remote CI and merge status are separate from these local results; check the
actual PR head rather than treating this receipt as a merge authorization.

## Reproduce

Use the [build guide](../../../docs/python/build-and-validation.md) to build and
install a private source wheel. The wheel metadata still reads 0.4.31.
`REQUEST` and `TORCH_REFERENCE` are the unchanged private companions of the
[original optimizer study](../2026-10-15-byte-study-optimizers/README.md).
Request SHA256: `208975bab0905a1ee968d5940975259491163680ce7bd2c0132bce23a637e523`.
Torch reference SHA256: `cec0bde4d5a6f8f57e5a9a5d5d3960e692f4fdf6c6c15672a5f57851d703385c`.
No new data or thresholds are selected here.

```sh
VENV/bin/python -I -B tools/qualify_byte_corpus_runs.py run \
  "$REQUEST" "$PRIVATE/python-recovery-reviewed" --stop 5 --wheel "$WHEEL"
VENV/bin/python -I -B benchmarks/results/2026-10-18-language-mlops-integration/qualify_selection.py \
  "$REQUEST" "$PRIVATE/python-selection-reviewed" \
  "$PRIVATE/python-recovery-reviewed/continuous" > "$PRIVATE/python-selection-reviewed.json"
python3 -I -B tools/byte_corpus_study.py compare "$REQUEST" "$TORCH_REFERENCE" \
  "$PRIVATE/python-recovery-reviewed/resumed/report-12.json" "$PRIVATE/python-torch-reviewed.json"
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo +1.99.0 test --locked --release \
  -p st-nn --no-default-features --features wgpu --lib resident::byte_study:: -- --test-threads=1
```

For the browser, repeat the [package qualification](../2026-10-18-packaged-language-mlops/README.md)
against a freshly built package, reusing all four pinned inputs. Run the main
page and resume on a fresh page, then verify before publication. Run GPU paths
serially. The small corpus, weights, generated bytes, private runs, wheels and
full logs remain local; only reviewed results and hashes are retained here.

`publish.py PRIVATE NEW_RESULTS` expects the exact paths listed in its `FILES`
map, validates the qualified wheel's helper/stub against source, rejects private
payload keys, and writes exclusively. It never uploads or promotes a model.
The [language MLOps guide](../../../docs/language_mlops.md) is the public entry
point for the actual APIs and their boundaries.
