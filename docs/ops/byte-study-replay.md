# Hosted byte-study replay diagnosis

Issue [#2289](https://github.com/RyoSpiralArchitect/SpiralTorch/issues/2289) holds
PyPI publication until the historical GPU numerical divergence is understood
and repaired. A successful diagnosis run does not close that issue or qualify
a release.

The manual **Byte Study Replay Diagnosis** workflow has read-only repository
permissions and no release, tag, package-upload, or PyPI job. It builds a private
wheel from one of two immutable revisions:

- `7ad0c100e9b5beddb224182b728e648d376ac8b8`: the prepared-study failure.
- `073254cf5dce4074a930ff29aef6e2e48aa7fce3`: the integrated hardening.

The harness comes from the workflow commit, independently of the native source.
It records both identities, the wheel hash, and verifies the installed extension
against the extension bytes in that wheel. The historical compiler, Python, and
maturin versions are pinned. `RUSTUP_TOOLCHAIN` overrides the source's `stable`
toolchain file; the build checks the effective compiler and logs its full version.
`macos-26` selects the OS family, **not** an immutable
runner image, GPU, or driver. Actual image/runtime identity is retained.

## Fixed protocol

1. In a clean no-Torch environment, run the entire existing
   `bindings/st-py/tests/test_nn_byte_study.py` suite once, in its original order.
2. Only if that phase succeeds, run its prepared-study replay test once more in
   the **same process**. This is a declared warm-process comparison, not a retry.
3. Stop at the first failure. Preserve its ordinary test log and the existing
   failure-only report/checkpoint diagnostics. Do not replace `run_wgpu()` with
   checkpoint capture, add per-step readbacks, relax equality, or retry to green.

The intentional CPU-artifact-only test skip is allowed. Any other skip, zero
tests, or missing completion record prevents a valid result. All foreign ML
imports are blocked before execution. The historical facade's blocked optional
Torch probe during module import is recorded separately; any import attempt
during the actual tests, or any loaded foreign ML module, invalidates the run.
The supervisor owns the worker process group, imposes a two-hour deadline, and
retains a terminal record even on timeout. The enclosing job has a larger
three-hour limit for compilation and artifact retention.

The first numerical outcome is saved before adapter metadata is collected in a
separate `runtime.json` postlude. If metadata stalls, the supervisor retains both
the earlier failure and the later timeout rather than replacing the first outcome.
Adapter identity is collected **after** numerical execution, so the harness does
not insert an earlier device initialization into the tested path. This protocol
does not replay the preceding Rust CI test processes and cannot reproduce an
ephemeral runner's full lifetime. Those are explicit scope limits.

## Evidence

Every run has a fresh output directory. Existing evidence is never overwritten.
The terminal outcome is `no_mismatch_observed`, `test_failure`, `execution_error`,
`timeout`, or `supervision_error`; `release_ready` is always false. Success is
limited to these synthetic cases, not LLM quality or general determinism.

The one-day Actions artifact contains synthetic public-fixture diagnostics,
logs, hashes, and runtime metadata, but not the built wheel, credentials, user
corpora, or model weights from external experiments. Download the first outcome
to private local validation storage before it expires. Publish only a bounded
summary, verification hashes, and reproduction procedure when discussing results.

For a local diagnostic using an isolated environment with the selected wheel
already installed:

```sh
python -I -B tools/run_byte_study_replay.py \
  --source-root /absolute/path/to/pinned-native-source \
  --wheel /absolute/path/to/spiraltorch.whl \
  --output /absolute/private/fresh-replay-directory \
  --timeout-seconds 7200
```

Use the same two-phase protocol for any comparison; do not silently replace the
historical source with a later build. No publication command belongs in this flow.
