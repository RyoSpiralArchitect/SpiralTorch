# Paired Byte-Corpus Studies From Python

**Requires Python package 0.4.31 or a source build; absent from 0.4.30.**
Check [PyPI release history](https://pypi.org/project/spiraltorch/#history) for
published wheels. Version metadata in this source tree is not proof of upload.
For a source build, use the [Python build guide](build-and-validation.md).
The model-level API introduced in 0.4.30 is described in
[Geometric Byte Language Models](byte-language-models.md).

## One Runner, Three Clients

`spiraltorch.nn.ByteCorpusStudy` calls the same Rust runner used by
`resident_byte_learning` and its standalone WASM qualification example.
Python does not implement a second training loop, geometry rule, checkpoint
validator or optimizer. This is a bounded, paired byte-LM study protocol, not
a general scheduler, HF trainer, streaming decoder or tokenizer API.

Use the [corpus protocol and preparation commands](../byte_corpus_learning.md)
to create a pinned `request.json`. It contains explicit documents, windows,
initial weights, case policies, rates and the evaluation schedule.
Requests and study checkpoints are limited to 64 MiB. Keep their exact JSON
bytes: even whitespace changes the request SHA-256. Accepting a request is not
evidence that its claimed calibration has been measured on the execution device.

```python
from pathlib import Path
import spiraltorch as st

raw = Path("request.json").read_bytes()
study = st.nn.ByteCorpusStudy.from_json(raw)  # CPU-only, no runtime selection.
print(study.request_sha256, study.total_updates)
study.validate_segment()                    # Includes worst-case checkpoint size.
print(study.checkpoint_size_bound())

first = study.advance_wgpu(1)                # Absolute accepted-update cursor.
with open("checkpoint-1.json", "x", encoding="utf-8") as out:
    out.write(first.checkpoint.to_json())

# Later, or in another client using the identical request:
loaded = study.checkpoint_from_json(Path("checkpoint-1.json").read_bytes())
finished = study.advance_wgpu(resume=loaded)  # Default stop: all requested updates.
with open("python-result.json", "x", encoding="utf-8") as out:
    out.write(finished.report_json)
```

The underlying runner rejects software CPU adapters. A CPU-only wheel still
provides construction, request/checkpoint validation, JSON transport and query
preparation; GPU execution and measured preparation raise `NotImplementedError`.
There is no silent fallback. Blocking Rust work releases Python's GIL.

`run_wgpu()` returns a report JSON string without capturing a checkpoint.
`advance_wgpu()` returns an immutable `ByteCorpusStudySegment` with
`report_json` and `checkpoint`. Pause/resume preserves the original evaluation
schedule; it does not insert validation at the pause boundary. Every case must
reach the requested accepted-update cursor before a new checkpoint is returned.
On failure the caller's old checkpoint remains unchanged. These checkpoints
include the common data cursor and complete scalar histories, unlike a bare
`ByteDecoderCheckpoint`. Both use stateless SGD, not Adam/momentum state.

Native and standalone WASM clients can consume these opaque checkpoint bytes
and vice versa. Do not round-trip through a decoded dict to transport them:
floating-point spelling and signed zero must not be silently changed.
Cross-device floating-point equality is not promised.

## From A Study To Inference

**Source-only addition after 0.4.31:** `case_checkpoint` validates the entire
study checkpoint against the exact request, then returns one existing typed
`ByteDecoderCheckpoint`. No GPU is acquired during extraction. Case names are
matched exactly; corruption in an unselected case still rejects the handoff.

```python
selected_name = "my-explicit-case-name"  # An exact name from request.json.
model_checkpoint = study.case_checkpoint(finished.checkpoint, selected_name)
with open("selected-model.json", "x", encoding="utf-8") as out:
    out.write(model_checkpoint.to_json())

# In a fresh process with an installed source candidate:
restored = st.nn.ByteDecoderCheckpoint.from_json(
    Path("selected-model.json").read_text(encoding="utf-8"),
).restore_wgpu()
print(restored.attempted_updates)        # Not reset to zero.
```

The model includes learned geometric coordinates, wave parameters, gains,
mixture/query routers and topology as well as ordinary LM weights. Restore uses
a fresh parameter owner. Partial checkpoints may also be extracted, but this
does **not** select a winner, attest past training or declare readiness for
deployment. Apply explicit held-out quality and operating-policy gates first.

Retain the exact request and full study checkpoint alongside the selected model
and case name: the bare model has no study identity, data cursor, histories or
frozen-rate policy. Resume the paired experiment through `advance_wgpu`, not
through unrestricted model SGD. This is a native byte-LM artifact, not an HF
adapter or an Adam/optimizer-state interchange format.

Rust uses `ByteCorpusStudy::case_checkpoint(&checkpoint, case_name)`. The
standalone WASM qualification module exposes the same validator through
`extract_resident_byte_learning_case(request_json, checkpoint_json, case_name)`,
returning opaque model JSON. This is not yet an export of packaged `st-wasm`.

For source-bound identity and an immutable inference owner, see
[Executable Byte-LM Artifacts](byte-model-artifacts.md). That source-only layer
adds a caller-pinned digest and an explicit no-update/no-external-bias policy;
it does not select or promote a case.

## Geometry Controls

Preparation also lives in Rust. Start from the unchanged five-arm v3 recipe:

```python
source = Path("v3-request.json").read_bytes()
calibration = st.nn.ByteCorpusBiasCalibration.from_json(
    source, [0, 1], fisher_rao_control=True,
)
matched = calibration.prepare_wgpu()  # v5: Poincare/flat/Fisher-Rao controls.
headwise = st.nn.ByteCorpusHeadwiseControls.from_json(
    matched.request_json,
).prepare_wgpu()                      # v6: ordered, heterogeneous two-head arms.
mixtures = st.nn.ByteCorpusMixtureControls.from_json(
    matched.request_json,
).prepare_wgpu()                      # v7: learned/frozen shared routers.
contextual = st.nn.ByteCorpusQueryMixtureControls.from_json(
    mixtures.request_json,
).prepare()                           # v8: CPU-only, zero-start query projections.

prepared_study = contextual.study()   # Same exact Rust-produced request bytes.
```

Choose 1..16 distinct training-window indices before examining results.
`fisher_rao_control=False` prepares v4 flat-only RMS controls. Headwise and
mixture recipes require two heads and a complete v5 source; they are separate
expansions of v5, not consecutive v6-to-v7 conversions. Query preparation does
not fit gains, draw randomness or consume training updates. All paths retain
the existing source arms. See the [calibration](../byte_corpus_calibration.md),
[headwise](../byte_corpus_headwise_controls.md),
[mixture](../byte_corpus_mixture_controls.md) and
[contextual](../byte_corpus_query_controls.md) protocols for exact controls.

`ByteCorpusPreparedBiasStudy` exposes `request_json`, `report_json` and
`study()`. Retain both JSON strings for reproducibility. The report is a
preparation receipt, not proof of a language-quality improvement.

## Compare Without Moving The Gate

Run the independent CPU-f32 PyTorch reference on the identical request, then
use the existing comparator on `python-result.json`, exactly as for the native
and browser reports. It checks the fixed numerical gates, frozen policies and
trajectories; no Python-specific acceptance tolerance is introduced here.
Keep backend numerical agreement separate from held-out quality and speed.
Geometry controls do not have equal parameter counts or compute budgets.

The [installed-wheel qualification](../../benchmarks/results/2026-10-14-python-byte-corpus-study/README.md)
replays the fixed real-corpus pilot and imports historical native/WASM completed
checkpoints. It records numerical compatibility, not a new quality advantage.

Portable and opt-in live-device tests:

```bash
python -I -B bindings/st-py/tests/test_nn_byte_study.py -v
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 python -I -B bindings/st-py/tests/test_nn_byte_study.py -v
```

Run these against an installed candidate wheel, not a source package shadowing
the extension. Tiny synthetic test fixtures exercise mechanics only.
