# Language MLOps With Executable Geometry

[Project entry](../README.md) | [Documentation index](README.md)

**Source-candidate APIs after 0.4.31, not a new package release.** Start with
the bounded native byte-LM path below. Rust owns model, optimizer, checkpoint,
selection and serving semantics; Python orchestrates local runs, and WASM is
another client of that core. This is not yet a distributed MLOps service.

## What Makes Geometry Operational

A model artifact contains the ordinary weights **and** its learned geometric
encoders and head/query routers, with any retained model optimizer state. The
full study checkpoint also preserves every case, frozen-geometry policy, data
cursor and histories; inference uploads weights, not optimizer moments.
Restoring a geometric model therefore restores computation, not just a report
describing an intervention. Available geometry depends on the declared study;
not every geometric crate is supported by this model family.

Keep these decisions separate:

| Stage | Shared contract | Working entry points |
| --- | --- | --- |
| Train and resume | Exact request, optimizer recipe and complete case state | [Rust study / Python](python/byte-corpus-studies.md), [WASM](wasm_byte_language_mlops.md) |
| Survive process interruption | Explicit immutable segment and parent identity | [Python local store and CLI](python/byte-study-runs.md) |
| Select saved weights | Minimum development CE among actual saved checkpoints | [Rust / Python selector](byte_corpus_checkpoint_selection.md), [WASM](wasm_byte_language_mlops.md#select-actual-saved-checkpoints) |
| Run the selected model | Geometry-bearing artifact and fixed-window inference | [Python artifacts](python/byte-model-artifacts.md), [WASM owners](wasm_byte_language_mlops.md) |
| Admit or roll back | Fresh language/geometry measurements on every named cohort | [Rust / Python gates](python/byte-serving-gates.md), [WASM gates](wasm_byte_language_mlops.md) |

The durable POSIX store is Python orchestration, not an alternative checkpoint
validator. Browser callers own persistence of opaque request/checkpoint JSON;
the packaged WASM selector uses the same Rust rule as Python. A serving
permit is process-local and is never reconstructed from a report after restart.

## Start With One Explicit Run

Build a [WGPU source wheel](python/build-and-validation.md). Pin a request and
choose retained evaluation boundaries before viewing results. For a study with
128 total updates and evaluation every 64 updates:

```python
import spiraltorch as st

st.run_byte_corpus_segments("request.json", "private-run", [0, 64, 128])
saved = [st.load_byte_corpus_segment("private-run", step)
         for step in (0, 64, 128)]
study = saved[0].study
artifact, selection_json = study.select_checkpoint(
    [item.checkpoint for item in saved],
    "my-explicit-case-name",  # Must be a case in the pinned request.
    policy="fixed_window_frozen.v1",
)
```

The parent directory must exist and `private-run` must be new. Loading and
selection need no GPU. Keep request, checkpoints and artifact JSON private and
unchanged; hashes alone are not recoverable weights or proof of their author.
If execution was interrupted after sealing step 64, recover into a different
directory using the selected seal's retained hash, as in the
[recovery recipe](python/byte-study-runs.md#run-and-recover).

Selection uses development observations; it does **not** deploy the model.
Measure a candidate against its incumbent on separately declared
[serving cohorts](python/byte-serving-gates.md#protect-named-evaluation-groups),
then explicitly activate an accepted permit. A cohort failure cannot be hidden
by a better overall average. Retain ordinary and frozen-geometry controls and
assess free-running generation separately from teacher-forced loss.

## Evidence And Boundaries

The [integrated source-candidate qualification](../benchmarks/results/2026-10-18-language-mlops-integration/README.md)
rebuilds both clients and checks real saved-model selection alongside recovery.
The [installed Python recovery qualification](../benchmarks/results/2026-10-18-byte-mlops-recovery/README.md)
compares uninterrupted execution with a killed worker and fresh-process resume.
The [packaged WASM qualification](../benchmarks/results/2026-10-18-packaged-language-mlops/README.md)
covers browser resume, inference, measured admission and rollback. These are
bounded mechanics checks, not evidence of a geometric language-quality gain.
The [longer numerical comparison](../benchmarks/results/2026-10-18-wide-rounding/README.md)
still has unresolved Key-bias differences against its fixed PyTorch reference;
the passing short run does not replace that result.

Existing [HF/PEFT workflows](python/build-and-validation.md) retain their own
adapter lineage and execution APIs. A native byte-LM artifact is not an HF
adapter, and this path does not yet supply KV caching, distributed scheduling,
authenticated model storage or durable serving ownership. The next integration
should connect these clients without recreating geometry or acceptance policy
in Python. Numerical agreement, recovery, serving mechanics, language quality
and speed remain separate gates.
