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
| Train and resume | Exact request, optimizer/clipping recipes and complete case state | [Rust study / Python](python/byte-corpus-studies.md), [WASM](wasm_byte_language_mlops.md) |
| Survive process interruption | Explicit immutable segment and parent identity | [Python local store and CLI](python/byte-study-runs.md) |
| Select saved weights | Minimum development CE among actual saved checkpoints | [Rust / Python selector](byte_corpus_checkpoint_selection.md) |
| Run the selected model | Geometry-bearing artifact and fixed-window inference | [Python artifacts](python/byte-model-artifacts.md), [WASM owners](wasm_byte_language_mlops.md) |
| Admit or roll back | Fresh language/geometry measurements on every named cohort | [Rust / Python gates](python/byte-serving-gates.md), [WASM gates](wasm_byte_language_mlops.md) |

The durable POSIX store is Python orchestration, not an alternative checkpoint
validator. Browser callers own persistence of opaque request/checkpoint JSON;
the packaged WASM API does not yet expose saved-checkpoint selection. A serving
permit is process-local and is never reconstructed from a report after restart.

## Bring Geometry To HF Models

The [portable geometry adapter](python/geometry-artifacts.md) extends the same
Rust-owned-state direction to existing HF language models: learned WaveGate,
Topos, Elliptic and Fractional tensors travel with their recipes, insertion paths
and explicit base-runtime identity. Python reconstructs the working stack for
inference or further learning; WASM validates/transports the same contract.
This is separate from the byte decoder's artifact, not an interchangeable model
format. HF base weights, optimizer, data/RNG and promotion decisions remain
owned by the existing HF experiment lifecycle.

The [HF geometry Trainer](python/hf-geometry-training.md) puts that stack into
the actual HF optimizer/accumulation/clipping/checkpoint path for frozen-base
float32 fine-tuning. Geometric weights are trained and restored, not merely
reported. Rust supplies a shared layout identity; HF retains its own optimizer,
scheduler, data cursor and RNG companions. This bounded integration does not
turn those files into a sealed multi-file artifact or prove a quality gain.

[Document-safe HF inputs](python/hf-geometry-data.md) keep geometric history
inside one document window. Rust plans the boundaries; Python supplies HF rows
and WASM exposes the identical plan. The offline pretrained-model example
connects those rows to geometric FT, checkpoint continuation and reconstructed
inference without introducing a second geometry or training implementation.

The [ordinary channel-affine control](python/channel-affine-control.md) now
uses that identical route. This makes a matched WaveGate ablation possible
without a separate Python-only training or checkpoint implementation. It is a
non-geometric adapter control, not full-model FT or a claimed quality win.

The [fixed book comparison](../benchmarks/results/2026-10-18-hf-book-controls/README.md)
extends that public route to a pinned chapter corpus and four controls across
three seeds. Its initial state, matched-gradient pairs and unused reserve are
qualified before training. Results must be retained for every declared arm;
the preflight itself is not evidence that geometry improves language quality.

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

The [resident clipping slice](resident_byte_decoder.md#resident-active-gradient-clipping)
now adds a Rust-owned stability mechanism to actual geometric byte-LM SGD and
AdamW, with frozen-geometry and exact-continuation tests. The source-only
[v10 study recipe](byte_corpus_optimizers.md#request-bound-clipping) makes that
policy durable for the same Rust/Python/WASM runner. It explicitly preserves
legacy unclipped studies and rejects resume with a changed clipping limit.
This integration does not resolve or relax the longer numerical gate above.
The [client qualification](../benchmarks/results/2026-10-18-byte-study-clipping/README.md)
retains both exact recovery checks and a small-SGD minimum-update failure: when
both runtimes round a geometric update to zero, numerical equality alone does
not qualify it as demonstrated geometric learning.

The [1,024-update comparison](../benchmarks/results/2026-10-18-byte-clipping-1024/README.md)
extends exact process recovery to the longer three-arm study. Explicit clipping
at 0.01 passes its unchanged Torch numerical gate, while disabled reproduces
the prior Key-bias failures. Evaluation BPB worsens in all three clipped arms:
numerical agreement and recovery are not quality improvement, and the limit
is not promoted to a default. Both results and private-artifact hashes remain.

Existing [HF/PEFT workflows](python/build-and-validation.md) retain their own
adapter lineage and execution APIs. A native byte-LM artifact is not an HF
adapter, and this path does not yet supply KV caching, distributed scheduling,
authenticated model storage or durable serving ownership. The next integration
should connect these clients without recreating geometry or acceptance policy
in Python. Numerical agreement, recovery, serving mechanics, language quality
and speed remain separate gates.
