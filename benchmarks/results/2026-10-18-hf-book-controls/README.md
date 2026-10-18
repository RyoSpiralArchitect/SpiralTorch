# Fixed Book Controls Through The Public HF Trainer

All twelve declared runs completed at step 512 without retries or tuning.
This larger follow-up to the [small ordinary-control probe](../2026-10-18-hf-affine-control/README.md)
uses the existing public `HfGeometryDataset`, `GeometryAdapterArtifact` and HF
Trainer integration. There is no study-specific optimizer or geometric formula.

## Observed Result

Lower development cross-entropy is better. The frozen base and all identity-start
adapters have CE **4.118154**. Full-precision per-run results, all case statuses,
learned-state deltas and private receipt hashes are in [results.json](results.json).

| Arm | Seed 41 | Seed 43 | Seed 47 | Mean |
| --- | ---: | ---: | ---: | ---: |
| Ordinary channel affine | 4.065830 | 4.064968 | 4.065067 | 4.065289 |
| Legacy WaveGate | 4.110890 | 4.110876 | 4.110872 | 4.110879 |
| WaveGate, fixed radius 1 | 4.110890 | 4.110876 | 4.110872 | 4.110879 |
| WaveGate, fixed radius 4 | 4.096088 | 4.095950 | 4.095968 | 4.096002 |

The primary WaveGate-minus-ordinary differences are **+0.045060, +0.045907,
+0.045805**: this WaveGate insertion underperforms the ordinary control in all
three seeds. Radius 4 improves over radius 1 by **0.014802, 0.014926, 0.014904**
within the same native arithmetic path, but does not catch the ordinary arm.
Legacy versus radius-1 CE differs by at most 4.77e-7 here; that observation does
not make their arithmetic bit-identical.

Every intended site learned; all fixed radius buffers remained unchanged. Each
run preserved the frozen base parameter hash and reproduced logits exactly after
export/reconstruction. These are lifecycle checks, not a new book-specific
interruption/resume experiment or proof of general language-quality improvement.
All six reserve chapters remain unused. No configuration is promoted.

The [saved-state check](saved-state-check.json) verifies all 24 retained step-256
and step-512 checkpoints: eight trainable tensors / 6,144 values, finite moments,
exact optimizer/scheduler steps, and final saved weights equal to the export
apart from training/evaluation mode flags. It reads trusted local outputs only.
[validation.json](validation.json) links tests, source pins and private log hashes;
[execution.json](execution.json) is the immutable pre-run execution declaration.

This provides a concrete next mechanism question: measure branch contribution
and projection saturation at the declared sites before changing geometry. The
radius comparison motivates that diagnosis; it does not by itself establish
saturation as the cause or justify searching until a favorable result appears.

## Frozen Comparison

- Local pretrained GPT-2 small, frozen float32 base on MPS; geometric operators
  remain Rust CPU operations transported through first-order Torch autograd.
- Four MLP sites, 6,144 trainable gate/bias values in each arm. Radius buffers
  are fixed and excluded from the trainable count.
- Ordinary channel affine versus legacy WaveGate is the primary comparison.
  Radius-1 versus radius-4 WaveGate is an exploratory comparison using the same
  wider-precision radius implementation on both sides.
- Seeds 41, 43, 47; 512 optimizer steps each, context 128, batch 2; identical
  AdamW, clipping, linear schedule and document ordering policy.
- Final-step development CE only. No best-checkpoint selection, adaptive search,
  retries, dropped failures or automatic promotion. Generated text stays local.

[plan.json](plan.json) is the complete predeclared recipe. Native radius VJP
uses f64 products/sums while legacy WaveGate and the ordinary control use f32
tensor primitives. The two **within-pair** initial parameter gradients match
bit-for-bit. Cross-path differences are measured, not silently waived or
attributed to geometry. Equal trainable count is not equal FLOPs or wall time.

## Data And Preflight

[Project Gutenberg's catalog](https://www.gutenberg.org/ebooks/1342) labels Jane
Austen's English *Pride and Prejudice* public domain in the USA. The acquired
file, its SHA-256 and normalization are pinned in the plan. This source was not
assumed identical to earlier local copies. No unseen-pretraining claim is made.

Numbered chapters only: 1-49 train, 50-55 development, 56-61 write-only reserve.
Header, preface, illustration blocks and final colophon are excluded. Balanced
bracket handling removes nested illustration metadata without deleting ordinary
bracketed prose. The preparation manifest and every selected chapter are pinned
before tokenization; symlinked split directories/chapters are rejected before
opening them. The reserve is never tokenized, trained on, evaluated or selected
by the preflight or replay. Preparation necessarily reads the source to split it.

| Split | Chapters | Tokenized tokens | Full 128-token windows | Dropped tail tokens |
| --- | ---: | ---: | ---: | ---: |
| Train | 49 | 137,399 | 1,047 | 3,383 |
| Development | 6 | 20,740 | 159 | 388 |
| Reserve | 6 | not tokenized | not planned | not measured |

Rust's document-window planner owns boundaries, including the shifted-label
target count. Each run samples 1,024 windows (130,048 shifted-label targets),
slightly less than one complete shuffled training epoch. It is not long-run
convergence evidence or a final held-out estimate.

[preflight-qualified.json](preflight-qualified.json) contains public pins,
window summaries, initial artifact identities and actual successful checks.
The first three-arm declaration and its failed preflight are retained in Git
commit `20dc23b29` and [initial-preflight-rejected.json](initial-preflight-rejected.json).
No book LM weights were loaded or updated before this explicit four-arm revision.
The rejected bitwise assumption and nested-illustration issue were not hidden.

## Replay

Use the qualified source wheel/runtime from the preceding control slice, not an
older PyPI package. Raw text, checkpoints, generated text and learned artifacts
remain in a new private directory. Acquire the pinned source according to the
catalog's applicable terms; `prepare.py` refuses a different source digest.

```bash
python -I -S -B benchmarks/results/2026-10-18-hf-book-controls/prepare.py \
  --raw "$PRIVATE/pg1342.txt" --output "$PRIVATE/chapters"

python -I -B benchmarks/results/2026-10-18-hf-book-controls/preflight.py \
  --expected-plan-id "$PLAN_ID" --expected-manifest-id "$MANIFEST_ID" \
  --data "$PRIVATE/chapters" --model-dir "$LOCAL_GPT2" \
  --output "$PRIVATE/preflight"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 \
python -I -B benchmarks/results/2026-10-18-hf-book-controls/replay.py \
  --expected-plan-id "$PLAN_ID" --expected-manifest-id "$MANIFEST_ID" \
  --data "$PRIVATE/chapters" --preflight "$PRIVATE/preflight" \
  --expected-preflight-id "$PREFLIGHT_ID" --model-dir "$LOCAL_GPT2" \
  --output "$PRIVATE/runs"
```

To repeat only the saved-state checks against these trusted private outputs:

```bash
python -I -B benchmarks/results/2026-10-18-hf-book-controls/verify_saved_state.py \
  "$PRIVATE/runs"
```

The three independently retained pins are in `preflight-qualified.json`; the
private full preflight hash differs from this compact public projection. A new
source/runtime qualification must retain its own preflight pin. The executor
checks the installed versions and source/installed-package hashes before and
after every case. It delegates each update to
`bindings/st-py/examples/hf_geometry_finetune.py` and retains all twelve case
statuses using the already-tested control manifest helper. Every child process
group has a one-hour timeout and cleanup on ordinary exit/cancellation. This
does not promise recovery from supervisor SIGKILL, power loss, or escaped daemons.

Only explicitly allowlisted metrics/hashes enter the publishable case manifest
and summary. `result.json`, full child logs and model exports are private. A
failure stops without retry and leaves completed, failed and unstarted cases in
the manifest. Numerical agreement, checkpoint mechanics, teacher-forced CE and
free-running quality remain separate questions.
