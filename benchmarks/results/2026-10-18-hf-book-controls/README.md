# Fixed Book Controls Through The Public HF Trainer

Pre-training declaration and qualified inputs; model results are not yet available.
This larger follow-up to the [small ordinary-control probe](../2026-10-18-hf-affine-control/README.md)
uses the existing public `HfGeometryDataset`, `GeometryAdapterArtifact` and HF
Trainer integration. There is no study-specific optimizer or geometric formula.

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
