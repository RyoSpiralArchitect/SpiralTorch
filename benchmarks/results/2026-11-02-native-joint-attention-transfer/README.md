# Native GPT-2: LoRA plus geometry at a shared attention output

This study asks whether the native WaveGate branch adds anything over ordinary
trainable channel-affine at the **same** pretrained attention output when the
LoRA projection is learned jointly. It is not a PyTorch speed test, full-model
fine-tune, radius-only intervention, or unconstrained architecture search.

The [primary protocol](plan.json) and [sample/engine freeze](execution.json)
preceded the first update and held-out forward. All nine fits use the same
Rust-owned CPU implementation. Raw training documents, downloaded books,
checkpoints and generated token IDs stay private. No release wheel is replaced.

## Matched design

The frozen model is GPT-2 small; a rank-4, alpha-8 LoRA update is attached to
layer 8's attention output projection. Each joint arm adds its branch after
that projection and before residual addition. The layer-8 channel parameters
from the [earlier calibrated study](../2026-10-31-native-book-ablation/README.md)
are relocated to this site, not learned at an MLP and transplanted afterward.
The two joint starts have exactly matching nonzero site output and train-probe
logits. LoRA alone starts exactly at the base function because B is zero.

| Arm | Trainable LoRA parameters | Trainable additional parameters |
| --- | ---: | ---: |
| LoRA alone | 6,144 | 0 |
| LoRA + channel-affine | 6,144 | 1,536 |
| LoRA + WaveGate, free radius | 6,144 | 1,537 |

All arms use fixed-order SGD at 0.01, one 64-token window per update, one
epoch and no clipping. Each sees the same 512 disjoint *Pride and Prejudice*
training windows per seed (131, 137, 139): 512 updates and 32,256 shifted
targets. The process stops after update 256 and a new process restores the
exact checkpoint against the same ordered documents. This tests a real cut
in every arm, not uninterrupted-versus-resumed equality for nine full fits.
All terminal IDs are sealed before scoring or generation.

The samples are [Austen's *Mansfield Park*](https://www.gutenberg.org/ebooks/141),
[Wells's *The War of the Worlds*](https://www.gutenberg.org/ebooks/36) and
[Darwin's *The Voyage of the Beagle*](https://www.gutenberg.org/ebooks/944).
Each uses six preselected chapters and twelve non-overlapping windows per
chapter, for 4,536 next-token targets per state. These are fresh study samples,
not known exclusions from GPT-2 pretraining. We do not report a pooled
three-book score, significance from repeated chapter measurements, or a direct
comparison to the earlier four-site MLP/AdamW training regime.

The primary contrast is joint WaveGate minus joint affine NLL (negative favors
WaveGate) on *Mansfield Park*, required to be strictly negative in all three
seeds. Joint versus plain LoRA is descriptive: the extra branch changes
the initial function. Subtracting initial NLL **does not** replace training
LoRA with the same nonzero branch frozen. Greedy divergence on three authored
prompts is not generation quality. The two joint arms compare whole
parameterizations, not just the single log-radius scalar.

## Reproduce

Use Python 3.12 in isolated mode. `OLD` is the private root of the original
calibrated study, including its encoded training data and initial artifacts.
`MODEL` is the pinned local GPT-2 snapshot in the original plan. The three
raw files in `RAW` are Gutenberg's `https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt`
with SHA-256 in this study's plan. `TOKENIZERS_SITE` contains only the required
preparation tokenizer dependency (`tokenizers==0.22.2`). For model execution
use the separately installed exact development wheel in `PY`; neither PyTorch
nor NumPy is required. Set `DEST` to a new private directory.

```sh
python3.12 -I -S -B -m unittest discover -s . -p 'test_*.py'
python3.12 -I -S -B joint_protocol.py "$RAW" "$OLD/data/train.json" "$MODEL" "$TOKENIZERS_SITE" "$DEST/data"
python3.12 -I -S -B joint_study.py run --old "$OLD" --data "$DEST/data" --out "$DEST/run" \
    --model "$MODEL" --python "$PY" --wheel "$WHEEL" \
    --plan-pin 944ed436b9d6eb40bba1ef0796d68fd091bbc33f40d9519912d00509fc1742f9 \
    --data-pin 4d99b6419cdb9153cc8e74b4873e566d510f6b2a95d5b70b4eb4ad118c380ad0 \
    --wheel-pin df0e099409048b35e8392db551dc0d675cce956c75ff5a8133b32bc25e0a4f98 \
    --native-pin 3f9d520c5afefe6fd6216b1d86e719ee0347b54d24c10f7ee31534ba1fbba069
python3.12 -I -S -B joint_study.py audit --old "$OLD" --data "$DEST/data" --out "$DEST/run" \
    --model "$MODEL" --wheel "$WHEEL"
```

The audit reads files only. Both joined parameter owners, the private step
chains, fresh-process boundary, data identities, target counts and all section
contrasts are checked. The scoring worker restores the inference states
without providing documents to the checkpoint loader; the training receipt
itself does not prove that an inference consumer validated a corpus.
