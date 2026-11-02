# Native GPT-2: LoRA plus geometry at a shared attention output

This study asks whether the native WaveGate branch adds anything over ordinary
trainable channel-affine at the **same** pretrained attention output when the
LoRA projection is learned jointly. It is not a PyTorch speed test, full-model
fine-tune, radius-only intervention, or unconstrained architecture search.

The [primary protocol](plan.json) and [sample/engine freeze](execution.json)
preceded the first update and held-out forward. All nine fits use the same
Rust-owned CPU implementation. Raw training documents, downloaded books,
checkpoints and generated token IDs stay private. No release wheel is replaced.

## What happened

**WaveGate did not beat joint affine on the primary book (0/3 seeds).**
Joint affine had lower terminal NLL on all nine book/seed pairs, but the
mean gap is only 1.0 to 3.3 millionths of a nat per predicted token.
All arms lowered NLL relative to their **own** start on every sample. Both
LoRA matrices and both channel vectors changed in every joint fit, and all
three WaveGate radii moved. That is evidence of learning, not a quality win.

| Sample | Frozen base NLL | LoRA | LoRA + affine | LoRA + WaveGate | Wave minus affine |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mansfield Park | 4.256685661 | 4.256656748 | 4.256663077 | 4.256666369 | +0.000003292 |
| The War of the Worlds | 4.131971598 | 4.131959560 | 4.132016776 | 4.132018228 | +0.000001453 |
| The Voyage of the Beagle | 4.082854913 | 4.082845264 | 4.083020354 | 4.083021354 | +0.000001000 |

Terminal columns are means across three order seeds, **within each book**.
Plain LoRA ended lowest on each sample. The matched joint insertion initially
raised NLL relative to base by 0.000025842, 0.000069075 and 0.000181145,
respectively. Joint training recovered that initial cost on Mansfield, but
not on Wells or Darwin. It would be misleading to attribute the whole
plain-LoRA versus joint gap to updating geometry.

All 13 states produced the **same 24 generated IDs per prompt** on the three
fixed greedy prompts; there was no paired divergence at any seed. This does
not mean the underlying distributions were equal. No sampling or open-ended
generation evaluation was performed.

The [paired results](results.json) include all condition and section NLLs,
parameter-group change flags, checkpoint identities, and continuation hashes.
Run `python3.12 -I -S -B verify_public.py` here (no model or ML library) to
reconstruct the 18 native score receipts from the [score roster](score_roster.json),
their 234 NLLs and the independently derived section IDs. All 4,608 updates,
nine fresh-process resumes and 290,304 shifted training targets also passed
the private file-only audit. Primary result SHA-256:
`3343e45654f20cc1547c4d2757dddc4acd5cd660984f6274deb2e64864f72fa8`.

## Where the difference follows the weights

The [secondary protocol](interchange_plan.json) was fixed while fits were in
progress, before any held-out scoring. Within each seed, restore the two
jointly trained snapshots, cross their final LoRA and geometry groups, then
score all four compositions. The [12-state crossover](interchange_results.json)
reproduced all 108 original diagonal scores exactly and measured 108 off-diagonal
scores. Each returned row was checked against its actual two adapter IDs and
section content ID. This was one [bounded CPU attempt](interchange_attempt.json)
with no further training. The observed worker finished in 355.5 seconds. Its
first-attempt record and 18 original native reports were retained.

Mean within-book contributions below are in **millionths of a nat per token**.
LoRA and geometry are the symmetric averages over the two swap orders; they
sum to the terminal Wave-minus-affine contrast. Interaction is a separately
reported difference in differences, not an additional term to add to the total.

| Sample | Total | LoRA group | Geometry group | Interaction |
| --- | ---: | ---: | ---: | ---: |
| Mansfield Park | +3.292 | -0.047 | +3.339 | -0.104 |
| The War of the Worlds | +1.453 | -0.011 | +1.464 | -0.141 |
| The Voyage of the Beagle | +1.000 | -0.081 | +1.081 | +0.015 |

In all nine book/seed pairs the larger contribution followed the **geometry
group**, not the co-trained LoRA group. This concerns the full terminal branch
(gate, bias, recipe and where applicable radius); it does not locate a causal
training mechanism or establish that radius alone made it worse.

## Practical handoff

The comparison above used the original one-window session wheel. Separately,
the newer source implements native `batch_windows` and removes gradient-vector
copies without changing the arithmetic. The [fixture validation](fixture_validation.json)
checks 5 legacy v1 pins, 36 single/batched checkpoint pins and 6 update-report
streams across LoRA, geometry and joint modes. The installed wheel passed
112 Python tests and the Rust library passed 1,069; WASM was compile-checked,
not used for a browser GPT-2 fit.

On [real GPT-2](batched_handoff.json), five 64-token training windows were
processed in batches of 2, 2 and 1, for two epochs. All six session updates
matched direct native batched gradients/updates. A separate **Rust executable**
restored the second checkpoint written in Python, took the 63-target tail
update and wrote exactly the same third checkpoint bytes. Python resumed at
epoch 1, window 0. Terminal corpus-free inference matched direct-model logits.
This functional probe is not included in the nine-fit quality study.

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
python3.12 -I -S -B freeze_binding.py run --old "$OLD" --data "$DEST/data" --out "$DEST/run" \
    --model "$MODEL" --python "$PY" --wheel "$WHEEL"
python3.12 -I -S -B freeze_binding.py audit --old "$OLD" --data "$DEST/data" --out "$DEST/run" \
    --model "$MODEL" --wheel "$WHEEL"
```

The audit reads files only. Both joined parameter owners, the private step
chains, fresh-process boundary, data identities, target counts and all section
contrasts are checked. The scoring worker restores the inference states
without providing documents to the checkpoint loader; the training receipt
itself does not prove that an inference consumer validated a corpus.
The original runner accepted caller-supplied pins. The wrapper above takes them
from the immutable [execution freeze](execution.json) and recomputes all 18
[section content IDs](input_binding.json) directly from the fixed windows. This
review addition authenticates the original run without retraining or rescoring.

To reproduce the secondary run in an independent primary run directory, invoke
`joint_interchange.py` using the isolated study interpreter and the common
`--old --data --out --model --wheel` arguments above. Supply the published
primary hash as `--result-pin` and a fresh private `--receipt` path. Within
that primary run directory there is one numerical attempt per primary-result
and crossover-plan hash pair, regardless of the export filename. A timed-out
or failed worker's staged result is not exportable. A successful supervisor
checks all 18 raw section reports before sealing the result. Use `--export-only`
for a new verified copy, without a new worker or overwriting another file.
Audit a new run with `verify_public.py --from-crossover RECEIPT --primary-run "$DEST/run"`.

The original measurement predated this lock: its supervisor keyed on the
receipt filename. The [later registration](interchange_registration.json) used
`--adopt-legacy` to check the retained first attempt, raw reports and signed-off
inputs, then reserve the canonical claim without starting a numerical worker.
It does not assert that a global or cross-machine lock existed during the study.
Audit the original via `verify_public.py --from-crossover ORIGINAL_RECEIPT`.
See the [batched session API and examples](../../../docs/python/native-training-session.md).
