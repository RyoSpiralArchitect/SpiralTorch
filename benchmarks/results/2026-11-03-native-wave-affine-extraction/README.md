# Does WaveGate beat its trainable affine control on answer-only SFT?

Prospective, CPU-only native GPT-2 small study, separate from the [initial
pilot](../2026-11-03-native-field-extraction-sft/README.md). That pilot compared
joint WaveGate training to a frozen identity, not to an equally trainable head.

## Fixed comparison

| Arm | Trainable geometry | LoRA | Total scalars |
| --- | ---: | ---: | ---: |
| Ordinary channel affine | 768 gates + 768 biases | 3,072 | 4,608 |
| WaveGate v2, radius fixed at 1 | 768 gates + 768 biases | 3,072 | 4,608 |
| WaveGate v2, learnable radius | same + 1 log radius | 3,072 | 4,609 |

At block 11's attention output, every arm adds 0.25 times its residual.
Gate and bias start at zero; the two Wave recipes have curvature -1, saturation
1 and porosity 0.05. The fixed log radius is a buffer, not an optimizer
parameter. At this origin the residual is zero, the gate/bias Jacobians match
affine, and the log-radius derivative is zero. Away from it, the Wave map uses
porous saturation and row-wise hyperbolic projection. This comparison tests the
combined map, not which individual operation caused a difference.

All arms use the same deterministic rank-2, alpha-4 attention output LoRA and
unchanged base. Joint SGD (lr 0.01), batch 1, context 64, supplied answer
labels only, one update per record. All 144 **triples** from the earlier
train/eval sets are excluded even if reworded or queried for a different field.
The new 96/48 split is balanced across 3 fields, 8 answers per field, requested
positions and templates. Three predetermined permutations of the same 96
training records are **order sensitivity**, not three independent test sets or
three parameter initializations. There are 864 arm-updates and 48 distinct
test answers. The initial score is shared; score again only at update 96 per
order, after pinned checkpoint read-back. No decoding, intermediate probes,
cross-run selection or post-hoc tuning.

Primary: final fixed WaveGate minus affine, mean of paired per-answer mean
unsmoothed target-token NLL (lower favors Wave). Secondary: learned minus fixed
radius, token-weighted NLL, signed per-answer counts and teacher-forced top-1.
Directional advantage requires all three ordered contrasts beyond
`log(1.001)` nats in the same direction with no fewer all-tokens-correct
answers for the favored arm. This descriptive margin corresponds to a 0.1%
likelihood ratio on the geometric-mean scale; it is not a significance level or
a proof of equivalence. Always report each raw signed contrast.

## Reproduce

The release number alone does not identify the development wheel. Exact wheel,
extension and data hashes are in `protocol.py`; model/tokenizer pins are in its
immutable predecessor. No Torch, NumPy, Transformers, PEFT or GPU is used.
Tokenization runs in a separate offline process. Both phases share one absolute
deadline (60 minutes total); inputs, outputs and attempt markers are exclusive.
Stop at first error, retain incomplete outputs, do not restart.

```sh
python3 -I -B -S protocol.py
"$VENV/bin/python" -I -B test_controls.py -v
COMMIT=$(git rev-parse HEAD)
DEADLINE=$(($(date +%s) + 3600))
"$VENV/bin/python" -I -B run.py prepare --output "$NEW_DIR" --model-dir "$MODEL" --wheel "$WHEEL" --tokenizers-site "$CODEC_SITE" --commit "$COMMIT" --deadline "$DEADLINE"
# Supply the digest printed by prepare, not a digest derived inside fit.
"$VENV/bin/python" -I -B run.py fit --output "$NEW_DIR" --model-dir "$MODEL" --wheel "$WHEEL" --manifest-pin "$PIN" --commit "$COMMIT" --deadline "$DEADLINE"
```

The pre-tokenization protocol and split hash (canonical JSON) is
`7ba48e87df11570ac065e1e5251cfe23e8f91a33af6430e002b0d01c7685a7e2`.
