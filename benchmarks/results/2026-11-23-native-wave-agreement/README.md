# Does learning a WaveGate change the odds of the correct verb?

This is a small, offline, preregistered agreement pilot on one frozen GPT-2
small. It tests a specific language contrast, not free-generation quality.
A previous one-prefix readout experiment found that most of a fixed
WaveGate's raw logit change was a common offset. A common offset has no
effect on the odds of one word versus another.

## Question and controls

For a prefix such as `The cup near the plates`, compare the logit of the
verb that agrees with the **subject** to that of the other verb (` is` vs
` are`). Their difference is the log probability ratio irrespective of
all other vocabulary entries. Both verbs and every word in the fixture
are single GPT-2 BPE tokens, checked against the pinned vocabulary and
merge table before model execution.

- Six training noun pairs produce 24 sentences; four **different** noun
  pairs produce 16 evaluation sentences. Each pair contributes all four
  subject/neighbor singular/plural combinations. This tests transfer to
  different nouns within one template, not to new syntax. The sentences
  are held out from this pilot, not known to be absent from GPT-2 pretraining.
- All arms have the same bare frozen decoder, no LoRA, and the same initial
  gate and bias on the last attention output. WaveGate uses curvature -1,
  saturation 1, porosity 0.05, fixed log radius 0 and residual strength 0.25;
  affine uses the same strength and trainable parameter count. Each starts
  from its own initial predictions: equal channel parameters do not imply
  equal initial functions.
- At each training position all arms differentiate the same Rust-owned CE
  objective, supervising only the final verb. The fixed WaveGate discards
  the gradient. The learned WaveGate and affine arms apply SGD at 0.025.
  Three prespecified example orders are paired across arms. No order or
  learning rate is chosen from the outcome.
- First finish every training arm and seal its exact adapter bytes. Only
  then reload the sealed adapters to predict all evaluation examples. Two
  initial states plus nine final states give 176 measured rows. In total:
  216 calls to compute gradients, 144 applied updates; the fixed control
  must reproduce its complete initial f32 logits bit for bit.

**Primary comparison:** for each order, average the 16 per-sentence changes
in correct-minus-rival margin relative to that arm's own start. A positive
pilot requires learned WaveGate to exceed both frozen WaveGate and learned
affine on this change in **every** order. Report null or negative results
without retuning or retrying. Also report target-token NLL over the full
vocabulary, strict pairwise wins and ties, and congruent/conflicting
subgroups. The orders reuse the same nouns; they are not independent test
sets. An increase in margin alone need not improve vocabulary-wide NLL.

## Running and replay

Requires an offline arm64 macOS Python environment containing the native
build with extension SHA-256
`69245aec6cb242848e282ee71228a0687bf8fb86d925648a063634a6047ad2d5`
for the real study. This pin is a particular saved development build, not
a promise about every wheel with the same package version. No Torch,
NumPy, Transformers, GPU or model downloads. The head is forced to the
original packed CPU path; only the vocabulary pullback uses the wide
backward accumulator. This is not a speed benchmark.

From this directory, use fresh private output directories. Preparation
validates the weights and BPE files and prints a content pin; it does not
instantiate a model or score sentences.

```sh
python3 -ISB test_probe.py -v
PIN=$(python3 -ISB probe.py prepare --kind gpt2 --model-dir "$GPT2" --directory "$OUT")
"$NATIVE_PYTHON" -IB probe.py run --kind gpt2 --model-dir "$GPT2" --directory "$OUT" --pin "$PIN"
python3 -ISB probe.py verify --kind gpt2 --directory "$OUT" --pin "$PIN"
```

Use `--kind rehearsal` and omit `--model-dir` for separate, deterministic
tiny weights, different token IDs and 8 applied updates. A rehearsal is
not a pretrained result. The real worker has a 600-second ceiling and
requires 4 GiB free throughout (plus 192 MiB at admission). A failed
attempt is kept; an existing attempt directory cannot be reused.

`attempt/worker-result.json` contains per-sentence scores, per-order
contrasts, geometry and raw-logit hashes. Only a clean parent-issued
`receipt.json` plus successful readback counts as a completed run.
Verification streams no model data through a framework: it reconstructs
the scores from stored f32 rows, checks the applied-update chain and
checks that the radius and frozen control were unchanged.
