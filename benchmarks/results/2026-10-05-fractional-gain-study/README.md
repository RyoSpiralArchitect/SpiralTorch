# Learned Gain Versus An Ordinary Short Filter

Completed offline CPU float32 learning study, not a speed benchmark.
Protocol: [matched gain study](../../../docs/fractional_gain_study.md).
Training revision: `f12c50a7ef0cf9bb6b7d26833716d4081860c024`.
Verifier revision: `2f963250ce973c7e2da48a76a0fe1a751a01d428`.

**The ordinary learned two-tap control beats both GL arms on both endpoint
sets in every seed.** Within GL, K=32 beats K=3 in every seed. All arms
improve on the frozen base here, but this study does not demonstrate a
distinctive quality advantage for GL geometry.

## Fixed Comparison

Local pretrained GPT-2 with a frozen base and first-MLP insertion. Each arm
has 1538 trainable parameters: two feature gates, one shape coordinate and
one log-gain. All start as identity with the same initial `[-2, 1]` strictly
past filter up to float32 rounding; gain starts at sqrt(5). GL uses Rust
normalized history at K=3 or K=32 with initial alpha=2. The ordinary control
uses independent Torch `gain * [-cos(angle), sin(angle)]` with two past taps.

Seeds 41/43/47, 512 updates each, Adam 0.001, strength 0.1, batch 2, context
128 and two CPU threads. All 4608 primary updates and 18 continuation-only
updates finished before fixed endpoint scoring. The separate pretrained
[preflight](../2026-10-05-fractional-gain-preflight/) used six auxiliary
training and six continuation-only updates, without endpoint scoring.
The model/data/update budgets match, not operator FLOPs or shape charts.

Training uses 1204 Pride blocks; the 16 development blocks are diagnostic
and did not select arms, checkpoints, duration or settings. The fixed
endpoints are 120 unused Pride-tail blocks and 32 Alice blocks. These are
reused exploratory books and may occur in pretraining, not pristine
held-out confirmation. Three minibatch-order seeds are not a significance
test and previous studies do not add independent seeds.

## All Outcomes

Mean next-token cross-entropy across three seeds; lower is better.

| Arm | Pride | Alice |
| --- | ---: | ---: |
| Frozen base | 4.073713269 | 4.011807486 |
| Ordinary learned short | 4.023867468 | 3.982157086 |
| GL learned-gain short | 4.029066804 | 3.985153474 |
| GL learned-gain full | 4.027346196 | 3.984107643 |

Positive differences below mean the first named arm is worse. Every
contrast has the same sign in all three seeds on both evaluation sets.
`summary.json` retains per-seed differences and descriptive sample SDs.

| Predeclared Contrast | Pride CE Difference | Alice CE Difference |
| --- | ---: | ---: |
| GL short - ordinary short | +0.005199336 | +0.002996388 |
| GL full - ordinary short | +0.003478728 | +0.001950557 |
| GL full - GL short | -0.001720608 | -0.001045831 |

The longer GL filter improves over its short counterpart under this fixed
budget, but does not recover the gap to the ordinary short control. This
does not uniquely isolate long memory: normalization changes the short
taps too, and coefficient norm does not measure hidden-state variance.

## Learned Coordinates

| Arm | Final Shape Coordinate Range | Final Gain Range |
| --- | --- | ---: |
| Ordinary short | angle -0.248459 to -0.236317 | 5.017912-5.202008 |
| GL short | alpha 0.980171-0.993912 | 4.971347-5.167671 |
| GL full | alpha 0.902527-0.915750 | 4.999610-5.206926 |

Gain and shape received nonzero gradients on all 511 updates after the
zero-gate first update in every run. Initial loss/gradient-norm criteria
passed; exact saved state is checked separately rather than inferred from
those norms. The gain rise is evidence that amplitude is actively learned,
not proof of a unique parameter interpretation or convergence. Feature
gates and gain remain redundant amplitude controls.

Angle and positive log-order are different optimizer charts with different
reachable shapes. Equal parameter count and initial filter do not make
their later learning dynamics equivalent. A useful next diagnostic is
chart-aligned short-filter learning, including parameter-gradient checks,
before attributing this gap to the GL kernel or adding longer history.
This is a proposed follow-up, not a condition retroactively selected here.

## Verification And Reproduction

Primary training, completed resume and saved-state verification exited zero.
Completed resume neither trained nor rescored; all 76 sealed output files,
including 72 checkpoints, were unchanged. Each final adapter was restored
with its exact recipe, scalar/feature dtype and shape, registration-bound
Adam state, final coordinates, cursor and batch schedule. Individual
parameter/moment hashes are public; tensors remain private.

The summary was reconstructed byte for byte. Seven frozen client files,
71 runtime files, seven model assets and two corpus hashes matched. The
423 earlier study/client/runtime files were rehashed unchanged. Native
gain observation uses Rust's checked f32 conversion, not a Python copy.

The verifier revision passed 646 Python regressions, including 42 new
verification cases, plus 34 benchmark correctness tests without timings.
Existing JIT deprecation warnings remain. Publication consistency tests are
separate from private saved-state verification: the final publication-inclusive
suite passed 649 tests, and 13 current/prior publication checks passed.

Rebuild the public numerical summary without Torch, Transformers, model weights or
the native extension, writing a new derived file:

```sh
python -B summarize_wave_gate_long_horizon.py \
  --plan plan.json.gz --results results.json.gz --journal journal.json \
  --output /tmp/fractional-gain-summary.json
cmp summary.json /tmp/fractional-gain-summary.json
shasum -a 256 -c SHA256SUMS
```

Private checkpoint verification additionally requires the exact frozen
client/runtime and local `study` directory. Follow the protocol's verifier
command. Both verifier files here must remain together; the factorial
module supplies only shared read-only helpers, not a previous-study replay.
The verification reads saved states and continuation receipts, not a new
training or scoring run. Unlike arms are not claimed to have equal states.

The original plan/results are compressed numeric records, including all
training/development/endpoint values, not text or weights. Build revision
`ab9f4ea35c9089cf57c7a11fb593555c2d57b668` and launch revision differ by the
preflight publication commit; executable hashes are checked against the
plan. No source, result or runtime of an earlier study was rewritten.

Weights, corpus, runtime packages and raw logs stay local. Public numeric
checks do not reproduce private-state verification. No claim is made about
speed, statistical significance, general LLM superiority, GPU/browser
residency, pristine held-out generalization or converged training.
