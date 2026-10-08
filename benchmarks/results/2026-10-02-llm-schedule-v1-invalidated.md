# Schedule Study v1: Invalid Matched Comparison

Status: **stopped for a confirmed denominator confound; no efficacy verdict**.
The original frozen protocol and private artifacts are unchanged. No outputs
from this attempt are resumed or pooled into the corrected v2 comparison.

## Cause and Reproduction

In the installed Transformers 4.57.6, stock `Trainer` counts every non-ignored
label for the accumulated loss denominator when the model accepts loss kwargs.
Causal model loss nevertheless discards position zero. The controlled Trainer
instead computes microbatch means over the shifted targets. At block length
128, the ordinary arm counted 128 labels while the controlled mean used 127.
Equal sequence lengths and masks therefore did not establish matched reduction.

A local, randomly initialized tiny GPT-2 LoRA reproduction used two identical
six-token rows, dropout zero, SGD, and accumulation two. The auxiliary candidate
count was zero, yet the ordinary denominator was 12 versus 10 shifted targets.
The ordinary/controlled LoRA update-norm ratio was `0.8333335518836975`, matching
the expected `10/12`. This is an execution confound, not evidence about whether
the periodic objective improves or harms language modeling.

## Preserved Attempt

- Protocol SHA-256: `63d9852cd718c6bbb4dcdf74a4fb11cf203f3ccd9be2cfd0d0a69a5000801001`.
- Private sealed-plan SHA-256: `377c207b03874f4cc3592c0662c56780327548cbcf22d4ad393ed6480b6eb092`.
- Seed 137 ordinary FT completed 256 updates and all four generation reports.
  It remains an ordinary-FT observation, not a matched intervention effect.
- Seed 137 periodic constant was explicitly interrupted after reproducing the
  confound. The last logged step and retained checkpoint were 64; this does not
  assert that exactly 64 updates had executed when interruption occurred.
- Remaining runs did not start. There is no `completed.json` or final assessment.
- Existing logs, cards, weights, generations and plan remain untouched. A separate
  `STOPPED-causal-label-denominator.json` records the interruption locally.

## Correction Before Restart

The opt-in `st.HfCausalLabelAlignmentCollator` clones labels and masks only the
unpredicted position zero before HF counts targets and before candidate planning.
Every arm uses it. It changes neither input IDs nor the causal prediction targets.
Preflight now checks both unshifted and shifted valid-label counts; completed run
cards must record the same alignment contract.

Real CPU regressions on tiny GPT-2, GPT-2 LoRA and Llama LoRA verified all-weight
equality at zero auxiliary candidates, including a final incomplete accumulation
group (`rtol=1e-6`, `atol=1e-7`). Fifty targeted tests passed with an isolated
native wheel. A separate local rerun without disabling automatic device patches
failed CPU/MPS device comparisons; repeating with those patches disabled passed.
Corrected study commands explicitly disable them before Python startup.

An additional cached-pretrained GPT-2 wiring smoke completed eight updates in
each of two arms with batch one / accumulation sixteen. Seed 7 and a two-block
evaluation subset make this a development check, **not efficacy evidence**.
Dataset and initial model-runtime identities matched; both cards recorded the
shared alignment adapter, and the treatment exercised 75 periodic candidates.

| Development arm | Initial CE | Final CE | Run-card SHA-256 |
| --- | ---: | ---: | --- |
| Ordinary FT | 4.857405662536621 | 4.855475425720215 | `8245b6a8db85ee11eb5d616049b9b83340ca0de568eefa14479b06ad5ffa4deb` |
| Periodic constant | 4.857405662536621 | 4.856139659881592 | `1560ae45b91a61b6c70c138cc5b7d97be4772a14d79d2fe707531a94b5116ec8` |

- Corrected isolated wheel SHA-256: `5cb72b2c3db8effaccd2a3472fc1b45edae3285734e351a8280208925416a553`.
- Private development verification record SHA-256: `3a291c9ddbcf8fd100c56e67de25938e114d56818fcad0565db23dc4d19cd5a7`.
- Nineteen runner/assessor tests and 22 runtime import tests also passed.

The [new v2 protocol](../../docs/benchmarks/hf_repetition_schedule_aligned_256step_prespec_20261002.json)
uses fresh seeds 151/157/163 and a fresh output directory. The 256-update horizon,
three arms, prompts, decoding settings and acceptance gates are unchanged. This
repair is not a global masked-token normalization guarantee for unequal
microbatches or distributed ranks. Legacy bridge defaults remain unchanged.
