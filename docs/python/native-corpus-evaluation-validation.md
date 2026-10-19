# Native Corpus Evaluation Qualification

Local qualification on 2026-10-19, Apple M4, Python 3.12.6. Source-only after
0.4.32; this is not a release or a repair of the separate #2284 study-resume
discrepancy. Rust owns evaluation; the native environment had no Torch,
Transformers, NumPy, PEFT, Accelerate or Python safetensors installed. Optional
reference execution used Torch 2.12.1 and Transformers 4.57.6 in another venv.

## Exact Inputs And Scoring Policy

The same local GPT-2 checkpoint and 1,303-token chapter from the
[earlier training qualification](../../benchmarks/results/2026-10-18-native-corpus-training/README.md)
were used. The adapted arm restores that run's saved update-20 checkpoint,
not newly fitted parameters. This chapter was already used for training: these
are **in-sample execution checks, not independent holdout or language-quality
evidence**. No weights, encoded corpus, private checkpoints or raw logs are
published here, and no new training occurred during this comparison.

Context length is 32 IDs, with 31 decoder inputs and one-token overlap between
windows. All 1,302 within-document targets are scored once. No smoothing,
sampling, padding or implicit EOS. The independent reference enumerates spans
and sums ordinary f32 Torch cross entropy, rather than reading a native plan.
The predeclared absolute mean-NLL gate is `5e-4`; all four conditions passed.

| Model | Native Batch | Native NLL | Reference NLL | Absolute Error |
| --- | ---: | ---: | ---: | ---: |
| Frozen base | 1 | 4.6850756293 | 4.6850497045 | 0.0000259248 |
| Frozen base | 2 | 4.6850756691 | 4.6850497045 | 0.0000259646 |
| Saved geometry, update 20 | 1 | 4.6798946119 | 4.6798689472 | 0.0000256648 |
| Saved geometry, update 20 | 2 | 4.6798945836 | 4.6798689472 | 0.0000256364 |

The two native batch sizes agree within `4e-8` in this run; that is an
observation, not a cross-device bitwise guarantee. Execution durations are not
speed evidence: compilation and other qualification work ran concurrently.

Retained input and execution identities (SHA-256):

```text
data.json       d95c8c6161a6ad91b67be1d477890daf3adc15b90f804a83a1350b955bd40fee
config.json     0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4
model.safetensors 248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707
checkpoint-20   1ed13924961efe03330d1e16407382e3fcb0b3255504939215215a9375f62e6e
local probe     66827516b2e935feac691dd197ed6fe4bb7f6bff58bb8bc66324f80f084cf36e
native library  7265bad2c2e94a717637a3f412d4d5e567d747fde9b5e06a05e278488037e001
```

## Regression Coverage

The public test fixtures require no downloaded weights or corpus. They cover
short documents/tails, custom labels, zero-target windows, token-weighted
reduction, numeric overflow, bounds before foreign token extraction, immutable
results, evaluation during training, and exact resumed optimizer/model state.
The packaged example test checks explicit/default intervals, an off-interval
final evaluation, a zero-update run, resumed logs and rejection before output
creation. An old installed wheel first failed with the expected missing
`evaluate_documents` attribute; the new implementation passes.

Optional reference tests cover the base and all 14 native geometry recipe
configurations with both full and masked targets: 30 cases, each with native
batch sizes 1, 2 and 8, under a `3e-6` absolute NLL gate. Native tests do not
import this reference stack.

From a checkout with the matching wheel installed:

```sh
python -I -B bindings/st-py/tests/test_native_causal_lm_evaluation.py -v
# Separate environment with the optional reference packages:
python -I -B -m pytest -q -o addopts= \
  bindings/st-py/tests/test_native_causal_lm_evaluation_parity.py
```

For the retained private chapter, reproduce the four scores by loading the
pinned base, restoring the pinned training checkpoint using
`NativeCausalLm.from_training_checkpoint`, then calling `evaluate_documents`
with context 32 and batch sizes 1 and 2. Verify the input hashes above first.
The public fixture suite is reproducible without those private inputs; it is
not a substitute for the retained real-checkpoint comparison.
