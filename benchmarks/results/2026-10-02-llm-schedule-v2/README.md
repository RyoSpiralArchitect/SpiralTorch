# Aligned GPT-2 Repetition Schedule Study

**Execution complete; `ready_but_negative`. Do not adopt the tested decay
schedule as a repetition improvement.** All nine 256-update runs, 36 generation
reports and 432 prompt continuations completed and passed the artifact checks.
Linear decay lost to both constant intervention and ordinary FT on the final
loop score in all three seeds. A working intervention is not evidence of benefit.

## Fixed Comparison

The [committed v2 protocol](../../../docs/benchmarks/hf_repetition_schedule_aligned_256step_prespec_20261002.json)
compares pretrained GPT-2 LoRA on the existing complete *Pride and Prejudice*
corpus: ordinary FT, periodic unlikelihood at strength 0.1, and the same
intervention with linear strength decay over slots 0 to 256. All arms use
active-position normalization where applicable, CPU float32, rank 4 / alpha 8,
batch 1 with accumulation 16, and the same learning-rate schedule and causal
label alignment. Selected held-out block counts are 124, 127 and 128 for seeds
151, 157 and 163. Each checkpoint generates the same twelve prompts greedily,
without inference controls. Every one of the 432 continuations has 96 tokens;
short output or early EOS does not explain the differences here.

The original [v1 attempt](../2026-10-02-llm-schedule-v1-invalidated.md) remains
invalidated for its causal-label denominator confound. It is not pooled with
this comparison. Development smokes and historical MPS results are also excluded.

## Final Results

Lower is better for both held-out causal-LM loss and the loop score. The loop
score is a sum of repetition ratios, not a probability or a general language
quality score. Only the prespecified final checkpoint decides the outcome.

| Seed | Arm | Held-out loss | Loop score | Active training positions |
| --- | --- | ---: | ---: | ---: |
| 151 | Ordinary FT | 4.672988 | 0.753471 | n/a |
| 151 | Constant | 4.700372 | 0.804058 | 0.4602% |
| 151 | Decay | 4.689689 | 0.808500 | 0.4625% |
| 157 | Ordinary FT | 4.694785 | 0.806477 | n/a |
| 157 | Constant | 4.716718 | 0.892932 | 0.4846% |
| 157 | Decay | 4.708444 | 0.908003 | 0.4863% |
| 163 | Ordinary FT | 4.649334 | 0.821483 | n/a |
| 163 | Constant | 4.670958 | 0.848809 | 0.4794% |
| 163 | Decay | 4.663124 | 0.905343 | 0.4801% |

- Ordinary-FT learning gate: **pass**. All seeds improve held-out loss; the mean
  before/after change is -0.329603, below the required -0.05.
- Decay versus constant: **fail**. Mean final loop difference is +0.025349,
  with zero of three seed wins; the protocol requires at least two.
- Decay versus ordinary FT: **fail**. Mean final loop difference is +0.080138,
  with zero of three seed nonlosses; the protocol requires at least two.
- Held-out-loss safety margin: **pass, not equivalence**. Decay minus ordinary
  FT is +0.014717 on average and +0.016701 at worst, within the prespecified
  +0.02 / +0.05 limits. All three differences are nevertheless unfavorable.

[assessment.json](assessment.json) contains the complete precision results,
all four checkpoints per run, all continuation tokens and texts, Rust-validated
generation evidence, treatment receipts, package versions, protocol and artifact
identities. No best checkpoint or favorable prompt subset replaces the final
endpoint. The active-position fractions are descriptive; they do not establish
that low candidate coverage caused this negative result.

## Verification And Provenance

The canonical assessor checked all 371 sealed source/input files, all run/report
hashes, exact condition coverage, the shared label alignment, completed update
horizons, paired initial evaluation/runtime identities, objective policy and
clock, actual saved adapter fingerprints, and token-to-text decoding for every
continuation. Raw logs, cards and all checkpoint adapters remain local. No
weights, private machine paths or full raw logs are published here.

- Execution source: `f30632db41982480998c95e80d52fdaff8f68858`.
- Protocol: `sha256:0cb91e84c56b53af66bb8a1aca4df5a9efb98718ca5c34415a4f0857dedd74b9`.
- Sealed plan: `2c4895935fc4cf86039d99eb6a13f5746613aea4c5324f17519875f8af30c74c`.
- Completion record: `1a41b440786908ff7edf9a6269830a80b42b9c2f83d1036ebf6929f3e5315769`.
- Tested native extension: `3625705db20d3980f050bd937e7a552320972f389750fd122b653e1702ab0341`.
- Tested wheel: `5cb72b2c3db8effaccd2a3472fc1b45edae3285734e351a8280208925416a553`.

The installed project version was 0.4.27, but that version number alone does not
identify this source-built wheel. Follow the source-build and isolated-runtime
instructions in the [binding guide](../../../bindings/st-py/README.md) rather
than assuming a same-version PyPI wheel has these changes.
The [study guide](../../../docs/hf_repetition_schedule_study.md) describes the
pinned local model/corpus inputs, runtime setup and full execution contract.

```bash
export SPIRALTON_MAGIC=0 SPIRALTON_TORCH=0
export SPIRALTON_MODEL_PATCHES=0 SPIRALTON_NUMPY=0
"$PYTHON" -I tools/run_hf_repetition_schedule_study.py \
  --model "$MODEL_SNAPSHOT" --corpus "$CORPUS" --output "$NEW_OUTPUT"
"$PYTHON" -I tools/assess_hf_repetition_schedule_study.py "$NEW_OUTPUT" \
  --output "$NEW_ASSESSMENT"
```

Use the recorded source and dependencies for reproduction; output paths must be
fresh. Bitwise reproducibility across devices or independently rebuilt wheels
is not established. From this result directory, `shasum -a 256 -c SHA256SUMS`
checks the published files.

## Next Mechanism And Execution Gates

This rejects the tested schedule, not every Z-Space mechanism. Diagnose how
Rust-owned candidate selection covers teacher-forced versus generated contexts
before another schedule sweep; do not train on these held-out prompts. New
learning candidates need a separately frozen comparison, including their extra
compute/token budget. Three seeds and a random same-book split do not establish
statistical significance, cross-book generalization or general language quality.

In parallel, connect resident causal attention with the existing Z-Space bias
semantics. Compare both plain attention and identically biased attention against
PyTorch, then measure the complete Q/K/V-to-output path including transfers.
Keep numerical correctness, execution performance and model-quality claims
separate. This CPU HF/PEFT study does not demonstrate native WGPU LLM execution
or a backend speedup; HF still owns the differentiable model and optimizer.
