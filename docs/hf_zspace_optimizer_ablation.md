# HF Z-Space Optimizer Ablations

This workflow asks a narrow question: when Z-Space changes a Hugging Face
Trainer learning-rate trajectory, is an observed loss difference caused by the
integrated learning-rate dose, by the time-varying shape, or by both?

It does not claim that three seeds or one short fine-tune establish general
model quality. It produces a matched, replayable diagnostic before a larger
study is justified.

## Semantic boundary

`st-core::runtime::zspace_optimizer` owns the trajectory contract. Given the
Rust-derived raw scales `r_i` and scheduler-owned nominal rates, it sets
`w_i = sum_g(lr_i,g)` across optimizer parameter groups and emits:

| Arm | Applied scale | Purpose |
| --- | --- | --- |
| `observe` | `1` | Baseline update plus calibration only |
| `dose_matched_constant` | `sum(w_i r_i) / sum(w_i)` | Raw integrated dose without its shape |
| `raw` | `r_i` | Original Z-Space control, dose and shape together |
| `dose_normalized` | `clamp(c r_i, min, max)` | Raw shape rescaled to baseline dose |

Rust solves `c` so the bounded normalized schedule satisfies
`sum(w_i clamp(c r_i, min, max)) = sum(w_i)`. The report includes the complete
step table, saturation counts, invariant residuals, expected non-identity update
counts, identity tolerances, and a SHA-256 trajectory identity. Python transports
the report, checks each live scheduler row and measured intervention count
against it, temporarily actuates `torch.optim.Optimizer.step`, and restores
nominal rates before the scheduler advances. A constant or dose-normalized arm
may legitimately be an identity control; the raw arm must still contain at least
one non-identity update for the factorized comparison to be ready.

If another Trainer callback stops a run early, SpiralTorch seals the realized
schedule instead of turning successful training into an exception. Generated
trajectories use the realized update count; a partially consumed calibrated
trajectory remains explicit `blocked` evidence and cannot enter the comparator.

This dose is the sum of parameter-group learning rates over optimizer updates.
It is not parameter-count weighted and does not claim to measure gradient norm,
parameter displacement, or useful learning by itself.

The separate `dose_preserving_complement` policy tests trajectory polarity
without changing that integrated dose. Rust computes the weighted raw center
`r_bar_w = sum(w_i r_i) / sum(w_i)`, then emits
`s_i = 1 - a (r_i - r_bar_w)`. It chooses the largest `a` in `[0, 1]` that
keeps every scale inside the shared `[0.1, 1.25]` bounds. Centering makes
`sum(w_i s_i) = sum(w_i)` by construction, while the sign reversal makes the
schedule weaker where the raw trajectory was above its center and stronger
where it was below. The policy and every step are owned, identified, and
revalidated by `st-core`; Python and WASM only transport the report.

## Run one matched seed

Run `observe` first. Keep every non-intervention argument identical across all
four runs, including model and tokenizer identity, corpus, selection, seed,
batching, optimizer, scheduler, precision, and step horizon.

The following abbreviated example assumes those shared arguments are in a
shell array. Use explicit values suitable for the model under test.

```bash
COMMON=(
  --model-name /path/to/local-model
  --tokenizer-name /path/to/local-model
  --train --train-file data/corpus.txt
  --finetune-mode lora --lora-rank 4 --lora-alpha 8
  --max-steps 16 --learning-rate 0.00005
  --eval-before-train --eval-after-train-policy always
  --seed 13
)

spiral-hf-finetune "${COMMON[@]}" \
  --output-dir runs/s13-observe \
  --run-card runs/s13-observe/run-card.json \
  --zspace-optimizer-control observe

TRAJECTORY=runs/s13-observe/spiraltorch-hf-zspace-optimizer-trajectory.json

spiral-hf-finetune "${COMMON[@]}" \
  --output-dir runs/s13-constant \
  --run-card runs/s13-constant/run-card.json \
  --zspace-optimizer-control apply \
  --zspace-optimizer-trajectory-arm dose_matched_constant \
  --zspace-optimizer-trajectory-json "$TRAJECTORY"

spiral-hf-finetune "${COMMON[@]}" \
  --output-dir runs/s13-raw \
  --run-card runs/s13-raw/run-card.json \
  --zspace-optimizer-control apply \
  --zspace-optimizer-trajectory-arm raw \
  --zspace-optimizer-trajectory-json "$TRAJECTORY"

spiral-hf-finetune "${COMMON[@]}" \
  --output-dir runs/s13-normalized \
  --run-card runs/s13-normalized/run-card.json \
  --zspace-optimizer-control apply \
  --zspace-optimizer-trajectory-arm dose_normalized \
  --zspace-optimizer-trajectory-json "$TRAJECTORY"
```

Calibrate independently for every seed. Do not reuse an `observe` artifact
after changing the scheduler, horizon, parameter groups, or control recipe.

## Compare run cards

```bash
spiral-hf-zspace-optimizer-factorized-compare \
  runs/s13-observe/run-card.json \
  runs/s13-constant/run-card.json \
  runs/s13-raw/run-card.json \
  runs/s13-normalized/run-card.json \
  --out runs/s13-factorized.json
```

Pass all four run cards for every seed in one invocation. Three matched seeds
allow the report to describe a directionally consistent bounded trend, not
statistical significance. Even a consistent improvement is reported separately
from `efficacy_claim_ready`, which remains false until a prespecified, powered,
multi-model evaluation exists.

## Run a resumable multi-seed study

The installed study runner generates the four commands for every seed, runs
`observe` before its three calibrated arms, and invokes the same comparator at
the end. First omit `--run` to create a recovery anchor without training:

```bash
spiral-hf-zspace-optimizer-factorized-study \
  --study-dir models/runs/zspace-factorized-64 \
  --seed 13 --seed 17 --seed 23 \
  --min-free-disk-gb 5 \
  -- \
  --model-name /path/to/local-model \
  --tokenizer-name /path/to/local-model \
  --train --train-file data/corpus.txt \
  --finetune-mode lora --lora-rank 4 --lora-alpha 8 \
  --max-steps 64 --learning-rate 0.00005 \
  --eval-before-train --eval-after-train-policy always
```

Inspect `study-plan.json`, then repeat the identical command with `--run` before
the separator. The plan is immutable inside one study directory and contains a
SHA-256 study identity over the scientific arguments, bridge content, package
source/native-extension fingerprint, available Git head/status, seeds, and arm
order. If the study directory is inside the repository, that generated subtree
is explicitly excluded from Git status so writing the plan does not change its
own recovery identity.

Each child completion is accepted only after the runner verifies its launch
command, seed and arm, before/after eval losses, complete optimizer horizon,
Rust trajectory identity, trainer and optimizer trace hashes, output directory,
and path-independent execution/runtime/input identities. Those identities must
also remain constant across every seed. The append-only `study-events.jsonl`
journal is fsynced before and after child execution, so a restart can recover a
child that finished immediately before the parent stopped without silently
adopting unrelated artifacts. `study-summary.json` records live progress and
`factorized-report.json` is written only after every verified arm is present.

If a failed attempt left artifacts that cannot be verified, the study fails
closed. `--retry-failed` preserves them under `quarantine/` before launching a
new attempt; it never overwrites them in place.

## Compare control gains

Run otherwise identical studies with an explicit
`--zspace-optimizer-control-gain`, then compare the completed study directories:

```bash
spiral-hf-zspace-optimizer-factorized-gain-compare \
  models/runs/zspace-factorized-gain-025 \
  models/runs/zspace-factorized-gain-050 \
  models/runs/zspace-factorized-gain-100 \
  --out models/runs/zspace-factorized-gain-response.json
```

The gain comparator verifies each plan identity, hash-chained completion
journal, factorized-report SHA-256, non-gain scientific arguments, seed set,
execution/runtime/input anchor, and exact observe losses. It rejects fewer than
three gains. The output reports per-contrast ordinary least-squares slope,
intercept, and `R²`; this is a descriptive response curve, not a significance
test.

### Audited 64-step result (2026-08-09)

The checked-in [gain-response artifact](benchmarks/hf_zspace_optimizer_gain_response_64step_20260809.json)
records one local GPT-2 LoRA run family on
`models/samples/spiral_corpus_en/06_spiral_longform.txt`: 64 optimizer updates,
seeds 13/17/23, CPU float32, and gains 0.25/0.5/1.0. Observe before/after losses
were exactly reproduced across all three studies.

| Contrast (left minus right) | gain 0.25 | gain 0.5 | gain 1.0 | `R²` |
| --- | ---: | ---: | ---: | ---: |
| raw minus observe | +0.000943 | +0.001855 | +0.003599 | 0.99988 |
| constant-dose minus observe | +0.000737 | +0.001457 | +0.002862 | 0.99996 |
| dose-normalized minus observe | +0.000217 | +0.000442 | +0.000902 | 0.99998 |
| raw minus constant-dose | +0.000206 | +0.000398 | +0.000737 | 0.99903 |

Lower validation loss is better, so every positive value favors the right arm.
For this single recipe the current progress-only, non-feedback control therefore
produced gain-correlated loss degradation. This is useful actuator diagnosis,
not evidence that Z-Space controls are generally harmful. It argues against
making this open-loop policy a default and motivates a separately ablated,
Rust-owned feedback gate before longer scale-up.

That gate is now available as the explicit
`--zspace-optimizer-feedback loss_guard` mode. It consumes the selected raw or
trajectory-arm proposal but keeps all loss projection, EMA, warmup, staleness,
halt/recovery state, and the final blend in
`st-core::runtime::zspace_optimizer_feedback`. Python and WASM expose the same
state-machine checkpoint/report contract; the HF receipt additionally seals
its observation and control lineage. It is deliberately not a fifth arm in
this frozen four-arm study: compare guarded and unguarded recipes in a new
matched multi-seed run, and retain the receipt's
`within_run_loss_guard_not_counterfactual_efficacy` boundary.

The frozen four-arm comparator fails closed unless each seed has exactly one
arm of every kind and all four share:

- the non-intervention training recipe and before-train loss anchor;
- training input, materialized dataset, tokenized blocks, model, and execution
  identities;
- one Rust trajectory, raw control sequence, and nominal scheduler sequence;
- complete optimizer actuation/restoration counts and a sealed actuation hash;
- measured nominal and effective-LR doses matching the selected Rust arm.

## Run the guarded feedback study

The feedback study freezes one Rust raw-control trajectory per seed and executes
three arms in this order:

| Arm | Raw trajectory | Loss guard | Question |
| --- | --- | --- | --- |
| `observe` | recorded, not applied | off | What does ordinary FT do? |
| `raw_unguarded` | applied | off | What is the total open-loop effect? |
| `raw_loss_guard` | applied | Rust `loss_guard` | Does the guard improve that same intervention? |

Create the immutable recovery anchor before training by omitting `--run`:

```bash
spiral-hf-zspace-optimizer-feedback-study \
  --study-dir models/runs/zspace-feedback-64 \
  --seed 13 --seed 17 --seed 23 \
  --min-free-disk-gb 5 \
  -- \
  --model-name /path/to/local-model \
  --tokenizer-name /path/to/local-model \
  --train --train-file data/corpus.txt \
  --validation-fraction 0.1 \
  --finetune-mode lora --lora-rank 4 --lora-alpha 8 \
  --max-steps 64 --learning-rate 0.00005 \
  --model-train-dtype float32 --training-use-cpu \
  --eval-before-train --eval-after-train-policy always
```

Inspect `feedback-study-plan.json`, then repeat the exact command with `--run`
before the separator. The runner owns `--seed`, output/run-card/trace paths,
trajectory transport, `--logging-steps 1`, feedback mode, and every feedback
parameter. Optional `--feedback-config-json config.json` accepts only a JSON
object; Rust validates the overrides and the plan seals the complete resolved
13-field config, contract owner, native-extension hash, and evidence boundary.

The final `feedback-report.json` exposes three lower-is-better contrasts:

- `unguarded_total_effect`: `raw_unguarded - observe`;
- `guarded_total_effect`: `raw_loss_guard - observe`;
- `guard_benefit`: `raw_loss_guard - raw_unguarded`.

The comparator additionally requires complete per-update loss/control lineage,
fail-closed warmup, no stale observations, a gate that actually became active,
and a guarded schedule distinct from the unguarded schedule. A short calibration
where the gate never actuates is preserved as useful `blocked` evidence rather
than being promoted into a vacuous comparison. Three consistent seeds permit a
bounded single-model, single-corpus trend only; `efficacy_claim_ready` remains
false by construction.

The study also requires tokenization to leave at least one evaluation block.
For a small corpus, verify that `validation_fraction * token_count` is not
smaller than `block_size`; otherwise the bridge stops at `dataset_fit` before
training rather than emitting an unanchored loss comparison.

### Audited feedback result (2026-08-09)

The checked-in [feedback artifact](benchmarks/hf_zspace_optimizer_feedback_readme_64step_20260809.json)
records one local GPT-2 LoRA run family over this repository's `README.md`: 64
optimizer updates, seeds 13/17/23, CPU float32, rank 4, alpha 8, batch size 2,
gradient accumulation 8, block size 128, 464 training blocks, and 16 capped
evaluation blocks. Each seed reused one frozen Rust raw-control trajectory
across all three arms.

| Seed | unguarded minus observe | guarded minus unguarded | guarded minus observe | active guard updates |
| ---: | ---: | ---: | ---: | ---: |
| 13 | +0.009914 | -0.008037 | +0.001878 | 39 / 64 |
| 17 | +0.008747 | -0.002035 | +0.006712 | 54 / 64 |
| 23 | +0.007768 | -0.004798 | +0.002970 | 49 / 64 |
| Mean | +0.008810 | -0.004957 | +0.003853 | 47.3 / 64 |

Lower validation loss is better. The open-loop raw arm was worse than ordinary
FT for 3/3 seeds, while the Rust loss guard improved that same intervention for
3/3 seeds. It recovered 56.26% of the mean unguarded harm, but the guarded arm
still lost to `observe` for 3/3 seeds. This is bounded evidence that the guard
mitigates this known failure mode, not evidence of absolute Z-Space efficacy or
statistical significance. The artifact therefore keeps
`efficacy_claim_ready=false`.

The frozen provenance anchors are:

- study ID `sha256:7558e02b75d2d696afa586af421e74328d8ec5f0e2f30c95f1742f1fdaf867e3`;
- original runner report SHA-256 `a5b5e8be0def36322737431ec3cc570815d0aeb99c5203886e5ff8bc924adcc3`;
- experiment Git commit `30513bf0532eb8f8ad118d362e6fac69de7c4096`;
- runtime source ID `sha256:ab4366caebae6808659177a60f81893571e1d6825a99513d39ccad5b825895e1`;
- native library SHA-256 `3981319b4d4d55c53c059b9a3cd7dc58884a9a79f8c9a1386645653131a8d908`.

## Run the dose-matched polarity study

The polarity runner executes `observe`, `dose_normalized`, and
`dose_preserving_complement` in that order for each seed. As with the other
study runners, omit `--run` first to write and inspect the immutable recovery
plan, then repeat the exact command with `--run`:

```bash
spiral-hf-zspace-optimizer-polarity-study \
  --study-dir models/runs/zspace-polarity-64 \
  --seed 13 --seed 17 --seed 23 \
  --min-free-disk-gb 5 \
  -- \
  --model-name /path/to/local-model \
  --tokenizer-name /path/to/local-model \
  --train --train-file README.md \
  --validation-fraction 0.1 \
  --finetune-mode lora --lora-rank 4 --lora-alpha 8 \
  --max-steps 64 --learning-rate 0.00005 \
  --model-train-dtype float32 --training-use-cpu \
  --eval-before-train --eval-after-train-policy always
```

The comparator requires all three arms to share one raw control sequence,
nominal scheduler sequence, Rust trajectory, before-train loss anchor, and
path-independent training identities. Both applied arms must match their Rust
planned dose and intervention count. The complement arm additionally requires
the policy contract, source trajectory, complete horizon, recipe identity, and
SHA-256-sealed policy artifact to agree.

### Audited polarity result (2026-08-10)

The checked-in [polarity artifact](benchmarks/hf_zspace_optimizer_polarity_readme_64step_20260810.json)
records one local GPT-2 LoRA run family over this repository's `README.md`: 64
optimizer updates, seeds 13/17/23, CPU float32, rank 4, alpha 8, batch size 2,
gradient accumulation 8, block size 128, 464 training blocks, and 16 capped
evaluation blocks. All nine run cards were ready and all three seeds shared the
same trajectory, raw control, nominal scheduler, model/runtime/input identities,
and exact before-train anchor within each seed.

| Seed | normalized minus observe | complement minus observe | complement minus normalized |
| ---: | ---: | ---: | ---: |
| 13 | +0.001717 | -0.000956 | -0.002672 |
| 17 | +0.001568 | -0.000861 | -0.002429 |
| 23 | +0.001608 | -0.000905 | -0.002513 |
| Mean | +0.001631 | -0.000907 | -0.002538 |

Lower validation loss is better. The original normalized shape was worse than
ordinary FT for 3/3 seeds. Its dose-preserving complement beat the normalized
shape for 3/3 and ordinary FT for 3/3. The Rust policy used weighted center
`0.9035117234`, maximal safe gain `0.5669527277`, first/last scales
`0.9013568801 / 1.25`, and exactly preserved nominal dose `0.00325`; every one
of its 64 updates was non-identity. This is bounded evidence for schedule
polarity in one short recipe, not statistical significance or general Z-Space
superiority, so `efficacy_claim_ready` remains false.

The frozen provenance anchors are:

- study ID `sha256:20bc9a3287cab5f6b5119034f8980a91f905221c3927b363e26edbae58451fdf`;
- original runner report SHA-256 `c63008dabb8699c2b5a5a45abdb9df6e4681e59fd7f83f29a3866716dd56e1a6`;
- experiment Git commit `4e3267bd1c97471e217a1552cd1f3f72f0e97ef3`;
- runtime source ID `sha256:bdb989da733086791143b7cea1a8001da41970d7938deb65e385b42b03ed45c3`;
- native library SHA-256 `9ccc426ed9807f09321c3d84aed1154a4313fbe0a61d10714021efe73546fc6f`;
- trajectory policy ID `sha256:b08588963b97bff33dc59c0bb0f9c830bc054aadc949b97be9324f1286b69553`.

## Cross-corpus polarity study

The single-corpus result above is a hypothesis anchor, not a generality result.
Use the corpus study runner to repeat the same three-arm, multi-seed protocol on
content-distinct local corpora without rebuilding aggregation semantics in
Python:

```bash
spiral-hf-zspace-optimizer-polarity-corpus-study \
  --study-dir models/runs/zspace-polarity-multicorpus-64 \
  --corpus fiction=data/dubliners.txt \
  --corpus psychology=data/psychology_of_the_unconscious_en.txt \
  --corpus encyclopedic=data/wiki_33.txt \
  --seed 13 --seed 17 --seed 23 \
  --min-free-disk-gb 5 \
  -- \
  --model-name /path/to/local-model \
  --tokenizer-name /path/to/local-model \
  --train --validation-fraction 0.1 \
  --finetune-mode lora --lora-rank 4 --lora-alpha 8 \
  --max-steps 64 --learning-rate 0.00005 \
  --model-train-dtype float32 --training-use-cpu \
  --eval-before-train --eval-after-train-policy always
```

Do not pass a dataset source after `--`; the runner owns one `--train-file` per
corpus. Run the command without `--run` to freeze and inspect all nested plans,
then repeat it exactly with `--run`. Recovery reuses the existing per-corpus
plan, hash-chained journal, run-card, trace, and policy receipts.

The final `polarity-corpus-report.json` is computed by the Rust
`st-core::runtime::zspace_evidence` contract. Rust requires the same seed set in
every corpus, recomputes `polarity = complement - normalized`, summarizes seeds
within each corpus, and only then gives every corpus mean equal weight. A
bounded trend is eligible only at three or more corpora and three or more seeds
per corpus, and only when every corpus mean has the same direction. The report
still sets `efficacy_claim_ready` to false: it is a corpus-level trend for one
model and recipe, not significance or general model superiority.

Existing completed studies can enter the same Rust contract independently:

```bash
spiral-hf-zspace-optimizer-polarity-corpus-compare \
  --study fiction=models/runs/polarity-fiction \
  --study psychology=models/runs/polarity-psychology \
  --study encyclopedic=models/runs/polarity-encyclopedic \
  --out models/runs/polarity-corpus-report.json
```

The comparator recomputes every source report from its run cards and verifies
the immutable plan, complete hash-chain, per-run card SHA-256, final report
SHA-256, runtime identity, trajectory, policy, control, and nominal schedule
before calling Rust. A mutable summary is status metadata only: its identity
anchor must exactly match the corpus, runtime, and execution identities
re-derived from those sealed run cards. The scientific protocol likewise uses
the path-independent execution/runtime identities, one content-addressed
training-recipe identity per seed and arm, and one path-independent
data-preparation identity reconstructed from sealed effective settings plus
materialization/tokenization receipt contracts. Raw model, tokenizer, and
corpus paths never enter that identity, and local source flag order is
canonicalized before hashing.
Per-study Git status and path-bearing receipts remain sealed audit and recovery
metadata, but do not define cross-study protocol equality; an earlier study's
output therefore cannot perturb a later standalone comparison. The report
identity hashes the Rust evidence ID, protocol and runtime identities, and
content-addressed corpus IDs in canonical order.

### Audited cross-corpus polarity result (2026-08-11)

The checked-in [multi-corpus polarity artifact](benchmarks/hf_zspace_optimizer_polarity_multicorpus_64step_20260811.json)
records 27 local GPT-2 LoRA runs: three content-distinct corpora, seeds
13/17/23, and the three matched polarity arms. Each run used 64 optimizer
updates, CPU float32, rank 4, alpha 8, batch size 2, gradient accumulation 8,
block size 128, and 16 capped evaluation blocks. The encyclopedic, fiction, and
psychology corpora supplied 1006, 383, and 212 training blocks respectively.

| Corpus | normalized minus observe | complement minus observe | complement minus normalized |
| --- | ---: | ---: | ---: |
| Encyclopedic | +0.000193 | -0.000104 | -0.000296 |
| Fiction | +0.001537 | -0.000892 | -0.002429 |
| Psychology | +0.001533 | -0.000877 | -0.002410 |
| Corpus-equal mean | +0.001087 | -0.000624 | -0.001712 |

Lower validation loss is better. The normalized shape was worse than ordinary
FT for 9/9 seeds and all 3/3 corpus means. The dose-preserving complement beat
ordinary FT for 9/9 and beat the normalized shape for 9/9; all three corpus
means agreed in both comparisons. The independent comparator regenerated the
same report byte-for-byte, and the Rust validator recomputed evidence ID
`sha256:ad2b604233dc2d23578c85d9f1e5fcb76026f077c085cd7c0621a6b8569381b8`.

This advances the earlier single-corpus observation to a balanced corpus-level
trend, but not to a general efficacy result. It is still one local GPT-2 model,
one short LoRA recipe, three corpora, and three seeds without a prespecified
power analysis or independent generation-quality endpoint. Accordingly,
`efficacy_claim_ready` remains false.

The frozen provenance anchors are:

- outer study ID `sha256:168487f38863898a1054587750fe6f8f5b18ccaea2c4be277a66791151b939ec`;
- path- and alias-independent protocol ID `sha256:39eb32b8f3b9b4039e6ad081e06457ac2c79b1e8352550f10732a3690a1a052f`;
- path-independent data-preparation ID `sha256:75f83e30bbd86776fee86b91cda4bf36624282cbbc880c1bb8df450a1bbe982f`;
- path- and alias-independent outer report ID `sha256:5800ac0e07d81dca84bb7528519cac7d05207ea42426b49968f6b5495e786f08`;
- stable outer report SHA-256 `e822f1030c2b447c34ad1d32af008b6afe48f3607386bef26dbc562b72e4c10e`;
- experiment Git commit `bedf582863abc60f72a6957163ea34cb75151ff1`;
- stable aggregation Git commit `c267de02f3612b5df07aedb43c6d575bd7a09906`;
- runtime source ID `sha256:ce31b284d9b88c15d6083ef5e16cbbb167d1df18684705dd99b3260b1a7ccbc2`;
- native library SHA-256 `7a5914318c835bc216a283f7f223e9c1a84b38707a8358e7c8607c61a11eafe1`.

## Pythia-70M model-family-transfer pilot

The next step changes model family without promoting the prior GPT-2 trend to
an efficacy claim. The checked-in [pilot protocol](benchmarks/hf_zspace_pythia70m_polarity_pilot_protocol_v1.json)
freezes one Pythia-70M snapshot, the fiction corpus, three seeds, three polarity
arms, and an independent eight-prompt generation endpoint. The generation
endpoint uses the Rust-owned token-sequence evidence contract and deliberately
disables inference-time Z-Space control, so it measures the trained adapters
rather than a mixture of training and decoding interventions.

The protocol pins generation-evidence behavior to commit
`32b552670bde887102420d340a7769d47e451e89`. A later checkout is acceptable
only when the functional paths are byte-identical; the study plan still seals
the actual Git head and status. Verify the model and corpus hashes in the
protocol before planning.

```bash
MODEL_CONFIGS="$PWD/bindings/st-py/examples/hf_finetune_model_configs.example.json"
PYTHIA_SNAPSHOT="/Users/ryospiralarchitect/.hf_home/hub/models--EleutherAI--pythia-70m-deduped/snapshots/e93a9faa9c77e5d09219f6c868bfc7a1bd65593c"
CORPUS="/Users/ryospiralarchitect/SpiralReality/data/dubliners.txt"
PROMPTS="$PWD/bindings/st-py/examples/hf_generation_evidence_pythia70m_pilot_prompts.json"
STUDY_DIR="$PWD/models/runs/hf-zspace-pythia70m-polarity-pilot-v1"

git diff --exit-code 32b552670bde887102420d340a7769d47e451e89 -- \
  crates/st-core/src/runtime/zspace_generation_evidence.rs \
  bindings/st-py/src/zspace_generation_evidence.rs \
  bindings/st-py/spiraltorch/generation_evidence.py \
  bindings/st-py/examples/hf_gpt2_zspace_generation_control_sweep.py \
  bindings/st-py/examples/hf_generation_evidence_pythia70m_pilot_prompts.json

STUDY_ARGS=(
  --study-dir "$STUDY_DIR"
  --seed 13 --seed 17 --seed 23
  --min-free-disk-gb 5
)

BRIDGE_ARGS=(
  --
  --model-configs "$MODEL_CONFIGS"
  --model-profile pythia-70m-local-smoke
  --model-name "$PYTHIA_SNAPSHOT"
  --tokenizer-name "$PYTHIA_SNAPSHOT"
  --train-file "$CORPUS"
  --validation-fraction 0.1
  --train
  --finetune-mode lora
  --lora-rank 4
  --lora-alpha 8
  --lora-dropout 0.05
  --lora-bias none
  --lora-target-module query_key_value
  --lora-target-module dense
  --max-steps 64
  --learning-rate 0.00005
  --model-train-dtype float32
  --training-use-cpu
  --per-device-train-batch-size 2
  --per-device-eval-batch-size 2
  --gradient-accumulation-steps 8
  --logging-steps 8
  --block-size 128
  --max-eval-blocks 16
  --save-steps 250
  --eval-steps 250
  --save-total-limit 1
  --dataloader-num-workers 0
  --dataloader-pin-memory false
  --eval-before-train
  --eval-after-train-policy always
)

# Freeze and inspect the immutable nine-run plan first.
spiral-hf-zspace-optimizer-polarity-study \
  "${STUDY_ARGS[@]}" "${BRIDGE_ARGS[@]}"

# Execute only after study-plan.json and all input hashes are accepted.
spiral-hf-zspace-optimizer-polarity-study \
  "${STUDY_ARGS[@]}" --run "${BRIDGE_ARGS[@]}"
```

After all nine run cards are ready, collect one greedy, baseline-only generation
report per adapter. Reusing the training seed makes the decoding identity
matched across the three arms within each seed; greedy decoding itself remains
deterministic.

```bash
for SEED in 13 17 23; do
  for ARM in observe dose_normalized dose_preserving_complement; do
    RUN_DIR="$STUDY_DIR/runs/s${SEED}/${ARM}"
    spiral-hf-zspace-generation-control-sweep \
      --model-configs "$MODEL_CONFIGS" \
      --model-profile pythia-70m-local-smoke \
      --model-name "$RUN_DIR/output" \
      --tokenizer-name "$PYTHIA_SNAPSHOT" \
      --model-artifact-kind peft-adapter \
      --prompt-set "$PROMPTS" \
      --baseline-only \
      --no-do-sample \
      --max-new-tokens 96 \
      --seed "$SEED" \
      --out "$RUN_DIR/generation-evidence.json"
  done
done
```

Every report must have status `complete`, exactly one baseline run, the fixed
prompt-set identity, and a report that round-trips through
`validate_zspace_generation_evidence()`. These nine pilot rows demonstrate
cross-family execution only; they are explicitly excluded from a later
prespecified efficacy analysis.

```bash
STUDY_DIR="$STUDY_DIR" python3 - <<'PY'
import json
import os
from collections import defaultdict
from pathlib import Path

import spiraltorch as st

root = Path(os.environ["STUDY_DIR"])
paths = sorted(root.glob("runs/s*/*/generation-evidence.json"))
assert len(paths) == 9, f"expected 9 generation reports, found {len(paths)}"

protocol_ids = set()
prompt_set_ids = set()
runtime_ids = set()
decoding_ids_by_seed = defaultdict(set)
for path in paths:
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["status"] == "complete" and report["run_count"] == 1
    row = report["runs"][0]
    assert row["status"] == "ok"
    evidence = row["generation_evidence"]
    assert st.validate_zspace_generation_evidence(evidence) == evidence
    request = evidence["request"]
    seed = int(path.parents[1].name.removeprefix("s"))
    protocol_ids.add(request["protocol_id"])
    prompt_set_ids.add(request["prompt_set_id"])
    runtime_ids.add(request["runtime_identity_id"])
    decoding_ids_by_seed[seed].add(request["decoding_config_id"])

assert len(protocol_ids) == 1
assert prompt_set_ids == {
    "sha256:4a5029e46c21915c4982c7eb005978ff7759cb04901c49f7d0e9f4b6aaaf94ed"
}
assert len(runtime_ids) == 1
assert set(decoding_ids_by_seed) == {13, 17, 23}
assert all(len(ids) == 1 for ids in decoding_ids_by_seed.values())
print("pythia70m_generation_evidence_ready", len(paths))
PY
```

## Read the contrasts

All contrasts use `left_arm loss change - right_arm loss change`; lower is
better.

- `dose_effect`: constant-dose arm minus observe.
- `shape_effect_at_raw_dose`: raw minus constant-dose.
- `dose_normalized_shape_effect`: normalized-shape minus observe.
- `raw_total_effect`: raw minus observe.

A positive `dose_normalized_shape_effect` means this particular normalized
shape was worse than its matched baseline. It does not establish that Z-Space,
other controls, longer horizons, or other models are worse. Treat it as a
calibrated signal for the next control design.

## Direct Python API

The trajectory operation is available without Trainer orchestration:

```python
import spiraltorch as st

trajectory = st.zspace_parameter_trajectory(
    raw_learning_rate_scales=[0.8, 1.1, 0.7],
    nominal_learning_rates=[[5e-5], [3e-5], [1e-5]],
)
assert trajectory["semantic_backend"] == "rust"

validated = st.validate_zspace_parameter_trajectory(trajectory)
assert validated["trajectory_id"] == trajectory["trajectory_id"]

policy = st.zspace_parameter_trajectory_policy(
    trajectory,
    policy="dose_preserving_complement",
)
assert policy["semantic_backend"] == "rust"
assert policy["planned_dose_ratio"] == 1.0
assert st.validate_zspace_parameter_trajectory_policy(policy) == policy
```

Changing any serialized report field causes Rust validation to reject the
artifact rather than accepting Python-reconstructed semantics.

## Resume compatibility

New checkpoints use optimizer-state contract v2 and retain every scheduler row
needed to reproduce the trajectory. A raw-control v1 checkpoint still resumes,
but v1 did not store historical nominal/effective LR rows. SpiralTorch records
that missing prefix explicitly, preserves optimizer actuation, and leaves the
trajectory and integrated-dose receipt fields unavailable instead of inventing
history. Such a migrated run cannot enter the four-arm comparator.
