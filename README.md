
# 🌀🕯️SpiralTorch🕯️🌀
trains where PyTorch can’t — inside the Z-space.
<p align="center">
  <img src="https://img.shields.io/badge/Rust-first-orange.svg" alt="Rust first">
  <img src="https://img.shields.io/badge/WGPU-supported-blueviolet.svg" alt="WGPU supported">
  <img src="https://img.shields.io/badge/MPS-ready-brightgreen.svg" alt="MPS ready">
  <img src="https://img.shields.io/badge/CUDA-enabled-lightblue.svg" alt="CUDA enabled">
  <img src="https://img.shields.io/badge/License-AGPL--3.0-blue.svg" alt="AGPL-3.0">
</p>
<p align="center">
  <b>SpiralTorch — a Rust-first learning framework for Z-space.<br>
  Runs natively on WGPU · MPS · CUDA · CPU.</b>
</p>

- SpiralTorch — Pure Rust AI core for Z-space exploration.**
- © 2025 Ryo ∴ SpiralArchitect — Licensed under AGPL-3.0-or-later.  
- Contact:(https://github.com/RyoSpiralArchitect/SpiralTorch/discussions) or kishkavsesvit@icloud.com
- Unauthorized derivations = non-compliant with AGPL §13.
- **For research collaborations or integration inquiries, please reach out directly.**
  
SpiralTorch is a Compact. Safe. Rust-native.
~10× smaller than PyTorch, yet feature-complete in AI training that keeps language,
geometry, and device heuristics in the same conversation. SpiralK orchestrates
the kernels, the hypergrad tape streams Z-space meaning, and the high-level
`st-nn` modules stay PyTorch-compatible without shipping NumPy or PyTorch.

The stack is comfortable living entirely in Rust—yet the Python wheel remains a
thin veneer that reuses the same planners, losses, and Z-space resonators. No
tensor shims, no translation layers, and no tracebacks.

```python
import spiraltorch as st
sess = st.SpiralSession(device="wgpu", curvature=-1.0, hyper_learning_rate=0.05)
sess.align_hypergrad(sess.hypergrad(1, 2), sess.barycenter([st.Tensor(1, 2, [0.7, 0.3])]))
```

**SpiralTorch is a Rust-first AI training framework** that keeps language,
geometry, and device heuristics in the same conversation. SpiralK orchestrates
the kernels, the hypergrad tape streams Z-space meaning, and the high-level
`st-nn` modules stay PyTorch-compatible without shipping NumPy or PyTorch.

The stack is comfortable living entirely in Rust—yet the Python wheel remains a
thin veneer that reuses the same planners, losses, and Z-space resonators. No
tensor shims, no translation layers, and no tracebacks.

## Why it’s different
 - **Training comes first:** Modules such as `Linear`, `Sequential`,
   `WaveGate`, the new `ToposResonator`, and `ZSpaceProjector` stream gradients
    into the hypergrad tape and expose a `train_epoch` loop that mirrors
    familiar `nn.Module` patterns.
  - **Open Z-space:** Gradient splits honour the A/B/C roundtable through the
    new `zspace_round` ops module so Above/Here/Beneath bands stay in sync with
    SpiralK plans without auxiliary buffers.
  - **Three-voice consensus:** SpiralK heuristics, DSL directives, and the
    generated WASM tuner table discuss every launch decision and keep the
    transcript in the roundtable log.
  - **Rust by default, Python ready:** Every feature—from WASM tuning to
    hypergrad curvature—is implemented in Rust and exposed unchanged through the
    Python bindings when needed.

---

## Hello SpiralSession quickstart

Kick the tires with the new end-to-end `hello_session` walkthrough. It seeds a
session, computes a barycenter, aligns a hypergrad tape, and runs a
session-level multi-epoch roundtable loop that restores the best row-weighted
checkpoint over a toy dataset.

```bash
cargo run -p st-nn --example hello_session
cargo run -p st-nn --example finetune_contract
cargo run -p st-nn --example byte_lm_finetune_contract
cargo run -p st-nn --example byte_lm_padded_contract
cargo run -p st-nn --example byte_lm_mlp_fit_contract
cargo run -p st-nn --example byte_lm_lora_adapter_contract
cargo run -p st-nn --example byte_lm_mlp_lora_adapter_contract
cargo run -p st-nn --example checkpoint_preflight_contract
cargo run -p st-nn --example byte_lm_replay_sweep_contract
cargo run -p st-nn --example byte_lm_zspace_compare_contract
```

`byte_lm_finetune_contract` is a tokenizerless next-byte smoke: it trains
windowed byte rows with sparse class-id targets, reloads a checked checkpoint,
then fine-tunes a head-only path through a source-retention guard while
auditing frozen/trainable movement and reporting micro-batches, optimizer
steps, active-row accuracy/perplexity, guarded best epoch, and source-retention
deltas after target fine-tuning. The guard is sparse-metric aware for the
byte-LM path, so validation-best selection can reject epochs that improve target
loss but drop source top-1 accuracy or spike source perplexity beyond the
configured ceiling.
`byte_lm_padded_contract` extends that path to variable-length byte spans padded
with `ignore_index=-1`, proving trainer histories count active target rows
instead of padded rows while also reporting active-row accuracy and perplexity.
`byte_lm_mlp_fit_contract` goes one step further with a tiny Rust-native
`Linear -> Relu -> Linear` byte MLP, gradient accumulation, validation-best
checkpoint restore, and requires row-weighted CE loss to improve across source
training and head-only fine-tuning while reporting active-row accuracy and
perplexity plus source-retention deltas. The byte-LM contracts validate their
tokenizerless samples before training, rejecting malformed one-hot rows,
out-of-range sparse byte targets, and all-padding batches before they can enter
the trainer.
`byte_lm_lora_adapter_contract` starts from a trained dense `Linear` checkpoint,
loads only its base weights into `LoraLinear`, then fine-tunes the low-rank
adapter while proving the dense base stayed frozen, adapter parameters moved,
and the FT-ready resume fingerprint still matches. It now emits the same
external-key preflight rows used by the standalone checkpoint contract before
the adapter base load, so non-scratch FT failures surface before training.
`byte_lm_mlp_lora_adapter_contract` lifts that path into a tiny
`Linear -> Relu -> LoraLinear` stack: a dense MLP checkpoint seeds the frozen
byte embedding and frozen head base, then only the LoRA head delta moves under
the same sparse source-retention audit. Its embed/head handoffs also print
per-entry source key, transform, and shape audit rows at the FT boundary.
`checkpoint_preflight_contract` isolates the external-checkpoint handoff: it
maps foreign keys into SpiralTorch module keys, applies transpose and
copy-overlap-zero transforms, prints per-entry source/shape/transform audit
rows, and only then loads the compatible embed/head subset. The Python
preflight helper mirrors this for `st.Tensor`, PyTorch-/NumPy-like arrays, and
plain 1D/2D sequences before adapter FT begins. It now also runs an
HF/PyTorch-style `transformer.wte.*` plus `lm_head.*` case so real LLM
state-dict imports can reuse the same transpose and overlap-copy audit path,
and the Python MLP adapter smokes seed their frozen embed/head base through the
same HF-style handoff helper. Its flat JSONL rows can also be compared before
FT, turning source-key, transform, hash, and shape drift into an early gate.
HF checkpoints that omit embedding or LM-head bias can be made explicit with
zero-row bias synthesis before preflight, keeping those loaded bias values
visible in the same audit trail instead of falling back to initializer state.
Python preflight helpers include GPT-2-, bare GPT-2-, LLaMA-, Gemma-wrapper-,
and GPT-NeoX-style key presets so common `transformer.wte.*`, `wte.*`,
`model.embed_tokens.*`, `model.language_model.embed_tokens.*`,
`gpt_neox.embed_in.*`, and `embed_out.*` layouts can share the same audit/load
path. External
HF/PyTorch state dicts can be materialized through only the configured
embed/head keys plus chosen audit extras, avoiding accidental conversion of an
entire large checkpoint just to preflight an adapter boundary. Python LoRA
adapter smokes can run with those presets via `--key-preset`, and checkpoint
summary gates track the selected preset alongside the key/transform hashes.
`write_byte_lm_hf_state_dict.py` can write a deterministic byte-compatible
`pytorch_model.bin` or train the tiny dense byte MLP first and externalize that
source checkpoint, giving the file-backed preflight and adapter paths a small
local checkpoint before pointing them at a larger external model. That file can
also feed the multi-case LoRA sweep with aggregate accepted/movement gates, so
the file-backed path is checked before graduating to real HF shards.
The standalone Python preflight can also point at a local `.safetensors`,
`pytorch_model.bin`, `.pt`, or indexed HF shard directory with `--hf-state-dict`
so real external checkpoint surfaces hit the same audit path before FT. It
can run `--shape-only --key-preset auto` first to detect the HF key layout and
log key presence, embed/head dimensions, byte-smoke resize requirements, and
projection policy without constructing SpiralTorch modules. If a checkpoint has
tied output embeddings and omits `lm_head.weight`, the audit marks
`lm_head_weight_synthesized_from_embed=true` and uses the embedding tensor as
the LM-head source for the bounded smoke path. Shape-only gates such as
`--require-shape-materializable`, `--require-exact-shape-match`, and
`--require-detected-key-preset` can turn unsafe or unexpected checkpoint
surfaces into an immediate failure. The normal preflight, adapter, and sweep
entrypoints also accept `--key-preset auto`; combined with
`--allow-overlap-resize` and `--checkpoint-projection-preset healthy`, this
creates the current real-HF smoke path from large indexed shards into the tiny
byte LoRA harness. Adapter and sweep rows can also carry
`--checkpoint-source-label`, keeping the human experiment label separate from
the resolved HF key preset when several local checkpoints share a layout.
`--checkpoint-source-gain` can deliberately increase or decrease mapped
embed/head checkpoint amplitude after projection and before module load, giving
large real-HF overlaps a controlled way to trade absolute FT movement against
target-vs-retention selectivity. It accepts the same
`--checkpoint-projection zspace` policy knobs as the adapter smoke, and writes
that projection policy into the flat preflight JSONL rows so
`--require-preflight-match` can catch geometry-policy drift before training.
The Python MLP LoRA adapter smoke can use the same file-backed handoff as its
frozen source via `--hf-state-dict` when the checkpoint has byte-smoke compatible
dimensions. Larger real LLMs should first pass the standalone shape audit, then
use explicit `--allow-overlap-resize` for a deliberate overlap-copy/zero-fill
projection into the byte harness; the audit keeps both the original external
shape and the adapted source shape visible. It can also apply
`--checkpoint-projection zspace` directly to the file-backed embed/head tensors
before adapter load; those projection strength/curvature/frequency values are
recorded in the same checkpoint gate, and transposed tensors are projected in
the target load orientation before being returned to their external layout.
`--checkpoint-projection-preset healthy` reuses the current projection-health
candidate (`strength=1.0`, `curvature=-0.04`, `frequency=0.65`) on both the
single MLP LoRA adapter and checkpoint-preflight entry points. The
companion
`byte_lm_handoff_strategy_compare.py` example compares exact, overlap-resized,
and Z-space-projected HF handoffs under the same LoRA/retention-guard gate
before moving those checkpoint policies into larger adapter experiments. The
Z-space handoff can sweep projection strength, curvature, and language-wave
frequency from the CLI, including comma-separated mini grids whose geometry
points become distinct strategy rows. `--zspace-preset healthy` carries the
current projection-health candidate (`strength=1.0`, `curvature=-0.04`,
`frequency=0.65`) into that handoff surface. It can also repeat the same policies
across small target cases keyed as `case::strategy`, and those geometry knobs
are included in the checkpoint audit gate. Its aggregate JSONL mode promotes
those rows into case-averaged strategy summaries, so Z-space handoff wins can be
gated on average behavior instead of a single strongest case; those strategy
aggregates reject duplicate cases and inconsistent case-label counts before
comparison. They also expose accepted/movement coverage counts and rates, so
handoff strategies can be gated on case coverage with `--min-aggregate-cases`
and repeated `--require-aggregate-case` before entering larger adapter sweeps.
The MLP LoRA
rank/alpha/LR sweep can also run with the same checkpoint projection policy, so
adapter-capacity comparisons stay tied to the same audited geometry-projected
source. That sweep can now start from a local HF/PyTorch state dict too, letting
file-backed checkpoints go straight into adapter-capacity comparisons without a
synthetic dense-source pretrain. It also accepts small checkpoint-projection
grids, producing `config::zspace_s...` rows that compare geometry knobs and
adapter capacity in one gateable JSONL surface; the
`--checkpoint-projection-preset healthy` shortcut reuses the same health
candidate without retyping the geometry knobs. It can also sweep
comma-separated `--checkpoint-source-gains`, producing
`config::...::gain_g...` rows so real-HF injection strength is ranked under the
same case coverage and retention gates as projection and adapter-capacity
candidates. `--adapter-weight-decays` now adds decoupled adapter-decay lanes
such as `config::wd0p01`, letting FT sweeps compare no-decay and mild-decay
adapter policies without changing the default smoke baseline. The same policy
surface can sweep `--max-grad-norms` and
`--gradient-accumulation-steps-list`, producing suffixes such as
`config::gn1p5::accum4` so clipping and accumulation choices are tracked beside
the adapter, projection, and checkpoint-gain settings. Longer-run FT controls
can also be swept with `--ft-epochs-list`, `--target-min-loss-deltas`,
`--patiences`, and `--lr-decay-patiences`/`--lr-decay-factors`, producing
suffixes such as `config::ep6::tmin1em06::pat3::ldp2::ldf0p8` while recording
early-stop and LR-decay outcomes in flat and aggregate JSONL rows. Flat and
aggregate rows also include `training_policy_key`, a stable grouping key for
the adapter-decay/clipping/accumulation/FT-control lane; the single MLP LoRA
adapter summary row emits the same key.
When only a few named FT-control policies should run, pass repeated
`--ft-control-jsonl` files with `byte_lm_ft_control` rows such as
`{"label":"strong_ep6","ft_epochs":6,"early_stopping_patience":3,"lr_decay_patience":2,"lr_decay_factor":0.8}`;
those sparse lanes avoid the cartesian FT-control grid while still flowing into
the aggregate, source-compare, profile, and promotion surfaces.
Sparse adapter-capacity lanes can likewise be supplied with repeated
`--lora-config-jsonl` files using `byte_lm_lora_config` rows such as
`{"label":"r18_a96_lr4p5","rank":18,"alpha":96,"adapter_lr_scale":4.5}`;
the built-in `r6_a32_lr3`/`r12_a64_lr4` configs stay unchanged, while heavier
promotion passes can be added without patching the sweep. The example
`bindings/st-py/examples/byte_lm_mlp_lora_capacity_lanes.jsonl` carries the
current bounded capacity probes (`r18_a96_lr4p5`, `r18_a128_lr5`,
`r24_a96_lr4p5`, and `r24_a128_lr5`) for promotion-ladder runs.
After those ladder runs emit aggregate JSONL, feed the aggregate files back into
`byte_lm_mlp_lora_source_compare.py --profile-jsonl ...` and then materialize
with `byte_lm_mlp_lora_profile_runner.py --lora-config-jsonl
bindings/st-py/examples/byte_lm_mlp_lora_capacity_lanes.jsonl` so the current
best capacity lane becomes a reusable profile rather than a one-off promotion
chain. When those aggregates will be profiled or materialized later, keep
`--checkpoint-source-label` on the original sweep and pass the same external
LoRA/FT-control JSONL files back to the profile runner so labels, source
identity, and training policy remain reproducible.
The same sweep can now repeat
those config/projection rows across multiple target corpora and emit
case-averaged `config_aggregate` JSONL rows, making adapter-capacity wins
gateable on average FT behavior instead of only a single lucky target. Weak
cases can also be supplied with `--case-jsonl` rows shaped like
`{"row_type":"byte_lm_case","label":"long_ft_probe","source_docs":[...],"target_docs":[...]}`,
then selected with repeated `--case` and pinned with
`--require-aggregate-case`, so longer tokenizerless corpora can enter the same
promotion path without patching the sweep. Weak
cases that produce no accepted improving config remain in those JSONL surfaces,
so transfer failures become inspectable data instead of disappearing mid-sweep;
aggregate gates also fail on case-scope drift so missing weak cases cannot make
the mean look healthier, and duplicate case rows are rejected before aggregation
so one corpus cannot be accidentally overweighted. Aggregate rows expose
target/retention loss margins, retention accuracy/perplexity margin means and
minimums, and accepted/movement coverage counts; `--min-aggregate-cases`,
`--require-aggregate-case`, `--min-aggregate-accepted-rate`,
`--min-aggregate-movement-ok-rate`, and `--require-aggregate-accepted-all` can
turn missing cases or weak coverage into a current-run failure, while
`--max-aggregate-*-rate-regression` gates coverage drops against saved aggregate
baselines. Aggregate compare also rejects inconsistent case-label counts,
duplicate labels, count sums, and count/rate mismatches before computing means.
`byte_lm_mlp_lora_source_compare.py` can ingest several real-HF
`config_aggregate` files and rank checkpoint sources on the same gateable JSONL
surface, reporting both absolute `target_loss_delta_mean` and
target-vs-retention selectivity (`target_retention_gap_mean` and
`target_retention_ratio`) while preserving `checkpoint_source_gain`,
adapter-weight-decay, trainer clipping/accumulation, and FT control policy. It
can use the same retention accuracy/perplexity margin floors as the aggregate
sweep gate, so gain/source winners are not selected only by large target
movement, and it can require a shared training-policy scope before ranking
source candidates. Ranked candidate/profile rows also carry a
`training_policy_key`, so dashboards can group source winners by the exact
adapter-decay/clipping/accumulation/FT-control lane. Use
`--min-profile-target-retention-ratio` when weaker comparison candidates should
remain visible but the selected profile winner must still clear a selectivity
floor. With
`--profile-jsonl`, it also emits reusable
`checkpoint_source_profile` rows for strong-effect, selective-gap, and
selective-ratio source/gain/decay/training-policy/FT-control lanes. The
companion `byte_lm_mlp_lora_profile_runner.py` turns those rows plus local
source-path mappings back into guarded sweep commands, so the bounded smoke path
has a reusable source comparison layer before graduating the same checks into
heavier FT runs. For a one-command local wiring check before a heavier FT pass,
run `python bindings/st-py/examples/byte_lm_profile_smoke.py`; it writes a tiny
byte-compatible HF state dict, sweeps it, emits a source profile, runs that
profile lane, and self-compares the resulting run summary through the
guard-aware promotion gates. It then feeds promotion JSONL back into the profile
runner and runs promoted follow-up rungs, proving that `input_promotion_*`
provenance and promotion-metric regression gates are wired before a heavier FT
pass. The promoted ladder defaults to one rung at one more FT epoch than the
initial diagnostic run; use `--promoted-ft-epochs` to choose the first promoted
rung explicitly, `--promoted-rungs` to climb multiple rungs, and
`--promoted-ft-epochs-step` to choose the increment between rungs. Each completed
rung is recorded in `promoted-rungs.jsonl` (or `--promoted-rungs-jsonl`) with its
input promotion, command, summary, and next-promotion artifact paths, so partial
ladder runs can be inspected or resumed deliberately. The smoke also writes a
top-level `profile-smoke-manifest.jsonl` (or `--manifest-jsonl`) that records
the checkpoint audit, sweep, profile, promotion, and final promoted-rung artifact
paths for downstream launchers. Use `--continue-manifest-jsonl` to add more
promoted rungs from that manifest without rerunning checkpoint preflight, sweep,
or source compare; continuation validates the recorded promoted-rung chain
before launching the next rung, and `--continue-plan-jsonl` can record the
planned rung artifact/epoch chain before child commands run. Use
`--validate-manifest-jsonl` when a long
experiment should first check the saved artifact paths and promotion chain
without launching another rung, and add `--manifest-validation-jsonl` when that
gate should also leave a machine-readable validation row. Pass `--hf-state-dict`, `--key-preset auto`, and usually
`--allow-overlap-resize` when the same bounded profile ladder should start from a
local real-HF checkpoint instead of the tiny fixture; the smoke first records
`checkpoint-shape-audit.jsonl` and `checkpoint-preflight.jsonl`, then forwards
the same checkpoint policy into the sweep and promoted rungs. Add
`--compare-checkpoint-preflight-jsonl` plus `--require-checkpoint-preflight-match`
when a saved checkpoint audit should gate the run before any FT work starts. Add
`--skip-promoted-follow-up` when only the first profile run is needed. The smoke
disables strict aggregate acceptance gates by default so diagnostic
guard-rejected lanes can still exercise the
profile/promotion plumbing; add `--strict-aggregate-gates` when that local check
must behave like the real acceptance gate.
Profile output paths include the selected or override
config/policy slug, so projection/gain/decay/clipping/accumulation lanes do not
collide, and FT-control suffixes travel with the selected or overridden base
config. Its
run-summary JSONL can then pin each profile lane to the
aggregate metrics produced by that execution; aggregate and command
`training_policy_key` values must agree when both are present. Later runs can
compare against that file with profile-level regression, margin, source/config,
training-policy, and input-promotion provenance match gates; profile run-summary
comparison keys rows by profile plus aggregate config so multiple override
configs can remain separate lanes. Add `--require-run-case-scope-match` when those lanes must also preserve
their exact `cases` and `case_labels`; the training-policy gate pins the stable
`training_policy_key`, while `--require-run-input-promotion-match` pins the
`input_promotion_*` rank/metric/value provenance so stale promoted summaries
fail loudly. Use `--min-run-target-retention-ratio` when profile runs should
keep a current-run target-vs-retention selectivity floor even before a baseline
exists, and pair it with `--min-run-accepted-rate` /
`--min-run-movement-ok-rate` when every profile lane must keep full case coverage
before promotion. Add `--require-run-guard-counts-available`,
`--min-run-guard-acceptance-rate-mean`,
`--max-run-guard-retention-rejected-epochs-mean`,
`--max-run-guard-retention-rejected-rate-mean`,
`--max-run-guard-target-stale-epochs-mean`, and
`--max-run-guard-target-stale-rate-mean` when profile summaries must prove that
their sparse-retention guard decisions came from exact backend diagnostics and
stayed within bounded rejected/stale epoch and rate budgets. When comparing to a
saved run summary, add `--max-run-guard-acceptance-rate-regression`,
`--max-run-guard-retention-rejected-rate-regression`, and
`--max-run-guard-target-stale-rate-regression` to catch acceptance-rate drops or
rejected/stale-rate increases before promotion. Add
`--promotion-jsonl`
after a run-summary compare or current-run gate to write
`checkpoint_source_profile_promotion` rows ranked by `--promotion-metric`
(default `target_retention_ratio`); tied best rows all remain promotion-ready so
follow-up FT runs can keep honest alternatives visible, and
`--promotion-ready-top-k` / `--promotion-ready-within` can deliberately widen
the promoted candidate set for heavier exploratory FT. Promotion-ready floors
such as `--promotion-ready-min-target-retention-ratio`,
`--promotion-ready-min-accepted-rate`, and
`--promotion-ready-min-movement-ok-rate`, plus
`--promotion-ready-max-input-promotion-metric-regression` for promoted chains,
and the guard-aware `--promotion-ready-require-guard-counts-available`,
`--promotion-ready-min-guard-acceptance-rate-mean`,
`--promotion-ready-max-guard-retention-rejected-epochs-mean`,
`--promotion-ready-max-guard-retention-rejected-rate-mean`,
`--promotion-ready-max-guard-target-stale-epochs-mean`, and
`--promotion-ready-max-guard-target-stale-rate-mean` guard gates, keep weak
candidates in the ranked JSONL for diagnosis while excluding them from
the default ready-only follow-up run; if every selected row is non-ready, the
follow-up materializer stops with the non-ready floor reasons instead of
launching an empty or stale FT pass. Add
`--min-promotion-ready-count` or `--min-promotion-ready-rate` when the promotion
file itself or a later materialize step should fail CI unless enough ready
candidates remain after floors; pair them with
`--min-promotion-ready-guard-policy-count` or
`--require-promotion-ready-guard-policy` when every launched ready lane must come
from a guard-aware promotion policy.
Feed those rows back
with `--promotion-input-jsonl` alongside the original profile JSONL to
materialize only promotion-ready profile commands for the next, heavier FT pass;
source, selected config, case scope, and training policy are required and
checked against the profile rows so stale promotions fail before launching. If
the profile was produced from external `--case-jsonl` corpora, pass the same
`--case-jsonl` path to the profile runner so promoted commands can resolve those
case labels without falling back to the built-in smoke set. If the next pass uses
external adapter-capacity labels, pass the same `--lora-config-jsonl` path so
promoted commands can resolve those labels too. Add
`--override-ft-epochs` and the companion `--override-target-min-loss-delta`,
`--override-patience`, `--override-min-delta`,
`--override-lr-decay-patience`, `--override-lr-decay-factor`, or
`--override-lr-decay-min-delta` when the promoted pass should intentionally run
a heavier FT-control policy than the profile row that selected it; the runner
updates the command slug, generated sweep flags, and `training_policy_key`
together so policy-match gates still describe the actual run. Chained promotion
rungs may advance those FT-control fields while preserving the non-FT training
policy; non-FT policy mismatches still fail before launch.
Add
`--promotion-selection-jsonl` on the materialize step to preserve how many
selected promotions were ready, non-ready, and actually command-materialized;
the audit row also counts guard-policy promotions and non-ready guard failures,
so guard-aware promotion files can be inspected before launch. The same ready
count/rate gates can be applied here when consuming promotion files from
elsewhere. Use `--run-events-jsonl` with `--run` to persist a
per-command execution event after each promoted lane, keeping partial failure
state inspectable during heavier FT passes.
Promoted run summaries carry `input_promotion_*` fields, preserving the upstream
rank/metric/value that selected the follow-up command, plus
`input_promotion_metric_current`, `input_promotion_metric_delta`, and
`input_promotion_metric_regression` for the heavier run's measured value; add
`--max-run-input-promotion-metric-regression` when the heavier promoted run must
not fall too far below that upstream promotion metric. Guard-readiness settings
also travel as `input_promotion_ready_*`, so `--require-run-input-promotion-match`
catches stale promoted chains whose guard-readiness policy changed.
Winner logs report `none` until a stable accepted aggregate winner exists.
`byte_lm_replay_sweep_contract` keeps the Python replay-ratio sweep honest in
Rust: target-only, target-per-replay-1, and target-per-replay-3 all start from
the same checked source byte-head state, then the winning replay ratio must beat
target-only on both target and source-retention loss deltas via
`SparseFineTuneReportSummary::compare_to`, while printing target/retention guard
margins so accepted-but-brittle FT states are easy to spot.
`byte_lm_zspace_compare_contract` runs the same tokenizerless route through a
small stateless `ZSpaceProjector` strength/placement/curvature sweep, requiring
every route to remain trainable while printing honest source/FT/retention
deltas across multiple corpus pairs, active-row accuracy/perplexity,
validation-best epochs, sparse retention-guarded FT epochs, optimizer steps,
and CSV-style aggregate metric summaries plus the current
loss/accuracy/perplexity and guard-margin winners. Route aggregate JSONL rows
also reject inconsistent case labels and route counts before comparisons, and
carry case labels plus accepted/movement coverage rates so route sweeps can gate
missing or non-moving cases before comparing winners. Python route sweeps also
emit `route_rank_summary` lines and `loss_delta_advantage_sum` so combined
source/FT/retention advantages are visible without hand-sorting JSONL.
They also emit `route_edge_check` to flag when the best route sits on the
maximum-strength or near-zero-curvature edge of the selected grid.
Route summaries and aggregate rows include projection diagnostics
(`projection_delta_input_l2_ratio`, `projection_output_input_l2_ratio`, and
`projection_output_input_col_variance_ratio`) so shallow-curvature wins can be
checked for identity-like behavior or hidden-state collapse before promotion.
`route_edge_check` also surfaces `projection_variance_collapse_risk` and
`projection_norm_expansion_risk` as diagnostic warnings for edge winners.
`route_health_rank_summary` separately ranks positive-advantage Z-space routes
with full coverage and no projection collapse or norm-expansion warning.
Use `--route-preset fine` to keep the default route set stable while exploring
strength/curvature neighbors around the current `zspace_post_s050_c025` winner.
Use `--route-preset ridge` after a fine run when `zspace_s075_c025` is leading;
it tests the nearby strength/curvature ridge without making the default smoke
path heavier.
Use `--route-preset crest` when the shallower `zspace_s075_c010` ridge is
leading; it brackets that crest with nearby strength and curvature candidates.
Use `--route-preset summit` when `zspace_s100_c010` leads; it keeps projector
strength within the supported `0.0..=1.0` range while testing shallower
curvature around the current peak.
Use `--route-preset horizon` before promoting a shallow-curvature route; it
checks whether gains keep growing near the zero-curvature edge.
Use `--route-preset health` after a horizon run when `zspace_s100_c005` is the
strongest healthy route; it refines the shallow-curvature band between the
healthy winner and the riskier near-zero edge.
Exploratory runs can pass
`--allow-zspace-nonadvantage` to log negative baseline advantages without
failing the coverage or JSONL checks. Set
`SPIRALTORCH_BYTE_LM_ZSPACE_SMOKE=1` to run the same wiring with the baseline
and representative `zspace_post_s050_c025` route before launching the full
sweep.

Enable the optional ψ telemetry layer (and CollapseDrive automation) directly
from the roundtable schedule:

```bash
cargo run -p st-nn --features "psi collapse" --example hello_session
```

The Python wheel mirrors the same flow for rapid notebooks:

```bash
python bindings/st-py/examples/hello_session.py  # enables psi+collapse by default
```

For a quick peek at the unified Rank-K planner (plus the CPU rank/compaction reference),
run the lightweight `st-core` examples:

```bash
cargo run -p st-core --example plan_rank
cargo run -p st-core --example compaction_cpu
cargo run -p st-core --example rank_select_cpu
cargo run -p st-core --features "wgpu wgpu-rt" --example compaction_wgpu
cargo run -p st-core --features "wgpu wgpu-rt" --example rank_select_wgpu
python bindings/st-py/examples/plan_rank.py
python bindings/st-py/examples/rank_select_cpu.py
```

Flip on the psychoid self-metrics layer when you want the full dream-engine
analysis (divergence, ritual rate, CTI, dream-pass/export events):

```bash
cargo run -p st-nn --features "psi psychoid collapse" --example hello_session
```

On the Python side, pass `psychoid=True` when building the roundtable and fetch
the latest reading via `spiraltorch.get_psychoid_stats()` to log the CTI score,
raw metrics, and z-scores emitted from the Rust meter.

Both variants print roundtable training diagnostics after aligning the
barycenter path with the hypergrad tape. On the Python side you can now spin up
the streaming loader without touching NumPy:

```python
trainer.set_gradient_accumulation_steps(2)
loader = st.dataset.from_vec(samples).shuffle(0xC0FFEE).batched(4).prefetch(2)
validation_loader = st.dataset.from_vec(validation_samples).batched(4)
eval_before = session.evaluate_epoch(trainer, model, loss, loader)
stats = session.train_epoch(trainer, model, loss, loader, schedule)
best = session.train_epochs_restore_best(trainer, model, loss, loader, schedule, epochs=3)
validation_best = session.train_epochs_restore_best_on_validation(
    trainer,
    model,
    loss,
    loader,
    validation_loader,
    schedule,
    epochs=3,
    patience=2,
    min_delta=0.0,
    lr_decay_patience=1,
    lr_decay_factor=0.8,
)
eval_after = session.evaluate_epoch(trainer, model, loss, loader)
```

The loader runs entirely in Rust—mini-batches stream straight into
`evaluate_epoch`, `train_epoch`, `train_epochs_restore_best`, and
`train_epochs_restore_best_on_validation`, then propagate errors as native
`TensorError`s when shapes drift.

### GoldenRetriever Training (distributed, data-race free)

Need to fan training across multiple local workers without sprinkling raw
`Arc<Mutex<...>>` or bespoke runtimes through your code? Enable the new `golden`
feature flag to pull in SpiralTorch’s Tokio/Rayon-style runtime and let the
**GoldenRetriever** orchestrator coordinate the fleet:

```bash
cargo test -p st-nn --features "golden" golden::tests::golden_retriever_trains_in_parallel -- --exact
```

The runtime exposes SpiralTorch-flavoured wrappers (`SpiralArc`, `SpiralMutex`,
`GoldenRuntime`) so modules, losses, and trainers stay inside the guard rails
while the scheduler spawns blocking steps and performs deterministic
Rayon-style reductions. A minimal Rust loop looks like:

```rust
use st_nn::{GoldenRetriever, GoldenRetrieverConfig, Linear, MeanSquaredError, ModuleTrainer};

let mut trainer_a = ModuleTrainer::new(caps, -1.0, 0.05, 0.01);
let mut trainer_b = ModuleTrainer::new(caps, -1.0, 0.05, 0.01);
let retriever = GoldenRetriever::new(GoldenRetrieverConfig::default(), vec![trainer_a, trainer_b])?;
let report = retriever.run_epoch(modules, losses, loaders, schedules)?;
println!("workers={} avg_loss={}", report.workers, report.average_loss);
```

GoldenRetriever keeps each trainer behind a poison-resistant mutex, launches the
epoch bodies on the shared runtime, and reduces the per-worker metrics using the
built-in parallel reducer so the roundtable stays deterministic. No additional
locking or thread book-keeping required.

## What you get for training

- **Rank-K family** (TopK / MidK / BottomK) with a **single entrypoint**
  Backends implement a `RankKExecutor`, decisions are made once via **unison heuristics**, and every exact-rank plan can now be rendered back into a SpiralK snippet via `choice.to_unison_script(kind)`.
  `MidK` means the centered `k`-wide exact-selection window in ascending value order; threshold/mask compaction uses its own `ops::compaction::plan_compaction(...)` surface and `backend::wgpu_rt::dispatch_compaction(...)` on WGPU.
  WGPU compaction apply can be forced with `ST_COMPACTION_APPLY=fallback|sg|sg2`; `sg2` is the experimental parallel subgroup-prefix variant for kernel A/B runs.
- **Introspectable compute plans**
  Unified `RankPlan`s expose their FFT stencil directly—call `plan.fft_plan()` to inspect the radix/segment shape, `plan.fft_wgsl()` to emit the ready-to-run WGSL kernel, or `plan.fft_spiralk_hint()` to log the same choice back into SpiralK.
- **SpiralK DSL** (K×Lisp-inspired)
  Hard assigns (`mk:`, `tile:`) and soft rules (`soft(mk, …)`, `soft(tile, …)`) that blend with measurements.
- **SoftLogic (finite-domain solver)**
  Explores a tiny discrete space (merge kinds, tiles) and scores candidates with your soft rules.
- **Pure Rust training core**
  `st-tensor::pure` ships dependency-free tensors, hyperbolic Z-space encoders,
  the new `UringFractalScheduler` for Tokio-uring style streaming, and the
  `AmegaHypergrad` tape so you can iterate on learning logic without
  PyTorch/Numpy while staying inside non-Euclidean geometry.
- **Open-topos hypergrad streaming**
  Parameters can now absorb complex Z-space waves or raw text directly into the
  hypergrad tape, so the roundtable can keep expanding meaning without Euclidean
  fallbacks or NumPy buffers.
- **TensorBiome canopies + spiral biomes**
  Curate rewrites with `TensorBiome`, weight individual shoots, stack the full
  harvest, and let SoT-3Dφ planners seed a ready-to-project biome via
  `SoT3DPlan.grow_biome(...)` before reinjecting it with `ZSpaceProjector`.
- **Rust-first modules & losses**
  `st-nn` now ships `Linear`, `Sequential`, the lightweight `Relu`, the
  hyperbolic `WaveGate`, `ToposResonator`, the new `ZSpaceMixer`, and the
  `ZSpaceProjector` alongside `MeanSquaredError`, `SoftmaxCrossEntropy`, and
  `HyperbolicCrossEntropy`
  losses. They stream gradients through the hypergrad tape, apply open-topos
  rewrites, and keep SpiralK planners one call away with roundtable-aware
  scheduling helpers. `SoftmaxCrossEntropy` accepts dense one-hot/probability
  targets as well as `(batch, 1)` sparse class-id targets for next-token style
  fine-tuning, with optional `ignore_index` masking for padded rows and
  `label_smoothing` for small-data FT regularisation; sparse targets can also
  emit active-row top-1 accuracy, mean loss, and perplexity diagnostics. Rust
  data helpers `byte_lm_windows`, `byte_lm_corpus_windows`,
  `padded_byte_lm_samples`, `byte_lm_sample_stats`,
  `validate_byte_lm_samples`, and
  `interleave_replay_samples` turn UTF-8 text or document lists into
  tokenizerless next-byte samples with byte vocabulary `BYTE_LM_VOCAB=256`,
  strict preflight validation, matching row accounting, and deterministic
  source replay for retention-guarded FT.
  `LoraLinear` provides a Rust-native low-rank adapter path with
  `Linear`-compatible base `name::weight` / `name::bias` checkpoint keys,
  frozen base parameters by default, trainable `name::lora_down` /
  `name::lora_up` parameters, and a checked `load_base_from_state_dict(...)`
  handoff for non-scratch adapter FT.
  Parameters can be frozen,
  learning-rate scaled, or assigned decoupled weight decay by exact
  state-dict name, prefix, suffix, or substring for checkpoint reloads and
  adapter/head-only fine-tuning: frozen tensors still load checkpoint values
  and propagate signals, but skip gradient accumulation and optimizer updates,
  while adapter/head groups can receive auditable LR boosts or explicit decay
  policies. Stable state
  fingerprints and checked checkpoint loads verify that a source `state_dict`
  actually landed; `load_state_dict_subset_checked` supports loading one module
  from the matching keys of a larger checkpoint when swapping dense heads for
  adapters. Use `state_dict_compatibility(...)` to audit missing keys, shape
  mismatches, and allowed extra checkpoint keys before loading; `LoraLinear`
  also exposes `base_state_dict_compatibility(...)` for dense-base adapter
  handoffs where the low-rank matrices are intentionally absent from the source
  checkpoint. External/LLM checkpoints can be bridged with a `source_key ->
  target_key` map via `state_dict_compatibility_with_key_map(...)`,
  `load_state_dict_subset_mapped_checked(...)`, or LoRA's
  `load_base_from_state_dict_mapped(...)`, so a foreign `lm_head.weight` can be
  audited and loaded into `head::weight` without mutating the module first.
  Key-map entries can also carry explicit tensor transforms:
  `transpose`, `copy_overlap_zeros`, or `transpose_copy_overlap_zeros`, covering
  common external LLM layouts and controlled vocab/head size drift. The
  compatibility report records each entry's external source key, transform,
  original source shape, and adapted source shape so preflight logs explain
  exactly what would be loaded.
  Built-in parameter
  movement audits compare a pre-FT snapshot against the current module and
  report whether frozen tensors stayed stable while trainable tensors actually
  moved. `training_state_fingerprint` and `ModuleTrainer::resume_fingerprint`
  add a second, optimizer-facing checkpoint audit for trainability, LR scaling,
  weight decay, hypergrad tape attachment, gradient accumulation, clipping, and
  trainer hook settings, so non-scratch FT resumes can prove the adapter/head
  metadata was reconstructed and not just the tensor values. `evaluate_epoch` returns the same row-weighted `EpochStats` as training
  without touching parameter state, and
  `evaluate_sparse_classification_epoch` returns active-row accuracy, mean
  loss, and perplexity for sparse next-token targets through either
  `ModuleTrainer` or `SpiralSession`; `SparseClassificationDelta` compares
  before/after metrics with consistent improvement-oriented deltas, so source
  training, target FT transfer, and post-FT source retention/forgetting can be
  logged with the same vocabulary, while
  `train_epochs_restore_best_on_validation` captures train and validation
  histories separately and restores the checkpoint selected by validation loss.
  `train_epochs_restore_best_with_retention_guard` extends that FT loop with a
  pre-FT source-retention baseline and only installs validation-best epochs
  whose retention loss stays within a configured ceiling; if every target epoch
  forgets too much, the pre-FT snapshot remains the selected state. Sparse CE
  runs can use `train_epochs_restore_best_sparse_with_retention_guard` plus
  `SparseRetentionGuardConfig` to guard source mean loss, active-row top-1
  accuracy, and optional perplexity at the same time; set
  `target_min_loss_delta` to avoid accepting near-zero target improvements as
  successful FT checkpoints. For FT harnesses that need a single Rust-native audit object,
  `train_epochs_restore_best_sparse_with_finetune_report` restores the guarded
  state and returns a `SparseFineTuneReport` with target/retention deltas plus
  frozen/trainable parameter movement status. Call `report.summary()` for a
  flat experiment digest containing status, deltas, movement, guard epoch,
  sparse-retention guard settings, movement audit tolerance, optimizer-step
  counts, state hashes, and FT-ready resume fingerprints.
  Rust harnesses can compare summaries with `SparseFineTuneRegressionLimits` /
  `SparseFineTuneSummaryComparison` to gate target-loss, retention-loss, status,
  guard-acceptance, guard-setting, movement-tolerance, and resume-fingerprint
  regressions; the
  Python façade exposes both the sparse guard and the report wrapper through
  `ModuleTrainer` and `SpiralSession` keyword arguments for notebook-scale FT loops.
  Optional validation early stopping with `patience` and `min_delta`, plus
  validation plateau LR decay with `lr_decay_patience` and `lr_decay_factor`,
  lets longer fine-tuning runs cool down before overfitting while still
  restoring the best validation checkpoint.
  Gradient accumulation lets small micro-batches produce fewer, larger effective
  optimizer updates, and `EpochStats` reports `batches` separately from
  `optimizer_steps` so FT runs stay auditable. Public max-grad-norm controls and
  non-finite loss/gradient guards keep fine-tuning updates from silently
  corrupting a run. The Python wheel exposes the training primitives, including
  `LoraLinear`, so you can stay NumPy-free while scripting experiments, with the new
  `spiraltorch.dataset.DataLoader` keeping shuffle/batch/prefetch entirely in
  Rust.
- **Optional WASM tuner table**
  Bake the JSON dataset offline and ship it to browsers/WASM. The runtime loads the table lazily, blends it with SpiralK, and keeps the optimiser in sync with the generated WGSL kernels.
- **Self-Rewrite**
  A/B/C conversations (Wilson CI) append `soft(...)` into
  `~/.spiraltorch/heur.kdsl` once the roundtable agrees a configuration is ahead, while transcripts land in
  `roundtable.log` so you can replay how every choice surfaced.
  
---

### Features (opt-in)

- `wgpu` / `wgpu-rt`: WebGPU backends + runtime wiring
- `mps`: macOS Metal (MPS)
- `cuda`: CUDA (planner surface today; exact rank runtime currently fail-fast)
- `hip`: ROCm HIP (planner/probe surface today; exact rank runtime currently fail-fast)
- **`hip-real`**: ROCm HIP + RCCL “real” path (requires ROCm toolchain & linker; gated on top of `hip`)
- HIP stub now probes `ROCM_PATH`/`HIP_PATH` and honours the
  `SPIRALTORCH_FORCE_HIP` override so simulated devices keep Z-space heuristics
  alive during CPU-only dev loops.
- **`kv-redis`**: enable Redis-backed consensus (soft hints); absent = **safe no-op**
- `logic` / `kdsl`: SoftLogic solver / SpiralK DSL

---

## Quick Start

### 1) Clone
```bash
git clone https://github.com/RyoSpiralArchitect/SpiralTorch.git
cd SpiralTorch
```

### 2) Build from source (Rust)

**CPU (default; no GPU deps)**
```bash
cargo build -p st-core --release
```

**WGPU (WebGPU; Windows/Linux/macOS)**
```bash
cargo build -p st-core --features wgpu --release
```

**MPS (macOS GPU)**
```bash
cargo build -p st-core --features mps --release
```

**CUDA (optional; needs NVRTC/Toolkit)**
```bash
cargo build -p st-core --features cuda --release
```

**HIP / ROCm (optional; real backend is feature-gated)**
```bash
export HIPCC=/opt/rocm/bin/hipcc
export ROCM_PATH=/opt/rocm
cargo build -p st-core --features hip,st-backend-hip/hip-real --release
```

### 3) Python wheels (optional)
```bash
pip install maturin==1.*

# CPU + WebGPU (default)
maturin build -m bindings/st-py/Cargo.toml --release --features wgpu

# Metal (macOS GPU)
maturin build -m bindings/st-py/Cargo.toml --release --features mps

# CUDA (toolchain on PATH)
maturin build -m bindings/st-py/Cargo.toml --release --features cuda

# HIP / ROCm (add hip-real for RCCL)
maturin build -m bindings/st-py/Cargo.toml --release --features "hip hip-real"
```

### 4) Python tensors & hypergrads

```python
from spiraltorch import Tensor, Hypergrad, LanguageWaveEncoder

encoder = LanguageWaveEncoder(-1.0, 0.6)
target = encoder.encode_z_space("SpiralTorch dances in Z-space")

weights = Tensor(*target.shape())
tape = Hypergrad(-1.0, 0.05, *target.shape())
tape.accumulate_pair(weights, target)
tape.apply(weights)
print("updated weights", weights.tolist())
```

### Canvas Pixel Transformer → Z-space feedback

- `CanvasProjector::refresh_with_vectors` now returns both the RGBA buffer and
  a colour vector field that carries normalised energy and chroma as
  Z-space-friendly coordinates.
- Use `CanvasProjector::emit_zspace_patch` to fold the canvas state back into
  the fractal scheduler without leaving Rust or allocating intermediate
  buffers.
- Blend chart priors with the new `z_space_barycenter` solver—available in
  Rust (`st_tensor::pure::measure`) and Python (`spiraltorch.z_space_barycenter`)—to
  wire colour energy directly into the Z-space roundtable.
- Follow the barycenter's loss-monotone intermediates and feed them straight into
  the hypergradient tape with `Hypergrad.accumulate_barycenter_path` so the
  optimiser converges along the same Z-space path as the solver.
- Drive the entire workflow from the high-level `SpiralSession` orchestrator in
  Rust (`st_nn::SpiralSession`) or Python (`spiraltorch.SpiralSession`) to pick
  devices, generate rank plans, synthesise barycentres, and align hypergrads via
  intuitive method calls.
- Launch `session.trace(tensor)` to compose non-commutative homotopy flows,
  functor linearisations, recursive barycenter gradients, and \(\infty\)-tower
  projections before calling `.resonate()` (or
  `.resonate_with_hypergrad(hypergrad)`) to surface a
  `DifferentialResonance` snapshot that binds the four differential layers
  together.
- Let the trace synthesise barycentres on demand via
  `trace.with_barycenter_from(weights, densities)` or override the coupling
  matrix with `trace.with_barycenter_with(weights, densities, Some(coupling))`
  before resonating, keeping Z-space orchestration entirely on the session.

---

## Minimal API

**Rust (TopK via unified entry + CPU execution)**
```rust
	use st_core::backend::device_caps::DeviceCaps;
	use st_core::backend::cpu_exec::CpuExecutor;
	use st_core::ops::rank_entry::{execute_rank, plan_rank, RankKind};

	// describe device
	let caps = DeviceCaps::cpu();
	// plan once (decisions: mk/mkd/tile/ctile/use_2ce)
	let plan = plan_rank(RankKind::TopK, rows, cols, k, caps);

	// dense row-major input buffer
	let x: Vec<f32> = vec![0.0; (rows * cols) as usize];
	let mut out_vals = vec![0.0; (rows * k) as usize];
	let mut out_idx = vec![0u32; (rows * k) as usize];

	// launch
	let mut exec = CpuExecutor::new(&x, cols, &mut out_vals, &mut out_idx);
	execute_rank(&mut exec, &plan)?;
```
**Modules**
- `Linear`, `Conv1d`, `WaveRnn`, `ReLU`, `ZSpaceProjector`
- `Sequential` composition and `ModuleTrainer`
- Fully Rust-native, Python-accessible via wheels

**Features**
- Dataset abstraction and serialization
- Hypergrad integration for every parameter
- WGPU · MPS · CUDA unified backends
```rust
use st_core::backend::device_caps::DeviceCaps;
use st_nn::{
    Linear, MeanSquaredError, ModuleTrainer, Relu, RoundtableConfig, Sequential, Tensor,
};

let mut model = Sequential::new();
model.push(Linear::new("encoder", 4, 3)?);
model.push(Relu::new());
model.push(Linear::new("head", 3, 2)?);

let mut trainer = ModuleTrainer::new(DeviceCaps::wgpu(32, true, 256), -1.0, 0.05, 0.01);
trainer.prepare(&mut model)?;

let schedule = trainer.roundtable(1, 2, RoundtableConfig::default());
let mut loss = MeanSquaredError::new();
let dataset = vec![
    (
        Tensor::from_vec(1, 4, vec![0.1, -0.2, 0.3, -0.4])?,
        Tensor::from_vec(1, 2, vec![0.0, 1.0])?,
    ),
    (
        Tensor::from_vec(1, 4, vec![0.2, 0.1, -0.3, 0.5])?,
        Tensor::from_vec(1, 2, vec![1.0, 0.0])?,
    ),
];

let stats = trainer.train_epoch(&mut model, &mut loss, dataset, &schedule)?;
println!("roundtable avg loss: {:.6}", stats.average_loss);
```

### Distributed roundtable consensus

SpiralTorch's roundtable now runs with a Blackcat moderator sitting between
local workers and the shared heuristics log:

1. **Local roundtable** — every worker runs the A/B/C negotiation locally and
   emits compact `DecisionEvent`s containing the winning band, score, and
   ψ-derived reliability. ψ stays internal to the trainer and is only used for
   automation.
2. **Blackcat meta moderator** — summaries flow into the moderator, which uses
   a dedicated Blackcat runtime to score support, publish moderator minutes,
   and forward evidence to the embedded `MetaConductor`. Once enough support
   accumulates a `GlobalProposal` is broadcast.
3. **heur.kdsl op-log** — proposals arrive as deterministic `HeurOp` entries
   that append soft rules, retract stale hints, or annotate strategies. The
   op-log is CRDT-safe so multiple nodes can merge without conflicts.

```rust
use st_core::backend::device_caps::DeviceCaps;
use st_nn::{DistConfig, ModuleTrainer, RoundtableConfig, Sequential, Linear, MeanSquaredError};

let mut trainer = ModuleTrainer::new(DeviceCaps::wgpu(32, true, 256), -1.0, 0.05, 0.01);
let dist = DistConfig {
    node_id: "node-a".into(),
    mode: st_nn::DistMode::PeriodicMeta,
    push_interval: std::time::Duration::from_secs(15),
    meta_endpoints: vec!["tcp://meta:5005".into()],
    summary_window: 8,
};
trainer.configure_distribution(dist);
trainer.install_blackcat_moderator(0.75, 2);

let mut model = Sequential::new();
model.push(Linear::new("encoder", 4, 4)?);
trainer.prepare(&mut model)?;

let mut cfg = RoundtableConfig::default();
#[cfg(feature = "psi")]
{
    cfg = cfg.enable_psi();
}
let schedule = trainer.roundtable(1, 4, cfg);
let mut loss = MeanSquaredError::new();
let dataset = vec![
    (
        Tensor::from_vec(1, 4, vec![0.0, 0.0, 0.0, 0.0])?,
        Tensor::from_vec(1, 4, vec![0.0, 0.0, 0.0, 0.0])?,
    ),
];
trainer.train_epoch(&mut model, &mut loss, dataset, &schedule)?;

// Inspect the deterministic op-log and the moderator minutes.
for op in trainer.heuristics_log().entries() {
    println!("meta op {:?}", op.kind);
}
for minute in trainer.blackcat_minutes() {
    println!("moderator: {} -> {:?} (support {:.2})", minute.plan_signature, minute.winner, minute.support);
}
```

**BlackCat runtime tap-in**

The derivative-free ZMeta ES and contextual bandits can ride alongside the
roundtable loop. Attach the runtime once and it will ingest per-step metrics,
log Above/Here/Beneath energy, estimate the BlackCat drift band, and
opportunistically promote winning `soft(...)` snippets behind a Wilson lower
bound. When you call `install_blackcat_moderator` a dedicated runtime is spun
up for the moderator so the training loop and the distributed consensus stay
decoupled.

```rust
use std::collections::HashMap;
use st_core::backend::device_caps::DeviceCaps;
use st_core::runtime::blackcat::{bandit::SoftBanditMode, ChoiceGroups, BlackCatRuntime};
use st_core::runtime::blackcat::zmeta::ZMetaParams;
use st_nn::{Linear, MeanSquaredError, ModuleTrainer, RoundtableConfig, Sequential, Tensor};

let mut trainer = ModuleTrainer::new(DeviceCaps::wgpu(32, true, 256), -1.0, 0.05, 0.01)
    .with_blackcat(BlackCatRuntime::new(
        ZMetaParams::default(),
        ChoiceGroups {
            groups: HashMap::from([
                ("tile".to_string(), vec!["128".into(), "256".into(), "512".into()]),
                ("merge".to_string(), vec!["bitonic".into(), "shared".into(), "warp".into()]),
            ]),
        },
        8,
        SoftBanditMode::TS,
        None,
    ));

let mut model = Sequential::new();
model.push(Linear::new("encoder", 4, 4)?);
let schedule = trainer.roundtable(1, 4, RoundtableConfig::default());
let mut mse = MeanSquaredError::new();
let dataset = vec![
    (
        Tensor::from_vec(1, 4, vec![0.4, -0.2, 0.1, 0.0])?,
        Tensor::from_vec(1, 4, vec![0.1, 0.2, 0.3, 0.4])?,
    ),
];
trainer.prepare(&mut model)?;
let _ = trainer.train_epoch(&mut model, &mut mse, dataset, &schedule)?;
// At this point rt.post_step() has consumed metrics and can append # blackcat heuristics.
```

**Rust (Z-space gating + projector)**
```rust
use st_core::backend::device_caps::DeviceCaps;
use st_nn::{ModuleTrainer, RoundtableConfig, Tensor, ToposResonator, WaveGate, ZSpaceProjector};
use st_tensor::pure::{topos::OpenCartesianTopos, LanguageWaveEncoder};

let encoder = LanguageWaveEncoder::new(-0.9, 0.7)?;
let topos = OpenCartesianTopos::new(-0.9, 1e-6, 1e4, 512, 16_384)?;
let projector = ZSpaceProjector::new(topos.clone(), encoder.clone())?;
let text = projector.encode_text("SpiralTorch keeps the open topos alive")?;

let mut gate = WaveGate::with_topos("gate", text.shape().1, encoder, topos.clone())?;
let trainer = ModuleTrainer::new(DeviceCaps::wgpu(32, true, 256), -0.9, 0.05, 0.01);
trainer.prepare_with_topos(&mut gate, topos)?;

let forward = gate.forward(&text)?;
let grad = forward.hadamard(&text)?.scale(1.0 / forward.shape().0 as f32)?;
let _ = gate.backward(&text, &grad)?;
trainer.step(&mut gate)?;

let (rows, cols) = forward.shape();
let mut resonator = ToposResonator::new("res", rows, cols)?;
resonator.parameter_mut().attach_hypergrad(-0.9, 0.02)?;
let activated = resonator.forward(&forward)?;
let (act_rows, act_cols) = activated.shape();
let schedule = trainer.roundtable(act_rows as u32, act_cols as u32, RoundtableConfig::default());
let bands = schedule.split(&activated)?;
let _ = bands.combine()?; // band-aware recomposition stays lossless
let energy = schedule.band_energy(&activated)?;
println!("above energy {:.3}, here {:.3}, beneath {:.3}", energy.above, energy.here, energy.beneath);
```

`DeviceCaps` now ships backend-specific constructors (`wgpu`, `cuda`, `hip`, `cpu`) and
builder-style setters (`with_subgroup`, `with_max_workgroup`, `with_shared_mem`) so you
can describe GPUs with realistic limits while still feeding the unified heuristic chooser
a compact struct. Extra helpers (`align_workgroup`, `preferred_tile`, `occupancy_score`)
let downstream tooling snap requested launches to warp-friendly shapes, reason about
effective occupancy, and auto-derive sweep/compaction tiles from the device limits.

**Python**
```python
import spiraltorch as st

plan = st.plan_topk(rows=8, cols=65_536, k=1_024, device="auto")
print(plan["choice"])  # unified merge-kind, tiles, and workgroup sizing
```

---

## Pure Rust training (zero PyTorch/Numpy deps)

Need a bootstrap-friendly learning loop without heavyweight dependencies?
`st-nn` layers sit directly on top of the `st-tensor::pure` stack so you can
train, schedule, and log every A/B/C decision entirely in Rust.

```rust
use st_core::backend::device_caps::DeviceCaps;
use st_nn::{
    HyperbolicCrossEntropy, Linear, MeanSquaredError, ModuleTrainer, Relu,
    RoundtableConfig, Sequential, Tensor,
};

fn main() -> st_nn::PureResult<()> {
    let mut model = Sequential::new();
    model.push(Linear::new("encoder", 3, 4)?);
    model.push(Relu::new());
    model.push(Linear::new("head", 4, 2)?);

    let mut trainer = ModuleTrainer::new(DeviceCaps::wgpu(32, true, 256), -0.95, 0.05, 0.01);
    trainer.prepare(&mut model)?;

    // Build a roundtable that splits gradients into Above/Here/Beneath bands.
    let schedule = trainer.roundtable(1, 2, RoundtableConfig::default());

    let dataset = vec![
        (
            Tensor::from_vec(1, 3, vec![0.3, -0.7, 0.1])?,
            Tensor::from_vec(1, 2, vec![1.0, 0.0])?,
        ),
        (
            Tensor::from_vec(1, 3, vec![-0.1, 0.4, -0.6])?,
            Tensor::from_vec(1, 2, vec![0.0, 1.0])?,
        ),
    ];

    let mut mse = MeanSquaredError::new();
    let epoch = trainer.train_epoch(&mut model, &mut mse, dataset.clone(), &schedule)?;
    println!("epoch loss: {:.6}", epoch.average_loss);

    // Inspect the logits with a hyperbolic cross-entropy probe.
    let mut hce = HyperbolicCrossEntropy::new(-0.95)?;
    let logits = model.forward(&dataset[0].0)?;
    let ce = hce.forward(&logits, &dataset[0].1)?;
    println!("hyperbolic CE: {:.6}", ce.data()[0]);

    Ok(())
}
```

Above/Beneath/Here gradients map directly onto TopK/MidK/BottomK roundtable
plans, so every update records which parts of the spectrum drove the change.
Hyperbolic losses run on the same tensors, meaning you can bounce between Z-space
encoders, Euclidean projections, and browser-friendly WASM canvases without
importing PyTorch or NumPy.

### Fractal uring scheduler + WASM canvas loop

Feed those spectra directly into an async-friendly fractal loop without ever
allocating more than a small ring buffer. The `UringFractalScheduler` keeps the
latest relation patches in a Tokio-uring style queue, blends them by coherence,
and hands the result straight to your browser front-end.

```rust
use st_tensor::pure::{Tensor, PureResult};
use st_tensor::pure::fractal::{FractalPatch, UringFractalScheduler};

async fn stream_waveforms(samples: Vec<Tensor>) -> PureResult<Tensor> {
    let scheduler = UringFractalScheduler::new(32)?;
    for (depth, relation) in samples.into_iter().enumerate() {
        let patch = FractalPatch::new(relation, 0.9, 0.7, depth as u32)?;
        // Works on any executor; tokio-uring, tokio, or synchronous loops.
        scheduler.push_async(patch).await?;
    }
    scheduler.fold_coherence()
}
```

For browser builds, wire the folded relation into a WebAssembly export that
paints onto `<canvas>` without tokenising text or duplicating buffers:

```rust
use st_tensor::pure::fractal::UringFractalScheduler;
use wasm_bindgen::prelude::*;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};

#[wasm_bindgen]
pub struct FractalCanvas {
    scheduler: UringFractalScheduler,
}

#[wasm_bindgen]
impl FractalCanvas {
    #[wasm_bindgen(constructor)]
    pub fn new(capacity: usize) -> Result<FractalCanvas, JsValue> {
        let scheduler = UringFractalScheduler::new(capacity)
            .map_err(|err| JsValue::from_str(&err.to_string()))?;
        Ok(Self { scheduler })
    }

    pub fn render(&self, canvas: HtmlCanvasElement) -> Result<(), JsValue> {
        let ctx: CanvasRenderingContext2d = canvas
            .get_context("2d")?
            .ok_or("missing 2d context")?
            .dyn_into()?;
        let frame = self
            .scheduler
            .fold_coherence()
            .map_err(|err| JsValue::from_str(&err.to_string()))?;
        let spectrum = frame.data();
        for (x, value) in spectrum.iter().enumerate() {
            let intensity = (value.clamp(0.0, 1.0) * 255.0) as u8;
            ctx.set_fill_style(&format!("rgb({0},{0},{0})", intensity).into());
            ctx.fill_rect(x as f64, 0.0, 1.0, canvas.height() as f64);
        }
        Ok(())
    }
}
```

And keep the JavaScript glue feather-light:

```html
<canvas id="zspace" width="512" height="32"></canvas>
<script type="module">
import init, { FractalCanvas } from "./pkg/spiraltorch_wasm.js";
const wasm = await init();
const canvas = document.getElementById("zspace");
const fractal = new FractalCanvas(64);
await fractal.render(canvas);
</script>
```

Pixels become Z-space relations, the scheduler keeps memory bounded, and the
entire loop stays panic-free even under aggressive streaming.

---

## Heuristics (SpiralK) — optional & powerful

SpiralK is a tiny runtime DSL for device-aware choices. Flip it on, then shape the policy per device.

```bash
export SPIRAL_HEUR_SOFT=1
export SPIRAL_HEUR_K='
  # mk: 0=bitonic, 1=shared, 2=warp (subgroup path on WGPU)
  mk:   sel(sg && (k<=128), 2, sel(k<=2048, 1, 0));
  # mkd: sub-strategy (auto/heap/kway/bitonic/warp_heap/warp_bitonic)
  mkd:  sel(mk==2,4, sel(mk==1,1,3));
  # TopK sweeping tile
  tile: sel(log2(c)>15.0, 2048,
        sel(log2(c)>13.0, 1024,
        sel(log2(c)>12.0,  512, 256)));
  # Mid/Bottom compaction tile
  ctile: sel(tile>=1024, tile/2, tile);

  # Soft hints (gently bias the solver)
  soft(mk,   2, 0.25, sg && (k<=128));
  soft(mk,   1, 0.20, (k>128)&&(k<=2048));
  soft(tile, 2048, 0.20, log2(c)>15.0);
  soft(tile, 1024, 0.15, (log2(c)>13.0)&&(log2(c)<=15.0));
'
```

**How the final choice is made (three-way roundtable)**

- **A** = SoftLogic best (your DSL soft + optional Redis soft)
- **B** = DSL **hard** assignment (if you set `mk:`/`tile:` explicitly, B wins)
- **C** = **Generated table** (tuner output)

Default policy: if **B** exists use it; otherwise the runtime invites **A** and **C** into a quick conversation. It scores both with backend-aware occupancy/tile metrics derived from `DeviceCaps`, then adds a gentle prior to **C** (`SPIRAL_HEUR_GEN_WEIGHT`, default `0.10`). When the discussion reaches a Wilson-backed agreement, **Self-Rewrite** appends the matching `soft(...)` into `~/.spiraltorch/heur.kdsl` so the next run starts from the shared insight.

Want to materialise the FFT path straight from the chosen plan? Call the new helpers and feed the result to your browser/WASM runtime:

```rust
use st_core::backend::wgpu_heuristics::{auto_fft_spiralk, auto_fft_wgsl};

let wgsl = auto_fft_wgsl(rows, cols, k, subgroup).expect("heuristics available");
let spiralk = auto_fft_spiralk(rows, cols, k, subgroup).unwrap();
// ship `wgsl` to your WebGPU runtime and persist `spiralk` if you want the DSL to learn it.
```

---

## Regenerating the WASM table (optional)

Run the offline baker to convert your latest measurements into a `WasmTunerTable`
that both native and browser builds can consume:
```bash
python3 tools/tuner/gen_generated_rs.py tools/tuner/tuner_results.json \
  > crates/st-core/src/backend/wgpu_heuristics_generated.rs
```

The generated module keeps the JSON embedded verbatim, parses it via
`st-core::backend::wasm_tuner`, and exposes a `choose(...)` helper that the
runtime queries after SpiralK/SoftLogic have spoken. Because the JSON format is
portable, you can ship the same file to a WebWorker, bake a table offline, and
let the browser pick overrides without re-running the tuner in production.

### Fractional FFT / SpiralK roadmap

- **Radix-2 → Radix-4 pipeline**: `st-frac::fft` still mirrors the GPU
  butterfly structure, and the new `SpiralKFftPlan` bridge turns the resulting
  `Choice` into auto-generated WGSL kernels for WebGPU.
- **Wilson-aware automation**: `st-kdsl::auto` turns latency deltas into
  high-confidence `soft(...)` rewrites, wiring tuned `radix`, `tile_cols`, and
  `segments` into `heur.kdsl` without manual editing.
- **ND GPU indexer**: A dedicated WGSL kernel materialises strided indices and
  per-segment IDs, unlocking fast fractional/FFT dispatches from WASM → Canvas.
- **WASM tuner baking**: `tools/tuner/tuner_results.json` keeps the measured
  overrides (`tile_cols`, `radix`, `segments`, `mode_*`) in one place so the
  generator can bake them into Rust **and** expose them to the Web via JSON.

**Example JSON**
```json
[
  {"rows": 256,  "cols_min": 0,     "cols_max": 4095,   "k_max": 128,  "sg": true,
   "wg": 128,    "tile": 512,  "tile_cols": 512,  "radix": 2, "segments": 1},
  {"rows": 512,  "cols_min": 4096,  "cols_max": 16383,  "k_max": 256,  "sg": true,
   "wg": 256,    "tile": 1024, "tile_cols": 1024, "radix": 4, "segments": 2},
  {"rows": 512,  "cols_min": 16384, "cols_max": 65535,  "k_max": 2048, "sg": false,
   "wg": 128,    "tile": 2048, "tile_cols": 2048, "radix": 4, "segments": 4, "use_2ce": true},
  {"rows": 1024, "cols_min": 65536, "cols_max": 262143, "k_max": 4096, "sg": false,
   "wg": 128,    "tile": 4096, "tile_cols": 4096, "radix": 4, "segments": 4, "use_2ce": true,
   "mode_bottomk": 2}
]
```

The generator bakes FFT-oriented hints (`tile_cols`, `radix`, `segments`) and
the ND compaction settings into the Rust table, while the same JSON remains
available for WASM workers that want to replay the optimisation flow offline.

---

## Amega Hypergrad (unrolled / implicit)

Rust utilities for hyper-parameter gradients (continuous relaxation):
- **Unrolled**: expand T updates and backprop
- **Implicit**: Neumann or **CG** to solve `(I − J) v ≈ g` efficiently

> See `crates/st-core/src/autograd/hypergrad*.rs`.
> Python glue is kept minimal; wheels can expose helpers.

The pure `st-tensor::pure::AmegaHypergrad` tape mirrors the same mindset in a
dependency-free package, letting you stage language diffusion experiments in
Rust and then feed the resulting curvature-aligned hints back into SpiralK.

---

## Safety & fallbacks

- Builds **CPU-only** by default (no GPU toolchains required).
- WGPU / CUDA / HIP are **feature-gated** and degrade safely.
- Heuristic chooser always returns a **safe** `Choice` (fills mk/tile from table or conservative defaults).

---

## Contributing

Issues & PRs welcome—especially:
- Backend kernels (WGPU subgroup variants, HIP/CUDA heap/k-way merges)
- Tuner recipes & generated tables
- New SpiralK sugar (e.g., `penalty_if(...)`, device-aware bands)

Run tests/benches on your device and share logs (latency / shapes / adapter caps).  
**AGPL-3.0-or-later** keeps it open and remix-able.

---

## Social preview

Upload a social preview PNG via **Repo → Settings → Social preview** (1200×630).  
Suggested caption: **“SpiralTorch — WGPU-first, Self-Tuning GPU Top-K (Rank-K)”**.

---

### Troubleshooting

- **No Redis?**  
  Build without `kv-redis` or leave `REDIS_URL` unset. The consensus chooser
  skips network calls and falls back to SpiralK / Generated-table safely.

- **ROCm not installed but `hip` enabled?**  
  Use `--features hip` only (stub path). The **real** path needs `hip-real`
  and a working ROCm + RCCL toolchain.

- **Wheels red?**  
  First build CPU+WGPU only: `maturin build -m bindings/st-py/Cargo.toml --release --features wgpu`
  to decouple GPU toolchain issues.

---

## License

**AGPL-3.0-or-later** for every crate and Python wheel. See `LICENSE`.
Unauthorized derivations will be treated as non-compliant with AGPL §13
