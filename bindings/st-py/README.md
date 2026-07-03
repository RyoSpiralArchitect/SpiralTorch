# SpiralTorch Python bindings

This package exposes a thin, dependency-light bridge to SpiralTorch's
Rust-first training stack. The wheel ships the same Z-space tensors,
hypergrad tapes, and unified rank planners that power the Rust API—no
NumPy, no PyTorch, and no shim layers.

## What's included

- `Tensor`, `ComplexTensor`, and `OpenTopos` for dependency-free
  geometry experiments.
- `LanguageWaveEncoder` + `Hypergrad` so Python callers can stream Z-space
  text, accumulate gradients, and project back into the Poincaré ball.
- `TensorBiome` to cultivate open-topos rewrites, weight shoots, stack the
  harvest, and guard tensors that can be re-imported into Z-space.
- Unified planning helpers (`plan`, `plan_topk`, `describe_device`) that
  reuse the same heuristics as the Rust executors.
- ROCm probing (`hip_probe`) so Python callers can reflect the stubbed
  device hints shared with the Rust runtime.
- Z-space barycentre solver (`z_space_barycenter`) to mix colour-field
  priors and chart couplings directly from Python.
- Loss-monotone barycenter intermediates (`BarycenterIntermediate`) that plug
  into `Hypergrad.accumulate_barycenter_path` so tapes converge along the
  same Z-space corridor as the solver.
- High-level orchestration via `SpiralSession` / `SpiralSessionBuilder` so
  callers can select devices, spawn hypergrad tapes, plan kernels, and solve
  barycentres with a few intuitive method calls. Structured results are
  returned through the new `ZSpaceBarycenter` class.
- Streaming dataset helpers via `spiraltorch.dataset`—build tokenizerless
  byte-LM samples plus a shuffle/batch/prefetch pipeline entirely in Rust using
  the native `DataLoader`.
- Non-commutative differential traces via `SpiralSession.trace(...)` which emit
  `SpiralDifferentialTrace` builders and `DifferentialResonance` snapshots to
  blend homotopy flows, functor derivatives, recursive barycenter energies, and
  \(\infty\)-tower projections—optionally wiring the result straight into a
  `Hypergrad` tape.
- SoT-3Dφ spiral planners (`spiraltorch.sot`) that collapse to Z-space tensors,
  grow full TensorBiomes via `SoT3DPlan.grow_biome(...)`, and stitch directly
  into `SpiralSession.trace(...)` for geometry-aware exploration loops.
- Z-space projector bindings (`spiraltorch.nn.ZSpaceProjector`) so spiral
  trajectories can be rendered onto the canvas or reused inside sequential
  transformer stacks.

## Building wheels

The binding mirrors the Rust feature flags. Pick the backend(s) you need
and maturin will bake the appropriate artefact:

```bash
pip install maturin==1.*

# CPU + WebGPU (default)
maturin build -m bindings/st-py/Cargo.toml --release --features wgpu

# Metal (macOS)
maturin build -m bindings/st-py/Cargo.toml --release --features mps

# CUDA (NVRTC toolchain expected on PATH)
maturin build -m bindings/st-py/Cargo.toml --release --features cuda

# HIP / ROCm (use hip-real for the RCCL path)
maturin build -m bindings/st-py/Cargo.toml --release --features "hip hip-real"
```

## Minimal usage

### Hello SpiralSession

```bash
python examples/hello_session.py
```

Aligns a barycenter with a hypergrad tape, prepares a Sequential module, and
finishes a roundtable epoch entirely from Python.

### Tokenizerless Byte-LM Fine-Tune

```bash
python examples/byte_lm_finetune.py
python examples/byte_lm_lora_adapter.py --jsonl /tmp/spiraltorch-byte-lm-lora.jsonl
python examples/byte_lm_lora_adapter.py --key-preset llama --jsonl /tmp/spiraltorch-byte-lm-lora-llama.jsonl
python examples/byte_lm_lora_adapter.py --compare-jsonl /tmp/spiraltorch-byte-lm-lora.jsonl --require-status-match --require-accepted-match --require-checkpoint-match
python examples/write_byte_lm_hf_state_dict.py --key-preset llama --trained-source --include-biases --out /tmp/spiraltorch-byte-hf/pytorch_model.bin
python examples/byte_lm_mlp_lora_adapter.py --jsonl /tmp/spiraltorch-byte-lm-mlp-lora-adapter.jsonl
python examples/byte_lm_mlp_lora_adapter.py --key-preset gpt_neox --jsonl /tmp/spiraltorch-byte-lm-mlp-lora-adapter-gpt-neox.jsonl
python examples/byte_lm_mlp_lora_adapter.py --hf-state-dict /path/to/tiny-byte-hf-state.pt --key-preset llama --include-extra-key model.layers.0.input_layernorm.weight --jsonl /tmp/spiraltorch-byte-lm-mlp-lora-adapter-real-hf.jsonl
python examples/byte_lm_mlp_lora_adapter.py --hf-state-dict /path/to/tiny-byte-hf-state.pt --key-preset llama --checkpoint-projection zspace --checkpoint-projection-strength 0.5 --checkpoint-projection-curvature -0.5 --checkpoint-projection-frequency 0.65 --jsonl /tmp/spiraltorch-byte-lm-mlp-lora-adapter-real-hf-zspace.jsonl
python examples/byte_lm_mlp_lora_adapter.py --hf-state-dict /tmp/spiraltorch-byte-hf/pytorch_model.bin --key-preset llama --checkpoint-projection-preset healthy --jsonl /tmp/spiraltorch-byte-lm-mlp-lora-adapter-file-backed-healthy.jsonl
python examples/byte_lm_mlp_lora_adapter.py --checkpoint-projection-preset healthy --jsonl /tmp/spiraltorch-byte-lm-mlp-lora-adapter-zspace-healthy.jsonl
python examples/byte_lm_mlp_lora_adapter.py --hf-state-dict /path/to/resized-byte-hf-state.pt --key-preset llama --allow-overlap-resize --jsonl /tmp/spiraltorch-byte-lm-mlp-lora-adapter-resized-hf.jsonl
python examples/byte_lm_mlp_lora_adapter.py --compare-jsonl /tmp/spiraltorch-byte-lm-mlp-lora-adapter.jsonl --require-status-match --require-accepted-match --require-checkpoint-match
python examples/byte_lm_handoff_strategy_compare.py --jsonl /tmp/spiraltorch-byte-lm-handoff-strategies.jsonl
python examples/byte_lm_handoff_strategy_compare.py --strategy hf_zspace_projected --zspace-strength 0.35 --zspace-frequency 0.8 --jsonl /tmp/spiraltorch-byte-lm-handoff-zspace-s035.jsonl
python examples/byte_lm_handoff_strategy_compare.py --strategy hf_zspace_projected --zspace-preset healthy --jsonl /tmp/spiraltorch-byte-lm-handoff-zspace-healthy.jsonl
python examples/byte_lm_handoff_strategy_compare.py --strategy hf_zspace_projected --zspace-strengths 0.25,0.5 --zspace-curvatures=-0.5 --zspace-frequencies 0.65,0.9 --jsonl /tmp/spiraltorch-byte-lm-handoff-zspace-grid.jsonl
python examples/byte_lm_handoff_strategy_compare.py --case adapter_ja --case route_cats --case geometry_tokens --strategy hf_zspace_projected --zspace-strengths 0.5 --zspace-curvatures=-0.5 --zspace-frequencies 0.65 --jsonl /tmp/spiraltorch-byte-lm-handoff-zspace-cases.jsonl --aggregate-jsonl /tmp/spiraltorch-byte-lm-handoff-zspace-cases-aggregate.jsonl
python examples/byte_lm_handoff_strategy_compare.py --compare-jsonl /tmp/spiraltorch-byte-lm-handoff-strategies.jsonl --require-status-match --require-accepted-match --require-checkpoint-match --require-resume-match --require-winner-match
python examples/byte_lm_handoff_strategy_compare.py --case adapter_ja --case route_cats --case geometry_tokens --strategy hf_zspace_projected --zspace-strengths 0.5 --zspace-curvatures=-0.5 --zspace-frequencies 0.65 --compare-jsonl /tmp/spiraltorch-byte-lm-handoff-zspace-cases.jsonl --compare-aggregate-jsonl /tmp/spiraltorch-byte-lm-handoff-zspace-cases-aggregate.jsonl --require-status-match --require-accepted-match --require-checkpoint-match --require-resume-match --max-aggregate-target-loss-regression 0.0 --max-aggregate-retention-loss-regression 0.0 --require-aggregate-winner-match
python examples/checkpoint_preflight.py --jsonl /tmp/spiraltorch-checkpoint-preflight.jsonl
python examples/checkpoint_preflight.py --source hf-style --jsonl /tmp/spiraltorch-checkpoint-preflight-hf.jsonl
python examples/checkpoint_preflight.py --source hf-no-bias --jsonl /tmp/spiraltorch-checkpoint-preflight-hf-no-bias.jsonl
python examples/checkpoint_preflight.py --source hf-llama --jsonl /tmp/spiraltorch-checkpoint-preflight-hf-llama.jsonl
python examples/checkpoint_preflight.py --source hf-gpt-neox --jsonl /tmp/spiraltorch-checkpoint-preflight-hf-gpt-neox.jsonl
python examples/checkpoint_preflight.py --source hf-style --compare-jsonl /tmp/spiraltorch-checkpoint-preflight-hf.jsonl --require-preflight-match
python examples/checkpoint_preflight.py --hf-state-dict /path/to/model.safetensors --key-preset auto --shape-only --vocab 256 --hidden 24 --target-classes 256 --allow-overlap-resize --require-shape-materializable --jsonl /tmp/spiraltorch-checkpoint-shape-audit-real-hf.jsonl
python examples/checkpoint_preflight.py --hf-state-dict /path/to/hf-checkpoint-dir --key-preset auto --vocab 256 --hidden 24 --target-classes 256 --allow-overlap-resize --checkpoint-projection-preset healthy --jsonl /tmp/spiraltorch-checkpoint-preflight-real-hf-bounded-healthy.jsonl
python examples/checkpoint_preflight.py --hf-state-dict /path/to/model.safetensors --key-preset llama --include-extra-key model.layers.0.input_layernorm.weight --jsonl /tmp/spiraltorch-checkpoint-preflight-real-hf.jsonl
python examples/checkpoint_preflight.py --hf-state-dict /path/to/model.safetensors --key-preset llama --checkpoint-projection zspace --checkpoint-projection-strength 0.5 --checkpoint-projection-curvature -0.5 --checkpoint-projection-frequency 0.65 --jsonl /tmp/spiraltorch-checkpoint-preflight-real-hf-zspace.jsonl
python examples/checkpoint_preflight.py --hf-state-dict /tmp/spiraltorch-byte-hf/pytorch_model.bin --key-preset llama --checkpoint-projection-preset healthy --jsonl /tmp/spiraltorch-checkpoint-preflight-byte-hf-healthy.jsonl
python examples/checkpoint_preflight.py --source hf-style --checkpoint-projection-preset healthy --jsonl /tmp/spiraltorch-checkpoint-preflight-hf-zspace-healthy.jsonl
python examples/checkpoint_preflight.py --hf-state-dict /path/to/model.safetensors --key-preset llama --vocab 256 --hidden 24 --target-classes 256 --allow-overlap-resize --jsonl /tmp/spiraltorch-checkpoint-preflight-resized-hf.jsonl
python examples/byte_lm_mlp_lora_adapter.py --hf-state-dict /path/to/hf-checkpoint-dir --key-preset auto --allow-overlap-resize --checkpoint-projection-preset healthy --jsonl /tmp/spiraltorch-byte-lm-mlp-lora-adapter-real-hf-bounded-healthy.jsonl
python examples/byte_lm_mlp_lora_sweep.py --jsonl /tmp/spiraltorch-byte-lm-mlp-lora.jsonl
python examples/byte_lm_mlp_lora_sweep.py --config r12_a64_lr4
python examples/byte_lm_mlp_lora_sweep.py --key-preset llama --config r12_a64_lr4
python examples/byte_lm_mlp_lora_sweep.py --hf-state-dict /path/to/tiny-byte-hf-state.pt --key-preset llama --include-extra-key model.layers.0.input_layernorm.weight --config r12_a64_lr4 --jsonl /tmp/spiraltorch-byte-lm-mlp-lora-real-hf.jsonl
python examples/byte_lm_mlp_lora_sweep.py --hf-state-dict /path/to/hf-checkpoint-dir --checkpoint-source-label llama-3.2-3b --key-preset auto --allow-overlap-resize --checkpoint-projection-preset healthy --checkpoint-source-gains 1.0,2.0,4.0 --case adapter_ja --config r12_a64_lr4 --jsonl /tmp/spiraltorch-byte-lm-mlp-lora-real-hf-bounded-healthy.jsonl --aggregate-jsonl /tmp/spiraltorch-byte-lm-mlp-lora-real-hf-bounded-healthy-aggregate.jsonl
python examples/byte_lm_mlp_lora_sweep.py --config r12_a64_lr4 --adapter-weight-decays 0,0.01 --max-grad-norms 1.5,2.0 --gradient-accumulation-steps-list 2,4 --jsonl /tmp/spiraltorch-byte-lm-mlp-lora-policy.jsonl --aggregate-jsonl /tmp/spiraltorch-byte-lm-mlp-lora-policy-aggregate.jsonl
python examples/byte_lm_mlp_lora_sweep.py --checkpoint-projection zspace --checkpoint-projection-strength 0.5 --checkpoint-projection-curvature -0.5 --checkpoint-projection-frequency 0.65 --config r12_a64_lr4 --jsonl /tmp/spiraltorch-byte-lm-mlp-lora-zspace.jsonl
python examples/byte_lm_mlp_lora_sweep.py --checkpoint-projection-preset healthy --config r12_a64_lr4 --jsonl /tmp/spiraltorch-byte-lm-mlp-lora-zspace-healthy.jsonl
python examples/byte_lm_mlp_lora_sweep.py --hf-state-dict /tmp/spiraltorch-byte-hf/pytorch_model.bin --key-preset llama --include-extra-key model.layers.0.input_layernorm.weight --checkpoint-projection-preset healthy --case adapter_ja --case route_cats --case geometry_tokens --config r12_a64_lr4 --min-aggregate-cases 3 --require-aggregate-case adapter_ja --require-aggregate-case route_cats --require-aggregate-case geometry_tokens --min-aggregate-accepted-rate 1.0 --min-aggregate-movement-ok-rate 1.0 --jsonl /tmp/spiraltorch-byte-lm-mlp-lora-file-backed-healthy-3case.jsonl --aggregate-jsonl /tmp/spiraltorch-byte-lm-mlp-lora-file-backed-healthy-3case-aggregate.jsonl
python examples/byte_lm_mlp_lora_source_compare.py --aggregate-jsonl /tmp/spiraltorch-real-gpt2-bare-bounded-healthy-3case-aggregate.jsonl --aggregate-jsonl /tmp/spiraltorch-real-gemma-bounded-healthy-gain-grid-3case-aggregate.jsonl --aggregate-jsonl /tmp/spiraltorch-real-llama32-bounded-healthy-gain-grid-3case-aggregate.jsonl --min-sources 3 --require-source gpt2-bare --require-source gemma-4-e4b-it --require-source llama-3.2-3b --min-cases 3 --require-case adapter_ja --require-case route_cats --require-case geometry_tokens --min-accepted-rate 1.0 --min-movement-ok-rate 1.0 --require-accepted-all --require-movement-ok-all --require-training-policy-scope-match --min-retention-accuracy-margin 0.5 --min-retention-perplexity-margin 50.0 --jsonl /tmp/spiraltorch-real-hf-source-gain-compare.jsonl --profile-jsonl /tmp/spiraltorch-real-hf-source-gain-profiles.jsonl
python examples/byte_lm_mlp_lora_profile_runner.py --profile-jsonl /tmp/spiraltorch-real-hf-source-gain-profiles.jsonl --source-path gemma-4-e4b-it=/path/to/gemma --source-path llama-3.2-3b=/path/to/llama --profile strong_effect --profile selective_ratio --commands-jsonl /tmp/spiraltorch-real-hf-profile-commands.jsonl --run-summary-jsonl /tmp/spiraltorch-real-hf-profile-run-summary.jsonl --min-run-target-retention-ratio 1.5 --min-run-accepted-rate 1.0 --min-run-movement-ok-rate 1.0 --min-run-retention-accuracy-margin 0.5 --min-run-retention-perplexity-margin 50.0 --require-run-guard-counts-available --min-run-guard-acceptance-rate-mean 0.75 --max-run-guard-retention-rejected-epochs-mean 0.0 --max-run-guard-retention-rejected-rate-mean 0.0 --max-run-guard-target-stale-epochs-mean 2.0 --max-run-guard-target-stale-rate-mean 0.25
python examples/byte_lm_mlp_lora_profile_runner.py --profile-jsonl /tmp/spiraltorch-real-hf-source-gain-profiles.jsonl --source-path gemma-4-e4b-it=/path/to/gemma --source-path llama-3.2-3b=/path/to/llama --profile strong_effect --profile selective_ratio --compare-run-summary-jsonl /tmp/spiraltorch-real-hf-profile-run-summary.jsonl --max-run-target-loss-regression 0.0 --max-run-target-retention-gap-regression 0.0 --max-run-target-retention-ratio-regression 0.0 --min-run-target-retention-ratio 1.5 --max-run-accepted-rate-regression 0.0 --min-run-accepted-rate 1.0 --max-run-movement-ok-rate-regression 0.0 --max-run-guard-acceptance-rate-regression 0.0 --max-run-guard-retention-rejected-rate-regression 0.0 --max-run-guard-target-stale-rate-regression 0.0 --min-run-movement-ok-rate 1.0 --min-run-retention-accuracy-margin 0.5 --min-run-retention-perplexity-margin 50.0 --require-run-guard-counts-available --min-run-guard-acceptance-rate-mean 0.75 --max-run-guard-retention-rejected-epochs-mean 0.0 --max-run-guard-retention-rejected-rate-mean 0.0 --max-run-guard-target-stale-epochs-mean 2.0 --max-run-guard-target-stale-rate-mean 0.25 --require-run-source-match --require-run-config-match --require-run-case-scope-match --require-run-training-policy-match --require-run-input-promotion-match
python examples/byte_lm_mlp_lora_profile_runner.py --current-run-summary-jsonl /tmp/spiraltorch-real-hf-profile-run-summary.jsonl --compare-run-summary-jsonl /tmp/spiraltorch-real-hf-profile-run-summary.jsonl --max-run-target-loss-regression 0.0 --max-run-target-retention-gap-regression 0.0 --max-run-target-retention-ratio-regression 0.0 --min-run-target-retention-ratio 1.5 --max-run-guard-acceptance-rate-regression 0.0 --max-run-guard-retention-rejected-rate-regression 0.0 --max-run-guard-target-stale-rate-regression 0.0 --min-run-accepted-rate 1.0 --min-run-movement-ok-rate 1.0 --require-run-guard-counts-available --min-run-guard-acceptance-rate-mean 0.75 --max-run-guard-retention-rejected-epochs-mean 0.0 --max-run-guard-retention-rejected-rate-mean 0.0 --max-run-guard-target-stale-epochs-mean 2.0 --max-run-guard-target-stale-rate-mean 0.25 --require-run-source-match --require-run-config-match --require-run-case-scope-match --require-run-training-policy-match --require-run-input-promotion-match --promotion-jsonl /tmp/spiraltorch-real-hf-profile-promotion.jsonl --promotion-ready-top-k 2 --promotion-ready-within 0.05 --promotion-ready-min-target-retention-ratio 1.5 --promotion-ready-min-accepted-rate 1.0 --promotion-ready-min-movement-ok-rate 1.0 --promotion-ready-require-guard-counts-available --promotion-ready-min-guard-acceptance-rate-mean 0.75 --promotion-ready-max-guard-retention-rejected-epochs-mean 0.0 --promotion-ready-max-guard-retention-rejected-rate-mean 0.0 --promotion-ready-max-guard-target-stale-epochs-mean 2.0 --promotion-ready-max-guard-target-stale-rate-mean 0.25 --min-promotion-ready-count 1 --min-promotion-ready-guard-policy-count 1 --require-promotion-ready-guard-policy
python examples/byte_lm_mlp_lora_profile_runner.py --profile-jsonl /tmp/spiraltorch-real-hf-source-gain-profiles.jsonl --promotion-input-jsonl /tmp/spiraltorch-real-hf-profile-promotion.jsonl --source-path gemma-4-e4b-it=/path/to/gemma --source-path llama-3.2-3b=/path/to/llama --commands-jsonl /tmp/spiraltorch-real-hf-promoted-profile-commands.jsonl --promotion-selection-jsonl /tmp/spiraltorch-real-hf-promoted-profile-selection.jsonl --min-promotion-ready-count 1 --min-promotion-ready-guard-policy-count 1 --require-promotion-ready-guard-policy
python examples/byte_lm_mlp_lora_sweep.py --checkpoint-projection zspace --checkpoint-projection-strengths 0.25,0.5 --checkpoint-projection-curvatures=-0.5 --checkpoint-projection-frequencies 0.65,0.9 --config r12_a64_lr4 --jsonl /tmp/spiraltorch-byte-lm-mlp-lora-zspace-grid.jsonl
python examples/byte_lm_mlp_lora_sweep.py --case adapter_ja --case route_cats --checkpoint-projection zspace --checkpoint-projection-strengths 0.25,0.5 --checkpoint-projection-curvatures=-0.5 --checkpoint-projection-frequencies 0.65 --config r12_a64_lr4 --jsonl /tmp/spiraltorch-byte-lm-mlp-lora-zspace-cases.jsonl --aggregate-jsonl /tmp/spiraltorch-byte-lm-mlp-lora-zspace-cases-aggregate.jsonl
python examples/byte_lm_mlp_lora_sweep.py --compare-jsonl /tmp/spiraltorch-byte-lm-mlp-lora.jsonl --require-status-match --require-accepted-match --require-checkpoint-match --require-winner-match
python examples/byte_lm_mlp_lora_sweep.py --case adapter_ja --case route_cats --checkpoint-projection zspace --checkpoint-projection-strengths 0.25,0.5 --checkpoint-projection-curvatures=-0.5 --checkpoint-projection-frequencies 0.65 --config r12_a64_lr4 --compare-jsonl /tmp/spiraltorch-byte-lm-mlp-lora-zspace-cases.jsonl --compare-aggregate-jsonl /tmp/spiraltorch-byte-lm-mlp-lora-zspace-cases-aggregate.jsonl --require-status-match --require-accepted-match --require-checkpoint-match --require-resume-match --max-aggregate-target-loss-regression 0.0 --max-aggregate-retention-loss-regression 0.0
python examples/byte_lm_replay_sweep.py
python examples/byte_lm_replay_sweep.py --jsonl /tmp/spiraltorch-byte-lm-replay.jsonl
python examples/byte_lm_replay_sweep.py --ratio target_per_replay_1
python examples/byte_lm_replay_sweep.py --compare-jsonl /tmp/spiraltorch-byte-lm-replay.jsonl --require-status-match --require-accepted-match --require-winner-match --min-target-loss-margin 0.001 --min-retention-loss-margin 0.5
python examples/byte_lm_zspace_compare.py --jsonl /tmp/spiraltorch-byte-lm-zspace.jsonl --aggregate-jsonl /tmp/spiraltorch-byte-lm-zspace-aggregate.jsonl
python examples/byte_lm_zspace_compare.py --case byte_patterns_to_jp --case routes_to_cats --case geometry_tokens --jsonl /tmp/spiraltorch-byte-lm-zspace-3case.jsonl --aggregate-jsonl /tmp/spiraltorch-byte-lm-zspace-3case-aggregate.jsonl --min-aggregate-cases 3 --require-aggregate-case byte_patterns_to_jp --require-aggregate-case routes_to_cats --require-aggregate-case geometry_tokens --min-aggregate-accepted-rate 1.0 --min-aggregate-movement-ok-rate 1.0
python examples/byte_lm_zspace_compare.py --route-preset fine --case byte_patterns_to_jp --case routes_to_cats --case geometry_tokens --jsonl /tmp/spiraltorch-byte-lm-zspace-fine.jsonl --aggregate-jsonl /tmp/spiraltorch-byte-lm-zspace-fine-aggregate.jsonl
python examples/byte_lm_zspace_compare.py --route-preset ridge --case byte_patterns_to_jp --case routes_to_cats --case geometry_tokens --jsonl /tmp/spiraltorch-byte-lm-zspace-ridge.jsonl --aggregate-jsonl /tmp/spiraltorch-byte-lm-zspace-ridge-aggregate.jsonl
python examples/byte_lm_zspace_compare.py --route-preset crest --case byte_patterns_to_jp --case routes_to_cats --case geometry_tokens --jsonl /tmp/spiraltorch-byte-lm-zspace-crest.jsonl --aggregate-jsonl /tmp/spiraltorch-byte-lm-zspace-crest-aggregate.jsonl
python examples/byte_lm_zspace_compare.py --route-preset summit --case byte_patterns_to_jp --case routes_to_cats --case geometry_tokens --jsonl /tmp/spiraltorch-byte-lm-zspace-summit.jsonl --aggregate-jsonl /tmp/spiraltorch-byte-lm-zspace-summit-aggregate.jsonl
python examples/byte_lm_zspace_compare.py --route-preset horizon --case byte_patterns_to_jp --case routes_to_cats --case geometry_tokens --jsonl /tmp/spiraltorch-byte-lm-zspace-horizon.jsonl --aggregate-jsonl /tmp/spiraltorch-byte-lm-zspace-horizon-aggregate.jsonl
python examples/byte_lm_zspace_compare.py --route-preset health --case byte_patterns_to_jp --case routes_to_cats --case geometry_tokens --jsonl /tmp/spiraltorch-byte-lm-zspace-health.jsonl --aggregate-jsonl /tmp/spiraltorch-byte-lm-zspace-health-aggregate.jsonl
python examples/byte_lm_zspace_compare.py --case routes_to_cats --route zspace_post_s050_c025
python examples/byte_lm_zspace_compare.py --compare-jsonl /tmp/spiraltorch-byte-lm-zspace.jsonl --require-status-match --require-accepted-match
python examples/byte_lm_zspace_compare.py --compare-aggregate-jsonl /tmp/spiraltorch-byte-lm-zspace-aggregate.jsonl --max-aggregate-ft-loss-regression 0.0 --require-aggregate-winner-match
```

Builds sparse next-byte samples with `spiraltorch.dataset.byte_lm_windows`,
trains a source byte head, reloads its state into a head-only fine-tune path,
then restores the target epoch selected by sparse source-retention guards over
loss, active-row accuracy, and perplexity. It prints source/FT/retention deltas
plus guarded-best, frozen/trainable movement status, and an FT-ready resume
fingerprint tying tensor values to trainer and parameter-training metadata.
`byte_lm_replay_sweep.py` reuses the same source checkpoint across target-only
and deterministic source-replay ratios, so FT loss improvements, guard
acceptance, retention deltas, and movement audits can be compared before moving
the setting into larger tokenizerless or LLM-adapter experiments.
`byte_lm_lora_adapter.py` takes the first adapter step directly in Python: it
trains a dense byte head, loads those base weights into `nn.LoraLinear`, and
fine-tunes only the low-rank delta through `ModuleTrainer` while
externalizing the dense head through the same HF-style `lm_head.*` handoff used
by the wider MLP adapter path. Pass `--key-preset llama` or
`--key-preset gpt_neox` to run the same adapter boundary with those source key
names instead of the default GPT-2-style keys. It logs checkpoint preflight rows,
frozen-base movement, and resume
fingerprints. Pass `--jsonl` to persist its single `SparseFineTuneReport`
summary row, then `--compare-jsonl` plus the shared status/accepted/margin and
checkpoint-audit gates to keep this smallest adapter FT path from quietly
regressing.
`byte_lm_mlp_lora_adapter.py` adds the Python MLP version now that `nn.Relu`
is exposed: it trains `Linear -> Relu -> Linear`, swaps the head to
`LoraLinear`, preflights the external embed/head key map, and keeps the
embedding/head base frozen while the adapter moves. It exposes the same
single-row JSONL and comparison-gate surface as the single-layer adapter smoke,
including `--require-checkpoint-match` for the embed/head preflight counts,
source/matched-subset hashes, key/transform signatures, load hashes, and the
selected checkpoint key preset. The checkpoint gate also tracks file-backed
source origin, loaded-file count, inferred checkpoint dimensions, and whether
overlap resizing was enabled, so exact-shape and projected handoffs do not
silently compare as the same source. It can also skip the dense source pretrain and
seed the frozen embed/head base from `--hf-state-dict`, which is intentionally
shape-strict for this byte smoke (`vocab=256`, `hidden=24`, byte head output
256) unless `--allow-overlap-resize` is explicit. Use
`checkpoint_preflight.py --hf-state-dict ... --key-preset auto --shape-only`
first for arbitrary full-size LLM checkpoints, then run the bounded adapter or
sweep with the same `--key-preset auto` once the shape audit is safe. Real HF
checkpoints that omit `lm_head.weight` because their output head is tied to the
embedding are accepted: the audit reports
`lm_head_weight_synthesized_from_embed=True`, and the bounded smoke path uses
the embedding slice as the head source. When a deliberate overlap-copy
projection is desired, pass `--allow-overlap-resize`; matched embed/head tensors
are copied into the target shape and zero-filled outside the overlap, while the
audit rows preserve both the original external shape and the adapted source
shape. Pass
`--checkpoint-projection zspace` to apply the current Z-space projector directly
to exact-shape or resized embed/head checkpoint tensors before load; the
projection policy is included in `--require-checkpoint-match`, and transposed
LM-head tensors are projected in their target load orientation before returning
to the external state-dict layout.
`byte_lm_handoff_strategy_compare.py` turns that boundary into a tiny strategy
comparison: a single trained dense byte MLP is externalized through HF-style
keys, then exact-shape, overlap-resized, and Z-space-projected handoffs are
fine-tuned under the same LoRA/retention-guard settings. Its JSONL rows use the
shared checkpoint/resume gate surface, so transform, shape, source-origin,
resize-policy, and projection drift can be gated before moving the handoff into
larger LLM adapter experiments. The Z-space projection can be swept with
`--zspace-strength`, `--zspace-curvature`, and `--zspace-frequency` for a single
custom handoff, or with comma-separated `--zspace-strengths`,
`--zspace-curvatures`, and `--zspace-frequencies` for a small grid. Grid points
become distinct strategy rows such as `hf_zspace_s0p5_cm0p5_f0p65`, and those
settings are written into the checkpoint audit row so `--require-checkpoint-match`
catches accidental geometry-policy drift. Use `--zspace-preset healthy` to carry
the current projection-health candidate (`strength=1.0`, `curvature=-0.04`,
`frequency=0.65`) from the route compare into the handoff strategy surface.
Pass `--case` repeatedly to run the
same handoff policies across the bundled `adapter_ja`, `route_cats`, and
`geometry_tokens` target cases; compare keys are `case::strategy`, so multi-case
JSONL files can be regression-gated without collapsing cases together. Add
`--aggregate-jsonl` to persist case-averaged `strategy_aggregate` rows; the
aggregate gate compares `target_loss_delta_mean`, `retention_loss_delta_mean`,
case scope, winner strategy, and checkpoint policy so a single strong case does
not hide weaker average behavior. Strategy aggregate rows also reject duplicate
cases and inconsistent case-label counts before comparison. They expose
`accepted_rate` and `movement_ok_rate`; use
`--min-aggregate-cases`, repeated `--require-aggregate-case`,
`--min-aggregate-accepted-rate`, `--min-aggregate-movement-ok-rate`, or
`--require-aggregate-accepted-all` for current-run coverage floors, and
`--max-aggregate-accepted-rate-regression` or
`--max-aggregate-movement-ok-rate-regression` with
`--compare-aggregate-jsonl` to gate coverage drops.
`checkpoint_preflight.py` isolates the checkpoint import preflight path for
external/LLM-style state dicts: it maps foreign keys, applies transpose and
copy-overlap-zero transforms, prints per-entry audit rows, and can save those
rows to JSONL before any FT run is attempted. The adapter examples reuse the
same helper at their actual load boundary, so the standalone audit and the FT
entrypoint stay aligned. Its `checkpoint_from_external_state(...)` helper
accepts `st.Tensor`, PyTorch-like tensors, NumPy-like arrays, and Python
1D/2D sequences, converting 2D weights directly and 1D bias vectors into
`(1, cols)` SpiralTorch tensors before preflight. Pass `--source hf-style` to
exercise the HF/PyTorch-style handoff shape: `transformer.wte.*` seeds the
SpiralTorch embedding, `lm_head.weight` is transposed from `(out, in)` to
`(in, out)`, and a shorter `lm_head.bias` is overlap-copied and zero-padded for
the target head. The reusable `hf_lm_key_rules(...)` helper exposes the same
mapping knobs for later real LLM state-dict imports, while
`hf_lm_handoff_from_spiraltorch_state(...)` lets the adapter smokes externalize
a dense SpiralTorch source checkpoint through those HF-style names before
loading the frozen embedding and LoRA head base.
`--source hf-no-bias` covers the common HF case where embedding/LM-head weights
exist but bias tensors do not: the helper synthesizes explicit zero-row bias
tensors before preflight so SpiralTorch's bias-bearing `Linear`/`LoraLinear`
base remains auditable instead of silently relying on initializer state.
`--source hf-llama` and `--source hf-gpt-neox` exercise common key-name presets
(`model.embed_tokens.weight` and `gpt_neox.embed_in.weight` / `embed_out.weight`)
with the same no-bias synthesis path. Real local checkpoints can also resolve
bare GPT-2 (`wte.weight`) and Gemma wrapper
(`model.language_model.embed_tokens.weight`) layouts through
`--key-preset auto`, with tied LM-head fallback when `lm_head.weight` is absent.
Use `hf_lm_key_preset("llama")`, `hf_lm_key_preset("gpt2_bare")`,
`hf_lm_key_preset("gemma")`, or `hf_lm_key_preset("gpt_neox")`, or pass
`key_preset=...` to
`hf_lm_handoff_from_spiraltorch_state(...)`, when a notebook or adapter smoke
needs those names without hand-writing the key map. For an already external
HF/PyTorch `state_dict`, use `hf_lm_handoff_from_external_state(...)` to
materialize only the configured embed/head keys, synthesize missing bias rows
when requested, and optionally include selected extra keys for audit without
converting an entire large checkpoint. The CLI can read a local HF/PyTorch
checkpoint directly via `--hf-state-dict`, including `.safetensors`,
`pytorch_model.bin`/`.pt`, and indexed HF shard directories. With shard indexes,
only shards containing the preset embed/head keys or requested
`--include-extra-key` audit fields are loaded. Add `--allow-overlap-resize`
only when explicit overlap-copy/zero-fill resizing is acceptable; without it,
shape drift stays a hard preflight failure. Add `--checkpoint-projection zspace`
to apply the same projection policy that the MLP adapter FT entrypoint uses;
report rows include the projection policy and geometry knobs, so
`--require-preflight-match` catches projection drift before any trainer state is
touched. Use `--shape-only --key-preset auto` first for larger HF shards when
key layout, embed/head shapes, byte-smoke resize requirements, and projection
policy should be logged without constructing SpiralTorch modules. Add
`--require-shape-materializable`, `--require-exact-shape-match`, or
`--require-detected-key-preset llama` when the shape audit should fail fast
instead of only logging an unsafe or unexpected checkpoint surface.
The same `--key-preset auto` resolver is available on normal preflight,
`byte_lm_mlp_lora_adapter.py`, and `byte_lm_mlp_lora_sweep.py`, so the checked
layout can be carried straight into bounded LoRA FT without manually repeating
`llama`, `gpt2_bare`, `gemma`, or `gpt_neox`. Use
`--checkpoint-source-label` on adapter and sweep runs when the resolved key
preset is not descriptive enough for cross-checkpoint dashboards, for example
when several local models share the same HF key layout.
`--checkpoint-source-gain` multiplies mapped embed/head checkpoint tensors after
optional projection and before module load; use it when a bounded real-LLM
overlap has good target-vs-retention selectivity but too little absolute
movement. The sweep form, `--checkpoint-source-gains 1.0,2.0,4.0`, expands the
config label with `::gain_g...` so source amplitude can be compared and gated
beside projection and LoRA capacity.
Use `--compare-jsonl` with `--require-preflight-match` on this standalone
preflight before FT when a saved external-checkpoint audit should fail on row,
hash, source-key, transform, or shape drift.
`byte_lm_mlp_lora_sweep.py` reuses that MLP checkpoint and compares a small
rank/alpha/adapter-LR grid, optionally writing flat summary rows to JSONL and
comparing a later run against them with the same summary regression gates. It
accepts the same `--checkpoint-projection zspace` knobs as the adapter entrypoint,
so rank/alpha/LR sweeps can be run against an audited geometry-projected source
checkpoint rather than only the exact HF-style handoff. It can also skip source
pretraining with `--hf-state-dict`, reusing the same exact-shape,
overlap-resize, missing-bias synthesis, and audit-extra controls as the single
MLP adapter smoke while preserving those source fields in each sweep row.
Comma-separated `--checkpoint-projection-strengths`,
`--checkpoint-projection-curvatures`, and
`--checkpoint-projection-frequencies` expand the sweep into rows keyed as
`config::zspace_s...`, so geometry knobs and adapter capacity can be compared in
the same regression-gated JSONL file. Comma-separated
`--checkpoint-source-gains` similarly expands source-amplitude candidates into
`config::...::gain_g...` rows, making it possible to increase real-HF injection
strength while retaining the same accepted/movement and retention gates.
Comma-separated `--adapter-weight-decays` adds decoupled adapter-decay lanes
such as `config::wd0p01` to the same JSONL surface without changing the default
zero-decay smoke. Comma-separated `--max-grad-norms` and
`--gradient-accumulation-steps-list` add trainer-policy lanes such as
`config::gn1p5::accum4`, making clipping and accumulation comparable under the
same retention gates. Longer-run FT controls can also be swept with
`--ft-epochs-list`, `--target-min-loss-deltas`, `--patiences`, and
`--lr-decay-patiences`/`--lr-decay-factors`, producing suffixes such as
`config::ep6::tmin1em06::pat3::ldp2::ldf0p8` while preserving early-stop and
LR-decay outcomes in flat and aggregate JSONL rows. Those rows also include a
`training_policy_key` for grouping the exact adapter-decay, clipping,
accumulation, and FT-control lane; the single MLP LoRA adapter summary row emits
the same key. Use repeated `--ft-control-jsonl` files with
`byte_lm_ft_control` rows, for example
`{"label":"strong_ep6","ft_epochs":6,"early_stopping_patience":3,"lr_decay_patience":2,"lr_decay_factor":0.8}`,
when only named FT-control lanes should run instead of the cartesian grid; those
lanes still reach aggregate, source-compare, profile, and promotion JSONL.
Use repeated `--lora-config-jsonl` files with `byte_lm_lora_config` rows, for
example
`{"label":"r18_a96_lr4p5","rank":18,"alpha":96,"adapter_lr_scale":4.5}`,
to add sparse adapter-capacity lanes beside the built-in `r6_a32_lr3` and
`r12_a64_lr4` configs without patching the sweep. The bundled
`examples/byte_lm_mlp_lora_capacity_lanes.jsonl` file contains the current
bounded probes (`r18_a96_lr4p5`, `r18_a128_lr5`, `r24_a96_lr4p5`, and
`r24_a128_lr5`) for promotion-ladder runs.
After ladder aggregates exist, pass them back through
`byte_lm_mlp_lora_source_compare.py --profile-jsonl ...`, then materialize the
selected lane with `byte_lm_mlp_lora_profile_runner.py --lora-config-jsonl
examples/byte_lm_mlp_lora_capacity_lanes.jsonl` so the winning capacity setting
becomes a reusable profile. Keep `--checkpoint-source-label` on the sweep runs
that produce those aggregates, and pass the same external LoRA/FT-control JSONL
files back to the profile runner so source labels and training-policy lanes stay
re-materializable.
Pass `--case` repeatedly to
run that same adapter-capacity/projection/gain/decay/policy/FT-control grid
across bundled target corpora; flat compare keys become `case::config` for
non-default cases. Add `--case-jsonl` with rows like
`{"row_type":"byte_lm_case","label":"long_ft_probe","source_docs":[...],"target_docs":[...]}`
to run longer tokenizerless source/target corpora through the same sweep; those
external labels can be selected by repeated `--case` and enforced with
`--require-aggregate-case`. Add `--aggregate-jsonl` to
persist case-averaged `config_aggregate` rows, then gate those means with
`--compare-aggregate-jsonl`, aggregate regression tolerances, case-scope drift,
and `--require-aggregate-winner-match` so one strong target case cannot hide
weaker average FT behavior or a missing weak case. Cases with no accepted
improving config are still emitted into the flat and aggregate JSONL surfaces,
making weak transfer targets visible instead of aborting the whole multi-case
sweep; winner logs print `none` when no accepted improving row exists. Add
`--require-aggregate-winner-match` only once every compared case has an accepted
aggregate winner that should remain stable. Aggregate rows reject duplicate
`case` entries for a config so accidental JSONL concatenation cannot overweight
one target corpus. They also include accepted/movement coverage fields such as
`accepted_cases`, `rejected_cases`, `accepted_rate`, and `movement_ok_cases`;
add `--min-aggregate-cases`, repeated `--require-aggregate-case`,
`--min-aggregate-accepted-rate`, `--min-aggregate-movement-ok-rate`, or
`--require-aggregate-accepted-all` when a run should fail unless the expected
cases and coverage clear the desired floor. Add
`--max-aggregate-accepted-rate-regression` or
`--max-aggregate-movement-ok-rate-regression` with `--compare-aggregate-jsonl`
when coverage should not drop versus a saved baseline. Aggregate rows also carry
target/retention loss-margin and retention accuracy/perplexity margin means and
minimums; pass `--min-aggregate-retention-accuracy-margin` or
`--min-aggregate-retention-perplexity-margin` when a gain/projection winner
should keep extra source-retention guard headroom. Aggregate compare also rejects
inconsistent case-label counts, duplicate labels, count sums, and count/rate
mismatches before computing means.
`byte_lm_mlp_lora_source_compare.py` can then ingest multiple
`config_aggregate` files from real-HF sweep runs and rank checkpoint sources on
the same JSONL surface. It reports the absolute `target_loss_delta_mean` winner
plus target-vs-retention selectivity (`target_retention_gap_mean` and
`target_retention_ratio`), preserves `checkpoint_source_gain`, adapter
weight-decay, clipping/accumulation policy, and FT-control policy, and can gate required source labels, case coverage,
accepted/movement rates, and all-case acceptance before a bounded smoke result
is treated as a reusable FT baseline. Add
`--require-training-policy-scope-match` when source/gain candidates should be
ranked only under a shared adapter-decay, clipping, accumulation, and FT-control
policy. Candidate/profile rows also preserve a `training_policy_key`, making
that exact lane easy to group in dashboards and profile-run summaries.
Use `--min-profile-target-retention-ratio` when weaker comparison candidates
should remain visible but the selected profile winner must still clear a
selectivity floor.
Use `--min-retention-accuracy-margin` and
`--min-retention-perplexity-margin` there too when source/gain comparison should
reject candidates that only win by consuming the retention guard's safety margin.
Add `--profile-jsonl` to emit reusable `checkpoint_source_profile` rows for
`strong_effect`, `selective_gap`, and `selective_ratio` lanes; each row preserves
the selected source label, gain, projection settings, trainer policy,
FT-control flags, and a CLI flag fragment excluding the environment-specific
`--hf-state-dict` path.
`byte_lm_mlp_lora_profile_runner.py` consumes those profile rows plus repeated
`--source-path source_label=/path/to/checkpoint` mappings, then materializes
per-profile `byte_lm_mlp_lora_sweep.py` commands with the saved
gain/projection/adapter-decay/training-policy/FT-control flags and aggregate
coverage gates. For a one-command local wiring check before a heavier FT pass,
run `python examples/byte_lm_profile_smoke.py`; it writes a tiny
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
The profile runner is dry-run by default; generated output paths include the
selected config/policy slug so projection/gain/decay/clipping/accumulation and
FT-control lanes do not collide, and `--config` overrides are reflected in that
slug. Pass `--run` only when the generated commands should execute. Add `--run-summary-jsonl` after a
dry-run with existing outputs or a live `--run` to merge each generated aggregate
row back into `checkpoint_source_profile_run` rows that preserve the lane name,
source/gain/decay/training-policy/FT-control choice, command shell,
target-vs-retention gap, and ratio; aggregate and command `training_policy_key`
values must agree when both are present.
Run-summary comparison keys each row by profile plus aggregate config, so one
profile lane can compare several override configs without collapsing them into a
single row.
Pass `--compare-run-summary-jsonl` with `--max-run-*-regression`,
`--min-run-target-retention-ratio`, `--min-run-retention-accuracy-margin`,
`--min-run-retention-perplexity-margin`, `--min-run-accepted-rate`,
`--min-run-movement-ok-rate`,
`--require-run-source-match`, `--require-run-config-match`,
`--require-run-case-scope-match`, or
`--require-run-training-policy-match`, or
`--require-run-input-promotion-match` to turn a saved profile run summary into a
regression gate for the next profile-lane execution; the case-scope gate pins
`cases` and `case_labels`, and the training-policy gate
also pins the stable `training_policy_key` so stale or mixed lane summaries fail
loudly. The input-promotion gate pins the `input_promotion_*` rank/metric/value
provenance when promoted FT runs must keep the exact upstream winner context.
Add `--require-run-guard-counts-available`,
`--min-run-guard-acceptance-rate-mean`,
`--max-run-guard-retention-rejected-epochs-mean`,
`--max-run-guard-retention-rejected-rate-mean`,
`--max-run-guard-target-stale-epochs-mean`, and
`--max-run-guard-target-stale-rate-mean` when current profile summaries must
prove their sparse-retention guard epoch counts came from exact backend
diagnostics and stayed inside bounded rejected/stale count and rate budgets.
When comparing to a saved run summary, add
`--max-run-guard-acceptance-rate-regression`,
`--max-run-guard-retention-rejected-rate-regression`, and
`--max-run-guard-target-stale-rate-regression` to catch acceptance-rate drops or
rejected/stale-rate increases before promotion.
The `--min-run-*` floors can also gate a freshly written `--run-summary-jsonl`
before a baseline exists.
For CI-style checks that already have current and baseline run-summary files,
use `--current-run-summary-jsonl` with `--compare-run-summary-jsonl`; this skips
profile JSONL/source-path command materialization and only performs the summary
gate. Add `--promotion-jsonl` after a run-summary compare or current-run gate to
write `checkpoint_source_profile_promotion` rows ranked by `--promotion-metric`
(default `target_retention_ratio`), preserving source/config/cases/training
policy/coverage context for the next FT pass; equal best rows stay
promotion-ready to keep honest ties visible, and `--promotion-ready-top-k` or
`--promotion-ready-within` can intentionally widen the ready set for heavier
exploratory FT. Add promotion-ready floors such as
`--promotion-ready-min-target-retention-ratio`,
`--promotion-ready-min-accepted-rate`, or
`--promotion-ready-min-movement-ok-rate`, plus
`--promotion-ready-max-input-promotion-metric-regression` for promoted chains,
and the guard-aware `--promotion-ready-require-guard-counts-available`,
`--promotion-ready-min-guard-acceptance-rate-mean`,
`--promotion-ready-max-guard-retention-rejected-epochs-mean`,
`--promotion-ready-max-guard-retention-rejected-rate-mean`,
`--promotion-ready-max-guard-target-stale-epochs-mean`, and
`--promotion-ready-max-guard-target-stale-rate-mean` guard gates, when weak lanes
should remain visible in the ranked promotion JSONL but be
excluded from default ready-only follow-up commands. If every selected promotion
is non-ready, materialization stops with the non-ready floor reasons. Add
`--min-promotion-ready-count` or
`--min-promotion-ready-rate` when a promotion JSONL generation or materialize
step should fail unless enough ready candidates remain after floors; pair them
with `--min-promotion-ready-guard-policy-count` or
`--require-promotion-ready-guard-policy` when every launched ready lane must come
from a guard-aware promotion policy. Feed that file back with
`--promotion-input-jsonl` and the original profile JSONL to materialize only
promotion-ready commands while reusing the profile row's checkpoint/projection
and training-policy flag fragment; source, selected config, case scope, and
training policy are required and checked against the profile rows so stale
promotions fail before launching. If those profiles reference external case
labels from `--case-jsonl`, pass the same `--case-jsonl` path to
`byte_lm_mlp_lora_profile_runner.py` so promoted sweep commands resolve the
longer tokenizerless corpora explicitly; if they reference external adapter
labels, pass the same `--lora-config-jsonl` path as well. Use
`--override-ft-epochs` and the companion `--override-target-min-loss-delta`,
`--override-patience`, `--override-min-delta`,
`--override-lr-decay-patience`, `--override-lr-decay-factor`, or
`--override-lr-decay-min-delta` when the promoted pass should intentionally run
a heavier FT-control policy than the profile row that selected it; the runner
updates the command slug, generated sweep flags, and `training_policy_key`
together so policy-match gates still describe the actual run. Chained promotion
rungs may advance those FT-control fields while preserving the non-FT training
policy; non-FT policy mismatches still fail before launch. Use
`--promotion-selection-jsonl` on this materialize step when the
ready/non-ready/materialized counts should be retained
as a small audit row; it also reports guard-policy promotions and non-ready guard
failures, so guard-aware promotion files can be inspected before launch. The
same ready count/rate gates can be applied here when consuming promotion files
from elsewhere. Promoted run summaries carry
`input_promotion_*` fields, preserving the upstream rank/metric/value that
selected the follow-up command, and add `input_promotion_metric_current`,
`input_promotion_metric_delta`, and `input_promotion_metric_regression` for the
heavier run's measured value. Use `--max-run-input-promotion-metric-regression`
to fail a heavier promoted run when its current value for that same metric drops
too far below the upstream promotion value. Guard-readiness settings travel as
`input_promotion_ready_*` too, so `--require-run-input-promotion-match` catches
stale promoted chains whose guard-readiness policy changed. When the materialized commands are actually
launched with `--run`, add `--run-events-jsonl` to write one execution event per
profile lane as soon as it finishes, so interrupted heavier FT passes still
leave a partial success/failure trail.
Add `--require-winner-match` when the winning LoRA rank/alpha/LR config should
stay unchanged. Use repeated `--config` filters for quick targeted LoRA checks.
Use `module.load_state_dict_subset_checked(full_state)` when a module should
load only its matching keys from a larger checkpoint before an adapter swap.
Call `module.state_dict_compatibility(full_state)` first to inspect matched,
missing, shape-mismatched, and extra checkpoint keys without mutating the module;
for `nn.LoraLinear`, use `base_state_dict_compatibility(full_state)` when a
dense checkpoint should seed only the frozen base weights.
When an external checkpoint uses different names, pass a `source_key ->
target_key` dict to `state_dict_compatibility_with_key_map(...)` and
`load_state_dict_subset_mapped_checked(...)`, or to
`LoraLinear.base_state_dict_compatibility_with_key_map(...)` and
`load_base_from_state_dict_mapped(...)` for dense-base adapter handoffs. Dict
values may also be `{"target": "head::weight", "transform": "transpose"}`; use
`copy_overlap_zeros` or `transpose_copy_overlap_zeros` only when explicit
row/column overlap copying and zero padding is acceptable. Compatibility report
entries include `source_name`, `transform`, `original_source_shape`, and
adapted `source_shape` for audit logs.
For replay sweeps, pass `--jsonl path` to persist one flat
`SparseFineTuneReport.summary()` row per ratio for later CSV/JSONL comparison.
Pass `--compare-jsonl previous.jsonl` plus optional `--max-target-loss-regression`,
`--max-retention-loss-regression`, `--min-target-loss-margin`,
`--min-retention-loss-margin`, `--min-retention-accuracy-margin`,
`--max-target-retention-gap-regression`,
`--max-target-retention-ratio-regression`, `--min-target-retention-ratio`,
`--require-status-match`, `--require-accepted-match`, `--require-guard-match`,
or `--require-movement-tolerance-match` to turn a
saved run into a small regression gate. The `--min-*-margin` gates reject
accepted runs that barely clear target-improvement or source-retention guard
budgets even when they did not regress versus the baseline; add
the target/retention selectivity gates when a fine-tune must keep improving the
target more than it disturbs retained source behavior. Add
`--require-resume-match` when the
FT-ready trainer/module resume fingerprint must also stay unchanged. Add
`--require-checkpoint-match` for adapter runs when checkpoint preflight matched
/ extra counts, source/matched-subset hashes, key/transform signatures, and
load hashes must stay unchanged. Add
`--require-winner-match` when the winning replay ratio should stay stable, and
use repeated `--ratio` filters for quick targeted replay checks. The same
backend comparison is available as
`spiraltorch.nn.compare_sparse_finetune_summaries(...)` for notebooks.
`byte_lm_zspace_compare.py` adds a tiny baseline-vs-Z-space route sweep across
multiple corpus cases, including the `geometry_tokens` transfer case used by
the adapter sweeps, using strength and curvature candidates surfaced by the Rust
contracts. It reuses the same sparse retention-guarded FT restore path as the
single-route example and prints per-case rows plus route-level aggregate means,
case labels, accepted/movement coverage, and winners.
It can also write route-level JSONL rows and compare later runs with the same
regression gates, so Z-space route changes can be treated like FT sweep changes.
Pass `--aggregate-jsonl` beside `--jsonl` when route-level means and
baseline-relative advantages should be saved for dashboards or longer sweep logs.
Aggregate rows include `loss_delta_advantage_sum`, and the CLI prints
`route_rank_summary` lines so a grid run immediately shows the strongest route
by combined source/FT/retention loss-delta advantage.
The CLI also prints `route_edge_check` when Z-space routes are present, flagging
whether the best route sits on the maximum-strength or near-zero-curvature edge
of the selected grid.
Route summaries and aggregate rows include projection diagnostics:
`projection_delta_input_l2_ratio`, `projection_output_input_l2_ratio`, and
`projection_output_input_col_variance_ratio`. These make shallow-curvature wins
easier to inspect before promotion: tiny delta suggests an identity-like route,
while a collapsing variance ratio suggests the projector is compressing hidden
structure rather than preserving useful geometry.
`route_edge_check` also surfaces `projection_variance_collapse_risk` and
`projection_norm_expansion_risk` as diagnostic warnings for edge winners; they
do not fail the run by themselves.
`route_health_rank_summary` ranks only Z-space routes that have positive
combined advantage, full accepted/movement coverage, and no projection collapse
or norm-expansion warning. This keeps the raw winner visible while surfacing the
strongest lower-risk candidate for follow-up runs.
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
Aggregate rows also carry target/retention guard margin means and minimums, so
route-level Z-space comparisons can surface accepted-but-brittle FT settings;
route aggregate rows reject inconsistent case labels and route counts before
comparison. They also expose `accepted_rate` and `movement_ok_rate`; use
`--min-aggregate-cases`, repeated `--require-aggregate-case`,
`--min-aggregate-accepted-rate`, `--min-aggregate-movement-ok-rate`, or
`--require-aggregate-accepted-all` for current-run coverage floors, and
`--max-aggregate-accepted-rate-regression` or
`--max-aggregate-movement-ok-rate-regression` with
`--compare-aggregate-jsonl` to gate coverage drops. Pass
`--allow-zspace-nonadvantage` for exploratory runs that should log negative
baseline advantages without failing after coverage and JSONL checks pass.
Pass `--compare-aggregate-jsonl` plus optional `--max-aggregate-source-loss-regression`,
`--max-aggregate-ft-loss-regression`, or `--max-aggregate-retention-loss-regression`
to gate route-level mean regressions directly; add
`--min-aggregate-target-loss-margin`,
`--min-aggregate-retention-loss-margin`,
`--min-aggregate-retention-accuracy-margin`, or
`--min-aggregate-retention-perplexity-margin` to reject routes whose aggregate
guard margin minimums fall below the floor; add
`--require-aggregate-winner-match` when the winning source/FT/retention routes
must stay unchanged.
Use repeated `--case` and `--route` filters for quick targeted checks; when a
Z-space route is selected, the baseline route is included automatically so
advantage metrics stay meaningful.

```python
from spiraltorch import Tensor, Hypergrad, LanguageWaveEncoder

z = Tensor(2, 4, [0.1, 0.2, 0.3, 0.4, 0.9, 0.8, 0.7, 0.6])
encoder = LanguageWaveEncoder(-1.0, 0.5)
wave = encoder.encode_z_space("SpiralTorch in Rust")

tape = Hypergrad(-1.0, 0.05, *z.shape())
tape.accumulate_pair(z, wave)
tape.apply(z)
print(z.tolist())
```

```python
from spiraltorch import SpiralSession, Tensor

session = SpiralSession(device="wgpu", curvature=-1.0, hyper_learning_rate=0.05)
densities = [Tensor(1, 2, [0.7, 0.3]), Tensor(1, 2, [0.2, 0.8])]
bary = session.barycenter(densities)
hyper = session.hypergrad(*bary.density.shape())
session.align_hypergrad(hyper, bary)
print(bary.objective, hyper.gradient())
```

```python
import spiraltorch as st
from spiraltorch.nn import Linear, MeanSquaredError, Sequential

session = st.SpiralSession(device="wgpu", curvature=-1.0)
trainer = session.trainer()
trainer.set_max_grad_norm(1.0)
schedule = trainer.roundtable(
    rows=1,
    cols=2,
    psychoid=True,
    psychoid_log=True,
    psi=True,
    collapse=True,
    dist=st.DistConfig(node_id="demo", mode="periodic-meta", push_interval=10.0),
)
trainer.install_blackcat_moderator(threshold=0.6, participants=1)
model = Sequential([Linear(2, 2, name="layer")])
loss = MeanSquaredError()
session.prepare_module(model)
model.set_parameters_trainable_by_suffix("::weight", False)  # freeze backbone-style tensors for head-only FT
model.set_parameters_learning_rate_scale_by_suffix("::bias", 1.1)  # gently boost the trainable head/bias group
model.set_parameters_weight_decay_by_suffix("::bias", 0.01)  # decoupled decay is explicit and parameter-group scoped
before = model.state_dict()
source_fingerprint = model.state_fingerprint()

loader = (
    st.dataset.from_vec([
        (st.Tensor(1, 2, [0.0, 1.0]), st.Tensor(1, 2, [0.0, 1.0])),
        (st.Tensor(1, 2, [1.0, 0.0]), st.Tensor(1, 2, [1.0, 0.0])),
    ])
    .shuffle(0xC0FFEE)
    .batched(2)
    .prefetch(2)
)
validation_loader = loader

trainer.set_gradient_accumulation_steps(2)
eval_before = session.evaluate_epoch(trainer, model, loss, loader)
stats = session.train_epoch(trainer, model, loss, loader, schedule)
history = session.train_epochs(trainer, model, loss, loader, schedule, epochs=3)
summary = st.summarize_epoch_history(history)
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
movement = model.parameter_movement(before, tolerance=1e-8)
print(
    f"roundtable avg loss {stats.average_loss:.6f} over "
    f"{stats.batches} batches / {stats.optimizer_steps} optimizer steps / {stats.rows} rows"
)
print(eval_before.average_loss_per_row, eval_after.average_loss_per_row)
print([round(epoch.average_loss_per_row, 6) for epoch in history])
print(summary.best_epoch, summary.best_loss_per_row, summary.best_improvement)
print(best.summary.best_epoch, best.best_fingerprint["hash"])
print(
    validation_best.validation_summary.best_epoch,
    validation_best.best_fingerprint["hash"],
    validation_best.early_stopped,
    validation_best.stop_epoch,
    validation_best.lr_decay_steps,
    validation_best.final_hyper_learning_rate,
)
print(source_fingerprint["hash"], movement["status"], movement["frozen_stable"])
print(st.get_psychoid_stats())
```

For next-token experiments, `spiraltorch.nn.SoftmaxCrossEntropy` accepts either
dense one-hot/probability targets or `(batch, 1)` sparse class-id targets, so
tokenizerless byte windows can stay NumPy-free end to end. Pass
`SoftmaxCrossEntropy(ignore_index=-1, label_smoothing=0.01)` when padded sparse
target rows should contribute no loss or gradient and small-data FT needs mild
regularisation. Use `loss.sparse_metrics(prediction, target)` to inspect
active-row `accuracy`, `mean_loss`, and `perplexity` without letting padding
rows dilute diagnostics, or call
`trainer.evaluate_sparse_classification_epoch(model, loss, loader)` or
`session.evaluate_sparse_classification_epoch(trainer, model, loss, loader)` to
aggregate the same metrics over a Rust `DataLoader`. Compare two returned metric
dicts with `spiraltorch.nn.sparse_classification_delta(before, after)`; positive
`loss_delta` and `perplexity_delta` mean the values decreased, while positive
`accuracy_delta` means top-1 accuracy improved. The same delta helper can
compare source pretrain, target fine-tune, and post-FT source-retention metrics
without changing sign conventions between reports.
For sparse next-token fine-tuning, call
`train_epochs_restore_best_sparse_with_retention_guard(...)` from either
`ModuleTrainer` or `SpiralSession` to select target-validation checkpoints that
also preserve source retention loss, accuracy, and optional perplexity ceilings.
Pass `target_min_loss_delta` when tiny near-zero target improvements should not
be accepted as FT checkpoints.
Use `train_epochs_restore_best_sparse_with_finetune_report(...)` when the same
run should return a `SparseFineTuneReport` with target/retention deltas,
guard-acceptance status, and frozen/trainable movement audit in one object.
`report.summary()` returns a flat dict for CSV/JSON experiment logs with status,
deltas, guard epoch, optimizer-step counts, movement status, movement audit
tolerance, and state hashes.
Use `training_state_fingerprint()` and `trainer.resume_fingerprint(model)` when
checkpoint/resume logs must also prove trainability, LR scales, hypergrad
attachment, accumulation, clipping, and trainer hook settings were restored.
Use `spiraltorch.nn.compare_sparse_finetune_summaries(current, baseline, ...)`
to compare two summary dicts with the same target/retention/status/acceptance
regression gate used by `byte_lm_replay_sweep.py`; pass
`require_guard_match=True` when
baseline and current rows must share the same sparse retention guard settings,
and `require_movement_tolerance_match=True` when their frozen/trainable
movement audits must use the same tolerance. Pass `require_resume_match=True`
when replay/FT comparisons must fail on trainer or parameter-training metadata
drift even if tensor checkpoint values still match.

```python
samples = st.dataset.byte_lm_windows("螺旋byte", context=4)
corpus = st.dataset.byte_lm_corpus_windows(
    ["spiral byte corpus", "猫byte corpus", ""],
    context=4,
)
replay_mix = st.dataset.interleave_replay_samples(
    corpus,
    samples,
    target_per_replay=1,
)
loader = st.dataset.from_vec(samples).shuffle(7).batched(4)
stats = st.dataset.byte_lm_sample_stats(samples)

padded = st.dataset.padded_byte_lm_samples(
    ["spiral", "猫byte", ""],
    pad_rows=8,
    ignore_index=-1,
)
active = st.dataset.byte_lm_sample_stats(padded, ignore_index=-1)["active_rows"]
```

`spiraltorch.nn.ZSpaceProjector(topos, encoder, strength=...)` can be swept from
identity (`0.0`) to the full Z-space projection (`1.0`) inside `Sequential`
models.

The `DistConfig` connects the local roundtable to a meta layer that exchanges
`MetaSummary` snapshots with peers. `install_blackcat_moderator` spins up a
moderator runtime that scores summaries, publishes Blackcat minutes, and funnels
evidence into the embedded meta conductor—all without exposing ψ readings to the
outside world.

```python
from spiraltorch import SpiralSession, Tensor

session = SpiralSession(device="wgpu", curvature=-1.0)
seed = Tensor(1, 2, [0.4, 0.6])
generator = Tensor(1, 2, [0.1, -0.2])
direction = Tensor(1, 2, [0.05, 0.07])
kernel = Tensor(2, 2, [1.0, 0.5, -0.25, 1.25])

weights = [0.6, 0.4]
densities = [Tensor(1, 2, [0.6, 0.4]), Tensor(1, 2, [0.5, 0.5])]

trace = session.trace(seed)
trace.deform(generator, direction)
trace.via(kernel)
trace.with_barycenter_from(weights, densities)
trace.with_infinity([densities[0].clone()], [])
resonance = trace.resonate()
print(resonance.homotopy_flow().tolist())
```

```python
from spiraltorch import LanguageWaveEncoder, SpiralSession, Tensor, TensorBiome
from spiraltorch.nn import ZSpaceProjector
from spiraltorch.sot import generate_plan

session = SpiralSession(device="wgpu", curvature=-1.0)
seed = Tensor(1, 8, [0.2] * 8)
trace = session.trace(seed, sot={"steps": 64, "radial_growth": 0.08})
plan = trace.sot_plan or generate_plan(64, radial_growth=0.08)

topos = session.topos()
encoder = LanguageWaveEncoder(session.curvature(), 0.5)
projector = ZSpaceProjector(topos, encoder)

spiral_tensor = plan.as_tensor()
canvas = projector.project_spiral(plan)
print(spiral_tensor.shape(), canvas.shape())

biome = plan.grow_biome(topos)
biome.absorb_weighted("canvas", canvas, weight=2.0)
stacked = biome.stack()
meaning = projector.reimport_biome(biome)
print("stacked", stacked.shape(), "reimported", meaning.shape())
```
