# SpiralTorch (Python) changelog

## 0.4.13

- HF PEFT runtime: add lazy `spiraltorch.hf_peft` helpers plus
  `prepare_hf_finetune_model(...)` for model-family-aware LoRA target
  resolution, parameter-freeze audits, gradient checkpointing, and adapter
  attachment without making PEFT an eager import dependency.
- HF adapter lifecycle: detect local adapter-only artifacts, reconstruct their
  base model/tokenizer/PEFT stack through `load_hf_causal_lm_artifact(...)`,
  expose compact load provenance to local trace, Z-Space inference/generation,
  and checkpoint audit, and add `spiral-hf-adapter-export` for atomic safe-merge
  export to a standalone full model.
- HF adapter continuation: let `spiral-hf-finetune` consume local or declared
  remote PEFT artifacts as trainable inputs, reuse their active adapter without
  double attachment, and preserve artifact/base/tokenizer/runtime-config
  provenance in the run card and model-profile surfaces.
- HF adapter lineage and promotion: fingerprint adapter config/weight content,
  write parent/root/run-card provenance after successful LoRA saves, expose
  `spiral-hf-adapter-lineage` / `spiral-hf-adapter-promote`, and optionally fail
  FT runs unless weights changed and before/after eval stays within a configured
  loss-regression bound.
- Promotion-aware HF sweeps: forward LoRA artifact/mode/promotion policy through
  every bridge command, retain blocked candidates in comparison evidence, and
  restrict selection, resume reuse, and scale-up command generation to adapters
  whose promotion gate is ready.
- Adapter-continuing HF scale-up: resolve promotion-ready sweep winners as the
  parent PEFT artifact by default, preserve lineage provenance through command
  replay, expose `auto` / `replay` / `continue` policy, and preflight local
  adapter weights plus lineage/promotion fingerprint and depth consistency.
- Multi-generation adapter chains: add `spiral-hf-adapter-chain` plus importable
  DAG reports that revalidate fingerprints, parent/root/depth continuity,
  promotion gates, run-card digests, forks, rejected generations, and a unique
  continuation tip. FT run cards now retain their exact launch command, and
  `spiral-hf-scale-up` accepts a ready chain report directly for the next depth.
- Evidence-driven adapter continuation: add opt-in maximum-depth, target-loss,
  and minimum-improvement/patience policies to promotion chains. Decisions are
  separately inspectable and persistable, missing eval evidence fails closed,
  and scale-up refuses to emit another generation after a policy stop.
- Resumable adapter executor: add `spiral-hf-adapter-executor` and an importable
  state machine that closes audit, policy, scale-up, preflight, execution, and
  live postflight promotion verification into one atomic artifact. Successful,
  failed, interrupted, and resumed generations retain command and lineage
  evidence, while per-invocation generation limits prevent accidental runaway.
- Executor observability: persist a unique owner-only combined subprocess log
  plus PID, hostname, and working-directory provenance for every real
  generation, and add `spiral-hf-adapter-executor-status` for read-only local
  liveness, artifact, and state-age checks. Quiet terminal mode keeps the
  durable log, while remote
  and legacy process identity remains explicitly unverified; recorded live or
  remote child processes and a per-output-root single-writer lock prevent
  duplicate FT launches.
- Executor stop control: add invocation-scoped, owner-only stop requests plus
  `spiral-hf-adapter-executor-stop`. The executor that owns the subprocess
  cooperatively terminates its isolated process group without signalling an
  unverified external PID, records in-flight work as cancelled, preserves
  completed promotion at generation boundaries, and fails closed when
  cancellation leaves partial output. Status reports expose healthy `stopping`
  transitions and durable request evidence.
- Detached executor lifecycle: let `spiral-hf-adapter-executor --run --detach`
  launch an isolated background executor, prove handoff through a new
  invocation plus matching single-writer-lock ownership, and retain owner-only
  launcher logs and durable launch history. Duplicate starts fail closed on
  unverified ownership, PID reuse alone does not block recovery, and
  `spiral-hf-adapter-executor-launch-status` combines launcher and executor
  health without mutating either artifact.
- Audited executor recovery: add
  `spiral-hf-adapter-executor-quarantine` plus importable plan/execute helpers
  for failed, cancelled, and postflight-failed outputs. Recovery CAS-checks the
  attempt ID, acquires the executor lock, atomically moves data into a sibling
  quarantine outside that executor root's chain discovery, rejects special
  filesystem entries, persists and revalidates a tree-metadata digest plus
  idempotent resolution history, blocks executor re-entry while an intent is
  unfinished or inconsistent, and returns status to a healthy
  `output_quarantined` / `resume_executor` handoff.
- Interrupted executor recovery: the same quarantine flow can now claim a
  stale `running` subprocess only after proving both its local leader PID and
  isolated POSIX process group are absent. The proof is rechecked under lock,
  persisted through intent/resolution history, exposed in live status, and
  remains fail-closed for remote, custom, live, or process-scope-unverified
  attempts.
- Durable executor resume: add `spiral-hf-adapter-executor-resume` and
  importable plan/execute helpers that replay the exact argv and working
  directory retained by detached launch history. New launches persist a
  SHA-256 replay contract binding argv, output root, executor state, and cwd;
  resume CAS-checks its source launch under the launch lock, requires a healthy
  terminal executor whose durable action is `resume_executor`, and rejects
  modified, stale, live, locked, policy-stopped, or cwd-missing requests.
- Bounded executor supervision: add
  `spiral-hf-adapter-executor-supervise` plus importable decision and execution
  helpers that wait across detached invocations and automatically replay only
  the exact `max_generations_per_invocation_reached` boundary. Resume budgets,
  timeout, a stale-reapable single-owner lock, atomic transition history, and
  launcher/executor handoff identity checks keep unattended continuation
  bounded; operator stop, policy stop, recovery, failed launch, remote owner,
  and unhealthy artifacts remain explicit terminal or manual boundaries.
- Detached supervisor control: extend the supervisor CLI with `--detach`, a
  durable launch history, PID-plus-lock handoff verification, idempotent
  duplicate launch handling, and separate launch/runtime status commands. A
  run-targeted cooperative stop artifact lets operators stop the controller
  without signalling a PID or terminating the independently owned executor;
  stale requests cannot cross into a restarted supervisor run.
- Integrated executor runtime control: add one read-only report over executor,
  launcher, supervision, supervisor, and detached-supervisor launch state with
  cross-layer path identity checks. Running
  `spiral-hf-adapter-executor-runtime --reconcile` idempotently starts an
  unmanaged controller, while any prior stop, timeout, interruption, or
  resume-budget boundary requires the explicit `--restart-supervisor` opt-in
  before a new bounded run can begin.
- Self-contained HF wheel CLIs: package every direct and transitive Python
  payload behind the installed `spiral-hf-*` and inference-distortion commands,
  validate that manifest before PyPI publication, and execute all 35 console
  entrypoints in the installed-wheel CI smoke. Generic run status now accepts
  the documented `--run-dir` form, while milestone capture spawns packaged
  sibling commands instead of repository-relative scripts. Adapter continuation
  also accepts an audited lineage-depth-zero seed without a promotion report and
  injects the promotion gate into required child scale-up commands.
- Release payload gate: reuse the manifest-backed wheel validator in both PyPI
  upload workflows and smoke every installed HF/Z-Space console command across
  the Linux, macOS, and Windows release matrix before publication.
- PyPI propagation guard: keep polling after uploaded wheels appear until the
  project latest-version index also exposes the release, avoiding false-red
  publication runs during the short CDN propagation window.
- Model-neutral HF artifact probe: add `hf_causal_lm_artifact_probe_report(...)`
  and `spiral-hf-artifact-probe` to reconstruct full models or PEFT adapters,
  run bounded generation, and archive device/token/timing/runtime evidence. A
  real Pythia 70M LoRA sample now covers non-GPT-2 train, promotion, reload, and
  MPS generation.
- Promotion-qualified artifact reload: Trainer outputs now retain their
  tokenizer, while `--adapter-promotion-gate` releases the trained model and
  accelerator cache, then requires a local-only fresh PEFT reload plus
  deterministic bounded generation. Sweep, run-card,
  promotion-chain, and scale-up artifacts preserve and revalidate the probe
  path, device, candidate identity, and generated-token evidence.
- Isolated artifact qualification: `spiral-hf-artifact-probe`, the importable
  subprocess probe API, and promotion-gated Trainer runs now reconstruct the
  saved artifact in a dedicated Python worker. Request JSON avoids repeating
  prompts in the worker argv, while PID, parent PID, exit code, timeout, and
  worker-module evidence are revalidated by promotion chains and retained by
  scale-up handoffs.
- Trainer resume audit: add `hf_finetune_checkpoint_resume_report(...)` and
  compact lines for optimizer/scheduler/RNG state availability, saved versus
  requested step horizons, and the exhausted-scheduler case where adapter
  weights-only warm start is safer than extending exact Trainer resume.
- Transformers compatibility: apply the generation `batch_size` compatibility
  shim through nested PEFT base models so adapter-backed generation-control
  sweeps follow the same path as full models.
- Generic HF bridge: add `--finetune-mode full|lora`, LoRA rank/alpha/dropout/
  target/module-save controls, adapter-aware profile launch preflight, and run
  card evidence for matched modules, trainable ratio, artifact kind, and saved
  adapter files. Built-in Pythia, GPT-2, Qwen2, and SmolLM2 LoRA profiles make
  local adapter smoke runs config-driven rather than script-specific.
- Packaging/preflight: add `hf-peft-finetune` as the adapter-training contract
  that combines PEFT with datasets, pyarrow, tqdm, and Trainer dependencies;
  keep `hf-peft` as the narrower adapter load/attach surface.
- Python API: add `spiraltorch.hf_ft` plus top-level
  `hf_gpt2_finetune_preflight_report(...)`,
  `hf_gpt2_finetune_rust_dependency_report(...)`,
  `hf_gpt2_finetune_zspace_probe(...)`, and
  `write_hf_gpt2_finetune_run_card(...)` helpers for local GPT-2-scale
  fine-tuning handoffs.
- Examples: add `examples/hf_gpt2_finetune_bridge.py`, a local
  `AutoModelForCausalLM` / `Trainer` bridge that records the `hf-gpt2-ft`
  dependency contract, SpiralTorch WGPU/CPU runtime readiness, optional
  Z-Space token probe metrics, and a JSON run card before long FT runs.
- Trainer trace: add HF Trainer callback helpers that write SpiralTorch JSONL
  rows for train/log/evaluate/save/end events and summarize loss/eval-loss
  telemetry back into the GPT-2 FT run card.
- Local corpus: let `examples/hf_gpt2_finetune_bridge.py` train from repeated
  `--train-file` / `--validation-file` inputs via the Hugging Face
  `text`/`json`/`csv` dataset builders, with lightweight file manifests and
  fingerprints written into the run card.
- Corpus scan: add `hf_gpt2_finetune_corpus_scan_report(...)` and
  `--corpus-scan` so large local GPT-2 FT runs can record streamed line/byte,
  empty-line, sample-text, and rough token-shape evidence before Trainer work.
- FT runtime hardening: make `--allow-remote` temporarily override local
  Hugging Face offline environment flags, record model-load/train failures into
  run cards, and filter `TrainingArguments` kwargs against the installed
  Transformers signature for 4.x/5.x compatibility.
- Dataset fit: add `hf_gpt2_finetune_dataset_fit_report(...)` and run-card
  `dataset_fit_report` so tokenized train/eval block counts, empty eval splits,
  and non-trainable block-size/data combinations are explicit before Trainer
  work proceeds.
- Generation samples: add `hf_gpt2_finetune_generation_report(...)` plus
  `--generation-prompt` / `--generation-max-new-tokens` so local GPT-2 FT run
  cards can compare before/after generated text, continuation text, token
  deltas, and hashes beside numeric Trainer telemetry.
- Eval reports: add `hf_gpt2_finetune_eval_report(...)`,
  `--eval-before-train`, and final run-card eval reports so local GPT-2 FT runs
  can compare eval loss/perplexity before and after training.
- Run-card comparison: add `load_hf_gpt2_finetune_run_card(...)`,
  `summarize_hf_gpt2_finetune_run_card(...)`, and
  `compare_hf_gpt2_finetune_run_cards(...)` so multiple local FT runs can be
  ranked by eval deltas, generation-sample changes, and trainer telemetry.
- FT sweep runner: add `examples/hf_gpt2_finetune_sweep.py` to execute local
  GPT-2 bridge grids across block size, learning rate, step budget, and seed,
  carrying corpus scans, generation/eval reports, Z-Space probes, WGPU gates,
  and run-card comparisons into `sweep-plan.json` / `sweep-report.json`.
- Sweep summaries: add `load_hf_gpt2_finetune_sweep_report(...)`,
  `summarize_hf_gpt2_finetune_sweep_report(...)`, and
  `summarize_hf_gpt2_finetune_sweep_report_lines(...)`; sweep reports now
  embed a compact status/top-run/scale-up-candidate summary.
- Sweep resume: add `--resume-existing` / `--force` to
  `examples/hf_gpt2_finetune_sweep.py` so interrupted local GPT-2 FT sweeps can
  reuse successful run cards, run only missing/failed rows, and still compare
  reused plus newly completed runs in one report.
- FT run-card hardening: keep Z-Space probes aligned with the current
  `OpenCartesianTopos` constructor and native Tensor `.tolist()` export, and
  make generation samples tolerate patched Hugging Face
  `_prepare_special_tokens(..., batch_size=...)` call sites without falling
  back to manual forward generation.
- FT eval controls: add `--max-eval-blocks`,
  `--eval-after-train-policy skip-if-final-step-eval`,
  `--dataloader-pin-memory auto|true|false`, `--dataloader-num-workers`, and
  `--eval-accumulation-steps` to make longer local GPT-2 FT runs easier to
  resume, audit, and keep responsive on MPS/CPU machines; sweep summaries use
  the final trainer-trace eval loss as the effective after-train loss when the
  duplicate final eval pass is intentionally skipped.
- FT trace telemetry: trainer trace summaries now include wall-clock duration,
  log-interval step/sec statistics, eval-loss series points, and eval-runtime
  stats so long local GPT-2 runs can expose throughput wobble beside loss
  improvements; sweep top-run summaries surface the same throughput/eval-series
  fields next to eval deltas.
- FT generation sample: add a sanitized local GPT-2 long-run sample artifact
  showing a fixed prompt shift from generic Torch/Tor wording before training
  toward SpiralTorch runtime / Python bindings wording after 640/1280/2560
  steps, with eval-loss and throughput evidence beside the text.
- HF generation control: add `spiraltorch.hf_generation` with
  `ZSpaceRepressionLogitsProcessor` and bridge/sweep
  `--generation-zspace-softmax` flags so local GPT-2 samples can apply a
  repetition-repression field before Z-Space entropy-temperature softmax and
  record bounded `generation_control` telemetry in run cards; run-card and
  sweep summaries surface top-token-change, entropy, temperature, and backend
  fields, and include a sanitized 2560-step GPT-2 generation-control sample
  artifact.
- HF generation sweep: add `examples/hf_gpt2_zspace_generation_control_sweep.py`
  for generation-only Z-Space/repression grids against an existing
  `AutoModelForCausalLM`, with generated text, loopiness metrics, and bounded
  control telemetry emitted without running Trainer or loading datasets; include
  a compact local GPT-2 2560-step grid sample showing softmax-only versus
  repression-driven loop relaxation, plus Python helpers to load and summarize
  those sweep artifacts with recommended processor kwargs plus bridge/sweep CLI
  arguments for the next decode run.
- Documentation: clarify the hard dependency boundary for larger local GPT-2
  FT runs: Rust keeps tensor/nn/text/logic/frac/rl/WGPU surfaces in the wheel,
  while Python explicitly brings Transformers, Torch, datasets, pyarrow,
  accelerate, safetensors, PEFT, and evaluation packages.

## 0.4.12

- Packaging: add Hugging Face / PyTorch optional dependency extras for local
  fine-tuning handoffs: `hf-runtime`, `hf-finetune`, `hf-peft`,
  `hf-gpt2-ft`, and `hf-trl-sft`.
- Runtime preflight: add `hf-gpt2-ft` and `hf-trl-sft` import presets so
  local GPT-2-scale FT runs can require `torch`, `transformers`,
  `tokenizers`, `datasets`, `accelerate`, `safetensors`, `pyarrow`, and
  related adapter/evaluation modules before a long run starts.

## 0.4.11

- Python API: expose runtime import preflight helpers directly from
  `spiraltorch`, including `runtime_import_preflight_report(...)`,
  `runtime_import_preflight_summary_lines(...)`, and
  `write_runtime_import_preflight_report(...)`, so fine-tune and interop
  notebooks can validate optional `torch` / `transformers` / `tokenizers`
  dependencies without importing the lower-level helper module first.
- Documentation: refresh the pip-facing README path around WGPU-first wheels,
  `SpiralSession`, Transformers co-import evidence, tokenizerless byte-LM
  diagnostics, and reload-pair/sweep runtime contracts.

## 0.4.10

- Packaging: pin the canonical root license and wheel license payload to LF
  line endings so Windows-built wheels embed the exact AGPL bytes checked by
  the release license report.

## 0.4.9

- Packaging: include the canonical AGPL license payload in wheels via
  `project.license-files` and require `maturin>=1.9` for PEP 639 support.
- Release: bump Python binding metadata to 0.4.9 after the unsigned 0.4.8
  recovery path exposed missing wheel license payloads.

## 0.4.7

- `nn`: add training-mode handling, norm-state propagation, and gradient-clipping support in module training workflows.
- `spiralk`: re-export `MaxwellFingerprint`, `MeaningGate`, `SequentialZ`, `MaxwellPulse`, `MaxwellProjector`, and the expectation helpers (`required_blocks`, `expected_z_curve`, `polarisation_slope`) at the top level so `import spiraltorch as st` quickstarts work as documented.
- Wheels CI: add Maxwell quickstart coverage in smoke tests and remove brittle trace/live-server checks from the cross-platform `Wheels` workflow.

## 0.4.6

- Wheels CI: replace the deprecated `--universal2` flag with `--target universal2-apple-darwin` (matches current maturin) on macOS, and switch Linux builds to `--compatibility manylinux2014 --zig` (with `maturin[zig]`) for more reliable cross-compilation. Applied to both `release_wheels.yml` and `wheels.yml`. No API changes.

## 0.4.5

- Wheels CI: install macOS cross-compilation targets (`aarch64-apple-darwin`, `x86_64-apple-darwin`) before building universal2 wheels — the prior pipeline assumed the targets were already present and failed on fresh runners. Mirrored in both `wheels.yml` and `release_wheels.yml`. No API changes.

## 0.4.4

- Wheels CI: consolidate the macOS matrix to a single `macos-latest` / macosx14 / universal2 wheel (covering arm64 + x86_64), drop the separate `macos-15-intel` and macosx11-arm64 jobs, and thread a `universal2` flag through the matrix. Linux (ubuntu-22.04) and Windows (windows-2022) entries unchanged. README install instructions updated to match the new wheel matrix. No API changes.

## 0.4.3

- _No source changes._ Version metadata + README pin bump only; re-publish of 0.4.2 to align wheel artifacts with the tagged version.

## 0.4.2

- _No source changes._ Version metadata + README pin bump only.

## 0.4.1

- Model zoo: add runnable Python baselines for `vision_conv_pool_classification.py` and `zspace_vae_reconstruction.py`.
- `nn`: expose `MellinBasis` + `ZSpaceVae` for the Coherence VAE reconstruction loop.

## 0.4.0

- DesirePipeline: surface `indices`/`probabilities`/`logit_offsets` in `step(...)` so callers can apply desire feedback during sampling.
- Model zoo: new attentionless character LLM baseline (`llm_char_wave_rnn_mixer.py`) built from `WaveRnn` + `ZSpaceMixer` + `WaveGate`.
- `nn`: expose `WaveGate`, `WaveRnn`, `ZSpaceMixer`, and `FeatureReorder2d` in the Python bindings.
- Dev: when running from a source checkout, auto-load the freshest native extension from `target/` after `cargo build -p spiraltorch-py`.
- Telemetry: add `trainer_events_to_atlas_route(...)` helpers and export them from `spiraltorch`.

## 0.3.9

- WGPU: remove unsupported WGSL `enable ...;` directives and harden compute-pipeline creation against validation panics.
- SpiralK: add `SpiralKContext.eval_with_trace(...)` and `RankPlan.rewrite_with_spiralk_explain(...)` for notebook-friendly explainability.
- Add `write_kdsl_trace_html(...)` / `write_kdsl_trace_jsonl(...)` + demo viewer.

## 0.3.8

- Fix `serve_zspace_trace(...)` live viewer (no PyO3 cross-thread `PluginQueue` panic).
- Fix HTML viewers (`write_zspace_trace_html`, live viewer) f-string escaping for JS template literals.
- Add wheel smoke coverage for trace viewers.

## 0.3.7

- Stream Z-space trace events into Python (payload + timestamp on `spiraltorch.plugin` custom events).
- Add `ZSpaceCoherenceSequencer.install_trace_recorder(...)` + `ZSpaceTraceRecorder` bindings.
- Add `write_zspace_trace_html(...)` / `serve_zspace_trace(...)` for notebook-friendly visualization (HTML + live SSE viewer).
- Add `zspace_trace_to_atlas_route(...)` to merge traces into `telemetry.AtlasRoute`.

## 0.3.6

- Route `nn.LayerNorm` forward through the tensor-layer `layer_norm` op (WGPU kernel available when the backend is enabled).
- Expose `Tensor.layer_norm_affine` / `Tensor.layer_norm_affine_add` for direct call sites in notebooks.
- Build `wgpu` with native backends enabled (Metal/Vulkan/DX12) so GPU adapter discovery works in release wheels.

## 0.3.5

- Add ops sugar (`ops.signature`, flexible `ops.register`) and allow variadic inputs for Python-registered operators.
- Add `ops.describe` plus varargs support for `ops.execute`.
- Add `plugin.record(...)` JSONL recorder plus queue helpers for notebook-friendly observability.
- Expand `nn` bindings with additional losses (HyperbolicCrossEntropy/CrossEntropy, Focal, Contrastive, Triplet) and broader `ModuleTrainer` coverage.
- Make `nn.save_json/load_json` accept state dicts and return them when called with `None` (plus bincode equivalents).
- Add `nn.save/nn.load` helpers with auto format detection and manifest generation.
- Allow `nn.Sequential.add` to accept NonLiner/Scaler/Dropout/Pool2d/ZPooling/ZConv/ZConv6DA layers (NCHW for 2d ops, NCDHW for ZConv6DA).
- Strengthen unittest smoke coverage for ops/plugin/serialization.

## 0.3.4

- Add `spiraltorch.ops` to register custom operators in Python and run them via the Rust registry.
- Upgrade `spiraltorch.plugin` with queue/listen helpers plus batch subscribe/unsubscribe.
- Add `spiraltorch.nn.save_json/load_json` (+ bincode) for module snapshots.
- Start the model zoo with Python + Rust MLP regression examples.
- Route core tensor ops (`add`, `matmul`, `row_softmax`, …) into plugin events via the new `st-tensor` observer hook.
- Expose `st_nn::ModuleTrainer` as `spiraltorch.nn.ModuleTrainer` with `RoundtableConfig`, `RoundtableSchedule`, and `EpochStats`.
