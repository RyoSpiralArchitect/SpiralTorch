# Learning stack and training recipes

[Documentation index](../README.md) | [Project entry](../../README.md)

This is the detailed source-tree reference, moved from the README.
Examples retain their individual feature, device, and optional-dependency requirements.
A source-tree API is not a claim that an older PyPI wheel exposes it.
Run repository commands from the repository root.

## Contents

- [🧱 Learning Stack (Model Zoo)](#-learning-stack-model-zoo)
- [Hello SpiralSession quickstart](#hello-spiralsession-quickstart)
- [GoldenRetriever Training (distributed, data-race free)](#goldenretriever-training-distributed-data-race-free)
- [SpiralTorchRL (agents)](#spiraltorchrl-agents)
- [SpiralTorchRec (open-topos recommendation lattice)](#spiraltorchrec-open-topos-recommendation-lattice)
- [Observation DAG calculus (Pólya-calibrated final coalgebra)](#observation-dag-calculus-pólya-calibrated-final-coalgebra)
- [What you get for training](#what-you-get-for-training)
- [Features (opt-in)](#features-opt-in)
- [Canvas Pixel Transformer → Z-space feedback](#canvas-pixel-transformer--z-space-feedback)
- [Minimal API](#minimal-api)
- [🌀 New: ZSpaceCoherenceSequencer](#-new-zspacecoherencesequencer)
- [Plugin Architecture](#plugin-architecture)
- [Why Not Attention?](#why-not-attention)
- [Pure Rust training (zero PyTorch/Numpy deps)](#pure-rust-training-zero-pytorchnumpy-deps)

### 🧱 Learning Stack (Model Zoo)

SpiralTorch’s “learning stack” is a set of minimal, runnable training baselines (Rust-first; no NumPy/PyTorch required). See `models/README.md`.

- Demo texts: `models/samples/spiral_demo_en.txt`, `models/samples/spiral_demo_ja.txt`
- Demo corpus folder: `models/samples/spiral_corpus_en/` (multiple `.txt` files)
- Run outputs: `models/runs/<timestamp>/` (e.g. `run.json`, `metrics.jsonl`, `summary.json`, `samples/`, `weights.json` / `weights.bin`)
- Optional (Python): `--backend cpu|wgpu|cuda|hip|auto` to pick the compute backend
- Optional (Python): `--events <path>` to record events (JSONL) + `--atlas` to emit `atlas_summary.json`
- Optional (Python): `--desire` to enable desire telemetry + apply desire offsets during sampling
- Optional (Python): tune SoftLogic band weighting via `SPIRAL_SOFTLOGIC_*`, `--softlogic-*` flags (saved into `run.json`), or `trainer.set_softlogic_config(st.nn.SoftLogicConfig(...))`
- Coherence-scan char LMs damp the scan context before the classifier head by default; use `--context-scale 0.05` to tune that initial-logit scale.
- Coherence scan/wave examples also accept the bigram prior and top-k guard flags; when shrinking `--steps` for smoke tests, set `--memory <= --steps`.
- Rust char-LM examples scale classifier weights by RMS by default; use `--head-rms 0.1` and, for scan/wave mixers, `--mix-rms 0.1` to tune update pressure.
- Rust char-LM examples add a learned smoothed train-token unigram prior before the softmax by default; use `--head-prior none` to start without that prior, or `--head-prior bigram|learned-bigram` to route a previous-token train bigram prior through the same head.
- Rust char-LM examples scale residual context logits before the head prior with `--head-residual-scale 1.0`; set it to `0` for a pure-prior ablation, or raise it to test whether context can beat the frequency/bigram prior.
- Rust char-LM examples can add a differentiable previous-token top-k preservation guard with `--bigram-topk-guard F --bigram-topk-guard-k N`, training on normal next-token CE plus a weighted CE over the smoothed train-bigram top-k row.
- The Rust raw-text fine-tune example can switch its recurrent core with `--recurrent spiral|lstm`; `tools/run_char_lm_sweep.py --architectures finetune,lstm ...` compares the SpiralRNN and stateless batched LSTM paths in the same table.
- One-command char-LM ablation report: `PYTHONNOUSERSITE=1 python3 -S -s tools/run_char_lm_ablation_report.py models/samples/spiral_corpus_en --preset smoke` runs LSTM, SpiralRNN, coherence scan/wave, and top-k guard variants across the default seeds, then writes `ablation_report.md`, `ablation_report.json`, `compare.json`, and `compare_summary.md`; add `--head-priors learned-bigram,learned-unigram,none` to sweep how much the output prior is carrying the result.
- Shape grids for recurrent comparisons use `--step-values`, `--hidden-values`, and `--embed-dim-values`; training-budget grids use `--epoch-values` and `--batches-values`; residual-logit grids use `--head-residual-scale-values`; bigram guard grids use `--bigram-topk-guard-values`; generated run names and aggregate groups include those dimensions so unlike shapes/scales/budgets/guards are not averaged together.
- Reproduce the validated guarded LSTM/SpiralRNN comparison with `tools/run_char_lm_sweep.py <text-or-dir> --recipe guarded-lstm`; override any generated grid flag explicitly when narrowing or expanding the recipe.
- Probe no-prior context learning with `tools/run_char_lm_sweep.py <text-or-dir> --recipe no-prior-context-pressure`, then focus the cheaper scan/wave budget path with `--recipe no-prior-coherence-budget`; use `--recipe no-prior-coherence-frontier` to compare the LSTM baseline, scan winner, and lite wave candidates on the same longer budget while applying scan/wave-only mix RMS normalization.
- Probe scan/wave expressiveness directly with `tools/run_char_lm_sweep.py <text-or-dir> --recipe no-prior-coherence-shape`, confirm the full grid with `--recipe no-prior-coherence-shape-confirm`, or rerun only the quick-probe scan winner plus promoted single-branch wave with `--recipe no-prior-coherence-shape-winners`; use `--recipe no-prior-coherence-wave-lite` and `--recipe no-prior-coherence-wave-lite-confirm` to trade wave branch count against route debt, then `--recipe no-prior-coherence-wave-promoted` to rerun the route-debt-selected lite wave shape. Use `--recipe no-prior-coherence-promoted-frontier` to compare that promoted wave against the scan winner on the same longer budget, `--recipe no-prior-coherence-wave-long` to rerun the promoted wave on the longer budget that improved no-prior evidence, `--recipe no-prior-coherence-wave-wide-corpus` with a widened docs bundle such as `models/samples/spiral_corpus_en models/samples/spiral_demo_en.txt models/README.md docs/getting-started.md docs/example-gallery.md docs/zspace_intro.md bindings/st-py/README.md`, `--recipe no-prior-coherence-wave-capacity-scout` to sweep hidden/memory capacity around that long wave recipe, `--recipe no-prior-coherence-wave-capacity` to rerun the h96/m24 single-dilation candidate, or `--recipe no-prior-coherence-wide-frontier` to compare the LSTM baseline, scan winner, and promoted wave on that widened long-budget setting. These vary or pin scan `context/query` scales and wave `kernel/dilations`, while compare artifacts keep coherence route/debt columns visible.
- Top aggregate char-LM rows rank by validation NLL first, then by trace latency and CPU-debt tie-breaks, so equal-quality recurrent shapes surface the cheaper path.
- Route-aware `compare_summary.md` emits `Bigram Guard Deltas` when a sweep contains both `bigram_guard=0` and guarded rows, making NLL, bigram lift, rank lift, and top-5 preservation tradeoffs visible against the unguarded baseline; `Bigram Guard Recommendations` floats clean guard candidates above mixed top-k tradeoffs.
- Char-LM `compare_summary.md` now includes a `Learning Scoreboard` section, ranking aggregate runs by positive NLL gain per traced step while keeping final/best NLL, bigram gap, CPU debt, and route status visible for scan/wave/SpiralRNN budget comparisons.
- Char-LM `compare_summary.md` also emits `Route Debt Decision` / `Route Debt Recommendations` when wave dilation variants reach neutral-or-better NLL with lower coherence route debt, surfacing lite wave shapes such as single-branch dilations before heavier-but-equivalent stacks; use `--compare-summary-sort-metric coherence_route_debt` to rank the main table by that route-debt column directly, or `--compare-summary-fail-on-route-debt-decision no_route_debt_recommendation` to gate a lite-wave confirmation sweep.
- Char-LM validation summaries include smoothed train-token unigram and bigram baselines, target-token rank, and context-lift metrics (`mean_target_logprob_lift`, rank lift, KL-to-unigram, plus bigram lift/KL/top-5 overlap fields), so runs can be checked against simple frequency and previous-token baselines before judging context learning; use `--compare-summary-sort-metric final_bigram_logprob_lift|final_bigram_rank_lift|final_top5_bigram_overlap` to shortlist by those guards instead of raw NLL.
- Compare char-LM runs: `PYTHONNOUSERSITE=1 python3 -S -s tools/compare_char_lm_runs.py --aggregate --curves --params 5 models/runs/<baseline> models/runs/<scan> models/runs/<wave>`
- **LLM (raw text, no tokenizer):** `cargo run -p st-nn --example modelzoo_llm_char_finetune -- <text.txt> [--recurrent spiral|lstm] [--head-rms 0.1 --head-residual-scale 1.0 --head-prior unigram|bigram|learned-bigram --bigram-topk-guard 0.05 --bigram-topk-guard-k 5] [--val-fraction 0.1 --eval-samples 256]`
- **LLM (Python, raw text, no tokenizer):** `PYTHONNOUSERSITE=1 python3 -S -s models/python/llm_char_finetune.py <text_or_dir> [<text_or_dir> ...]`; add `--preflight-only --runtime-import-preset hf-full-finetune --require-runtime-imports --runtime-device-backend wgpu --runtime-preflight-json-out ft-runtime.json` before a heavier local HF handoff to capture the SpiralTorch NN/backend surface plus same-process `transformers`/`torch`/`tokenizers`/`datasets`/`accelerate`/`safetensors`/`pyarrow`/`evaluate`/`peft` import evidence and WGPU device readiness in `preflight.json`, `runtime_preflight.json`, and `summary.json`; add `--require-runtime-device-ready-backend wgpu` for strict accelerator gating. On a real run, the same runtime contract is also recorded in `run.json`.
- **LLM (Python, tokenizerless FT profile ladder):** `PYTHONNOUSERSITE=1 python3 -S -s bindings/st-py/examples/byte_lm_profile_smoke.py --hf-state-dict <local-hf-state-dict-or-dir> --key-preset auto --ft-readiness-preset hf-wgpu-balanced` audits local HF/PyTorch-style checkpoints, enables the checkpoint + Transformers trace + produced-manifest runtime contract, records same-process `transformers`/`torch`/`tokenizers` co-import evidence, captures `describe_device("wgpu")` runtime readiness evidence, checks the Transformers/trainer runtime bridge, and gates bounded byte-LM LoRA/source/profile comparisons with WGPU run-summary and promotion readiness thresholds before heavier FT. The FT preset expands to `--runtime-contract-preset hf-runtime --wgpu-readiness-preset balanced`; use `hf-wgpu-observed` to only require WGPU metrics/report presence or `hf-wgpu-strict` for a high-readiness bar that also requires WGPU runtime-ready evidence. Lower-level `--runtime-contract-preset`, `--wgpu-readiness-preset`, explicit `--runtime-device-report-backend`, `--min-run-epoch-wgpu-*`, `--max-run-epoch-wgpu-*`, promotion-ready, or manifest trainer/device WGPU flags override the recipe defaults. Direct trace/import flags such as `--transformers-trace-runtime-import-preset torch-transformers`, `--require-transformers-trace-runtime-import torch`, and `--require-manifest-transformers-trace-runtime-import-preset torch-transformers` remain available for narrower runtime audits; use `hf-finetune` to also require `datasets`/`accelerate`/`safetensors`, or `hf-peft` to include `peft`.
- **LLM (Python, Transformers logit trace):** `PYTHONNOUSERSITE=1 python3 -S -s bindings/st-py/examples/byte_lm_transformers_trace.py --model-path <local-transformers-model> --prompt "spiral route" --jsonl trace.jsonl` records runtime config/tokenizer/model metadata plus next-token top-k logits/probabilities and hidden-state summaries before FT; add `--zspace-project` to attach a bounded Z-space projection probe, `--runtime-contract-preset hf-runtime` to require same-process `transformers`/`torch`/`tokenizers` co-import evidence directly, `--runtime-import-preset torch-transformers|hf-runtime|hf-finetune|hf-peft` or repeated `--runtime-import <module>` to audit narrower same-process imports, persist preset module expansion plus satisfied/failed/coimport status contracts and install hints for known missing HF modules, gate direct traces with `--require-runtime-import torch` or `--require-runtime-import-preset torch-transformers`, and `--require-runtime-metadata-match` during `--compare-jsonl` checks to catch model/tokenizer/runtime swaps.
- **LLM runtime preflight (installed wheel):** `spiral-runtime-preflight --preset hf-full-finetune --require --runtime-device-backend wgpu --json-out ft-runtime.json` checks the local `transformers`/`torch`/`tokenizers` plus `datasets`/`accelerate`/`safetensors`/`pyarrow`/`evaluate`/`peft` stack and records `describe_runtime_devices(["wgpu"])` readiness before a heavier local HF FT run; add `--require-runtime-device-ready-backend wgpu` for strict accelerator gating, `--json` for stdout-based CI output, use `--preset hf-peft` for narrower PEFT adapter workflows or `--preset hf-trl-sft` for TRL SFT handoffs, and call `python -m spiraltorch.runtime_imports ...` when console scripts are unavailable. Runtime-device direct/surrogate readiness, native availability, fallback identity, evidence-preserving `ready`/`not_ready`/`unknown` states, and the ordered executable selection are evaluated by the versioned Rust `spiraltorch.runtime_device_route.v5` contract. Probe success is reported separately from native availability; Python/WASM transport the committed selection, canonical evidence, request/output SHA-256 commitments, and replay result without rebuilding them. Install the strong local FT surface with `pip install "spiraltorch[hf-full-finetune]"`; `hf-gpt2-ft` remains as a legacy extra/preset alias.
- **LLM local HF FT bridge:** `PYTHONPATH=bindings/st-py python3 bindings/st-py/examples/hf_finetune_bridge.py --model-configs bindings/st-py/examples/hf_finetune_model_configs.example.json --model-profile qwen2-0.5b-local-smoke --metadata-only --allow-remote --zspace-probe --run-card ft-run-card.json` loads `AutoModelForCausalLM`/tokenizer/dataset metadata after the full HF contract passes, records SpiralTorch WGPU/CPU readiness plus optional Z-Space token-probe metrics, and writes a run card. Model-specific defaults live in `hf_finetune_model_configs.example.json`; start with `causal-lm-local-smoke`, choose `gpt2-local-smoke`, `distilgpt2-local-smoke`, `pythia-70m-local-smoke`, `qwen2-0.5b-local-smoke`, or copy `local-causal-lm-template` for a local checkpoint. Profiles now carry model-family, parameter-scale, and activation-hook hints alongside tokenizer/training/generation defaults, so widening beyond GPT-2 should usually mean editing config rather than script code. Add `--train --train-file data/corpus.txt --validation-fraction 0.02 --max-train-samples 50000 --block-size 128 --output-dir runs/hf-finetune` to enter a real local `AutoModelForCausalLM` / `Trainer` FT loop from local text files; `spiral-hf-profile --launch-plan --train` resolves its default `--mode auto` to `full-finetune`, so launch bundles audit the same stronger dependency surface that real training needs. Use repeated `--train-file` / `--validation-file` plus `--dataset-format text|json|csv` for larger corpora, and add `--corpus-scan` before long runs to stream line/byte/sample stats into the run card without loading the model. Train runs emit `spiraltorch-hf-finetune-trainer-trace.jsonl` from the generic bridge unless overridden; legacy `hf_gpt2_*` Python helper names remain available, while `hf_finetune_*` core helpers now normalize run-card, eval, generation, telemetry, trace, and summary row types for model-neutral imports.
- **LLM local HF LoRA/PEFT:** select `causal-lm-lora-local-smoke`, `gpt2-lora-local-smoke`, `qwen2-0.5b-lora-local-smoke`, or `smollm2-135m-lora-local-smoke` with `spiral-hf-finetune --model-profile <profile> --train ...` to attach a model-family-aware PEFT adapter through the same audited Trainer path. `spiral-hf-profile --launch-plan --train` resolves these profiles to the Trainer-ready `hf-peft-finetune` preflight; direct runs can use `--finetune-mode lora --lora-rank 16 --lora-alpha 32 --gradient-checkpointing`. Run cards retain matched target modules, trainable/frozen counts and ratio, PEFT version, artifact kind, and adapter-save evidence. Successful Trainer saves place the tokenizer beside both full-model and adapter outputs. Adapter-only outputs can return directly to `spiral-hf-finetune --model-name <adapter> --finetune-mode lora` for weights-only continuation without double attachment; each successful save now writes a content-addressed parent/root lineage manifest. Audited continuations carry `--expected-parent-adapter-id`, `--expected-parent-lineage-depth`, and `--expected-root-adapter-id` for the parent adapter plus `--expected-training-input-id` for the path-independent bundle of local model config, ordered corpus/validation files, distortion artifacts, and recursive resume checkpoint. Exact checkpoint resumes also write collision-safe immutable trainer-trace segments, seal parent/current digests in the run card, and revalidate them through adapter-chain and executor audits instead of truncating historical telemetry. Remote Hub corpora independently resolve aliases/branches to a canonical repository commit and propagate `--expected-dataset-input-id`; config, split, text-column, repository, or revision drift is rejected before model loading. After row selection, local and remote runs additionally adopt `--expected-dataset-materialization-id`, hashing the exact ordered train/eval text bytes so mutable external builders, shuffle changes, or row-content drift fail before tokenization and Trainer work. Training runs then adopt `--expected-tokenized-dataset-id`, hashing every post-grouping block and column so changed token IDs, masks, labels, or block boundaries fail before model preparation or Trainer construction. The bridge fingerprints these contracts at their corresponding load boundaries and carries them through adapter lineage, scale-up, and executor continuations. `--eval-before-train --adapter-promotion-gate` requires changed weights plus bounded before/after eval regression, releases the training model/accelerator cache, then launches a separate Python worker for local-only PEFT reload and deterministic bounded generation before promotion. Tune that qualification with `--adapter-promotion-probe-prompt`, `--adapter-promotion-probe-max-new-tokens`, `--adapter-promotion-probe-device`, and `--adapter-promotion-probe-timeout-seconds`; sweeps forward the same knobs, promotion chains revalidate gated roots as well as descendants, and scale-up artifacts retain the exact probe path/device/token/PID/exit evidence they trusted. `spiral-hf-adapter-lineage` and `spiral-hf-adapter-promote --require-artifact-probe` apply the same contracts to existing artifacts, and `st.hf_finetune_checkpoint_resume_report(...)` distinguishes fresh-schedule warm start from exact optimizer/scheduler resume and warns when a saved schedule is already exhausted. `st.load_hf_causal_lm_artifact(...)` reconstructs adapters for local trace, Z-Space inference, generation sweeps, and checkpoint audit; `spiral-hf-adapter-export --adapter <adapter-dir> --output-dir <merged-dir>` creates an atomic standalone model with merge provenance. The lazy attachment API remains `st.prepare_hf_finetune_model(...)`.
- **LLM Z-Space optimizer factorized ablation:** the [HF ablation guide](../hf_zspace_optimizer_ablation.md) calibrates one Rust-owned LR trajectory, separates integrated dose from schedule shape, verifies the optimizer's actual effective-LR dose, and compares matched `observe`, `dose_matched_constant`, `raw`, and `dose_normalized` run cards. The Rust-owned `dose_preserving_complement` policy reverses the centered trajectory at exactly matched dose; its audited local GPT-2 study beat both ordinary FT and the original normalized shape for 3/3 seeds while retaining a bounded single-recipe evidence label. The audited multi-corpus study repeated that protocol over three content-distinct corpora: normalized was worse than ordinary FT for 9/9 matched seeds, while the complement beat both arms for 9/9, with all 3/3 corpus means agreeing and a Rust-owned corpus-equal polarity mean of `-0.001712`. This remains a single-model, short-recipe trend rather than an efficacy claim. The runner delegates balanced seed validation and corpus-equal aggregation to `st-core`, rather than pooling seed rows in Python. `--zspace-optimizer-feedback loss_guard` separately routes a proposal through a checkpointed Rust loss/delta gate. The factorized, polarity, corpus-polarity, and feedback study runners freeze and resume their evidence with content, Git, journal, policy-artifact, and run-card identities.
- **LLM Z-Space optimizer feedback evidence:** the [audited three-seed study](../hf_zspace_optimizer_ablation.md#audited-feedback-result-2026-08-09) found that the Rust `loss_guard` reduced the same open-loop harm for 3/3 seeds and recovered 56.3% of its mean loss penalty. The guarded arm still beat ordinary FT for 0/3 seeds, so the checked artifact supports mitigation of this failure mode, not an efficacy claim.
- **LLM immutable trainer-trace lineage:** every completed bridge run rebuilds the sealed root-to-tip trace lineage, writes its report plus cumulative summary into the run card, and carries the lineage ID through sweep selection, adapter transitions, and executor telemetry evidence. `st.hf_finetune_trainer_trace_lineage_report(...)` revalidates segment/receipt IDs, parent digests, paths, and boundaries; `st.load_hf_finetune_trainer_trace_lineage(...)` returns ordered rows annotated with segment identity; and `st.summarize_hf_finetune_trainer_trace_lineage(...)` restores the full loss/eval/telemetry curve across exact resumes. Repeated steps from retrying the same checkpoint remain visible as overlap warnings instead of corrupting integrity. Live geometry guards continue to evaluate only the current segment, while generation curves and run-artifact handoffs use the verified cumulative lineage and active tip.
- **LLM local HF recipe identity:** run the intended command once with `--training-recipe-only` to resolve effective Transformers optimizer/scheduler defaults, batch/seed/precision, applied LoRA/full-FT trainability, dtype, resume state, collator, and loss-guard control without constructing Trainer. The run card rewrites its canonical command to `--train --expected-training-recipe-id <sha256:...>`; exact replay fails before `Trainer(...)` on recipe drift, while scale-up explicitly reissues this layer instead of incorrectly enforcing the parent's shorter schedule.
- **LLM local HF full replay identity:** the same adoption pass now binds adapter lineage, local or Hub data source, selected raw rows, tokenized Trainer blocks, model/tokenizer runtime, software/device execution, and the effective recipe into one path-independent `--expected-finetune-replay-id`. Canonical replay still carries each layer's expected ID for earlier diagnostics, then verifies the composite immediately before `Trainer(...)`; intentional scale-up strips and reissues the parent composite because it represents a new run.
- **LLM local HF artifact probe:** `spiral-hf-artifact-probe runs/hf-finetune --prompt "SpiralTorch is" --max-new-tokens 16 --out artifact-probe.json` reconstructs either a full model or PEFT adapter in an isolated worker process, runs bounded generation, and records model family/classes, resolved base/tokenizer, device, token counts, continuation text, timing, worker/parent PIDs, exit status, and timeout evidence. The parent passes probe inputs through a private JSON request rather than repeating the prompt in the worker argv. It is local-files-only by default; add `--allow-remote` when the adapter's base model is not cached, `--device cpu|mps|cuda` for an explicit runtime, `--timeout-seconds 900` for the worker bound, or use `st.hf_causal_lm_artifact_subprocess_probe_report(...)` directly. `st.hf_causal_lm_artifact_probe_report(...)` remains the lower-level same-process diagnostic. Promotion-gated Trainer runs invoke the isolated route automatically after saving and archive it as `spiraltorch-hf-artifact-probe.json`. The [Pythia 70M LoRA qualification sample](../../bindings/st-py/examples/hf_pythia70m_lora_artifact_probe_sample.json) records a qualified root plus ten real promoted non-GPT-2 continuation generations through target resolution, warm-start continuation without double attachment, runtime release, isolated local-only MPS reload/generation, promotion revalidation, transition audit, executor postflight, policy stop, canonical wheel-native continuation, parent-adapter and local-training-input gates, a model/tokenizer runtime identity adopted at depth seven then enforced at depth eight, and a path-independent software/device execution identity adopted at depth nine then enforced at depth ten.
- **LLM local HF FT sweep:** `PYTHONPATH=bindings/st-py python3 bindings/st-py/examples/hf_finetune_sweep.py --model-configs bindings/st-py/examples/hf_finetune_model_configs.example.json --model-profile pythia-70m-local-smoke --train-file data/corpus.txt --validation-fraction 0.02 --corpus-scan --generation-prompt "SpiralTorch is" --generation-from-inference-distortion --eval-before-train --zspace-probe --trainer-telemetry --inference-distortion-probe runs/inference-distortion/live-probe.json --block-size-values 64,128 --learning-rate-values 0.0001,0.00005 --seed-values 7,13 --out-dir runs/hf-finetune-sweep` runs multiple local bridge configurations, writes `sweep-plan.json` before launching, writes `sweep-report.json` afterward, and embeds comparison plus summary payloads so block size, learning rate, seed, eval deltas, generation changes, Z-Space probes, inference-distortion handoff, provider request dropped/sent keys, desire/psi trainer telemetry, and trainer traces can be compared before scaling the corpus. For LoRA sweeps, add `--eval-before-train --adapter-promotion-gate`: blocked/evidence-incomplete rows remain auditable, but only promotion-ready adapters can be selected or reused, and `scale-up-command.json` automatically continues the winning adapter weights as the next lineage parent rather than restarting from its original model. Use `spiral-hf-scale-up ... --adapter-continuation replay` for configuration-only replay or `--adapter-continuation continue` to require a non-gated adapter handoff; preflight checks adapter files, input/output separation, lineage depth/fingerprint, and promotion evidence. A sweep with no ready candidate exits nonzero. The generic bridge/sweep wrappers rewrite run-card, sweep-plan, sweep-report, and scale-up-command artifacts to model-neutral row types. Use `--inference-distortion-sweep-report runs/zspace-inference-distortion-sweep/sweep-report.json` instead when a multi-probe grid has already been ranked. Add `--dry-run` to inspect commands, `--resume-existing` to continue interrupted sweeps from successful run cards, `--force` to rerun every row intentionally, or `--require-wgpu-ready` to gate each run on SpiralTorch WGPU readiness; use `st.summarize_hf_finetune_sweep_report(...)` to recover the selected scale-up run, command/run-card/trace paths, and top candidates from Python.
- **LLM local HF adapter promotion chain:** `spiral-hf-adapter-chain runs/hf-finetune-study --out runs/hf-finetune-study/promotion-chain.json --require-continuation-ready` re-fingerprints every discovered adapter, validates parent/root/depth and promotion/run-card evidence, preserves failed branches, and selects a unique deepest ready tip. Its `transitions` rows make each parent-to-child depth step, fingerprint/weight check, eval handoff and improvement, promotion revalidation, isolated-probe PID/exit evidence, and pinned training, model/tokenizer, and execution-environment identities directly auditable instead of requiring node-by-node inference. Once a generation adopts the composite fine-tune replay identity, every descendant must publish a verified replacement: reusing or dropping the parent composite blocks the edge because adapter input and effective recipe are generation-scoped. Pass that report directly to `spiral-hf-scale-up ... --write-command next-generation.json --require-ready`; the command artifact and preflight retain the selected transition, inject the expected parent ID/depth/root and input IDs into the child launch, and resolve recorded relative model-config/corpus/distortion/checkpoint inputs against the source run card's `launch_cwd`. New FT run cards record that working directory and their exact launch command, so a promoted generation can become the next parent without returning through a sweep or depending on the executor's current directory. Known historical bridge-script and `spiral-hf-finetune` commands are rewritten to the current interpreter's `python -m spiraltorch.hf_finetune_entrypoint`, while the original prefix remains in provenance and unknown custom launchers remain untouched. `spiral-hf-adapter-executor` carries runtime, all input identities, recipe/composite reissue contracts, and the selected edge through state, pending plans, attempts, recovery, postflight, and read-only status/runtime output, and refuses promotion when transition, an enforced identity contract, or module evidence is not ready. Equal-depth forks require `--select-adapter-id`, while pre-lineage local seeds are inferred only after their live fingerprint matches the declared root ID.
- **LLM local HF FT ops (installed wheel):** `spiral-hf-run-status runs/hf-finetune`, `spiral-hf-status-history runs/hf-finetune/direct-run-status-history.jsonl`, `spiral-hf-wait-launch --manifest next.json --dry-run -- spiral-hf-finetune ...`, `spiral-hf-wait-launch-summary runs/hf-finetune/long-run-wait-launch-history.jsonl`, `spiral-hf-milestone-capture runs/hf-finetune --milestone-step 4096`, `spiral-hf-milestone-runtime runs/hf-finetune --execute`, `spiral-hf-run-artifacts runs/hf-finetune`, and `spiral-hf-run-ops runs/hf-finetune` keep long local HF runs monitorable and handoff-ready through generic `hf_ft_*` line prefixes and artifact names; the importable `spiraltorch.hf_finetune_*` ops helpers now emit model-neutral row types too, while legacy GPT-2 scripts remain compatible.
- **LLM local HF FT long-run controls:** add `--max-eval-blocks N`, `--eval-after-train-policy skip-if-final-step-eval`, and the default `--dataloader-pin-memory auto` when local MPS/CPU evaluation becomes the bottleneck; the generic bridge records both pre-cap and post-cap eval block counts plus the resolved dataloader settings in the run card for any configured `AutoModelForCausalLM` profile.
- **LLM local GPT-2 FT generation sample:** [`bindings/st-py/examples/hf_gpt2_finetune_generation_sample.json`](../../bindings/st-py/examples/hf_gpt2_finetune_generation_sample.json) records a sanitized long-run sample where the fixed prompt shifts from generic Torch/Tor wording before training to SpiralTorch runtime / Python bindings wording after 640/1280/2560 local GPT-2 FT steps, alongside eval-loss and trace-throughput evidence.
- **LLM Z-Space generation control sample:** [`bindings/st-py/examples/hf_gpt2_zspace_generation_control_sample.json`](../../bindings/st-py/examples/hf_gpt2_zspace_generation_control_sample.json) records the same 2560-step local GPT-2 prompt after `ZSpaceRepressionLogitsProcessor` control, showing native `spiraltorch_zspace_softmax` telemetry and repetition-repression top-token changes that relax the greedy wrapper loop.
- **LLM Z-Space generation control sweep:** `PYTHONPATH=bindings/st-py python3 bindings/st-py/examples/hf_zspace_generation_control_sweep.py --model-configs bindings/st-py/examples/hf_finetune_model_configs.example.json --model-profile pythia-70m-local-smoke --prompt "SpiralTorch is a tensor and geometry runtime that" --zspace-entropy-target-values none,3.0 --repression-strength-values 0.0,0.75,1.25 --last-token-repression-values 0.0,1.0 --out runs/hf-zspace-generation-control-sweep.json` loads one local or remote `AutoModelForCausalLM` once, runs a small Z-Space/repression decode grid without Trainer or datasets, and writes generated text, bounded `generation_control` telemetry, and loopiness metrics for fast post-FT decoding experiments. Swap profiles to move beyond GPT-2, or pass `--model-name runs/gpt2-small-zspace-ft --tokenizer-name gpt2` for a direct checkpoint replay; dry-run artifacts now include `generation_control_profile_config`, `generation_control_grid`, and reusable sweep/bridge CLI args resolved from the selected profile. Use `st.load_zspace_generation_control_sweep(...)`, `st.summarize_zspace_generation_control_sweep(...)`, or `st.summarize_zspace_generation_control_sweep_lines(...)` to rank runs by loop score and top-token changes from Python, including `recommended_config`, `recommended_processor_kwargs`, `recommended_sweep_cli_args`, and `recommended_bridge_cli_args` for the next decode run. The Python helpers `st.zspace_generation_control_profile_config(...)`, `st.zspace_generation_control_processor_kwargs(...)`, `st.zspace_generation_control_sweep_cli_args(...)`, and `st.zspace_generation_control_bridge_cli_args(...)` also accept or resolve HF profiles/runtime plans, so profile generation defaults can be reused directly without hand-copying Z-Space knobs.
- **LLM Z-Space profile generation defaults:** `spiral-hf-profile --model-configs bindings/st-py/examples/hf_finetune_model_configs.example.json --model-profile pythia-70m-local-smoke --generation-control-config --json` resolves the selected HF profile into reusable `processor_kwargs`, sweep CLI args, and bridge CLI args, keeping model-specific generation/Z-Space knobs in config samples instead of GPT-2-specific scripts.
- **LLM checkpoint generation-control plans:** `spiral-hf-checkpoint-generation-control --dry-run --run-dir runs/hf-finetune --checkpoint checkpoint-4096 --model-configs bindings/st-py/examples/hf_finetune_model_configs.example.json --model-profile pythia-70m-local-smoke --run-card checkpoint-control.json` builds checkpoint replay, compare, and optional curve commands while recording the same `generation_control_profile_config` plus reusable sweep/bridge CLI args at the plan level.
- **LLM Z-Space generation control grid sample:** [`bindings/st-py/examples/hf_gpt2_zspace_generation_control_grid_sample.json`](../../bindings/st-py/examples/hf_gpt2_zspace_generation_control_grid_sample.json) records a compact 13-run local GPT-2 2560-step grid where softmax-only keeps the greedy wrapper loop, while repression changes top-token ranks and drives the measured loop score from `13.0` to `0.0`.
- **LLM Z-Space inference distortion probe:** `PYTHONPATH=bindings/st-py python3 bindings/st-py/examples/zspace_inference_distortion_probe.py --model-configs bindings/st-py/examples/hf_finetune_model_configs.example.json --model-profile pythia-70m-local-smoke --prompt "SpiralTorch is a tensor and geometry runtime that" --out runs/zspace-inference-distortion-probe.json` applies one shared desire/psi distortion adapter to local HF logits, local activation hooks, and an API-model-shaped runtime context before re-FT; installed wheels expose the same route as `spiral-zspace-inference-distortion-probe`. Model/tokenizer and local activation-hook defaults now come from the same profile file used by FT (`causal-lm-local-smoke`, `gpt2-local-smoke`, `distilgpt2-local-smoke`, `pythia-70m-local-smoke`, `qwen2-0.5b-local-smoke`, or a copied `local-causal-lm-template`), so Pythia uses `gpt_neox.layers.0`, Qwen/SmolLM-style models use `model.layers.0`, OPT uses `model.decoder.layers.0`, and GPT-2 profiles keep `transformer.h.0` unless overridden. From Python, `st.zspace_inference_distortion_runtime_plan(model_profile="qwen2-0.5b-local-smoke")` resolves the same local model, tokenizer, generation, and activation-hook defaults. Use `--local-model <checkpoint-or-model>`, `--tokenizer-name <tokenizer-or-dir>`, or `--activation-name-contains <module-substring>` only when overriding the profile or targeting a freshly trained checkpoint. Without a local model it still runs a keyless fake API path; use `--api-provider openai-responses|openai-chat|anthropic --api-model <model>` to send the same request/context distortion through a live provider, add `--from-sweep-report runs/zspace-inference-distortion-sweep/sweep-report.json` to replay the sweep-recommended prompt/runtime/config, `st.summarize_zspace_inference_distortion_probe(...)` to flatten changed-text, top-token-change, activation, and API telemetry into one comparison row, or `st.compare_zspace_inference_distortion_probes(...)` to rank several probe artifacts before choosing the next FT/decode setting.
- **LLM Z-Space inference distortion sweep:** `PYTHONPATH=bindings/st-py python3 bindings/st-py/examples/zspace_inference_distortion_sweep.py --model-configs bindings/st-py/examples/hf_finetune_model_configs.example.json --model-profile qwen2-0.5b-local-smoke --prompt "SpiralTorch is a tensor and geometry runtime that" --desire-pressure-values 0.45,0.8 --psi-total-values 0.5,0.75 --coherence-values 0.35,0.55 --api-provider fake --out-dir runs/zspace-inference-distortion-sweep` writes a reusable `sweep-plan.json`, per-setting probe artifacts, `sweep-report.json`, and `sweep-report.md` with `compare_zspace_inference_distortion_probes(...)` results, a recommended probe/config, replay commands, installed replay commands for `spiral-zspace-inference-distortion-probe` / `spiral-zspace-inference-distortion-sweep`, and Python summaries via `st.load_zspace_inference_distortion_sweep(...)` / `st.summarize_zspace_inference_distortion_sweep(...)` so local HF internal hooks and API-model request/context distortion can be compared before the next FT run. Profile-derived replay commands preserve `--model-configs` / `--model-profile` and only spell out explicit overrides, keeping model-specific runtime choices in the config sample. Swap profiles to move beyond GPT-2 without editing scripts; the selected profile carries the default activation hook substring unless you pass `--activation-module-name` or `--activation-name-contains` explicitly. Pass `--local-model runs/gpt2-small-zspace-ft --tokenizer-name gpt2` for a direct checkpoint replay, swap `--api-provider` to `openai-responses`, `openai-chat`, or `anthropic` with `--api-model <model>` when you want the same pressure grid to hit a live hosted model, add `--resume-existing` to continue an interrupted/costly sweep without re-calling matching successful probes, `--report-only` to rebuild comparisons from existing probe JSON, `--from-probe runs/inference-distortion/live-probe.json` to promote one or more saved local/API probe artifacts into a reusable pre-FT sweep report without re-calling providers, or `--force` to intentionally rerun every row.
- **LLM (API model + Z-space runtime):** `PYTHONNOUSERSITE=1 python3 -S -s bindings/st-py/examples/api_llm_zspace_runtime.py` shows how an OpenAI-compatible response or arbitrary hosted-model callable becomes a Z-space partial trace with device preflight evidence, usage/latency telemetry, and posterior confidence. With `OPENAI_API_KEY` plus `pip install openai`, run `PYTHONPATH=bindings/st-py python3 bindings/st-py/examples/openai_api_llm_zspace_runtime.py --prompt "Describe SpiralTorch entering Z-space runtime."` to hit the OpenAI Responses API through the same bridge. With `ANTHROPIC_API_KEY` plus `pip install anthropic`, run `PYTHONPATH=bindings/st-py python3 bindings/st-py/examples/anthropic_api_llm_zspace_runtime.py --prompt "Describe SpiralTorch inference as bipolar geometry."` to route Anthropic Messages into the same trace path.
- **LLM (API model prompt suite):** `PYTHONNOUSERSITE=1 PYTHONPATH=bindings/st-py python3 -S -s bindings/st-py/examples/api_llm_prompt_suite.py` runs several hosted-model-shaped prompts through one Z-space runtime, writes a compact JSONL artifact, and compares the suite without network access. `PYTHONNOUSERSITE=1 PYTHONPATH=bindings/st-py python3 -S -s bindings/st-py/examples/api_llm_provider_suite_compare.py` runs the same prompts through OpenAI-shaped and Anthropic-shaped callables, writes one JSONL artifact per route, and ranks the provider matrix by the same Z-space route score.
- **LLM (API model open-topos sweep):** `PYTHONPATH=bindings/st-py python3 bindings/st-py/examples/api_llm_topos_sweep.py --prompt-limit 3 --context-prompt --out-dir /tmp/spiraltorch-topos-sweep` compares open/contextual/guarded topological runtime adapters around one provider-shaped callable, writes one JSONL trace per route plus `report.json` with bounded response previews, route scorecards, pairwise route deltas, response-side route winners, and balanced/quality/grounded/efficiency/latency selection profiles, and works offline by default. Add `--live-provider openai-responses|openai-chat|anthropic --model <model>` with the matching SDK/key installed to replay the same sweep through a hosted model.
- **RL (stAgent policy trace):** `PYTHONPATH=bindings/st-py python3 bindings/st-py/examples/stagent_policy_trace.py --steps 200 --jsonl-out /tmp/stagent-policy.jsonl` runs a tiny native DQN bandit loop, emits per-step `select_action_trace()` records with Q-values/epsilon/explore-vs-greedy metadata, and prints a final `policy_report()` summary for handoff into Z-space/topos route selectors.
- **LLM × RL (topos route policy):** `PYTHONPATH=bindings/st-py python3 bindings/st-py/examples/api_llm_topos_stagent_route_policy.py --out-dir /tmp/spiraltorch-topos-policy --profile grounded --policy-out /tmp/spiraltorch-topos-policy/policy.json` turns an open-topos sweep report into bounded route rewards, trains an stAgent-shaped policy, and prints the selected route plus Q-value trace, reusable `policy_selection`, request controls, and runtime route. The Rust-owned v2 score contract treats missing metrics neutrally, applies sample-count shrinkage, omits unobserved routes, and revalidates reward source evidence during resolution. Stored v1 reward arrays do not contain that witness and must be rebuilt from their original sweep rows before resolution; legacy rows without a positive observation `count` must be remeasured. Pass `--report /tmp/spiraltorch-topos-sweep/report.json` to reuse a live or offline sweep without re-calling providers.
- **LLM (live API provider matrix):** `PYTHONPATH=bindings/st-py python3 bindings/st-py/examples/api_llm_live_provider_matrix.py --prompt-limit 12 --repeat 3 --near-best-tolerance 0.02 --out-dir /tmp/spiraltorch-live-matrix` replays a larger prompt population through OpenAI Responses plus any available Anthropic routes (`claude-opus-4-8`, `claude-fable-5` by default), records per-route JSONL traces, and writes `report.json` with route settings, route score, near-best routes, quality/efficiency/text-quality score breakdowns, prompt coverage, repetition, latency, token, refusal, empty-text, completion-rate comparisons, and `selection_profiles` for balanced, quality, grounded, efficiency, or latency-sensitive routing. Claude 5/Opus 4.8 routes use adaptive thinking with `output_config.effort` rather than non-default sampling parameters, so keep `--anthropic-max-tokens` comfortably above the visible answer size to avoid measuring budget truncation instead of model behavior.
- **LLM (live API provider matrix sweep):** `PYTHONPATH=bindings/st-py python3 bindings/st-py/examples/api_llm_live_provider_matrix_sweep.py --prompt-limit 12 --repeat 3 --budget-pairs 192:768,256:1024 --resume-existing --out-dir /tmp/spiraltorch-live-matrix-sweep` runs multiple live matrix budgets, writes one `report.json` per budget pair, then writes `sweep-report.json` with `compare_api_llm_matrix_reports(...)` so profile-winner stability and route tradeoffs can be audited across repeated sweeps. Use `--dry-run` to inspect the planned matrix without calling provider APIs, `--resume-existing` to avoid re-calling APIs for completed budget pairs, or `--force` to rerun them intentionally.
- **LLM (live OpenAI + WASM context):** `PYTHONPATH=bindings/st-py python3 bindings/st-py/examples/openai_api_llm_wasm_context_runtime.py --wasm-report report.json --include-context-prompt --write-wasm-context-artifact /tmp/spiraltorch-wasm-context.json --trace-jsonl /tmp/spiraltorch-openai-wasm-trace.jsonl` sends a browser-exported WASM learning report through the OpenAI Responses bridge, optionally prepends bounded Z-space/WASM telemetry to the hosted-model prompt, persists the selected context handoff, and records trace telemetry such as `wasm.loss` and `wasm.webgpu_device_ready` beside the hosted response.
- **LLM (API model trace comparison):** `PYTHONNOUSERSITE=1 PYTHONPATH=bindings/st-py python3 -S -s bindings/st-py/examples/api_llm_trace_compare.py` compares multiple API LLM trace JSONL artifacts by route score, quality score, efficiency score, deterministic text-quality score, prompt coverage, confidence, latency, token use, runtime readiness, health penalties, near-best routes, profile-specific route selections, Z-space metrics, and attached WASM context signals such as browser-side loss and WebGPU readiness. The typed Rust `spiraltorch.api_llm_route_policy.v1` contract owns every normalization, score, winner, rank, and near-best decision; Python gathers traces and renders its evidence witness, while WASM exposes the same contract to browser clients.
- **LLM (API model report comparison):** `PYTHONNOUSERSITE=1 PYTHONPATH=bindings/st-py python3 -S -s bindings/st-py/examples/api_llm_report_compare.py` compares multiple live provider-matrix `report.json` files with `compare_api_llm_matrix_reports(...)`, then summarizes profile-winner stability, route-score means, latency/token tradeoffs, skipped providers, client-error counts, carried WASM context loss/WebGPU readiness, and whether selected browser contexts were consistent across repeated sweeps.
- **LLM (coherence scan, raw text, no tokenizer):** `cargo run -p st-nn --example modelzoo_llm_char_coherence_scan -- <text.txt> [--context-scale 0.05 --mix-rms 0.1 --head-rms 0.1 --head-residual-scale 1.0 --head-prior unigram|bigram|learned-bigram --bigram-topk-guard 0.05 --bigram-topk-guard-k 5] [--val-fraction 0.1 --eval-samples 256]`
- **LLM (Python, coherence scan, raw text, no tokenizer):** `PYTHONNOUSERSITE=1 python3 -S -s models/python/llm_char_coherence_scan.py <text_or_dir> [<text_or_dir> ...] [--context-scale 0.05]`
- **LLM (coherence wave, raw text, no tokenizer):** `cargo run -p st-nn --example modelzoo_llm_char_coherence_wave -- <text.txt> [--mix-rms 0.1 --head-rms 0.1 --head-residual-scale 1.0 --head-prior unigram|bigram|learned-bigram --bigram-topk-guard 0.05 --bigram-topk-guard-k 5] [--val-fraction 0.1 --eval-samples 256] [--infuse \"spiral\" --infuse-every batch --infuse-mode separate]`
- **LLM (Python, coherence wave, raw text, no tokenizer):** `PYTHONNOUSERSITE=1 python3 -S -s models/python/llm_char_coherence_wave.py <text_or_dir> [<text_or_dir> ...] [--infuse \"spiral\" --infuse-every batch --infuse-mode separate]`
- **LLM (Python, WaveRnn+Mixer, attentionless):** `PYTHONNOUSERSITE=1 python3 -S -s models/python/llm_char_wave_rnn_mixer.py <text.txt>`
- Example (Python, desire + atlas): `PYTHONNOUSERSITE=1 python3 -S -s models/python/llm_char_coherence_wave.py models/samples/spiral_demo_en.txt --desire --events models/runs/demo_desire/events.jsonl --atlas --run-dir models/runs/demo_desire`
- WGPU quickstart (build + run): `bash scripts/wgpu_quickstart.sh`
- **Vision (Conv/Pool):** `cargo run -p st-nn --example modelzoo_vision_conv_pool_classification`
- **Vision (Python, Conv/Pool):** `PYTHONNOUSERSITE=1 python3 -S -s models/python/vision_conv_pool_classification.py`
- **GNN (roundtable graph regression):** `cargo run -p st-nn --example modelzoo_gnn_graph_regression`
- **GNN trace (band replay coefficients):** `cargo run -p st-nn --example gnn_trainer_band_trace_demo`
- GNN graph-level training uses `ZSpaceGraphBatchRegressor` for fixed-size graph mini-batches, so row-concatenated node tensors read out to one prediction row per graph instead of collapsing the whole batch into one graph.
- GNN band-trace sweeps can grid `--epoch-values`, `--batch-values`, `--node-values`, `--feature-values`, graph-count values, `--lr-values`, and roundtable axes such as `--top-k-values` / `--here-tolerance-values`; generated run names, group averages, `Top Validation Candidates`, and `Roundtable Axis Deltas` include those axes so validation-wide readout MSE, band replay deltas, step time, and CPU debt can be compared without mixing unlike shapes or schedules. The same candidate and delta records are stored under `comparison` in `sweep.json`, and `--follow-up-from previous/sweep.json` replays a ranked top validation candidate while still allowing explicit grid overrides such as new seeds or wider `--top-k-values`. Add `--follow-up-neighborhood --follow-up-neighborhood-axes lr,top_k,bottom_k` to automatically fan out a local schedule search around that candidate without hand-copying the winning axes; follow-up reports also emit `Follow-Up Result` / `comparison.follow_up_result` with the new best-vs-source validation MSE verdict, `Follow-Up Promotion` / `comparison.follow_up_promotion` naming the candidate to carry forward, and `--follow-up-fail-on-verdict regressed,unknown` writes `Follow-Up Gate` / `comparison.follow_up_gate` while turning that verdict into an optional non-zero gate. A later `--follow-up-from` uses promotion by default when present (`--follow-up-source auto`), with `--follow-up-source top-candidate` available when you want to ignore the conservative promotion choice; `Next Follow-Up Command` / `comparison.follow_up_next_command` provides a runnable template for the next chained sweep, and `config.follow_up.lineage` records parent path/run root plus follow-up generation for multi-step chains.
- Multi-seed GNN comparisons also add seed stability fields to `Top Validation Candidates` and mirror them as `Stable Validation Candidates` / `comparison.stable_validation_candidates`, ranking schedules by `avg_validation_readout_mse + validation_mse_stddev` so low-average but volatile candidates can be separated from repeatable wins before promotion. Trace and compare artifacts also include `validation_readout_nmse` / `avg_validation_readout_nmse`, normalizing validation MSE by target mean-square energy so seed shifts caused by harder target draws can be diagnosed separately from schedule effects.
- Follow-up compare artifacts also emit `Follow-Up Chain` / `comparison.follow_up_chain`, `Follow-Up Ancestors`, `Follow-Up Chain Guidance` / `comparison.follow_up_chain_guidance`, and `Guided Next Follow-Up Command` / `comparison.follow_up_guided_next_command`, so parent generation, source mode, selected candidate source, prior verdicts, promotion actions, verdict streaks, candidate stability status, raw/NMSE replay deltas, and a replay/neighborhood command template with explicit `NEXT_RUN_ROOT` / `NEW_SEEDS` placeholders stay visible across multi-step GNN tuning runs. If an `improved` candidate is still `single_seed_probe` or `volatile`, guidance asks for fresh seeds before continuing promotion; if repeated improvements remain `volatile`, it switches to `widen_stability_search` with `--follow-up-neighborhood --seeds NEW_SEEDS`, and if volatility survives that neighborhood pass it emits `increase_sample_budget` with doubled epoch/train/validation graph values. When a larger-budget run regresses on average but surfaces a more stable top candidate, `review_stability_tradeoff` reruns that top candidate with fresh seeds instead of silently discarding the stability win; if the review only finds a tiny average improvement that is less stable than the source, `keep_source_stability_guard` keeps the seed-stable source and guides another fresh-seed confirmation. Once repeated improvements are `multi_seed_stable` / `watch_spread`, `explore_stable_neighborhood` anchors the promoted candidate and adds `--follow-up-neighborhood --seeds NEW_SEEDS` so the next run broadens locally instead of replaying the same stable point. A stable neighborhood win uses `confirm_stable_promotion` with fresh seeds before broadening again. Follow-up results also surface `source_replay_*` fields when the new sweep re-runs the source schedule; if a neighborhood looks regressed only because the source replay shifted upward on fresh seeds while the best neighbor matches or beats that replay, `review_seed_shift_neighborhood` repeats the source-anchored neighborhood with fresh seeds before locking the old source. If raw MSE regresses but source replay NMSE does not, `review_target_scale_shift` reruns with fresh seeds and a wider validation graph budget before treating the run as a schedule loss. If seed-shift evidence is itself `volatile`, `increase_seed_shift_validation_budget` keeps the source anchor but doubles `--validation-graph-values` before another promotion decision; if seed-shift regressions repeat, `widen_seed_shift_neighborhood` widens locally with explicit fresh seeds instead of replaying an old seed surface, and after persistent seed-shift regressions `audit_seed_sensitivity` pauses widening to remeasure the source with a broader seed list and validation budget. The same guided command is written as `next_follow_up_command.sh` inside the run directory, using environment variables for required placeholders.
- GNN tensor-utility threshold grids use `tools/run_gnn_threshold_grid.py --thresholds 1,1024 ...`; WGPU rows preflight once by default, `grid.json` keeps preflight/failure rows, and `compare.md` reports predicted CPU/WGPU routes, actual utility op routing, validation-wide readout graph counts, and CPU debt.
- **Coherence (ZSpace VAE):** `cargo run -p st-nn --example modelzoo_zspace_vae_reconstruction`
- **Coherence (Python, ZSpace VAE):** `PYTHONNOUSERSITE=1 python3 -S -s models/python/zspace_vae_reconstruction.py`
- **Coherence (Python, Text→ZSpace VAE):** `PYTHONNOUSERSITE=1 python3 -S -s models/python/zspace_text_vae.py models/samples/spiral_corpus_en --mellin ramp`
- **Coherence (Python, Text VAE on/off compare):** `PYTHONNOUSERSITE=1 python3 -S -s models/python/zspace_text_vae_compare.py models/samples/spiral_corpus_en --mellin ramp`
- **LLM (Python, VAE context features):** `PYTHONNOUSERSITE=1 python3 -S -s models/python/llm_char_vae_context.py models/samples/spiral_corpus_en --features raw,reconstruction,latent,raw_latent,reconstruction_latent --feature-normalize-modes blocks,vector --hybrid-latent-scales 0.5,1.0 --seeds 7,13`; the sweep report surfaces mean NLL/accuracy plus win and near-win stability by seed, including hybrid raw+latent and reconstruction+latent context probes with normalization/latent-scale config grids, `best_config`, and a chained `next_follow_up_command.sh` fresh-seed confirmation script. Add `--latent-dims 8,16,32 --hidden-sizes 64,128` to compare VAE/head capacity in the same sweep; config reports and follow-up commands preserve the winning capacity. A later `--follow-up-from previous/summary.json` reuses that best config and source feature set by default, then writes `follow_up_result` / `Follow-Up Result` plus `follow_up_chain` / `Follow-Up Chain` with generation, ancestors, verdict history, and streaks. It also emits `follow_up_ancestors` / `Follow-Up Ancestors`, `follow_up_trajectory` / `Follow-Up Trajectory`, `follow_up_guidance` / `Follow-Up Guidance`, and `guided_next_follow_up_command` / `Guided Next Follow-Up Command` so parent run summaries, cumulative NLL movement, trajectory/source-feature conflict flags, promote/continue/stop decisions, tie-aware runner-up uncertainty, and the executable next-step script stay visible in the report; unsafe trajectory promotions disable the guided next command until the source-feature tradeoff is audited. Add `--follow-up-fail-on-verdict regressed,unknown` when automation should write the report but return non-zero for weak follow-up outcomes; generated follow-up scripts preserve that gate through `FOLLOW_UP_FAIL_ON_VERDICT`.
- **LLM (Python, live VAE run monitor):** `PYTHONNOUSERSITE=1 python3 -S -s tools/summarize_char_vae_live_runs.py previous/mainline_scale_up` summarizes long in-progress context runs, including remaining seeds, current feature, remaining active-seed features, completed-seed winners, and active best-so-far deltas.
- **LLM (Python, VAE context chain runner):** `PYTHONNOUSERSITE=1 python3 -P tools/run_char_vae_context_chain.py models/samples/spiral_corpus_en --preset small --follow-ups 1` runs a parent context-feature sweep, follows generated guidance with fresh seeds, and writes `chain.json` / `chain_report.md` so improved confirmations and seed-shift gate stops are both preserved as first-class experiment evidence. When the current best and runner-up are within combined seed uncertainty, generated follow-up commands boost confirmation to five fresh seeds; the chain runner records seed precedence in the chain artifacts and uses those command defaults unless a matching `--follow-up-seed-groups` entry explicitly overrides that follow-up, with `follow_up_seed_group_plan` mapping group slots to attempted / unused-after-stop / extra status, `follow_up_seed_resolution` plus its summary recording the actual per-follow-up seeds/source used, and `extra_explicit_seed_groups` / `unused_explicit_seed_groups` staying disjoint. Compare multiple chains with `PYTHONNOUSERSITE=1 python3 -P tools/summarize_char_vae_context_chains.py models/runs --recursive --markdown-out comparison.md --json-out comparison.json --command-out-dir commands --write-command-inspection`; the comparison report highlights the accepted champion separately from the absolute best chain when gate-stop evidence needs review before promotion, then emits a recommendation such as `continue_from_accepted` or `review_absolute_best` with the relevant summary paths and a command directory containing `recommended_next.sh`, runnable follow-up/review scripts, README/manifest context, bundled comparison JSON/Markdown artifacts, input chain sources, and strict inspection reports. Re-inspect an existing command bundle with `PYTHONNOUSERSITE=1 python3 -P tools/inspect_char_vae_command_bundle.py commands --strict --write-report`, or run the inspected next step through `PYTHONNOUSERSITE=1 python3 -P tools/run_char_vae_command_bundle.py commands --write-inspection-report --write-run-report --append-run-history --write-run-history-report`; regenerate `run_history.md` / `run_history_summary.json` without executing or appending via `PYTHONNOUSERSITE=1 python3 -P tools/run_char_vae_command_bundle.py commands --history-report-only`; run a bounded automation pass with `PYTHONNOUSERSITE=1 python3 -P tools/run_char_vae_history_loop.py commands --max-steps 3 --fail-on-max-steps-continuation --write-loop-report`, which defaults to non-zero for review/inspect final actions and can also fail when the step bound is exhausted while a runnable continuation remains. Generated bundle README/manifest files include the absolute runner command plus `run.json` / `run.md` paths, an append-only `run_history.jsonl`, a human-readable `run_history.md`, and a machine-readable `run_history_summary.json` for automation handoff. Generated recommendation wrappers record and `cd` back to the comparison-time working directory before running, while generated manifests/README files use absolute handoff paths so later inspection does not depend on the caller's cwd. The `small`/`base` presets scout hybrid latent scales through `4.0`; `capacity_scout` adds `--latent-dims` / `--hidden-sizes` grids around the hybrid4 recipe, `capacity_zoom` expands the upper-edge signal with `--latent-dims 12,16,24 --hidden-sizes 32,64` at scale `4.0`, `capacity_lockin` reruns the confirmed `latent_dim=12 hidden=64` raw+latent setting on five fresh seeds, and `capacity_train` keeps that capacity fixed while increasing the train/eval budget for longer learning passes. Each feature head saves both final `head_<feature>.json` and validation-best `head_<feature>_best.json` weights, with checkpoint health in the report; reload a run for evaluation with `--vae-load previous/text_vae_weights.bin --head-load-dir previous --eval-only`, or omit `--eval-only` to continue training from those heads. Add `--allow-gate-stop` when a gate stop should be recorded as evidence without failing the outer automation.
- **LLM (Python, promoted VAE recipe eval):** `PYTHONNOUSERSITE=1 python3 -P tools/run_char_vae_promoted_recipe.py previous/summary.json --json` inspects a promoted `mainline_scale_up_command.promoted_learning_recipe` and reports the per-seed eval-only reload commands; add `--ready-only --complete-only` to skip seeds whose VAE / requested best feature-head checkpoints or source run summary are not present yet, `--execute --seed 1043` to run a selected seed, or `--write-report` to persist `promoted_recipe_eval_run.json` / `.md` beside the source summary. Summarize executed reload evidence with `PYTHONNOUSERSITE=1 python3 -P tools/summarize_char_vae_promoted_eval_runs.py previous/promoted_recipe_eval_run.json`; once reload evidence promotes and the mainline run summary exists, the summary also surfaces the next mainline scale-up command plus a `screen` launch command that writes `run.log`.
- **Training (Lightning/selfsup):** `cargo run -p st-nn --example modelzoo_lightning_selfsup_minimal`


## Hello SpiralSession quickstart

The wheel ships a self-contained demo script that exercises the currently
supported Python surface:

```bash
python bindings/st-py/examples/hello_session.py
```

Minimal session + rank planning:

```python
import spiraltorch as st

session = st.SpiralSession()  # backend="auto" by default
print("backend:", session.backend, "device:", session.device)

plan = session.plan_topk(rows=128, cols=256, k=32)
print("plan:", plan.kind, plan.merge_strategy, "tile", plan.tile, "workgroup", plan.workgroup)
print("SpiralK hint:\n", plan.fft_spiralk_hint())

session.close()
```

Minimal dataset loader (runs entirely in Rust):

```python
import spiraltorch as st

session = st.SpiralSession()
pairs = [
    (st.Tensor(1, 2, [1, 0]), st.Tensor(1, 2, [1, 0])),
    (st.Tensor(1, 2, [0, 1]), st.Tensor(1, 2, [0, 1])),
]
loader = session.dataloader(pairs, batch_size=2, shuffle=123, prefetch=2)
for x, y in loader:
    print("batch:", x.shape(), y.shape())
session.close()
```

Minimal `ModuleTrainer` loop:

```python
import spiraltorch as st

trainer = st.ModuleTrainer(input_dim=2, output_dim=2)
inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
targets = [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]]
loss = trainer.train_epoch(inputs, targets, learning_rate=0.1, batch_size=2)
print("avg loss:", loss)
```

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
let mut retriever = GoldenRetriever::new(GoldenRetrieverConfig::default(), vec![trainer_a, trainer_b])?;
let report = retriever.run_epoch(modules, losses, loaders, schedules)?;
println!("workers={} avg_loss={}", report.workers, report.average_loss);
```

Need curvature-aware consensus from multiple Z-space patches? Call
`GoldenRetriever::sync_z_barycenter(&partials, rank)` to fold worker tensors
into a single Leech-biased barycenter before broadcasting the result back to the
fleet. Pair it with `GoldenRetriever::install_blackcat_moderator(threshold,
participants)` to seed cooperative moderators on every trainer without touching
individual mutexes—perfect for multi-node runs where you want Redis-backed
guards and SpiralK hints to stay aligned.

For an auditable path, use `sync_z_barycenter_with_receipt(...)`, or build a
Golden-specific Black Cat controller with
`golden_barycenter_adaptation_session(...)` before passing its validated plan to
`sync_z_barycenter_with_plan(...)`. Golden candidate identity follows the FFT
tile actually consumed by the guard, not unrelated rank-k knobs. Before reward
credit, call `validate_golden_barycenter_output(...)` to compare the result with
an independently recomputed partial/guard reference. The returned
`GoldenBarycenterReceipt` captures the effective rank, guard, and complete plan
snapshot. It also states `plan_executed: false` because the plan parameterizes
the Golden guard rather than claiming a rank-k launch.

GoldenRetriever keeps each trainer behind a poison-resistant mutex, launches the
epoch bodies on the shared runtime, and reduces the per-worker metrics using the
built-in parallel reducer so the roundtable stays deterministic. No additional
locking or thread book-keeping required.

Need the distributed run to keep every local Blackcat moderator in sync? Toggle
the cooperative switches on the config:

```rust
let config = GoldenRetrieverConfig {
    sync_blackcat_minutes: true,
    sync_heuristics_log: true,
    coordinate_blackcat: true,
    exploration_bias: 1.5,
    optimization_boost: 0.75,
    synergy_bias: 1.25,
    reinforcement_bias: 1.1,
    ..GoldenRetrieverConfig::default()
};
let mut retriever = GoldenRetriever::new(config, vec![trainer_a, trainer_b])?;
let report = retriever.run_epoch(modules, losses, loaders, schedules)?;
assert!(!report.moderator_minutes.is_empty());
if let Some(pulse) = &report.cooperative_pulse {
    use std::time::Duration;
    println!(
        "dominant_plan={:?} exploration={} optimization={} synergy={} reinforcement={}",
        pulse.dominant_plan,
        pulse.exploration_drive,
        pulse.optimization_gain,
        pulse.synergy_score,
        pulse.reinforcement_weight
    );
    let directive = pulse.directive(Duration::from_secs_f32(2.0), 48);
    println!(
        "retune: push_interval={:.2}s summary_window={} reinforcement_weight={:.2}",
        directive.push_interval.as_secs_f32(),
        directive.summary_window,
        directive.reinforcement_weight
    );
}
```

Every epoch collects the union of moderator minutes and heuristics ops across
workers, rebroadcasting them before the next round so proposals and soft rules
stay aligned. With `coordinate_blackcat` flipped on, GoldenRetriever also emits
an aggregated **GoldenBlackcatPulse** that nudges every worker’s distributed
node. The pulse now captures cooperative synergy (`synergy_score`), the amount
of shared reinforcement from heuristics and moderator minutes
(`reinforcement_weight`), confidence-weighted coverage, and raw op-log
composition. Each pulse can synthesize a `GoldenCooperativeDirective`, which
Golden retrievers and trainers use to retune push intervals and summary windows
without guessing at scaling factors. Trainers expose both
`last_blackcat_pulse()` and `last_blackcat_directive()` so downstream tooling
can inspect exactly how the synergy evolved during the run.

Blackcat now keeps a running scoreboard for every plan signature it moderates.
Each entry tracks observation counts, mean support, reward, ψ, and confidence so
dashboards can highlight sustained winners instead of relying on a single
minute. Access the aggregated view with `ModuleTrainer::blackcat_scoreboard()`
from Rust or call `trainer.blackcat_scoreboard()` in Python to retrieve a list
of dictionaries (plan signature, script hint, averages, and timestamps). The
scoreboard honours the moderator history window and can be capped via
`BlackcatModerator::set_scoreboard_limit()` when you only care about the top-N
plans.

Need runtime telemetry without wiring into a dashboard? The embedded Blackcat
runtime now keeps exponential moving averages for step time, memory pressure,
retry rate, and the reward distribution. Call
`ModuleTrainer::blackcat_runtime_stats()` to fetch a
`BlackcatRuntimeStats` snapshot that includes the latest reward mean/stddev and
all tracked extra metrics. Python callers can access the same data via
`trainer.blackcat_runtime_stats()`, which returns a rich object with dict-like
extras for quick printing or logging.

`GoldenRetrieverConfig` picked up `synergy_bias` and `reinforcement_bias` knobs
to tilt how aggressively the aggregated metrics should respond to support vs.
heuristic weight. Bumping `synergy_bias` favours exploration-heavy, confidence
driven pulses while `reinforcement_bias` amplifies heuristics and reward
signals when tightening distributed synchronization. When you want Golden to
renegotiate those biases automatically, hand it a
`GoldenSelfRewriteConfig`. The retriever stages a four-party council (explorer,
optimizer, harmoniser, reinforcer) that blends the latest cooperative pulse
with scheduler depth to rewrite the coordination biases in-place:

```rust
use st_nn::{GoldenRetriever, GoldenRetrieverConfig, GoldenSelfRewriteConfig};

let mut retriever = GoldenRetriever::new(
    GoldenRetrieverConfig::default().with_self_rewrite(
        GoldenSelfRewriteConfig::default()
            .with_schedule_weight(0.8)
            .with_negotiation_rate(0.45)
            .with_inertia(0.5),
    ),
    vec![trainer_a, trainer_b],
)?;
let before = retriever.coordination_biases();
// modules/losses/loaders/schedules prepared as shown above
let report = retriever.run_epoch(mods.clone(), losses.clone(), loaders.clone(), schedules.clone())?;
let after = retriever.coordination_biases();
println!("biases before={before:?} after={after:?}");
if let Some(pulse) = &report.cooperative_pulse {
    let mut persisted = GoldenRetrieverConfig::default().with_self_rewrite(
        GoldenSelfRewriteConfig::default().with_schedule_weight(0.8),
    );
    persisted.rewrite_with_scheduler(&schedules, Some(pulse));
}
```

`GoldenRetriever::coordination_biases()` exposes the live negotiation result so
Dashboards can visualise the four delegates converging. The
`rewrite_with_scheduler` helper mirrors the runtime logic in case you need to
persist the negotiated configuration or replay it in another process.

For longer runs the self-rewrite council now keeps a rolling transcript. The
`GoldenSelfRewriteConfig` gained `with_council_memory`,
`with_schedule_resonance`, and `with_synergy_pressure` knobs so you can tune how
aggressively schedule depth and Blackcat energy bend the delegates. Every epoch
emits a `GoldenCouncilSnapshot` that summarises the negotiated biases,
resonance, and stability alongside the pulse that triggered it. The snapshot now
tracks the epoch watermark, the heuristics log ranges that still need
reconciliation, the top soft-rule winners, and a `CouncilEvidence` bundle that
captures band energy, graph flow, ψ, and geometric cues used for the vote.
Inspect it via `GoldenEpochReport::council_snapshot()` or the new
`GoldenRetriever::last_council()` helper to plot convergence, detect
oscillations, or persist the negotiated state for a follow-up run. Consumers who
need streaming updates can subscribe with `GoldenRetriever::subscribe_digest()`
and replay `CouncilDigest` events as nodes fall in and out of the cluster.

Python note: Golden/Blackcat runtime taps are currently Rust-first; Python
wrappers for council/pulse snapshots are on the roadmap.

### SpiralTorchRL (agents)

The wheel exports lightweight RL agents under `spiraltorch.rl` (alias of
`spiral_rl`) so you can prototype bandits, PPO, or SAC loops without leaving the
native stack.

```python
import spiraltorch as st

agent = st.rl.stAgent(state_dim=1, action_dim=2, discount=0.0, learning_rate=5e-2)
agent.set_epsilon(0.1)

trace = agent.select_action_trace(0)
action = int(trace["action"])
agent.update(0, action, reward=1.0, next_state=0)

print("ok (stAgent)")
print("policy:", agent.policy_report(0))
print("trace:", trace)
print("also available:", [name for name in ("PpoAgent", "SacAgent") if hasattr(st.rl, name)])
```

Rust projects can pair the policy with the new geometric feedback module to
ground the update scale in observability measurements. Feed a
`DifferentialResonance` snapshot into `GeometryFeedback` and the learner will
adapt its learning rate according to the coalgebra efficiency.

```rust
use st_core::theory::observability::{ObservabilityConfig, SlotSymmetry};
use st_spiral_rl::{GeometryFeedback, GeometryFeedbackConfig, SpiralPolicyGradient};

let mut policy = SpiralPolicyGradient::new(6, 3, 0.01, 0.99)?;
let feedback = GeometryFeedback::new(GeometryFeedbackConfig {
    observability: ObservabilityConfig::new(1, 5, SlotSymmetry::Symmetric),
    z_space_rank: 24,                 // Maryna Viazovska's Leech shell as default
    leech_density_weight: 0.5,        // densify η with Λ24 packing pressure
    ramanujan_iterations: 4,          // refine π via Ramanujan's fast series
    softening_beta: 0.6,              // keep the projection memory-light
    max_learning_rate_scale: 2.8,     // pre-clamped to stay in the 2..3 stable band
    ..GeometryFeedbackConfig::default_policy()
});
policy.attach_geometry_feedback(feedback);
let resonance = session.trace(state.clone())?
    .generator(direction.clone())?
    .barycenter(barycenter.clone())?
    .resonate()?; // DifferentialResonance snapshot
let (report, signal) = policy.finish_episode_with_geometry(&resonance)?;
if let Some(signal) = signal {
    println!("η̄={:.3}, scale={:.2}", signal.averaged_efficiency, signal.learning_rate_scale);
}
let telemetry = policy.telemetry();
if let Some(geo) = telemetry.geometry {
    println!("rank~{:.1} pressure~{:.4} scalē~{:.2}", geo.rolling_rank, geo.rolling_pressure, geo.rolling_scale);
}
```

Hypergradient loops can now fuse loss variance with the geometric controller via
the **HyperSurprise** pipeline. Attach a `LossStdTrigger` and SpiralTorch injects
η̄ pulses whenever the episode's return standard deviation breaches the guard:

```rust
use st_spiral_rl::{HyperSurpriseConfig, LossStdTrigger, SpiralPolicyGradient};

let mut policy = SpiralPolicyGradient::new(4, 2, 0.05, 0.9)?;
policy.attach_hyper_surprise_with_config(
    LossStdTrigger::new(0.12)
        .with_warmup(2)
        .with_max_ratio(2.5)
        .with_deadband(0.15),
    HyperSurpriseConfig::default()
        .with_smoothing(0.35)
        .with_reversion(0.55)
        .with_lr_floor(1e-4),
);
// ...record transitions...
let report = policy.finish_episode()?;
if let Some(surprise) = &report.hyper_surprise {
    println!(
        "σ={:.3} inject={:.2} η̄={:.3} gauge={:.2} lr={:.4} σ̂={:.3}",
        surprise.loss_std,
        surprise.inject_ratio,
        surprise.eta_bar,
        surprise.gauge,
        surprise.learning_rate,
        surprise.rolling_std
    );
}
```

`LossStdTrigger` keeps an EMA of the observed loss standard deviation, applies a
configurable deadband before clamping surprise pulses, and modulates both the
learning-rate and gradient gauge inside the episode update.
`SpiralPolicyGradient::last_hyper_surprise()` exposes the latest packet so
telemetry dashboards can correlate η̄ spikes with emergent behaviour.

`HyperSurpriseConfig` now includes builder helpers for smoothing, gauge floors,
and floor clamps on both η̄ and the learning rate. A dedicated reversion factor
lets gauges glide back to baseline instead of snapping when shocks subside, and
the emitted telemetry now shares the rolling standard deviation alongside the
imposed clamps. Ratio telemetry is derived from the post-clamp, smoothed gauge
so `HyperSurpriseSignal::inject_ratio` mirrors the actual scaling applied to
η̄ and the learning-rate. These guards keep legacy pipelines untouched (defaults
mirror the previous behaviour) while unlocking telemetry-rich packets via
`HyperSurpriseSignal::gauge`, `HyperSurpriseSignal::learning_rate`, and
`HyperSurpriseSignal::rolling_std`.

`HyperSurpriseConfig` now includes builder helpers for smoothing, relaxation
back to the baseline gauge, ratio smoothing, cooldown windows, gauge floors, and
floor clamps on both η̄ and the learning rate. These guards keep legacy
pipelines untouched (defaults mirror the previous behaviour) while unlocking
telemetry-rich packets via `HyperSurpriseSignal::gauge` and
`HyperSurpriseSignal::learning_rate`.

The controller uses the ratio smoother to bleed off residual pulses instead of
snapping gauge/η̄ immediately back to baseline, and the cooldown window prevents
short bursts of volatility from hammering the learner every frame. Relaxation
lets you pick how gently the gauge returns to neutral once the surprise subsides
so you can trade responsiveness for smoothness depending on your training
regime.

Chrono loop signals now feed directly into the controller: every
`SpiralSession::resonate_over_time` call plants a `ChronoLoopSignal` in the
telemetry hub, and `SpiralPolicyGradient::finish_episode_with_geometry` consumes
the latest signal before measuring a resonance snapshot. Harmonic gain and decay
estimates tighten the learning-rate clamps, bump the Λ₂₄ pressure when collapse
drive pulses flare, and publish the live loop gain/softening factor through
`PolicyTelemetry.geometry`. In other words, Z-space temporal dynamics, SpiralK
heuristics, and collapse drive pressure now close a tidy feedback loop without
additional plumbing.

The loop no longer stops at a single node: every roundtable summary and collapse
intervention now broadcasts a bounded `LoopbackEnvelope` through the telemetry
hub. SpiralK meta-summaries attach their script hints, the softlogic observer
threads its live Z-space bias, and PSI collapse totals hitch a ride so other
policies can replay the same temporal context. `GeometryFeedback::absorb_loopback`
blends the envelopes into a synthetic chrono signal, boosts clamp tightening
according to peer support, and preserves the strongest SpiralK script so the
controller can keep rewriting its own limits. Callers don’t need extra wiring—
the policy gradient automatically drains the envelope queue before every
resonance measurement and folds the distributed telemetry back into Z-space.

### SpiralTorchRec (open-topos recommendation lattice)

SpiralTorchRec factors implicit-feedback matrices under open-cartesian topos
guards so embeddings stay psychoid-safe during long training arcs. The Rust
crate exposes a deterministic SGD loop with saturation-aware updates while the
Python view (`spiraltorch.rec.Recommender`) mirrors the same ergonomics for
notebooks and serving pipelines. User and item embeddings remain regularised by
the curvature guard, ensuring they can be re-imported into SpiralTorch modules
without violating the Z-space contract.

```python
from spiraltorch.rec import Recommender

rec = Recommender(users=10, items=20, factors=5, learning_rate=0.03, regularization=0.002)
epoch = rec.train_epoch([(0, 0, 4.0), (0, 3, 5.0), (1, 0, 3.5)])
print(epoch.rmse, rec.predict(0, 1))
```

### Observation DAG calculus (Pólya-calibrated final coalgebra)

The new `st_core::theory::observability` module formalises the experimental
setup behind our DAG compression runs. Observation trees are treated as the
final coalgebra of the endofunctor `F(X) = R × Orb_{G_Λ}(X^b)` where `R` is the
root alphabet and the child slots are quotiented by a symmetry group (S₍b₎,
C₍b₎, or D₍b₎). The helper exposes both the Pólya upper bound and the efficiency
`η = observed / expected` so you can tell exactly how much structure survives a
given symmetry choice.

```rust
use st_core::theory::observability::{
    ColorAction, ColorSymmetry, ObservabilityConfig, ObservationalCoalgebra, SlotSymmetry,
};

let config = ObservabilityConfig::new(
    1,                // structural root variants (without colour)
    3,                // b: ternary branching
    SlotSymmetry::Dihedral,
)
.with_color_action(ColorAction::new(2, ColorSymmetry::Symmetric));
let mut coalgebra = ObservationalCoalgebra::new(config);
let theoretical = coalgebra.unfold(3);          // free-branching upper bound
let measured = vec![1, 4, 52, 1_368];
let assessment = coalgebra.assess(&measured);
println!("theoretical counts: {:?}", theoretical);
println!("η per depth: {:?}", assessment.efficiency);

// Symmetric colour action identifies {a,b}, so "pure a" is invisible until symmetry is broken.
let colour_gate = ColorAction::new(2, ColorSymmetry::Symmetric);
assert_eq!(colour_gate.singleton_observable().unwrap(), false);
```

Pair the output with your `roundtable.log` counts to see exactly which depth or
symmetry regime causes drops in observability. Lowering the symmetry (e.g.
S₍b₎→C₍b₎→Exact) or enriching the alphabet instantly changes the theoretical
sequence, making it trivial to reason about how much “pure a” signal can ever be
observed before symmetry breaking. Likewise, switching the colour action to
`ColorSymmetry::Trivial` raises the observable root count and restores singleton
visibility—the exact manoeuvre the theoretical note predicts when constructing
`c′`.

## What you get for training

- **Rank-K family** (TopK / MidK / BottomK) with a **single entrypoint**
  Backends implement a `RankKExecutor`, decisions are made once via **unison heuristics**, and every plan can now be rendered back into a SpiralK snippet via `choice.to_unison_script(kind)`. Hard SpiralK rewrites return a newly validated Rust `RankPlan`; malformed workgroups, FFT settings, or conflicting 1CE/2CE directives are rejected rather than repaired by a binding.
  On WGPU, `choice.use_2ce` now executes the Rust-owned exact two-command path: planner-sized tiles are finite-filtered and total-ordered first, then merged into TopK, MidK, or BottomK with the same lower-index tie-break and `(NaN, -1)` padding as the CPU reference. Python only projects this plan and kernel report; it does not reconstruct rank semantics.
- **Introspectable compute plans**
  Unified `RankPlan`s expose a validated FFT execution contract directly. Call `plan.fft_plan()?` to inspect the radix/segment shape and ordered ping-pong dispatches, `plan.fft_wgsl()?` to emit their shared WGSL module, or `plan.fft_spiralk_hint()?` to log the same choice back into SpiralK. Invalid radix, non-power-of-two tiles, and out-of-range segment counts fail closed in Rust.
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
  `st-nn` now ships `Linear`, `Sequential`, the lightweight `Relu`, sequence
  cores like `SpiralRnn`/`WaveRnn`, hyperbolic-friendly `ZSpaceSoftmax`, the
  hyperbolic `WaveGate`, `ToposResonator`, the new `ZSpaceMixer`, and the
  `ZSpaceProjector` alongside `MeanSquaredError` / `HyperbolicCrossEntropy` /
  `CategoricalCrossEntropy` losses. They stream gradients through the hypergrad
  tape, apply open-topos rewrites, and keep SpiralK planners one call away with
  roundtable-aware scheduling helpers. Every primitive is exported through the
  Python wheel so you can stay NumPy-free while scripting experiments—with the
  new `spiraltorch.dataset.DataLoader` keeping shuffle/batch/prefetch entirely
  in Rust.
- **Optional WASM tuner table**
  Bake the JSON dataset offline and ship it to browsers/WASM. The runtime loads the table lazily, blends it with SpiralK, and keeps the optimiser in sync with the generated WGSL kernels.
- **Self-Rewrite**
  A/B/C conversations and BlackCat training evidence use the canonical Rust
  Wilson interval before appending `soft(...)` rules. The shared store honours
  `SPIRAL_HEUR_FILE`, otherwise writes `~/.spiraltorch/heur.kdsl`, and returns a
  durable append-once receipt rather than hiding persistence failures. Transcripts
  land in `roundtable.log` so you can replay how every choice surfaced. Opening
  the default store imports the legacy `~/.spiraltorch/heur/heur.kdsl` history
  with a hashed byte cursor and leaves the source file intact. Later append-only
  legacy updates import only their new suffix; a rewritten prefix fails closed.

---

### Features (opt-in)

- `wgpu`: shared WebGPU runtime plus tensor execution kernels
- `wgpu-rt`: additional rank/linear runtime primitives on the same WGPU context
- `mps`: honest macOS placeholder with Rust-selected WGPU/CPU surrogate routing
- `cuda`: CUDA (NVRTC/PTX loader expected)
- `hip`: ROCm HIP (stub-safe)
- **`hip-real`**: ROCm HIP + RCCL real path, including dense, scaled, lhs-transpose-scaled, device-resident fused bias/residual activation GEMM, owning `RcclCommGuard::allgather_u64()`, and safe row compaction through `compact_rows_f32()` (requires ROCm toolchain & linker; gated on top of `hip`). The backend-neutral shape, output, validation, and CPU oracle live in `st-kernel-contracts`; `st_tensor::compaction` exposes explicit CPU and HIP entrypoints without making a runtime routing decision.
- HIP stub now probes `ROCM_PATH`/`HIP_PATH` and honours the
  `SPIRALTORCH_FORCE_HIP` override so simulated devices keep Z-space heuristics
  alive during CPU-only dev loops.
- **`kv-redis`**: enable Redis-backed consensus (soft hints); absent = **safe no-op**
- `logic` / `kdsl`: SoftLogic solver / SpiralK DSL

---

### Canvas Pixel Transformer → Z-space feedback

- `CanvasProjector::refresh_with_vectors` now returns both the RGBA buffer and
  a colour vector field that carries normalised energy and chroma as
  Z-space-friendly coordinates.
- `FractalCanvas::vectorFieldFft(false)` surfaces the per-row FFT spectrum as
  interleaved energy/chroma pairs so Canvas Transformer pipelines can ingest
  frequency features without leaving Rust.
- `CanvasProjector::accumulate_hypergrad` and
  `CanvasProjector::accumulate_realgrad` stream the refreshed canvas tensor
  directly into SpiralTorch's Riemannian or Euclidean optimisers without
  additional copies.
- `FractalCanvas::relation()` mirrors the projector's tensor output as a
  `Float32Array` so browser call-sites can feed the raw relation into custom
  pipelines or training loops.
- `FractalCanvas::hypergradWave(curvature)` and `FractalCanvas::realgradWave()`
  surface curvature-aware hypergrad updates alongside Euclidean gradients so the
  Canvas Transformer can keep hypergrad/Realgrad buffers in sync by default.
- `FractalCanvas::gradientSummary(curvature)` condenses both tapes into shared
  L1/L2/∞ norms plus RMS/mean-absolute magnitudes so monitoring dashboards can
  watch gradient health without shipping the full relation buffers across the
  WASM boundary.
- `FractalCanvas::desireInterpretation(curvature)` lifts the paired gradient
  summaries into Desire-ready feedback metrics (pressure, balance, stability)
  so automation layers can steer the Desire Lagrangian without leaving WASM.
- `FractalCanvas::desireControl(curvature)` extends that pipeline with
  ready-to-apply Desire gradient control packets—penalty gains, bias/observation
  mixers, and tuned hyper/Realgrad learning-rate scales—mirroring the Rust
  automation layer on the browser side.
- `FractalCanvas::hypergradOperatorUniformFromControl(control)` and
  `FractalCanvas::hypergradOperatorUniformAuto(curvature)` map those Desire
  control packets directly into the WGSL uniform payload, saving JavaScript
  callers from recomputing the blend/gain heuristics before dispatching the
  GPU hypergrad operator.
- `FractalCanvas::vectorFieldFftKernel(true)` returns the ready-to-dispatch
  WGSL compute shader (including uniform layout) so WebGPU call-sites can bind
  the vector field and accumulate the spectrum fully on-GPU.
- `FractalCanvas::hypergradOperatorKernel(false)` emits the complementary WGSL
  pass that accumulates relation tensors into hypergradient buffers directly on
  the GPU, with `hypergradOperatorUniform(mix, gain)` +
  `hypergradOperatorDispatch(subgroup)` mirroring the uniform payload and
  workgroup math for WebGPU callers.
- `FractalCanvas::vectorFieldFftUniform(false)` packages the `CanvasFftParams`
  uniform (width, height, inverse flag, padding) as a `Uint32Array` so the WGSL
  kernel can be dispatched without manual byte packing.
- `FractalCanvas::vectorFieldFftLayout()` reports the byte lengths and strides
  for the `FieldSample`/`SpectrumSample` storage buffers plus the uniform block
  so WebGPU callers can allocate resources without hard-coding struct sizes.
- `FractalCanvas::vectorFieldFftDispatch(true)` computes the workgroup triplet
  for the generated WGSL so callers can hand the counts directly to
  `computePass.dispatchWorkgroups(...)` (or the Rust equivalent) without
  duplicating the ceil division logic.
- Use `CanvasProjector::emit_zspace_patch` to fold the canvas state back into
  the fractal scheduler without leaving Rust or allocating intermediate
  buffers.
- Blend chart priors with the new `z_space_barycenter` solver—available in
  Rust (`st_tensor::z_space_barycenter`) and Python (`spiraltorch.z_space_barycenter`)—to
  wire colour energy directly into the Z-space roundtable.
- Inspect the Tesla tail spectrum via `st_tensor::tesla_tail_spectrum` (Rust) or
  `spiraltorch.tesla_tail_spectrum` (Python), then adapt roundtable weights with
  `spiraltorch.nirt_weight_update` to follow the similarity- and coherence-aware
  NIRT rule.
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

**Rust (TopK via unified entry)**
```rust
use st_core::backend::device_caps::DeviceCaps;
use st_core::ops::rank_entry::{RankKind, plan_rank, execute_rank};

// describe device
let caps = DeviceCaps::wgpu(32, true, 256); // lane, subgroups, max_wg
// plan once (decisions: mk/mkd/tile/ctile/use_2ce)
let plan = plan_rank(RankKind::TopK, rows, cols, k, caps);

// choose a backend executor (WGPU/CUDA/HIP); CPU fallback exists
use st_core::backend::wgpu_exec::WgpuExecutor;
let exec = WgpuExecutor::default();

// launch
execute_rank(&exec, &plan)?;
```

## 🌀 New: ZSpaceCoherenceSequencer

**NOT Attention. NOT Transformer.**

Instead of Q·K^T softmax:
- **Maxwell pulses** detect phase synchronization
- **Desire Lagrangian** applies semantic bias (no RLHF needed)
- **Hyperbolic geometry** naturally encodes hierarchy
- **Fractional operators** replace dot products

```python
import spiraltorch as st
from spiraltorch.nn import ZSpaceCoherenceSequencer

# Toy input (same API as larger models)
x = st.Tensor.rand(1, 768, seed=1)
model = ZSpaceCoherenceSequencer(dim=768, num_heads=12, curvature=-1.0)

out, coherence, diagnostics = model.forward_with_diagnostics(x)
print("channels:", diagnostics.preserved_channels, "preserved,", diagnostics.discarded_channels, "discarded")
print("top report:", diagnostics.channel_reports[0].channel, diagnostics.channel_reports[0].backend)

# Pre-discard sequencing + snapshot history
model.configure_pre_discard(dominance_ratio=0.35, energy_floor=1e-3, min_channels=3)
model.configure_pre_discard_memory(limit=64)

out, coherence, diagnostics = model.forward_with_diagnostics(x)
if diagnostics.pre_discard:
    print("pre-discard:", diagnostics.pre_discard.discarded, "discarded of", diagnostics.pre_discard.total)

latest = model.pre_discard_snapshots[-1]
print("snapshot step", latest.step, "survivors", latest.survivors)
model.clear_pre_discard_snapshots()
model.disable_pre_discard()
```

Telemetry now tracks both survivor/discard counts and their energy share,
so you can monitor whether the discard policy is merely trimming duplicates or
aggressively stripping away signal. Snapshot entries expose the raw
`survivor_energy_ratio`, `discarded_energy`, and even the dominant pre-discard
weight so plugins can adapt thresholds dynamically.

[See example](../../examples/05_new_layers/zspace_coherence_demo.py)

### Plugin Architecture

`ZSpaceCoherenceSequencer` now exposes a lightweight plugin system so other
subsystems can tap into each stage of the pipeline without forking the core
implementation. Plugins are notified when tensors move through projection,
coherence measurement, geometric aggregation, semantic window derivation,
canonical concept selection, Maxwell desire emission, distribution fusion, and
language bridging. They also receive callbacks when backends or linguistic
profiles change, when contours/reports are emitted, and when PSI telemetry is
published (with the `psi` feature).

Key stages:

- `Projected`, `CoherenceMeasured`, `PreDiscardApplied` *(with survivor + discard indices)*, `Aggregated`
- `SemanticWindowDerived`, `SemanticDistributionDerived`, `CanonicalConceptSelected`
- `MaxwellBridgeEmitted`, `SemanticWindowFused`, `LanguageBridged`
- `BackendConfigured`, `LinguisticProfileRegistered`, `LinguisticProfilesCleared`
- `LinguisticContourEmitted`, `ChannelsDescribed`
- `PsiTelemetryPublished` *(when compiled with `psi`)*

Implement the `ZSpaceSequencerPlugin` trait and register it on a sequencer to
receive callbacks:

```rust
use st_nn::{
    zspace_coherence::{
        CoherenceBackend, ZSpaceCoherenceSequencer, ZSpaceSequencerPlugin, ZSpaceSequencerStage,
    },
    OpenCartesianTopos, PureResult, Tensor,
};

struct TelemetryPlugin;

impl ZSpaceSequencerPlugin for TelemetryPlugin {
    fn name(&self) -> &'static str { "telemetry" }

    fn on_stage(&self, stage: ZSpaceSequencerStage<'_>) -> PureResult<()> {
        match stage {
            ZSpaceSequencerStage::Aggregated { diagnostics, .. } => {
                println!("entropy: {:.2}", diagnostics.coherence_entropy());
            }
            ZSpaceSequencerStage::SemanticWindowDerived { window, .. } => {
                println!("window tokens: {}", window.len());
            }
            ZSpaceSequencerStage::BackendConfigured { backend } => {
                println!("backend -> {}", backend.label());
            }
            _ => {}
        }
        Ok(())
    }
}

fn main() -> PureResult<()> {
    let topos = OpenCartesianTopos::new(-1.0, 1e-5, 10.0, 256, 8192)?;
    let mut sequencer = ZSpaceCoherenceSequencer::new(768, 12, -1.0, topos)?;
    sequencer.register_plugin(TelemetryPlugin);
    sequencer.set_backend(CoherenceBackend::Fftw)?;
    let (_out, _, _) = sequencer.forward_with_diagnostics(&Tensor::zeros(1, 768)?);
    Ok(())
}
```

### Why Not Attention?

| Aspect | Attention | ZSpaceCoherence |
|--------|-----------|-----------------|
| Token weighting | Q·K^T softmax | Maxwell pulses |
| Geometry | Euclidean (dot product) | Hyperbolic (geodesic) |
| Semantic bias | External (RLHF/DPO) | Intrinsic (Desire Lagrangian) |
| Operators | Softmax | Fractional calculus |
| Hierarchy | Implicit | Explicit (curvature) |

**Features**
- Dataset abstraction and serialization
- Multi-epoch validation, best-state restore, and deterministic epoch reshuffling
- Hypergrad integration for every parameter
- Optional Realgrad accumulation via `ModuleTrainer::with_realgrad`
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

**BlackCat runtime tap-in**

The derivative-free ZMeta ES and contextual bandits can ride alongside the
roundtable loop. ZMeta now runs a guarded ask/tell `(1+1)` contract: the first
credited selection establishes a baseline, each later candidate applies a
temperature-scaled, bounded latent deformation to the actual bandit context,
and only that selection's observed free-energy utility may accept the candidate.
Retry, gradient-norm, and loss-variance telemetry controls the next proposal
radius rather than a disconnected diagnostic. Observation-only reports and
explicitly abandoned selections do not reward-train the ES. The
selection and update traces retain the base/effective contexts, evaluated Z,
reward comparison, acceptance decision, and fractional penalty.

Black Cat contextual-bandit witness contract v3 separates selection attempts
from credited observations, records forced exploration and quarantine state,
and transports Thompson's `u64` RNG seed as decimal text so WASM clients retain
its exact value.

Attach the runtime once and it will ingest per-step metrics, evaluate the
canonical Rust variational free-energy report, log Above/Here/Beneath energy,
estimate the BlackCat drift band, and accumulate both successful and unsuccessful
step rewards for each canonical `RankPlan.choice` script. The default adoption
epoch requires at least eight observations and promotes a rule only when the
95% Wilson lower bound clears the configured baseline. The interval, threshold,
decision, and typed persistence result are retained in
`SoftHeuristicAdoptionReport`; a store error is reported without pretending that
the rule was adopted, and a later observation retries the write. Observations
continue to update the Wilson witness after adoption without rewriting the rule;
automatic retraction is a separate policy. Persisted rules use a stable SHA-256
identity and an atomically synced append-once snapshot.
ZMeta's fractional penalty
reuses the same periodic Sobolev evaluator as `runtime::zspace_optimizer`, so a
CPU/WGPU route request cannot change the mathematical objective. Invalid reward,
adaptation, or context inputs are rejected before ES, bandit, statistics, or
telemetry state changes. Reward shaping is configurable before learning starts
and then locked, preventing one runtime from mixing ZMeta incumbents, bandit
posteriors, or statistics learned under different utility functions. When you call
`install_blackcat_moderator` a dedicated runtime is spun up for the moderator
so the training loop and the distributed consensus stay decoupled.

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
let report = trainer
    .blackcat_free_energy_report()
    .expect("BlackCat committed a guarded free-energy report");
println!("F={} p_accept={}", report.free_energy, report.acceptance_probability);
let evidence = trainer
    .blackcat_heuristic_adoption_report()
    .expect("BlackCat retained the cumulative heuristic evidence");
println!(
    "rule={} wins={}/{} lower={} decision={:?}",
    evidence.rule_id,
    evidence.interval.successes,
    evidence.interval.trials,
    evidence.interval.lower,
    evidence.decision,
);
```

**Rust (Z-space gating + projector)**
```rust
use st_core::backend::device_caps::DeviceCaps;
use st_nn::{
    ModuleTrainer, RoundtableConfig, Tensor, ToposResonator, ToposResonatorConfig,
    WaveGate, ZSpaceProjector,
};
use st_tensor::{topos::OpenCartesianTopos, LanguageWaveEncoder};

let encoder = LanguageWaveEncoder::new(-0.9, 0.7)?;
let topos = OpenCartesianTopos::new(-0.9, 1e-6, 1e4, 512, 16_384)?;
let projector = ZSpaceProjector::new(topos.clone(), encoder.clone())?;
let text = projector.encode_text("SpiralTorch keeps the open topos alive")?;

let mut gate = WaveGate::with_topos("gate", text.shape().1, encoder, topos.clone())?;
let mut trainer = ModuleTrainer::new(DeviceCaps::wgpu(32, true, 256), -0.9, 0.05, 0.01);
trainer.prepare_with_topos(&mut gate, topos.clone())?;

let forward = gate.forward(&text)?;
let grad = forward.hadamard(&text)?.scale(1.0 / forward.shape().0 as f32)?;
let _ = gate.backward(&text, &grad)?;
trainer.step(&mut gate)?;

let (rows, cols) = forward.shape();
let resonance = ToposResonatorConfig::new(0.25, 4)?;
let mut resonator = ToposResonator::with_config_and_topos(
    "res",
    rows,
    cols,
    resonance,
    topos.clone(),
)?;
// The same topos now guards both the finite Picard response and its hypergradient.
resonator.attach_open_topos(-0.9, 0.02, topos)?;
let activated = resonator.forward(&forward)?;
println!("resonance audit: {:?}", resonator.latest_audit());
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

plan = st.plan_topk(rows=8, cols=65_536, k=1_024, backend="auto")
print("choice:", plan.merge_strategy, plan.tile, plan.workgroup, "lanes", plan.lanes)
```

---

## Pure Rust training (zero PyTorch/Numpy deps)

Need a bootstrap-friendly learning loop without heavyweight dependencies?
`st-nn` layers sit directly on top of the `st-tensor::pure` stack so you can
train, schedule, and log every A/B/C decision entirely in Rust.

For direct Rust use, `st-tensor`'s `wgpu_dense` feature owns the dense Tensor
GPU APIs and dispatch; `wgpu` is a compatibility alias, not an additional
requirement. Defaults include the dense provider, while `wgpu_frac` alone does
not claim dense execution. See [Tensor backend features](../development/tensor_backend_features.md)
for the isolated build matrix and runtime checks.

Geometry-aware policy loops now broadcast their feedback as loopback envelopes,
so reinforcement learners automatically feed their learning-rate modulation
into the global telemetry hub for other SpiralTorch nodes to replay.

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
