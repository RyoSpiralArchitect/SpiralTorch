# Repetition Objective Control: Bounded Validation

Implementation source: `c94c0d3cc5b4428cca09045ccfb1c62134a452e0`.
See the [contract and usage guide](../../docs/hf_repetition_objective_control.md).

This validates execution of the new Rust-owned objective coefficient. It does
not establish better language quality, an optimal schedule, or a WGPU HF backend.
The preceding negative long-horizon study remains unchanged.

## Correctness Checks

- 25 Rust repetition-planner/objective tests passed.
- Nine native WASM-binding tests passed; the wasm32 module was also built.
- 56 targeted Python tests passed with a freshly built native wheel.
- 96 native Rust/Python requests matched actual wasm32 outputs exactly through
  both JSON and object APIs, including policy identity. Invalid requests were
  rejected. This checks WASM semantics under Node, not browser GPU execution.
- Actual CPU Trainer tests matched a manual masked-microbatch gradient update.
  Uninterrupted versus 2/2 resumed training matched all weights and objective
  slots for tiny GPT-2, GPT-2 LoRA and Llama LoRA. The models were initialized
  locally; these are correctness tests, not pretrained-model efficacy evidence.
- A periodic model-top-k test exercised nonzero auxiliary loss and adapter
  updates. Changed recipe, clock, accumulation and missing checkpoint metadata
  were rejected before loading weights. Evaluation matched stock Trainer.
- Rust 1.99 native/wasm32 core Clippy, wasm32 all-target checking, nightly
  formatting, Python F-rule lint and diff whitespace checks passed.

Whole-binding strict WASM Clippy additionally reports 19 existing diagnostics
in seven unchanged source files (`cobol`, `cosmology`, `fractal_field`,
`scale_stack`, `cobol_bridge`, `fft`, `mellin`). It is not claimed clean. The
changed core passed strict checking and the WASM build/runtime tests above pass.

## Pretrained GPT-2 Smoke

Cached GPT-2 revision `607a30d783dfa663caf39e06633721c8d4cfcd7e` and the existing
English *Pride and Prejudice* corpus were used offline. Corpus SHA-256:
`df06eafcd4c1793dc0bf12211401f4cfb0477ef6d68fe278642fd434a6f56f9c`.
No models or corpus were downloaded.

Environment: Python 3.12.6, Torch 2.12.1, Transformers 4.57.6, PEFT 0.19.1,
Accelerate 1.14.0, local SpiralTorch 0.4.27 candidate wheel, CPU training.
LoRA rank 4 / alpha 8 / dropout 0.05 targets `c_attn,c_proj`: 405,504 trainable
parameters and 124,439,808 frozen base parameters.

The first two-update wiring smoke selected too little text and no validation
split: it had zero active positions and skipped evaluation. Its retained run
card therefore does not establish nonzero intervention or held-out behavior.
The second, explicitly bounded smoke used all corpus rows, a 10% split, 128-token
blocks, batch 2, accumulation 2, seed 7 and eight updates. Evaluation was capped
at two blocks. No claims are selected from either smoke's language quality.

| Second smoke observation | Result |
| --- | ---: |
| Materialized training blocks | 1,141 |
| Completed update slots | 8 |
| Training microbatches | 16 |
| Active microbatches | 11 |
| Active positions / periodic candidates | 17 |
| Eligible targets | 4,064 |
| Mean actual weighted auxiliary loss | 0.029276492074131966 |
| Last slot / schedule scale / effective strength | 7 / 0.125 / 0.0125 |
| Two-block evaluation CE before / after | 4.819457054138184 / 4.8185930252075195 |

The canonical recipe, eight-slot trainer trace and adapter save completed.
This is an uncontrolled eight-update smoke; the tiny CE difference is **not**
efficacy evidence. No generated-text comparison or matched long-horizon trial
was performed in this validation.

Reproduce the second smoke with a wheel built from the implementation source,
existing `MODEL_SNAPSHOT` and `CORPUS` paths, and a fresh `OUTPUT` directory:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 \
"$PYTHON" -I bindings/st-py/examples/hf_gpt2_finetune_bridge.py \
  --model-name "$MODEL_SNAPSHOT" --tokenizer-name "$MODEL_SNAPSHOT" \
  --train-file "$CORPUS" --output-dir "$OUTPUT" --run-card "$OUTPUT/run-card.json" \
  --train --training-use-cpu --seed 7 --finetune-mode lora --lora-rank 4 --lora-alpha 8 \
  --learning-rate 0.00005 --max-steps 8 --block-size 128 --validation-fraction 0.1 \
  --max-train-samples 0 --max-eval-samples 0 --max-eval-blocks 2 \
  --per-device-train-batch-size 2 --per-device-eval-batch-size 2 \
  --gradient-accumulation-steps 2 --save-steps 4 --save-total-limit 2 --logging-steps 1 \
  --eval-before-train --eval-after-train-policy always \
  --zspace-repetition-unlikelihood-strength 0.1 \
  --zspace-repetition-unlikelihood-candidate-source model-topk-periodic \
  --zspace-repetition-unlikelihood-normalization active-positions \
  --zspace-repetition-unlikelihood-decay-end-update 8
```

Private artifact SHA-256 values (raw cards contain local paths and are not published):

- Initial smoke card: `1f4304cab681281dea717a8085ebcc66002e5bd175a0c2a20007c13584a86abc`.
- Second smoke card: `4fa5e233c1481d2009a68d4f71b08af6981618f5e2203101db0f34a4be60979f`.
- Native wheel: `1bd7f833613b9ce6738193bdefc36ce302b10f173d2e8815b252dd7b8ba3d814`.

Hashes identify the retained artifacts; they are not independent attestation or
a substitute for the raw files. The next experiment must retain ordinary FT,
the frozen intervention and one prespecified candidate, fresh seeds, a fixed
longer horizon, held-out loss and generated-text assessment.
