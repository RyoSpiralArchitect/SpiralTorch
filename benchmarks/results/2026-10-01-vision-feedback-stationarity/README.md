# Fixed-Model Feedback Stationarity Probe

## Question And Result

Can the current loss gate react strongly even when the model does not change?
Yes, in all six measured cases. This establishes that changing batch composition
is sufficient to trigger gate activity. It does not establish that every gate
closure during actual training is wrong, or that aggregation improves learning.

We freeze the initial and final **baseline** models from the
[three-seed ablation's final build](../2026-10-01-vision-feedback-ablation/final-head/README.md).
Each consumes the same 400 recorded training batches: 1,280 CIFAR-10 images,
batch size 16, five shuffled passes, and no augmentation. Model checkpoint
strings are unchanged before/after inference, and no update is submitted.
The model, normalization, CE and gate remain Rust-owned. An independent Torch
MPS model receives the same frozen parameters and normalized inputs.

| Seed | Frozen weights | `halt` actions / 400 | Halted observations / 400 | Nonidentity shadow controls / 400 | Spread of full-pass mean CE |
| --- | --- | ---: | ---: | ---: | ---: |
| 17 | Initial | 67 | 209 | 133 | 2.68e-8 |
| 17 | Final | 77 | 240 | 108 | 3.43e-8 |
| 29 | Initial | 83 | 259 | 99 | 1.49e-8 |
| 29 | Final | 92 | 260 | 93 | 2.38e-8 |
| 43 | Initial | 71 | 256 | 97 | 2.98e-8 |
| 43 | Final | 113 | 262 | 104 | 1.49e-8 |

All 2,400 native/Torch CE comparisons pass the existing `2e-4` scaled-error
bound; maximum observed error is `1.59e-7`. Native input hashes match the retained
training batches. Reversing the same loss sequence changes gate history; a
synthetic constant-loss control produces no halt or nonidentity controls.
Neither shadow control is applied to the model. A halted gate suppresses the
external half-rate proposal, **not training**, and would return to nominal SGD.

## Implication

The gate currently averages adjacent **relative changes**, not comparable
population losses. The global objective of a frozen model is stable here while
individual batch means differ. We should next add an explicit, opt-in Rust-owned
observation window and evaluate it on the same controls, rather than select new
thresholds or seeds. One full input pass (80 accepted observations in this task)
is a coverage-derived candidate, not a tuned winning window size.

That candidate must checkpoint partial-window statistics, preserve acceptance
and staleness clocks, and retain the existing single-observation default. Longer
windows delay detection of real regression; they are not automatically safer.
The frozen-model check alone cannot decide that tradeoff or establish a quality
win. Actual matched training, dose controls and the 37/363 restart remain required.
No observation-window policy or default change is implemented by this probe.

## Evidence And Replay

- [Summary](summary.json) contains all cases, passes, shadow controls and raw-file receipts.
- [Independent verification](verification.json) recomputes retained input/metric comparisons and exactly replays all 7,200 shadow observations through the native Rust gate. It does not rerun model inference.
- The measured probe source is commit `69709820`; its digest is recorded in the summary. The native wheel and retained ablation sources are capture `70c6f8eb`, not a newly modified runtime.
- Full observations, images and model checkpoints remain local. The earlier ablation and its frozen artifacts are unchanged. This is not browser execution, throughput, geometric-update evidence or improved learning quality.

With the measured `nn,wgpu` wheel and the original NumPy/Torch/torchvision environment:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 python -I tools/probe_vision_feedback_stationarity.py \
  --ablation "$FINAL_ABLATION_DIR" --source-ref 70c6f8eb \
  --data-root "$CIFAR_ROOT" --output "$NEW_PROBE_DIR"
python -I tools/verify_vision_feedback_stationarity.py "$NEW_PROBE_DIR" \
  --ablation "$FINAL_ABLATION_DIR" --probe-ref 69709820 --output "$NEW_VERIFICATION_JSON"
python -I tools/test_vision_feedback_stationarity.py
```

The probe never downloads datasets and refuses to overwrite its output.
Eight offline admission/negative tests cover input membership, malformed losses,
core clock mismatches, reference metrics, missing records and tampered bit fields.
