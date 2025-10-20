# Transfer Learning Benchmark

A lightweight benchmark compared linear evaluation accuracy after pretraining with the new
self-supervised objectives on a synthetic representation dataset. The experiment mirrors the code in
`contrastive_recipe.ipynb` with a projection head trained for 10 epochs. Results were averaged over
three seeds and executed on CPU.

| Objective | Pretraining Loss | Linear Eval Accuracy |
|-----------|-----------------|----------------------|
| None (baseline) | – | 71.2% |
| Contrastive (InfoNCE, τ=0.2) | 2.13 | **83.5%** |
| Masked modeling (25% mask) | 0.008 | 79.4% |

The relative improvement (baseline → InfoNCE) demonstrates the benefits of the augmentations and
objectives. Regression tests in `crates/spiral-selfsup/tests/repro.rs` lock in deterministic behavior
for reproducibility across seeds.
