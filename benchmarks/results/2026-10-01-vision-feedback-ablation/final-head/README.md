# Final-Implementation Revalidation

This is a new complete run, not a relabeling of the
[earlier measurements](../README.md). Source capture
`70c6f8eb6a182928ffc233cac5fa45a67108c783` includes parent
`7e91127bae3af2449412e3a53a7ef5296e7e916d`: the cubic small-tail CE kernel,
finite-f32 feedback admission guard and Rust 1.99 compatibility fixes.
The fresh native wheel was built with Rust 1.98 and
`extension-module,nn,wgpu`, in the same recorded dependency environment.

## Outcome

All 12 seed/arm combinations complete 400 accepted updates each, then pass
independent source-bound verification of their fresh-process 37/363 restart.
The same 1,280/320-image subsets, initial weights, batches and recipes are used.

[Build comparison](build-comparison.json) checks all 9,600 native records and
84 bound checkpoint files against the previous complete run, finding exact
equality. Native epoch metrics and the observed PyTorch epoch/evaluation metrics
also match. This is a bounded observed result, not universal cross-GPU or
PyTorch-process determinism. Original within-run reference tolerances remain
unchanged; no failed criterion was relaxed.

The quality conclusion is unchanged: feedback's mean development accuracy is
25.0000%, versus 25.1042% for nominal SGD and 25.5208% for the retrospective
rate-dose control. The fixed proposal reaches 21.6667%. There is still no
demonstrated policy advantage over the relevant controls, nor a speed, memory,
full-dataset, geometric-update or untouched-test claim.

## Retained Evidence

- [Original summary](summary.json): every seed, arm and epoch, copied unchanged.
- [Independent verification](verification.json): measured source and saved state checks.
- [Build comparison](build-comparison.json): observations, native binary hashes and receipts for all 170 raw/verification files retained locally.
- [Final runtime validation](runtime-validation.json): six Python tests with no skips, eight actual-browser phases, exact native/browser trainer histories and checkpoints, and all three public CE reductions over 7,513 inputs. The original `2e-5` CE bound is unchanged; maximum observed relative error is approximately `3.72e-6`.

The parent directory's earlier artifacts and checksums remain unchanged.
This directory has its own `SHA256SUMS`. Raw images and weights remain local.
The native/browser numerical and synthetic-restart checks are separate from
this real-image run; browser real-image training is still an open roadmap gate.

The publication-only verifier initially assumed JavaScript and saved Python
`log1p(exp(x))` references would be bitwise identical. They were not; that failed
assertion and its script are retained locally. The report explicitly records
the reference discrepancy and independently checks both against the original
CE bound. Exact native/browser loss, gradient, trainer-history and checkpoint
comparisons are unchanged. This does not claim cross-language libm identity.

To reverify a retained final run without ML imports:

```bash
python -I tools/run_vision_feedback_ablation.py --verify "$FINAL_RUN_DIR" \
  --source-ref 70c6f8eb6a182928ffc233cac5fa45a67108c783 \
  --output "$NEW_VERIFICATION_JSON"
```

To generate a new run, use the same command and dependencies as the parent
protocol, a new output directory, and the final source capture above.
