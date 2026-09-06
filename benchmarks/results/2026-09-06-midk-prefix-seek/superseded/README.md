# Superseded Or Non-Admitted Runs

These files are preserved byte-for-byte, including their original `passed`
fields. That status describes the checks implemented at the time; it does not
admit the files into the final matched comparison in the parent directory.

- `seek-resident-*.json`: the first A/B/B/A comparison used `torch.topk` for tied
  TopK/BottomK controls and checked CUDA values/gather consistency but not exact
  source-index equality. The native kernel still matched the CPU stable reference,
  but the timed CUDA controls had a weaker contract. PR #2076 review identified
  this issue. Commit `817af465` uses stable sort for tied controls and checks all
  CUDA output indices before and after timings. The final comparison is rerun
  with the corrected, identical harness on both kernel revisions.
- `seek-canonical-baseline-82bdd5e2.json`: corrected canonical checks pass, but
  candidate compilation overlapped this run on Furnace. It is retained as a
  correctness preflight only. The final A/B/B/A sequence starts after both
  executables are built and preserved; no compilation is launched during it.

Do not pool these samples with the admitted results or select the best run.
