# Native selected-target readout: preserve the regression, then fix it

Local offline GPT-2 CPU loss evaluation, not a model-quality or training-speed
benchmark. All six cases, both attempts, raw timing samples and matched-input
checks are retained. Source-only after PyPI 0.4.32; development wheels used here
must not be uploaded under that version.

## What Changed

Native causal loss now gathers only scored terminal hidden rows before final
LayerNorm, vocabulary projection and CE. Geometry learning scatters their
adjoints back into the complete decoder context. This does not prune attention,
remove prompt context, freeze prompt gradients or change the target denominator.
With explicit labels, the logit budget covers selected targets rather than all
context rows. All-selected objectives avoid the new gather/scatter.

The first benchmark found a regression: at 16 context rows / 4 selected targets,
old native took 104.911 ms and compact readout took 131.469 ms. CPU autotuning v5
rounded 1 and 4 rows to the same 8-row trial and cached `m8n12`; four real rows
then used scalar tails. A test reproduced that shape alias before the fix.

CPU autotuning v6 keeps row counts 1..64 exact in both keys and trials and
invalidates old cached selections. In the follow-up the 4-row head selected
`m4n16`. No arithmetic kernel, backend policy, reduction order or architecture
flag was changed. Both packed kernels have exact-reduction tests for partial
rows and column tails. Larger row counts still use the existing coarse buckets.

## Complete Follow-Up Results

Warm median milliseconds, five samples per case on this one Apple M4 host:

| Context / selected | Old native | Native + compaction + v6 | Torch full head | Torch selected head | Old / new native |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 / 1 | 102.940 | 84.735 | 18.572 | 15.022 | 1.215 |
| 16 / 4 | 103.057 | 79.341 | 18.396 | 17.034 | 1.299 |
| 16 / 16 | 104.078 | 103.473 | 18.305 | 17.427 | 1.006 |
| 64 / 1 | 401.358 | 285.101 | 38.536 | 31.296 | 1.408 |
| 64 / 4 | 402.302 | 279.144 | 39.756 | 33.100 | 1.441 |
| 64 / 64 | 413.820 | 410.824 | 38.509 | 38.058 | 1.007 |

Torch remains substantially faster. Near-1 all-selected ratios are not evidence
of an improvement. These are descriptive local measurements, not confidence
intervals, cross-hardware results or a claim that native is generally faster.

Both engines load and pack weights before timing. The measured boundary is
host integer lists to a host loss scalar, frozen f32 CPU, no geometry, smoothing
0.1, identical synthetic IDs and labels. Both decode the entire context. Torch
selected-head compacts after its final norm; native compacts before it. One
first call and one warmup precede five randomized within-engine trials. First
calls are retained separately. OMP/MKL/OpenBLAS/Rayon settings request one thread;
Torch also requests one intra/inter-op thread. Physical CPU utilization and
other BLAS thread pools were not measured.

Processes ran serially, not paired or interleaved across engines. Initial order:
native-before, torch-selected, native-after, torch-full. Follow-up order:
torch-full, native-after, torch-selected, native-before. Host load, thermal state
and order can confound timing ratios. This is not a complete factorial isolation
of the autotune change. No training/backward, GPU, RSS or language quality is
timed or claimed here.

The predeclared loss gate is absolute error <= 2e-4 against Torch full-head.
All cases passed in both attempts. The comparator requires matched model hashes,
script, IDs/labels, policy, context/selection accounting and execution conditions.
It also checks reference Torch/Transformers versions and reported intra-op thread
counts, preserving both reference runtime records.
Native runs forbid importing Torch, NumPy, Transformers, Accelerate and the Python
safetensors package; native Rust owns model loading and computation.

## Learning Regression

The unchanged [selected-target probe](../2026-10-18-native-gpt2-objective/probe.py)
was rerun with the final development wheel. Three accumulated geometry AdamW
updates, ten selected targets per group, smoothing 0.1, passed; the frozen-base
probe was unchanged and the next update after restoring the artifact/optimizer
pair was exact within this runtime. The mean smoothed objective moved from
3.36368413 to 3.36163301. This tiny reused fixture is not SFT quality evidence or
an across-version bitwise-resume promise. `geometry-report.json` contains results
and hashes; raw gradients, geometry and optimizer states remain private.

## Evidence and Reproduction

- `initial/` retains the first four reports and comparison, including the slow
  16/4 result. `autotune-v6/` retains the second four reports and comparison.
- `provenance.json` identifies source/binary hashes, invocation order and the
  actual cached head-kernel choices. Timing scripts are unchanged between runs.
- Before is commit `8a6422cfde00649a2aa5bf309d5627e1fd318861`; first after is the
  selected-readout source before the v6 fix; final after is this change set.
- Both development binaries report version 0.4.32 but are **not** the published
  0.4.32 release. Use the recorded binary hashes, not the version string.

Recheck either published cohort without ML dependencies:

```sh
python3 -I -S tools/compare_native_lm_readout.py \
  benchmarks/results/2026-10-18-native-lm-selected-readout/autotune-v6 \
  --output /tmp/readout-rechecked.json
python3 -I -S tests/test_native_lm_readout_bench.py
```

For fresh measurements, use separate native-before/native-after environments
with their corresponding built wheels and an optional Torch/Transformers
reference environment. No model download is performed; point `MODEL` to the
locally available GPT-2 snapshot with the hashes recorded in each report. Use a
new output directory for every invocation; do not overwrite prior evidence.

```sh
"$NATIVE_AFTER_PYTHON" -I tools/bench_native_lm_readout.py \
  --engine native --model-dir "$MODEL" --output "$OUT/native-after"
"$NATIVE_BEFORE_PYTHON" -I tools/bench_native_lm_readout.py \
  --engine native --model-dir "$MODEL" --output "$OUT/native-before"
"$REFERENCE_PYTHON" -I tools/bench_native_lm_readout.py \
  --engine torch-full --model-dir "$MODEL" --output "$OUT/torch-full"
"$REFERENCE_PYTHON" -I tools/bench_native_lm_readout.py \
  --engine torch-selected --model-dir "$MODEL" --output "$OUT/torch-selected"
python3 -I -S tools/compare_native_lm_readout.py "$OUT" --output "$OUT/comparison.json"
```
