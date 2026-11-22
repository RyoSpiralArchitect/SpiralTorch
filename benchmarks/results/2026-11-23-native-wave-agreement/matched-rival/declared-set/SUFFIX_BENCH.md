# Frozen-prefix A/B: one finite geometry step

Compare the previous CPU wheel (extension SHA-256
`ac7120c27d914423c4496fefb2d2adf0110ea0d904f7e0f5692a9842c06a32e7`)
with the optimized build of the same training API. No external model, data,
ML runtime, GPU, network or production candidate. A fresh local output
directory and a single bounded invocation (CPU wall time 240 seconds, 4 GiB
free-space floor) preserve failures. No rerun for an unfavorable time.

* Artificial decoder, 12 blocks, width 64, four heads, 512 vocabulary
  classes, explicit untied head. Weights from a fixed integer RNG; no
  pretrained checkpoint. WaveGate R=1 at block 0 (control) and 11 (late).
  Use original-packed forward and f32-wide-backward on both builds.
* Four fixed batches, two rows each, 16 context tokens and one supervised
  final token in each row. One declared rival and the same requested
  parameter distance 1/256. The returned model feeds the next update,
  including an unchanged origin on finite rejection. No tuning or retry.
* Each fresh process warms one four-step chain at each site, then times
  three four-step chains from the same initial snapshot. Measure only
  the native step through the Python API (including its receipt conversion);
  exclude fixture creation, independent checking, hashing and logging.
  The control and late site use the same batches but are different models.
* Four paired process blocks: old/new, new/old, new/old, old/new. Alternate
  which site goes first across blocks. One worker at a time, library thread
  caps set to one. Preserve raw timings, paired ratios and failures; do not
  infer a general speedup from the microbenchmark. No performance gate in CI.
* Equivalence comes first: full numerical receipts, chosen geometry IDs
  and final logits on a training input must match exactly between and within
  builds. Do not publish a speed comparison on mismatched work. Mechanical
  counts of forward/replayed blocks are tested separately in Rust. This is
  not a held-out comparison or a claim of improved learning.

Run `suffix_bench.py compare OLD_PYTHON NEW_PYTHON NEW_EXTENSION_SHA OUTPUT`
with the two already installed local wheels. OUTPUT must not already exist.
