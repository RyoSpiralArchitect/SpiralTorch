# Document-held-out byte learning pilot

This record extends the [complete-model geometry owner](../2026-10-09-byte-causal-geometry-owner/README.md)
from synthetic fixtures to frozen windows of repository prose. Native Metal
and actual browser WebGPU each run six models for 128 updates through the
same Rust runner. Independent CPU-f32 PyTorch checks the learning trajectories
and final parameters. This is a learning-correctness pilot, not evidence of
LLM quality, pretrained-model fine-tuning or a speed improvement.

## Outcome

Held-out bits per byte (BPB, lower is better) falls from roughly 8.1 to 5.13
for both ordinary and causal-geometry arms. The paired final differences are
small and mixed; no winning mechanism is selected from this result.

| Seed | Native ordinary BPB | Native geometry BPB | Geometry minus ordinary BPB |
| --- | ---: | ---: | ---: |
| 11 | 5.13633765 | 5.13636467 | +0.00002702 |
| 23 | 5.12804811 | 5.12497554 | -0.00307257 |
| 37 | 5.12810194 | 5.12610018 | -0.00200176 |

The ordinary model has 11,216 scalar parameters; geometry has 11,290.
Common initial parameters are bitwise identical within each pair, but the
arms are not parameter- or compute-matched. Two small improvements and one
small regression on three seeds and 2,048 held-out target bytes do not
establish general benefit. The initial losses also differ because geometry
is enabled from the first forward pass.

## Fixed Protocol

- Corpus revision: `01e12ac8cc9bb10a1b151a8d7b1fda6b157fea64`.
- Three train and two validation documents, separate by file; exact duplicate
  documents rejected, near-duplicate prose and boilerplate not deduplicated.
- Batch 2, context 32 bytes, width 16, MLP width 32, two heads, one block;
  geometry dimension 4, curvature -0.75, float32 SGD rate 0.05.
- Seeds 11, 23 and 37; 128 updates / 8,192 training target bytes per arm;
  validation at revisions 0, 64 and 128 on the same 2,048 target bytes.
- Fixed 256-byte alphabet, shifted byte targets, no subword dictionary;
  no EOS insertion, cross-document windows, padding or text normalization.
- Topos is off in this pilot so that the causal wave/metric addition is the
  only mechanism changed. Shared head gains are learned, not context-dependent
  routing or an established ability to switch geometry off when unhelpful.
- Request SHA-256:
  `fe4f21d72ce419703c3fd8c6bedc6eb89011d4a04f992f4085ad845b4006eba3`.

Every training CE, held-out batch CE and final parameter is checked with the
predeclared `3e-6 + 5e-5 * abs(reference)` numerical gate. Each geometry
parameter tensor must have a nonzero final change, reference change-vector L2
norm greater than `1e-8`, and relative change-vector error at most `0.002`.
No numerical threshold was relaxed after observing results. These checks
establish agreement with this reference, not superiority of the architecture.

## Evidence And Reproduction

See [preparation.json](preparation.json) for source hashes, window selection,
common initial parameter hashes and frozen criteria.
[native.json](native.json) and [browser.json](browser.json) retain every scalar
training and validation record; only final parameter arrays are omitted and
explicitly labelled. [native-comparison.json](native-comparison.json) and
[browser-comparison.json](browser-comparison.json) include all numerical error
summaries and individual geometry change-vector checks.

[validation.json](validation.json) records tested source/artifact hashes,
environments, commands, regression checks and review repairs. Full raw request,
reference, measured outputs, logs and built artifacts are retained locally.
The public scalar records are not accepted as full comparison inputs because
they omit the final weights; regenerate the full reports with the
[reproduction recipe](../../../docs/byte_corpus_learning.md).

The runner reads update acceptance and scalar losses only at bounded resident
checkpoints. Initial/final parameter readback verifies construction and the
independent comparison; there is no per-update activation or gradient
readback. No timing or throughput claim is made.

Larger independent corpora, capacity/compute controls, generation evaluation,
and complete-model checkpoint/resume remain separate work. Earlier synthetic
records are unchanged; this pilot does not retroactively broaden their scope.
