# Byte corpus: learned versus frozen geometry

This adds a third, identical-initialization **frozen geometry** arm to the
[previous paired pilot](../2026-10-09-byte-corpus-matched-learning/README.md).
Rust owns all three learning paths, including freeze policy and checkpoint
validation. Native Metal and actual browser WebGPU each execute nine cases
(three seeds, 128 updates per case). Independent CPU-f32 PyTorch is the
numerical reference, not a SpiralTorch training implementation.

## Result

All nine cases pass the original numerical criteria in both runtimes. Every
frozen geometry tensor retains its initial float32 bits. Geometry remains in
the backward graph: the embeddings and ordinary backbone still learn.

Held-out bits per byte (BPB, lower is better):

| Seed | Native ordinary | Native learned geometry | Native frozen geometry | Learned minus frozen |
| --- | ---: | ---: | ---: | ---: |
| 11 | 5.13633765 | 5.13636467 | 5.13645244 | -0.00008776 |
| 23 | 5.12804811 | 5.12497554 | 5.12486331 | +0.00011223 |
| 37 | 5.12810194 | 5.12610018 | 5.12614373 | -0.00004355 |

Browser differences have the same signs, but are measured separately rather
than declared bitwise equal across runtimes. All three contrasts, including
frozen minus ordinary, are retained for every seed in the comparison files.
The extra benefit of learning geometry is small and mixed. In this recipe,
most of the difference from ordinary attention is already present with fixed
geometry weights. This does **not** establish that learned geometry is useless,
nor that geometry is better than a capacity/compute-matched ordinary mechanism.
No arm or seed is removed for giving an unfavorable result.

| Maximum error against Torch | Native | Browser |
| --- | ---: | ---: |
| Training CE | 9.536743e-7 | 9.536743e-7 |
| Held-out batch CE | 1.430511e-6 | 9.536743e-7 |
| Final parameter | 5.960464e-7 | 3.576279e-7 |
| Learned geometry change-vector relative L2 | 1.928225e-5 | 9.870353e-6 |

The gates are unchanged: `3e-6 + 5e-5 * abs(reference)` for every training CE,
held-out batch CE and final weight; each learned geometry tensor must have
reference change-vector norm greater than `1e-8`, a nonzero measured change,
and relative L2 error at most `0.002`. Frozen tensors instead require exact
initial float32 bits in both reference and measured outputs.

## Fixed Design

- Source revision, documents, windows, SGD rate, backbone initializations and
  six existing arms are unchanged from the paired pilot.
- Seeds 11, 23, 37; batch 2; 32 byte positions; width 16; hidden width 32;
  two attention heads; one block; geometry width 4 and curvature -0.75.
- Float32 SGD 0.05; 128 updates, 8,192 training target bytes per arm;
  validation at 0/64/128 on 2,048 target bytes. Topos remains off.
- Ordinary has 11,216 parameters, all trainable. Both geometry arms have
  11,290 parameters; learned trains all, frozen trains the 11,216 backbone
  scalars. Equal trainable counts do not imply equal architecture or compute.
- Learned/frozen initial parameters match bit for bit, not merely by seed.
  Frozen means zero update rates for geometry weights, **not** detached
  coordinates, detached embedding pullbacks, or fixed coordinates.
- Fixed 256-byte alphabet, not removal of discrete symbols. This is a tiny
  repository-prose study, not pretrained-model FT or general LLM quality.
- Request SHA-256:
  `0782ac7a078005b6b4b20c4c7e56c9629363a995228d2d0cd654541623e456ee`.

The three-arm request/result/partial/checkpoint schemas are v2. v1 keeps its
two-arm contract and rejects an explicit policy. The generic model checkpoint
and outer browser segment envelope are unchanged. The request hash binds the
policy; Rust also checks frozen tensor bits before restoring any saved cursor.

## Resume

Each runtime also ran a separate 37-update prefix and resumed it to 128 in a
new process or fresh browser page. The entire resumed report matches that
runtime's uninterrupted baseline with strict, signed-zero-sensitive equality
for all nine cases. The saved parameter bits match the paused readback, all
frozen tensors match their initial bits, and no extra evaluation is inserted
at the pause. Both resumed outputs independently pass the original Torch gate.

[native-resume-verification.json](native-resume-verification.json) and
[browser-resume-verification.json](browser-resume-verification.json) retain
these checks and input hashes. This is within-runtime continuation evidence,
not a new native-to-browser handoff claim or cryptographic attestation.

## Evidence And Reproduction

[preparation.json](preparation.json) records the frozen input recipe and hashes.
[native.json](native.json), [browser.json](browser.json) and
[torch.json](torch.json) retain all nine scalar trajectories and held-out batch
losses, with omitted final weights explicitly marked. They are **not** complete
comparator inputs. Full raw requests, weights, checkpoints, logs and binaries
remain local; public hashes identify them but do not attest honest execution.

[native-comparison.json](native-comparison.json) and
[browser-comparison.json](browser-comparison.json) retain every arm's numerical
errors and all three seedwise contrasts. See [validation.json](validation.json)
for source hashes, review repairs, regression and resume verification scope.
Earlier pilot artifacts are not rewritten.

The new native binary also reran the original v1 request: its full raw output
is byte-identical to the retained v1 baseline and passes the original Torch
comparison. In v2, all six pre-existing case reports remain strictly equal to
their own native/browser v1 baselines after removing the two new policy/count
fields. The new frozen arms do not change the old arms' execution.

Use the [corpus recipe](../../../docs/byte_corpus_learning.md) with a new output
directory and add `--frozen-geometry-control` to the existing preparation command.
Keep the pinned source revision, all five document paths and all three seeds.
Generate the independent reference before GPU runs. Use `--checkpoint-out` for
the full native run, pause a separate run with `--stop-after 37`, and resume that
checkpoint in a new process. For browser runs use
`byte_learning_resume_browser.html`, then `?stop=37` and a fresh `?resume=1` page.
Download the original Rust report/checkpoint strings without reserializing them
in JavaScript. Run both `compare` and `verify_byte_corpus_resume.py` separately
for native and browser, using their own uninterrupted baseline.

The retained protocol tests cover invalid policies, missing triplet arms,
unequal initial tensors, changed frozen checkpoint weights and signed zero.
Independent review additionally found and closed Python acceptance of invalid
resume policies, loss of raw JSON `-0`, and duplicate JSON policy keys.
Reference regeneration after those repairs is byte-identical to the frozen
reference; no numerical threshold or selected condition changed.

Larger independent corpora, capacity/compute-matched non-geometric controls,
generation quality and throughput remain separate work. This record measures
learning correctness and the bounded effect of trainability, not a speed win.
