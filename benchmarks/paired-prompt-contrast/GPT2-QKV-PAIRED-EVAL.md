# GPT-2 QKV Paired Read-Only Diagnostic

One bounded, read-only CPU evaluation completed on the Rust paired path. This
is a separate run from the three unsuccessful independent-score attempts
described in [the timeout record](GPT2-QKV-EVAL-TIMEOUT.md); their first
failures remain retained. The paired path shares only the 11 decoder blocks
before the trained QKV layer.

## Fixed Conditions

The source was GPT-2 small with frozen LoRA and WaveGate at layer 11. The
candidate added exactly two saved updates to that layer's fused QKV matrix.
The same 48 previously scored, QKV-untrained documents supplied 74 selected
answer targets to both arms. CPU, one Rayon thread, f32 readout, context length
64, microbatch 1 and `SPIRALTORCH_AUTOTUNE=0` were fixed before execution.
The run had a 1,500-second wall ceiling, no retry, and no weight update or
generation. It used a private development wheel, not a published release.

| Arm | Target-weighted unsmoothed NLL | Perplexity |
| --- | ---: | ---: |
| Frozen LoRA/WaveGate source | 3.6262775144061528 | 37.57269216805 |
| Source plus two QKV updates | 3.54073221941252 | 34.492165780072064 |

Trained-minus-source NLL was `-0.08554529499363284`. The receipt records
QKV layer 11, step 2, serving ID
`sha256:96b14dac3d178805d83f0bc08421a6d06ef5a012608cb0cb39d6e792dce619c7`,
11 shared prefix layers, and equal corpus and plan IDs for both arms:

| Identity | SHA-256 |
| --- | --- |
| Corpus data ID | `3a96c113bf292966198cbaa16a9a65c1830295da264703ac32baf2093456ed47` |
| Evaluation plan ID | `478bca7751403cb3d1a13aacd9c4fe09023f27524d04e3b958f2c03fc5dcc0d4` |
| Frozen protocol | `2cd3a6f9e127d35d01fa7d459a43a1a39454b9d0207f820a8a09c11c608e35f3` |
| Complete result JSON | `60f31aca0c98813834e0c0dd7163ed0e2fa7c61fb6aa0ef8597e7b9d361492c0` |
| First stderr | `237ee7b5b50e8eaaa8b43ef7abd0381e982b38fde8b0295445269ec7615c7d8f` |
| Run script | `febc228355b756508c95bed76c3f873af58805b69d0b56b25fe4472d5ca239e2` |

The original QKV snapshot and evaluation JSON kept their pre-run hashes
`c8e07d12ee4b94b8a7dc5b67682394860fb674c315f27fbb4bb9ca057c989760`
and `13e8d368fb61e5e355bc86453879f6d17e5688c7eb629d02cf90dcf751f35c37`.
The other pinned input hashes were config
`0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4`,
base weights `248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707`,
frozen source `dc1dbd48507e505c179707edc8a1bf505fd7e90e8d2ae72ef0439e720574fd4c`,
and source boundary `174699cef128650b9d7502dd29b08b7236966b0677f3f736a29952d2538d43b4`.
The captured stderr contained only the start marker, with no autotune message
or error. The private originals, script, model and corpus are retained locally;
the hashes identify bytes but do not authenticate them or make this exact run
reproducible from the public repository. The wheel was built from a pre-commit
worktree near commit `86362fa4a54ce89517552262c1d80aebcfe05870` (wheel SHA-256
`4ef856a5ec1e47ff1a9ba25402faa87df603ffe6b3385aa38f4a213ecd883cbb`),
so this is not signed source/build provenance.

## Interpretation

This is one within-task diagnostic on an evaluation set already examined in a
different pilot. It is not a fresh blind confirmation, a matched ordinary-FT
ablation, evidence that WaveGate caused the change, or a general language
quality or throughput claim. The newer
[`evaluate_against_source_by_answer`](../../docs/python/native-qkv-run.md)
was not used in this original run; its separate follow-up is reported below.

## Answer-Level Follow-Up

The same 48 previously scored documents and 74 selected targets were evaluated
once through `evaluate_against_source_by_answer`, with the same frozen source,
two-update QKV snapshot, context 64, microbatch 1, f32 readout, one Rayon
thread and disabled autotune. No training, generation or GPU execution occurred.
The first attempt stopped **before model construction**: the HF config path was
a symlink, but the new runner passed it to a regular-file-only reader without
resolving it. Its empty stdout and first stderr were retained. A separately
versioned runner resolved that path as the successful aggregate runner had;
its first attempt exited 0 under a 1,500-second wall ceiling. No scored
outcome was retried.

| Answer-level measure | Frozen source | Source plus two QKV updates |
| --- | ---: | ---: |
| Top-1 correct target tokens | 31 / 74 | 31 / 74 |
| Fully top-1-correct answers | 5 / 48 | 5 / 48 |

Candidate-minus-source token-weighted answer NLL was
`-0.08554528779211315`; the answer-weighted mean was
`-0.09502612483759613`. Each answer's delta was negative, ranging from
`-0.19252084172089656` to `-0.023251323823020087`. No answer gained or lost
a top-1-correct token. The follow-up's parsed aggregate comparison object
exactly matched the original paired receipt, and its QKV checkpoint and
evaluation corpus retained their original hashes. Thus this is a consistent
within-set probability shift, **not** an observed accuracy improvement.

The private v2 result JSON SHA-256 is
`9a63eb8d55e91c10de4e0316ed557cbdb1e78db9153cf56e0852b139852bac98`;
v2 stderr is `4579fd99e7b2bd7c2000325e64cf40f39a24546b32188396bff2a0e17338845f`,
v2 runner is `9aeff657a74e8b40ee4f04ce2985b6961ab1503019e5d1a3eb7afc553f892d71`,
and the failed first stderr is
`8e33e894c3f0e2473fb50b16cdd78ead1535c514fe4ad32ceb3244b60b6cb718`.
The engineering wheel SHA-256 is
`e5d99071de25bc7568eba737ecb8485b0d103500f533990605d0b7fcace25053`.
Its PR-test merge tree equals the merged main tree
`678f6f933dac99e6ee2eb79fcbcf9764ff98637b`, but the wheel is not a
signed release artifact. Private inputs and outputs remain local, so hashes
alone do not make the exact follow-up publicly reproducible. The result is
still not a fresh blind confirmation, a matched ordinary-FT ablation, or
evidence that WaveGate caused the improvement.
