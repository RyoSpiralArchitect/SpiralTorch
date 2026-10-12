# Frozen Contextual Corpus Screen

Declared before v8 preparation and learning. Source: exact v7 request
`7733e760748e14944744a91a6c55a16ad21982a53af0375a4a2ac95d00e9a010`.
Preserve all 51 source arms and every data selection. Add 24 contextual arms:
four metric subsets, train/frozen query weights, three seeds (11, 23, 37).
Both new arms train the shared router. Query affine bias is always frozen +0.
All query weights start +0; no calibration or parameter tuning is performed.

Batch 2, 32 byte positions, width 16, hidden 32, two heads, one block,
geometry width 4, curvature -0.75, Topos off. 128 updates at the source f32
rate (~0.05), evaluations at 0/64/128 with all 32 held-out batches.
Every arm sees 8,192 training targets and 2,048 held-out targets per evaluation.
This is the original tiny repository-prose corpus, not a new corpus or LLM FT.

Run native Metal, independent single-thread CPU f32 Torch, and browser WebGPU
serially for GPU workloads. Bind all to the same Rust-prepared request hash.
No result-dependent changes of source, seed, rate, initialization, gate,
update count or held-out selection.

Use unchanged corpus gates: all loss/parameter scalars within
`3e-6 + 5e-5 * abs(reference)`; every trained geometry-family final delta
and every individual shared-router delta must be nonzero, with reference
norm/absolute magnitude > `1e-8` and relative error <= `0.002`.
Every frozen parameter must preserve its original f32 bits.
Retain any failure and all completed raw reports, without threshold relaxation.
These are corpus final-delta gates, not a replacement for the separately
failed contextual v1 browser per-step rounded-delta gate.

Primary descriptive contrasts, all 12 retained per runtime:
contextual minus frozen-query and contextual minus shared-only final BPB.
Also report frozen-query minus shared-only BPB as a path-rounding control.
Do not rank signs at the scale of engine disagreement as a learning advantage.
Compare all inherited v7 case reports with the retained original runtime
reports; record differences rather than presuming bitwise identity.

After complete runs, exercise native pause-37/fresh-process resume and
browser pause-37/fresh-page resume if runtime availability permits. Partial
completion must be labeled partial, not silently promoted. No-op resume is
separate. Resume requires exact within-runtime reports and frozen bits.

Publish scalar results, all contrasts, checks, hashes, source/reproduction
instructions and independent review. Raw documents, weights, checkpoints,
executables and full logs remain local. No speed or general-quality claim.
