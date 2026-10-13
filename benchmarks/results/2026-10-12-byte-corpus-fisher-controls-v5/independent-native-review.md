# Native v5 Fisher-Rao Evidence Re-Audit

**Assessment: Share with caveats. No P1/P2 defect or fixed-criterion failure found in the audited native artifacts.**

Reviewed on 2026-10-12, starting from the artifact snapshot inspected at 22:56 JST. This is an artifact-only re-audit, not a new model execution. No GPU work, build, network, source edit, cleanup, or original-artifact write was performed. Only this new review directory was created.

## Scope and Method

The question is whether the native 27-arm v5 run satisfies the unchanged numerical and control contracts, and what its Fisher comparisons actually support. It is not whether Fisher is generally better or whether browser learning/resume is complete.

Artifacts are under `$RAW`. The retained source is `$V3/request.json`. The prior v4 request and native result are under `$V4`.

The standard-library-only checks did the following:

- Re-executed `byte_corpus_study.compare` on the opaque native-prepared request bytes, `torch.json`, and `native.json`. The recomputed comparison exactly equals `native-comparison.json` as a typed JSON value.
- Re-executed `verify_byte_corpus_calibrated_controls.verify` against the retained v4 request/result and `native-first-step.json`. Its result exactly equals `native-controls.json` after excluding the CLI-added input-hash map; that map was independently recomputed and matched.
- Compared all 21 inherited initial cases and the non-case recipe directly to v4. The initial cases, non-case recipe, and all 21 inherited full native case reports are unchanged. All 12 within-native train/freeze pairs pass the first-step checks.
- Verified the native and browser preparation packets independently against the retained v3 source. The regenerated browser verification, including its packet hash, exactly equals `browser-preparation-reviewed.json`.
- Confirmed that `pre-registered.json` and `frozen-measurement-plan.json` are byte-identical and that their criteria, seeds, arm count, update count, and calibration selection agree with the request/comparator.
- Independently recomputed all 21 contrasts for each of three seeds from the final raw batch losses, using linear combinations of arm-level bits/byte rather than trusting summary fields. Recomputed all Fisher geometry change norms, per-head gain float32 bit changes, and observed native/Torch contrast discrepancies.
- Ran in-memory no-op and ULP perturbation negative controls on the real Fisher gain readbacks. These mutations were never written to the original reports. No Torch module was imported.

## Identity and Completion

The common native-prepared request hash is `95ef8c21c6c9c8221d7aced6adcda9d80ce4c0dccf6ec3819e40475c52c1db38`. Native, Torch, and native first-step reports bind to it. Both extracted native request copies (`input` and `input-reviewed`) are byte-identical to the request string in the native packet. The local browser learning input at `$REPO/target/resident-byte-learning-web/request.json` has the same hash. This checks staged browser input, not a completed browser execution.

The browser-prepared request hash is `439259281daf4a2370a52145971883a0c2a284c435c9a980f7411d264b90ada6`. It is deliberately not the comparative input. Its differences from the native request are limited to matched gain values: 12 scalar occurrences across eight trained/frozen cases, each differing by one float32 ULP. Those are six distinct fitted head values duplicated in the trained/frozen arms. Non-gain parameters and the non-case recipe agree. Its successful preparation does not establish browser qualification of the common native-fitted request.

Native and Torch each contain all 27 cases, 128 training losses per case, and held-out evaluations at revisions 0, 64, and 128. The record declares native Apple M4/Metal and independent PyTorch 2.12.1, CPU float32, one thread. These are artifact metadata, not independent execution attestation.

The recipe has three training documents, two held-out documents, 256 distinct selected training windows, and 64 distinct selected validation windows. Each window is 32 target bytes; batch size is two. Each arm therefore has 8,192 training target-byte exposures and 2,048 evaluation target-byte exposures per checkpoint. Overlap between different windows is not ruled out; these counts are not unique corpus-byte counts or independent observations. The model has width 16, hidden width 32, two heads, four geometry columns, one non-Topos block, curvature -0.75, and float32 learning rate 0.05000000074505806.

Ordinary has 11,216 parameters. Geometric arms have 11,290 parameters; frozen geometry leaves 11,216 trainable. Matching geometric parameter totals does not match trainable counts between train/freeze arms or computational cost.

## Fixed-Criterion Checks

| Check | Recomputed value | Frozen criterion / interpretation |
| --- | ---: | --- |
| Maximum training CE discrepancy | 9.5367431640625e-7 | Elementwise atol 3e-6 plus rtol 5e-5 |
| Maximum held-out batch CE discrepancy | 1.430511474609375e-6 | Same elementwise criterion |
| Maximum final parameter discrepancy | 5.960464477539062e-7 | Same elementwise criterion |
| Maximum geometry change-vector relative L2 error | 1.928224657832324e-5 | At most 0.002 |
| Minimum learned-geometry reference change L2 | 5.152136679664534e-4 | Strictly greater than 1e-8 |
| Native preparation maximum relative RMS error | 2.0909285103343223e-7 | At most 1e-5 |
| Independent Torch qualification maximum relative RMS error | 1.3065411996748821e-7 | At most 1e-5 |
| Separate browser preparation maximum relative RMS error | 2.1105778835739386e-7 | At most 1e-5 on its own fitted request |
| Inherited full native reports | 21 exact | All seven old arms for each seed |
| First-step trained/frozen pairs | 12 passed | Four geometric pairs for each seed |

Every individual check was rerun, rather than inferring passage from the maxima alone. The largest geometry relative error belongs to seed 11 original Poincare `geometry.raw_phase`, not Fisher. Within each qualification record set, both metric records share exactly the same reference RMS matrix for each seed. Calibration uses only batches [0, 1, 2, 3], with 4,224 causal pairs per head. Native and Torch need not have bit-identical measured RMS values across engines; their own fixed relative gates are the contract.

## Fisher Gain Updates

All six learned Fisher final raw-gain values match Torch bit-for-bit. Each initial raw gain is between about 9.30 and 15.80; float32 spacing throughout is 9.5367431640625e-7. The observed changes are real representable updates, not decimal-to-float conversion or a no-op hidden by a loose final-weight tolerance.

| Seed | Final gain-vector change L2 | Relative native/Torch change error | Final per-head ULP changes | First-step per-head ULP changes |
| --- | ---: | ---: | --- | --- |
| 11 | 5.152136679664534e-4 | 0 | -532, -94 | -6, -43 |
| 23 | 1.0391707276594456e-3 | 0 | -1070, -206 | +39, +36 |
| 37 | 5.833720193510401e-4 | 0 | -595, +142 | -21, +18 |

The smallest gain-vector change is over 51,000 times the 1e-8 reference-change floor. Its smallness relative to the gain magnitude is not a failure of the predeclared absolute-change floor. The 0.002 relative change-vector gate permits only about 1.0304e-6, 2.0783e-6, and 1.1667e-6 of gain-vector error for seeds 11, 23, and 37 respectively.

In-memory negative controls confirmed these limits: altering one final gain by 1, 2, and 1 ULP respectively still passes; increasing those perturbations to 2, 3, and 2 ULP rejects at the geometry-delta gate. Replacing the final gain vector with its initial values rejects for all three seeds. Seeds 11 and 37 specifically reject at the geometry-delta gate despite the looser final-weight tolerance; seed 23 already rejects at the final-weight check.

These are family-level L2 criteria, not separate relative-error guarantees for every head. First and final readbacks do not prove that every intervening update was nonzero or that the trajectory was monotonic. Seed 23's first update and final net change even have opposite signs. Bit-identical final gains also do not imply bit-identical gradients at every step.

| Fisher family | Seed 11 relative change error | Seed 23 | Seed 37 |
| --- | ---: | ---: | ---: |
| Projection weight | 4.7631e-6 | 3.4146e-6 | 2.9816e-6 |
| Projection bias | 1.3427e-6 | 6.9297e-7 | 6.8655e-7 |
| Raw decay | 0 | 0 | 0 |
| Raw phase | 1.8320e-5 | 1.0443e-6 | 2.9532e-6 |
| Raw gain | 0 | 0 | 0 |

Thus Fisher learning is not evidenced by a gain-only change: projection, decay, phase, and gain families all have qualified nonzero end-to-end changes. All frozen geometric parameter bits are preserved in both full reports. The first-step native controls show matching train/freeze backbone updates and moving token and position embeddings; they are within-native checks, not separate first-step Torch-gradient measurements.

## All Seed Contrasts

Values are final held-out bits/byte differences at revision 128, recomputed from raw batch losses. Lower bits/byte is better. In a direct A-minus-B contrast, positive means A is worse. In an interaction, positive means the first treatment's train-minus-freeze effect is more adverse than the second treatment's; it does not by itself say which absolute model is better.

Notation: O = ordinary; P = original Poincare; L = original flat; M = RMS-matched flat; F = RMS-matched Fisher. A suffix f means frozen geometry. Other geometric symbols mean trained geometry. Every seed and all 21 contrasts are included, including all adverse values. Last column is the largest observed absolute native/Torch discrepancy across the three seeds, not a confidence interval or an error bound for future runs.

| Contrast | Seed 11 | Seed 23 | Seed 37 | Max Torch discrepancy |
| --- | ---: | ---: | ---: | ---: |
| P - O | +0.0000270228 | -0.0030725668 | -0.0020017598 | 9.674e-8 |
| Pf - O | +0.0001147877 | -0.0031847962 | -0.0019582052 | 1.290e-7 |
| P - Pf | -0.0000877649 | +0.0001122294 | -0.0000435546 | 1.290e-7 |
| L - P | -0.0000978366 | +0.0005727882 | +0.0003343128 | 5.374e-8 |
| Lf - Pf | -0.0001209575 | +0.0006296177 | +0.0003224352 | 4.300e-8 |
| L - Lf | -0.0000646440 | +0.0000553999 | -0.0000316771 | 1.182e-7 |
| (L - Lf) - (P - Pf) | +0.0000231209 | -0.0000568295 | +0.0000118776 | 5.374e-8 |
| M - L | -0.0000385026 | -0.0002461717 | -0.0003146100 | 9.674e-8 |
| Mf - Lf | -0.0000200897 | -0.0002619081 | -0.0003129332 | 1.935e-7 |
| M - P | -0.0001363392 | +0.0003266166 | +0.0000197028 | 6.449e-8 |
| Mf - Pf | -0.0001410473 | +0.0003677097 | +0.0000095020 | 2.365e-7 |
| M - Mf | -0.0000830569 | +0.0000711363 | -0.0000333539 | 2.365e-7 |
| (M - Mf) - (L - Lf) | -0.0000184129 | +0.0000157364 | -0.0000016768 | 2.795e-7 |
| (M - Mf) - (P - Pf) | +0.0000047080 | -0.0000410931 | +0.0000102007 | 3.010e-7 |
| F - P | +0.0012378021 | +0.0019964069 | -0.0001091015 | 8.599e-8 |
| Ff - Pf | +0.0012286763 | +0.0020447017 | -0.0001314915 | 1.827e-7 |
| F - M | +0.0013741414 | +0.0016697903 | -0.0001288043 | 1.505e-7 |
| Ff - Mf | +0.0013697236 | +0.0016769921 | -0.0001409935 | 6.449e-8 |
| F - Ff | -0.0000786391 | +0.0000639346 | -0.0000211646 | 5.374e-8 |
| (F - Ff) - (P - Pf) | +0.0000091258 | -0.0000482949 | +0.0000223900 | 9.674e-8 |
| (F - Ff) - (M - Mf) | +0.0000044178 | -0.0000072018 | +0.0000121893 | 2.042e-7 |

All 63 contrast signs agree with Torch. The smallest effect-to-observed-discrepancy ratio is about 19.5 for seed 37's flat-calibration-by-training interaction (-1.67683e-6 versus 8.59913e-8 discrepancy). Among Fisher contrasts, the smallest ratio is about 35.3 for seed 23's Fisher-versus-matched-flat training interaction (-7.20177e-6 versus 2.04229e-7). These ratios establish separation from this observed cross-engine discrepancy only, not statistical significance, robustness, or separation from the entire allowed tolerance envelope.

The adverse results are substantive within this frozen pilot:

- Fisher is worse than both Poincare and matched flat on seeds 11 and 23, for both trained and frozen geometry. Seed 37 goes the other way.
- Training geometry worsens seed 23 held-out bits/byte for all four geometry families, including Fisher. The modest mean Fisher train-minus-freeze improvement does not erase that adverse seed.
- Fisher's train-minus-freeze benefit is less favorable than Poincare's and matched flat's on seeds 11 and 37, despite the favorable seed 23 interactions. An interaction advantage on seed 23 does not mean Fisher training is beneficial there: its own train-minus-freeze contrast remains positive.
- The equal-seed mean F-minus-P is +0.0010417025 bits/byte and mean F-minus-M is +0.0009717091. This is not evidence of a Fisher winner. Mean F-minus-Ff is only -0.0000119564 bits/byte, with mixed signs across seeds.
- Earlier adverse results remain visible: Poincare versus ordinary is adverse at seed 11; original flat versus Poincare is adverse at seeds 23 and 37; matched flat versus Poincare is also adverse at seeds 23 and 37. Matched flat versus original flat improves in all three seeds, but that does not imply general superiority.

## Limits and Handoff

This is strong bounded numerical implementation evidence for the recorded native run: complete histories, independent end-state agreement, trained/frozen controls, unchanged prior arms, and actual small gain updates passing the original gates. It is not evidence of superior geometry, broad language quality, generation quality, statistical significance, matched compute, speed, or large-corpus training.

The pilot has only three initialization seeds, one small fixed corpus/recipe, one block, two heads, and 128 updates. Seeds share the same document/window selections; batch observations and seed contrasts are not independent corpus replications. Calibration matches one initial RMS statistic, not bias patterns, attention distributions, gradient conditioning, optimization dynamics, or effective parameter-step scales. All matrices have been checked at the measured shape; this artifact audit does not newly establish multi-block or other-device behavior.

Browser preparation passed for its separately fitted request, while browser comparative learning was still outside this audit. No browser learning result, v5 stop-37 resume, fresh-process continuation, or no-op resume result was asserted here. Retaining v4 results and passing native first-step controls do not substitute for those tests. No waiting or new model execution was performed.

Input hashes bind the exact artifacts inspected, but self-reported engine/adapter fields and hashes alone are not a proof of execution provenance. Recorded losses and readbacks were checked, not independently regenerated by rerunning the models. Successful numerical agreement does not erase the retained failed initial preparation attempt; that historical failure was not counted as a current gate pass or as an unresolved defect after the helper fix.

No required fix was identified for the audited native numerical evidence. Sharing should keep the native-only qualification and mixed/adverse results explicit. Charts were not involved; visualization QA is not applicable.

## Audit Hashes

All following v5 filenames are in the v5 artifact directory stated above.

| Artifact | SHA-256 |
| --- | --- |
| Native prepared request bytes | 95ef8c21c6c9c8221d7aced6adcda9d80ce4c0dccf6ec3819e40475c52c1db38 |
| V3 source request bytes | 759ed1282bad72c2351fcc9c78fef0f661087bf57f578f96e741882763204849 |
| Prior v4 request bytes | 276a2245277cc36995eef024a99614811fe12c5be1a0efd897eda2d27b11afff |
| Prior v4 native.json | 99661962b1df6fd60c97682a564f6d30acbe4c02178b3e6d394124d4b571316e |
| native-prepared-packet.json | 692c1d367f5e39ef59da16491bacd99284aa3fc8a16fee6cdb7ff0cfdaa85e03 |
| native.json | 48a2d82b6408633f9cc188e3f2aff178abec47a526d155b5781f7cd336f4ecc6 |
| torch.json | 9480a3e0840bae8a021f029bf920494953b1b12081dbf0698edd05333b074d99 |
| native-comparison.json | c7172b955dffd18451f889580387073add8299a16c4d2a4c979260dc2e7a205a |
| native-first-step.json | 561822007427216c7eb39e548f9ce383ab35e860af58ec37382cda622d58ab21 |
| native-controls.json | 64519d0f4554d132bd59ad924565ff2f31da36f4bd5154e83f6acc94e00ceb3a |
| pre-registered.json and frozen-measurement-plan.json | da5f6b19b92cb04eb35516bf4d441fc3b01d102e98d4cffee088d74428272308 |
| browser-prepared-packet.json | c0c1427abab60d59b0d302831107d7e6e0cdf387466d1778a1fdd9569d124eb6 |
| browser-preparation-reviewed.json | d2afbfe61bd69d4db37a322d9859b40e4ad7f2f0de9016c30dd35f5f6a637483 |

Verifier source root: `$REPO/tools`. These are the uncommitted verifier contents used, not an assertion that HEAD alone reproduces v5.

| Verifier | SHA-256 |
| --- | --- |
| byte_corpus_study.py | 1f890a8fe24708a47cc57f00789a61de4b9bbc58da9fd62ee2e222e6c015b766 |
| byte_corpus_calibration.py | e793da1202f6b2cfa185aac63e2ab0a268f4ad47eb75a9423d628ace7383d7bc |
| verify_byte_corpus_preparation.py | bd6e5092034f7fd473d60dd2c9b55e13d5cb7807aee6c56782fbc6735542c919 |
| verify_byte_corpus_calibrated_controls.py | 3e1a8ab50d7b91c29e68d4d9d458a7b3305d497a15a425873e9c32eda256195e |
| verify_byte_corpus_resume.py | 3bc1e91f9e229e13b16a619e5ff28f57a9d9b37fbf393d9c909971cec192e8c5 |

The audit used `python3 -I -S -B` with that tools directory added explicitly for local imports. Formulas: held-out bpb = `fsum(batch_losses) / number_of_batches / log(2)`; geometry change = final float32 parameter minus initial float32 parameter; relative change error = `L2(native_final - torch_final) / L2(torch_final - initial)`. The contrast table gives all linear combinations. ULP probes changed only one head's positive float32 bit pattern in memory and called the existing comparator without changing any criterion.
