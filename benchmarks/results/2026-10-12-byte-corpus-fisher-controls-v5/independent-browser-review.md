# v5 Browser Artifact Audit

**2026-10-12. PASS WITH LIMITS. No P1/P2 findings in the requested artifact checks.**

Scope: browser full learning, first-step controls, 37-to-128 resume, and 128-to-128 no-op. Native resume is excluded. CPU standard-library validation only; no simulations, Torch execution, GPU work, builds, network, or original/source edits. This report contains scalar results, relative artifact labels, and hashes, not private paths or raw tensors.

## Recomputed Checks

- The opaque native-prepared request is byte-identical to `input/request.json`. Preparation was reverified against the retained v3 source. All browser reports/checkpoints bind to this common request, not a separately browser-fitted request.
- The original and frozen measurement-plan files are byte-identical. Request seeds [11, 23, 37], nine arms per seed, 128 updates, calibration batches [0, 1, 2, 3], and numeric criteria agree with the plan.
- `byte_corpus_study.compare` was rerun on full, resumed, and no-op browser reports against `torch.json`. All 27 cases pass; recomputed full/resumed comparisons exactly equal the saved comparisons as typed JSON values. No-op yields the same comparison.
- `verify_byte_corpus_calibrated_controls.verify` was rerun against the retained v4 browser result. All 21 inherited initial cases and full case reports remain exact, and all 12 first-step train/freeze pairs pass. The non-case recipe is unchanged. The saved control result and its input-hash map were independently reproduced.
- `verify_byte_corpus_resume.verify` was rerun for 37-to-128 and no-op. Checkpoint 37 binds all 27 parameter readbacks, metric identities, frozen bits, model revision clocks, and complete histories to the partial report. Checkpoint 128 and the no-op checkpoint each bind to the final report. Both saved verification results and their input-hash maps were independently reproduced.
- Additional report validation rechecked coverage, parameter counts, finite values, frozen geometry, and arithmetic summaries at cursors 1, 37, and 128. The partial report has 37 training points and only revision-0 validation; final reports have 128 training points and validations [0, 64, 128]. No extra pause evaluation appeared.
- **Full, resumed, and no-op report files are byte-identical. Final and no-op checkpoint files are byte-identical.** These are direct byte comparisons, not trusted `passed` flags.

## Fixed Numeric Results

| Quantity | Recomputed maximum/minimum | Unchanged gate |
| --- | ---: | --- |
| Training CE maximum error | 1.430511474609375e-6 | atol 3e-6 + rtol 5e-5 |
| Held-out batch CE maximum error | 1.430511474609375e-6 | Same elementwise gate |
| Final parameter maximum error | 3.5762786865234375e-7 | Same elementwise gate |
| Geometry change-vector maximum relative L2 error | 9.870352922683123e-6 | <= 0.002 |
| Minimum reference geometry change L2 | 5.152136679664534e-4 | > 1e-8 |
| Independent Torch initial RMS maximum relative error | 1.3065411996748821e-7 | <= 1e-5 |

All individual guards were executed, including the exact shared Poincare target check across metric records. Sixty trained geometry parameter families qualify; frozen geometry bits are preserved. All six trained Fisher final gain values match Torch float32 bits. Their gain-vector change norms, by seed 11/23/37, are 5.152136679664534e-4, 1.0391707276594456e-3, and 5.833720193510401e-4. All 63 old/new seed-contrast signs agree with Torch, including adverse outcomes.

## Limits

This establishes consistency and fixed-gate numerical agreement for the recorded browser artifacts. It does not independently rerun model mathematics, attest execution provenance, or establish that archived binaries were built from the current source snapshot. Full checkpoint topology validity remains the Rust preflight's responsibility; the Python audit rechecks stored envelopes, metric identities, values, shapes, clocks, histories, and report bindings. Browser device-side initial RMS was not independently remeasured here; the RMS statistic above is the stored independent Torch qualification, whose arithmetic and identities were rechecked.

Three seeds on a small fixed corpus, one block, two heads, and 128 updates do not establish statistical significance, general quality, speed, matched compute, or a winning geometry. Fisher remains worse than Poincare and RMS-matched flat on seeds 11 and 23 for trained and frozen arms; Fisher training is worse than freezing on seed 23. Exact resume/no-op results do not change those limitations. Native 37-to-128 completion is not claimed.

## Input SHA-256

Unprefixed labels identify files in the v5 evidence bundle; `v3/` and `v4/` identify retained predecessor bundles. No input was overwritten.

| Input | SHA-256 |
| --- | --- |
| input/request.json | 95ef8c21c6c9c8221d7aced6adcda9d80ce4c0dccf6ec3819e40475c52c1db38 |
| native-prepared-packet.json | 692c1d367f5e39ef59da16491bacd99284aa3fc8a16fee6cdb7ff0cfdaa85e03 |
| torch.json | 9480a3e0840bae8a021f029bf920494953b1b12081dbf0698edd05333b074d99 |
| pre-registered.json | da5f6b19b92cb04eb35516bf4d441fc3b01d102e98d4cffee088d74428272308 |
| frozen-measurement-plan.json | da5f6b19b92cb04eb35516bf4d441fc3b01d102e98d4cffee088d74428272308 |
| browser.json | 1556b99e2b82c7a0c1254f9ecea1226d697424a33e4ba4d7a3b4a75d79dd5de6 |
| browser-comparison.json | 122190d4aac6f4a23e72092ea457e1fcb7c4b3625934836f4068e8a1d95e0cd4 |
| browser-first-step.json | ca0361fa147079db526ca6fac1b5a88278e122c14d14207b61a00858fec6aef0 |
| browser-controls.json | 4486b16807d0a5a5f7fb29b6f3f3638ce7657ec5d89e2416727e0727f10aa32e |
| browser-partial-37.json | 3f390265e81aef04dd9d0cdcabd7e9fb2b65341bafc4ba7bd367861ad962b14d |
| browser-checkpoint-37.json | a74c7144f5d4112170f8fd8a6c57f5e4526523a4f72762113f764fce14c957b9 |
| browser-resumed.json | 1556b99e2b82c7a0c1254f9ecea1226d697424a33e4ba4d7a3b4a75d79dd5de6 |
| browser-checkpoint-128.json | 9df40f0888269f7b584d2bc85ae91976c385de3b920c06036286def115327c99 |
| browser-resume-verification.json | a610b010631a1b0ca2c7453464eb4bc9eb39adde7c4d6ea2d29809e933abcd36 |
| browser-resumed-comparison.json | 122190d4aac6f4a23e72092ea457e1fcb7c4b3625934836f4068e8a1d95e0cd4 |
| browser-noop.json | 1556b99e2b82c7a0c1254f9ecea1226d697424a33e4ba4d7a3b4a75d79dd5de6 |
| browser-noop-checkpoint.json | 9df40f0888269f7b584d2bc85ae91976c385de3b920c06036286def115327c99 |
| browser-noop-verification.json | 52a290fb3367b6c4fa50f3dd21e95ce085deeee9559c81dd9661473aad354308 |
| v3/input/request.json | 759ed1282bad72c2351fcc9c78fef0f661087bf57f578f96e741882763204849 |
| v4/input/request.json | 276a2245277cc36995eef024a99614811fe12c5be1a0efd897eda2d27b11afff |
| v4/browser.json | 7f02ec86718b25ad68d946c7888645b7ee2b4f7d3257e8347471e2d52d5cc571 |

## Source and Bundle SHA-256

Source labels are repository-relative. Base HEAD is `025b100746b40fa7fc21e53314e1fb72b4852518`; the listed current source contents include uncommitted v5 work. This is a bounded source snapshot, not a complete dependency/build-provenance manifest. The oracle source files were hashed without importing or executing Torch.

| Source | SHA-256 |
| --- | --- |
| tools/byte_corpus_study.py | 1f890a8fe24708a47cc57f00789a61de4b9bbc58da9fd62ee2e222e6c015b766 |
| tools/byte_corpus_calibration.py | e793da1202f6b2cfa185aac63e2ab0a268f4ad47eb75a9423d628ace7383d7bc |
| tools/verify_byte_corpus_preparation.py | bd6e5092034f7fd473d60dd2c9b55e13d5cb7807aee6c56782fbc6735542c919 |
| tools/verify_byte_corpus_resume.py | 3bc1e91f9e229e13b16a619e5ff28f57a9d9b37fbf393d9c909971cec192e8c5 |
| tools/verify_byte_corpus_calibrated_controls.py | 3e1a8ab50d7b91c29e68d4d9d458a7b3305d497a15a425873e9c32eda256195e |
| tools/fisher_rao_reference.py | 842faae185b0289365d621ed1e292ac0e6d5e25d83d0666846c2f4a80b4026e4 |
| tools/generate_resident_byte_decoder_torch_fixture.py | 47b5af2f4bfbf5ce6bf367d9d32f762db86d6b5ac2117c8e5a264376dcd78430 |
| tools/generate_causal_zspace_wave_torch_fixture.py | 517cae06fd27833d15e3bb574b959e7c2e4696a7a43271efe6e9ba454a66b8fa |
| tools/generate_poincare_bias_torch_fixture.py | ba71c5410cd69316e9161f4ae9c975555751216be8dfa289e9010d982251dd18 |
| crates/st-nn/src/resident/byte_study.rs | b85750253effa3d93ef97f07fdb6515dfae5c12b0c240b8618cebda5c1d3bf7f |
| crates/st-nn/src/resident/byte_study/calibration.rs | 522956e1a1bf14a991b52b30c9b72eadde67fdb89e341e50fa8bf77bd391977c |
| crates/st-nn/src/resident/byte_study/policy.rs | f74acfbcbbbfecde0bb8e6b87c25ccf06b009602f26f5e10707677f9b8986773 |
| crates/st-nn/examples/resident_byte_learning_browser.rs | 3d38f8d881ee371778631a28f73d5c88febc462e8447235f3442e3a6f27cdfc3 |
| crates/st-nn/tests/byte_learning_browser.html | 11b26b6ea7ba2f321ddcacc5b4b039ba705931392a9f85d45c6a90d8bc53a3e9 |
| crates/st-nn/tests/byte_learning_prepare_browser.html | 8cbd72f3efd40b895fa00b1e60c6c2974bf7fb2b4d7524d9381172df2b0a8841 |
| crates/st-nn/tests/byte_learning_resume_browser.html | 9f8c226ec97c3db0d0df4ae0cf5fccf847a9f9a9a84ed894b7d1da76f96e2d02 |
| resident_byte_learning_browser.js | 4156e91fe9df51f4673cbef0815ddf84f5f3a1db3f73228c61cc43ca47e12efa |
| resident_byte_learning_browser_bg.wasm | 98f46d913d29474b471b0d99f2be7f9fbbb316cc67e3513aa5d516f2fe1cc517 |

Audit mode: `python3 -I -S -B`, local standard-library verifier imports, explicit raw-file hashes, typed JSON equality, float32 parameter-bit comparisons, direct byte equality, and original-byte rechecks. No native-resume files were read as audit inputs. All input and source hash rows were checked against their files after report creation.
