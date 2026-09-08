# Frozen Stage Specialization Experiment

- Baseline source: 00a955a701aa21821fdca34a7c16d27d712aa997. Its production code matches main f9f5784d; only its opt-in profile test differs.
- Candidate source: e15f463d8f257406fb0d0ef4cbc94a026ef6c161.
- Preserve baseline native/WASM products in the preceding transpose experiment logs; verify their manifests against the baseline commit.
- Build fresh native/WASM candidate products from the clean candidate commit and retain them separately.
- Execute the existing nine fixed recipes, two cadences, two warmups and eight retained samples.
- Native lanes are baseline/candidate/eager PyTorch MPS. Browser lanes are baseline/candidate, numerically checked against the native PyTorch captures.
- Run the full matrix twice on each client, regardless of the first result. No selected-shape retries or changed tiles/parameters/tolerances.
- Serialize owned GPU timing jobs. No compilation or other owned GPU work during timing. External macOS contention and power state remain uncontrolled.
- Independently run native/browser training correctness and CPU/MPS PyTorch replay, including finite guards and rollback/recovery.
- Specialization is not proof of a speedup. Preserve adverse results and compare setup costs too; only required forward activation variants are compiled, at most two.
- Do not promote the rejected transpose-loading candidate or change generic host-tensor shader semantics.
