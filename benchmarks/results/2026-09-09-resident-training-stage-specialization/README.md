# Resident Training Stage Specialization Evidence

Baseline: `00a955a701aa21821fdca34a7c16d27d712aa997`.
Candidate: `e15f463d8f257406fb0d0ef4cbc94a026ef6c161`.

See [implementation, measurements and limitations](../../../docs/resident_nn_training_stage_specialization.md).
`measurement-plan.md` fixes both complete native/browser matrices before timing.
`summary.json` retains all seed/run median ranges, numeric summaries and setup
measurements, including the adverse small/native and first/browser results.
`validation.json` and `validation-repeat.json` contain source bindings and
read-only full-state validation. `final-checks.json` records all ten local checks.

The complete raw captures and build logs remain in the durable local directory
identified by `local-raw-manifest.json`; the large raw arrays are **not bundled**
in this compact repository record. Their hashes do not allow remote readers to
independently validate absent arrays. Reproduction uses the existing commands in
`docs/resident_nn_training_benchmarks.md` with the commits above. No selective
retry, prompt/data change, or GPU fallback contributed a successful result.

Source identity, measured numeric agreement and device attestation are separate.
The result supports the bounded resident Linear/GELU + mean-MSE/SGD path, not
general LLM quality, universal acceleration, or equality of PyTorch guard costs.
