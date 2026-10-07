# Resident Topos Shared-Gate Learning

Correctness-only measurement of source `20686ce788052fef3767fd64549a32e23552d53b`,
whose immediate parent and comparison base are both
`4bdcfe757cd88fc4c4409cf2d0e081049d172d35`.

On Apple M4 / Metal, two 100-update trajectories keep Topos forward, mean MSE,
input/gate VJPs, shared reduction and immutable SGD updates on the GPU. The
first host observation occurs after each complete 100-update trajectory.
Inputs/targets are still uploaded and every step still allocates buffers.
The probe retains owning outputs for comparison; it is not a throughput test.

Independent CPU Torch 2.12.1 carries its own weights throughout both runs.
Maximum absolute errors: output/gate VJP/loss `1.1920929e-7`, input VJP
`5.9604645e-8`, updated gate `2.9802322e-8`, within unchanged `rtol=5e-4`,
`atol=3e-5`. The two local native runs produced identical JSON bytes. This is
one deterministic synthetic case family, not multi-seed or model-quality evidence.

## Contents And Validation

- `native-learning.json.gz`: all 200 synthetic inputs, outputs, gradients, losses
  and updated gates; no pretrained weights or corpus text.
- `torch-check.json`: independent comparison result and source/receipt hashes.
- `verification.json`: tested source and retained binary/log hashes, commands'
  scope, successful checks, initial failures and unresolved validation limits.
- `SHA256SUMS`: public file integrity, not proof of execution provenance.

Kernel contracts 49, core Topos 34, tensor Topos 88, NN 776, resident pointwise
9, and comparison guards 31 tests passed locally. The real-GPU markers include
24 Topos cases, three execution policies, repeated VJPs, 3-D input, strided
views, empty rows, saturation boundaries, inherited failures, clean retries
and overflow guards. Backend wasm32 all-target compilation and formatting
passed; neither constitutes browser execution.

Strict native all-target Clippy stopped at seven **unchanged** source locations
containing the unknown `clippy::chunks_exact_to_as_chunks` lint under local
Clippy 0.1.97. No warning suppression or unrelated lint edits were added.
Vendor WGPU warnings and the existing `block` future-incompatibility notice
also remain. Full failure/success logs and the executable are retained locally.

## Reproduction

Use fresh output paths and an isolated Torch environment without startup
monkey patches. The initial local non-isolated comparison was patched to MPS
and failed on float64; it is not counted as a successful comparison.

```sh
cargo build --locked --release -p st-backend-wgpu --example topos_resident_learning
target/release/examples/topos_resident_learning /tmp/topos-resident-new.json
python tools/check_topos_shared_gate_learning.py /tmp/topos-resident-new.json /tmp/topos-resident-check-new.json
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo test --locked --release -p st-backend-wgpu --lib resident_tensor::pointwise -- --nocapture
cargo test --locked --release -p st-kernel-contracts
cargo test --locked --release -p st-core --lib topos
cargo test --locked --release -p st-tensor --lib topos
cargo test --locked --release -p st-nn --lib
cargo check --locked -p st-backend-wgpu --target wasm32-unknown-unknown --all-targets
python -I -m pytest --import-mode=importlib --confcutdir=tools --rootdir=tools -q tools/test_topos_shared_gate_learning.py
```

This is a low-level Rust/WGPU building block. It does not yet expose a portable
graph Topos stage or Python/WASM resident NN constructor. It does not replace
the core geometry/depth/volume admission, audit, or transactional graph SGD.
VJP recomputes the recurrence rather than retaining sensitivity. Existing
host-Tensor NN and scalar-WASM APIs keep their existing semantics. No CUDA,
browser, pretrained-model, speedup or quality claim follows from these results.
