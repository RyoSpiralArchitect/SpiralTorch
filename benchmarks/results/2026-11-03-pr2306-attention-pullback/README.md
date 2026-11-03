# First PR 2306 Metal failure: resident attention pullback

The [first CI job](https://github.com/RyoSpiralArchitect/SpiralTorch/actions/runs/41558261535/job/125662358042)
on `dd3905cae` failed in `attention_gradients_and_resident_sgd_match_pytorch`,
before the detached byte-study recovery gate. Nine other jobs passed. No
same-head retry. This is **not** the earlier PR 2301 embedding mismatch or a
failure of the CPU GPT-2 experiment.

The log reports `query[0]: 0 != -0.036814254`. Among the 18 frozen Torch
cases, the nonzero expected f32 word `0xbd16ca8b` identifies
`mask_2_bias_2`: causal offset 2, pair bias, no Z bias, Q shape [2,2,3,7],
K/V shape [2,2,5,7]. The old message does **not** distinguish contiguous
from strided inputs, record the other values or capture intermediate GPU
statistics. The backend, test and fixture at the initial PR head were
unchanged from its main base.

- Full retained GitHub job log SHA-256: `47cac9ca685f84f9efe6881a98c96b4338fc5af56167611fa360c851a196b881`.
- [Frozen Torch fixture](../../../crates/st-backend-wgpu/tests/fixtures/resident_attention_vjp_torch.json)
  SHA-256: `60fd7df22a1ce7641ffa614bfb2cf63082c11b2718017b1debd2523eb0654ef3`.
- The independent Rust CPU reference agrees on the exact first query word.
  The maximum absolute query error across the case is 9.32e-9 (rounded up).
  A separate f64 calculation of its first query gradient gives -0.03681425468848.

## Candidate and limits

The original scratch and gradient allocation was declared as float storage.
It transported both signed exponents and reference-key indices by reinterpreting
their integer bits as floats. For the first row, the reference denominator is
about 2.42531: exponent 2 becomes `0x00000002`, a subnormal if treated as f32.
Other signed exponent words can look like NaNs. Bit reinterpretation itself
is not evidence that a particular Metal device altered those bits.

The candidate uses u32 storage words for that same allocation. Finite float
components and gradients are bitcast only at their typed load/store boundary;
integer metadata stays integer. Buffer size, bind group, dispatch order, the
math, the frozen oracle and its error tolerances are unchanged. A device-free
Naga storage-type regression fails on the original shader and passes on this
representation. That establishes a representation change, **not historical
root cause**. The regular full test still runs every original case and layout;
a failure now also identifies the case, layout, first differing bits and zero counts.

```sh
cargo test --locked --release -p st-backend-wgpu --test resident_attention_vjp causal_pair_bias_cpu_reference_matches_frozen_torch -- --exact --nocapture
cargo test --locked --release -p st-backend-wgpu --lib packed_stats_keep_integer_metadata_in_word_storage
cargo test --locked --release -p st-backend-wgpu --lib shader_abi_and_workgroup_storage_validate_without_an_adapter
```
