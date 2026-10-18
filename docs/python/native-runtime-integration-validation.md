# Native Runtime Integration Follow-Up

Local qualification on 2026-10-18 after integrating reviewed main
`169aa5b18febbf569cb112958b0f238ed528215b` into the native runtime branch at
`1a01fdb56cd8512a8dc4295911c06e909cb311e8`, then applying the selective-pullback
fix. This is source-only work after the fixed 0.4.32 candidate, not an official
release or PyPI publication record.

## Review Fix

An independent read-only integration review found that the earliest geometry
site still computed an unused input adjoint. A zero-hidden-state frozen-GPT-2
fixture with a finite `f32::MAX` affine gate had valid loss and parameter
gradients, but failed with `mul_row_output = -inf` on the discarded input path.
That regression failed before the fix and passes afterward.

`NativeGeometryBatch::vjp_parameters` and its Python sequence/buffer bindings
reuse the same Rust kernels, requesting no input adjoint. Standalone SGD and
the earliest decoder geometry site use this path. Intermediate sites still
request the complete VJP to propagate loss to earlier placements. All 14 recipe
configurations retain parameter gradients, including frozen parameters for
inspection, while omitting fixed buffers. Invalid requested gradients still
fail; updates remain atomic. This does not introduce a second Python formula,
Torch fallback or a new WGPU execution path.

Follow-up review closed the implementation finding and caught an undersized
affine test fixture. The fixture now admits its 12 input values; runtime limits
were not relaxed. Final bounded source review reported no remaining P0/P1/P2
findings. The reviewer did not run builds, tests or models; execution below was
performed separately.

## Qualification

Apple M4 / macOS 26.4.1, Python 3.12.6. The development wheel's SHA-256 is
`daacc0607834f9b14a5447639e43530962404f08de1043e52fea23430b0f5081`.
Its development metadata is still 0.4.32; it is **not** the fixed candidate's
wheel and must not be uploaded as an official 0.4.32 release asset.

| Targeted checks | Result |
| --- | --- |
| Rust decoder/learning, geometry, WaveGate, affine and Topos | 15 + 21 + 20 + 4 + 25 passed |
| Installed no-Torch geometry, model, weights and optional-import boundary | 16 + 7 + 8 + 1 passed |
| Optional Torch/reference/artifact and HF observation/Trainer integration | 222 passed, including CPU/MPS observation and continuation checks |
| Release/compatibility contracts | 32 passed |
| Forward/learning probe comparators and publication boundary | 11 passed |
| Installed core smoke programs | Autograd, learning stack and FFT passed |
| WASM core compilation / strict WGPU-feature Rust documentation | Passed; not browser or native-decoder GPU execution |
| Clippy / targeted Ruff, rustfmt and whitespace checks | Completed; Clippy retains warnings |

The new installed-wheel test covers all 14 configurations with zero/nonzero
strength and frozen/unfrozen parameters, comparing parameter-only output to the
complete VJP. It also checks owned exported buffers and rejects malformed and
non-finite upstreams. Native regressions separately check valid parameter
gradients despite input overflow and rejection of required parameter overflow.

## Real-Model Recheck

The [three-step GPT-2 learning probe](../../benchmarks/results/2026-10-18-native-gpt2-learning/README.md)
was rerun in separate native/no-Torch and optional Torch/HF environments against
the existing local checkpoint. Three simultaneous Wave, fractional and Topos
placements learn through mean next-token NLL. No model download or provider
call was made.

The newly produced native report, reference report, private comparison and
public result were byte-identical to the previous validated measurement:

- Native report: `b90bcb44a9e11e5a4acd6032a0be6a73b8225cca387ca855f1e4cdf4f73c223d`.
- Reference report: `a4a5a0be58156a6901cedf8bcd6c9712079f0c5d768c6b323b83b7ffee123dbe`.
- Private comparison: `4959f0064554e731d1c98ed0a9c90f868622c44d00862dc3db4b6d1f93eb1527`.
- Public comparison: `c35bf5ffa89d9a7e6ac645bb2f79df201bf4ce86c6d2dab0bb7c2d2badf8e5c3`.

Maximum absolute NLL/gradient/post-update-parameter differences remain
`3.147125e-5`, `2.348935e-5`, and `1.126810e-6`; native artifact restoration
reproduces the next update exactly. These are observed discrepancies, not
thresholds selected from the outcome. Original receipts, gradients and adapter
states remain local; the public comparison retains every case/error metric
without operand norms that reveal scalar weights. Historical results were
not rewritten or relabeled as measurements of the new build.

This verifies integration and numerical agreement on three short windows. It
does not establish held-out quality, preferred geometry, full-model FT,
arbitrary HF import, GPU residency or a speed advantage. The native decoder is
still CPU f32, full-prefix and unpadded, with no KV cache.
