# Native Geometry Runtime Candidate Validation

Local qualification on 2026-10-18, based on main commit
`271aa97fd0b5682afd35725399ed40751f788737` plus this native-runtime patch.
These checks are not PyPI publication or all-platform release qualification.

## Installed Candidate

The default-feature release wheel includes WGPU, but this runtime deliberately
executes its selected placements on f32 CPU. The tested local artifact is
`spiraltorch-0.4.32-cp38-abi3-macosx_14_0_arm64.whl`, SHA-256
`3c93f7a7b6e2dc5e583afc956285fbf50f1a63d07db6ef96a2f0c8926b2ff13c`.
This arm64-only local build is not the official universal2 release wheel.

| Check | Result and scope |
| --- | --- |
| Rust runtime tests | 5 pass: finite differences, atomic rejection, saved continuation, zero-strength/empty inputs, target bounds before copying |
| Installed native tests | 6 pass under isolated CPython 3.12, with Torch/Transformers imports forbidden and Torch, Transformers, NumPy and Accelerate absent |
| Optional Torch comparison and artifact regressions | 42 pass; includes four affine/Wave/fixed-radius/learned-radius forward, VJP and SGD comparisons at `atol=rtol=1e-6` |
| Installed core smokes | Autograd, learning stack and FFT pass without optional ML packages |
| Native and binding compilation | Offline locked Rust checks and final default-feature release-wheel build pass |
| WASM core | `st-nn` compiles for `wasm32-unknown-unknown`; no new browser binding or browser execution is claimed |
| Rust documentation and lint | Strict WGPU-enabled rustdoc passes; Clippy has existing warnings but no new runtime warning; targeted pinned formatting, Python lint and version consistency pass |

The first core-autograd smoke stopped because three tracked examples were
absent from the storage-pruned local checkout. The same examples and smoke
scripts were extracted from the base commit into an isolated validation
directory; all three core smokes then passed with the final installed wheel.
The working tree's unrelated deletions were not restored or included.

## Synthetic Learning

The shipped `bindings/st-py/examples/native_geometry_training.py` exercises a
two-row, three-feature synthetic WaveGate activation fit, not a language model:

```json
{
  "backend": "rust_f32_cpu",
  "first_mse": 0.08500000834465027,
  "last_mse": 0.045734889805316925,
  "steps": 40,
  "continuation_steps": 1,
  "restored_continuation_exact": true,
  "artifact_id": "sha256:a42616f4ee8de0db18f4d46efdd7ab81c06ae958ae57207da781209a4153ca1e",
  "scope": "synthetic_activation_fit_not_language_quality"
}
```

Restore uses the same next inputs, target and learning rate. This checks plain
stateless SGD continuation, not HF optimizer/RNG/data-cursor restoration.
No speed benchmark, whole-HF-model execution, GPU residency or language-quality
improvement is inferred from these tests. Optional Torch integration remains
separate; the native route does not silently delegate unsupported recipes.

Independent read-only source review identified the direct-Rust target-length
allocation ordering. It was fixed before the final wheel and covered by the
fifth Rust test. The reviewer reported no further actionable findings in the
scoped implementation, docs, examples and release gates. CI and the official
three-platform wheel workflow remain separate integration/release gates.
