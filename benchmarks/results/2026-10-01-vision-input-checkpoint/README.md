# Vision Input Restart: Bounded Verification

The Rust-owned input checkpoint now reaches Python and WASM transform clients.
See [contract and replay commands](../../../docs/vision_input_checkpoint.md).

## Executed Checks

| Check | Result |
| --- | --- |
| Rust CPU, no default features, `input-checkpoint` | 54 passed |
| Rust `st-vision/wgpu`, live native GPU, serial tests | 96 passed, 1 existing manual diagnostic ignored |
| Fresh release-wheel Python vision suite | 23 passed |
| Public input checkpoint tests, explicit GPU enabled | 3 passed; included in the vision suite |
| Python -> actual wasm32 Rust under Node | 20 transforms and final RNG state exact |
| Generated and shipped TypeScript surface | Both passed |
| CPU-only Python binding compile | Passed |
| WASM webgpu release build | Passed |
| Scoped st-vision library Clippy, `-D warnings`, no-deps | Passed |
| Workspace rustfmt | Passed |

Native GPU verification uses an Apple M4 adapter, rejecting CPU fallback.
The learning fixture has a two-stage classifier, shuffle and resident image
transforms. It compares 100 consecutive update attempts with a restart after
attempt 37. Invalid CE targets reject attempts 37 and 72; all remaining input
identities/image bits and final classifier JSON/input state match exactly.
The model and input checkpoints are captured at an observed, settled boundary.

The public `python-to-wasm-fixture.json` contains synthetic input values 0..15,
the native transform checkpoint, its 20 expected outputs and final RNG state.
Run the checked-in Node fixture against a newly generated nodejs WASM module
to replay it. This is a CPU wasm32 portability check, not browser WebGPU
learning, model/input transactionality or cross-device bitwise training.

`source-sha256.txt` identifies the core measured sources. `local-log-sha256.txt`
identifies raw logs retained locally; local machine paths are not published.
Wheels, compiled WASM binaries and build logs remain outside Git. The known
vendor cfg warnings are not suppressed or misreported as a clean workspace
Clippy run. Only the explicitly listed scoped Clippy check passed.

No new throughput, real-image quality, schedule/optimizer-state resume or
policy advantage is claimed. Those remain separate roadmap gates.
