# spiraltorch-wasm bindings

The `types/spiraltorch-wasm.d.ts` file ships hand-crafted TypeScript declarations for the
wasm-bindgen surface exposed by this crate. Copy the file next to the generated
JavaScript glue (e.g. into the `pkg/` directory produced by `wasm-pack`) and reference it
from your bundler's `types` field to enable editor completion and static checking.
