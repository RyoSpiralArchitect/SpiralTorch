# spiraltorch-wasm bindings

The `types/spiraltorch-wasm.d.ts` file ships hand-crafted TypeScript declarations for the
wasm-bindgen surface exposed by this crate. Copy the file next to the generated
JavaScript glue (e.g. into the `pkg/` directory produced by `wasm-pack`) and reference it
from your bundler's `types` field to enable editor completion and static checking.

## High-level Canvas utilities

`types/canvas-view.ts` implements an opinionated orchestration layer around the raw
`FractalCanvas` wasm bindings. It manages requestAnimationFrame loops, pointer-based
navigation (pan + zoom), palette overrides, and gradient statistic sampling.

Add the file to your bundler entrypoint (for example by copying it next to the generated
JavaScript glue or importing it from your TypeScript project) and instantiate the helper:

```ts
import init, { FractalCanvas } from "spiraltorch-wasm";
import { SpiralCanvasView } from "./canvas-view";

await init();

const canvas = document.querySelector("#fractal") as HTMLCanvasElement;
const fractal = new FractalCanvas(64, 512, 320);
const view = new SpiralCanvasView(canvas, fractal, {
    palette: {
        stops: [
            { offset: 0, color: "#0b1026" },
            { offset: 0.5, color: "#3967ff" },
            { offset: 1, color: "#ffce69" },
        ],
        gamma: 0.8,
    },
});

view.on("stats", ({ summary }) => {
    console.log("hyper RMS", summary.hypergradRms);
});
```

The helper stays framework-agnostic so it can be wrapped inside React/Vue components or
connected to bespoke UI panels. Use the exposed `on("pointer", …)` hooks to keep external
controls synchronized with the navigation state.
