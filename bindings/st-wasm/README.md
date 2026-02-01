# spiraltorch-wasm bindings

The `types/spiraltorch-wasm.d.ts` file ships hand-crafted TypeScript declarations for the
wasm-bindgen surface exposed by this crate. Copy the file next to the generated
JavaScript glue (e.g. into the `pkg/` directory produced by `wasm-pack`) and reference it
from your bundler's `types` field to enable editor completion and static checking.

## Building

Build the WebAssembly package (and copy the bundled TypeScript declarations) with:

```bash
./scripts/build_wasm_web.sh --dev
# or: ./scripts/build_wasm_web.sh --release
```

If you have `vcpkg`-style host linker flags exported in your shell (for example via
`RUSTFLAGS`), prefer the helper script above (it sanitises the environment for wasm
builds) or run `wasm-pack` via `env -u RUSTFLAGS -u LIBRARY_PATH -u PKG_CONFIG_PATH …`.

## Examples

- COBOL dispatch console: `bindings/st-wasm/examples/cobol-console/`
- Canvas hypertrain demo (FractalCanvas + hypergradWave): `bindings/st-wasm/examples/canvas-hypertrain/`
- Mellin log grid demo (evaluateMany): `bindings/st-wasm/examples/mellin-log-grid/`

## High-level Canvas utilities

`types/canvas-view.ts` implements an opinionated orchestration layer around the raw
`FractalCanvas` wasm bindings. It manages requestAnimationFrame loops, pointer-based
navigation (pan + zoom), palette overrides, and gradient statistic sampling. The helper
exposes getters and setters so UI layers can tune stats sampling, pointer navigation, and
device pixel ratios at runtime.

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

For teams that want a ready-to-go diagnostics HUD, `types/canvas-dashboard.ts` exposes a
vanilla DOM controller that wires `SpiralCanvasView` up to palette selectors, curvature
sliders, pointer toggles, and a stats grid:

```ts
import { FractalCanvas } from "spiraltorch-wasm";
import { SpiralCanvasView } from "./canvas-view";
import { SpiralCanvasDashboard } from "./canvas-dashboard";

const canvas = document.querySelector("#fractal") as HTMLCanvasElement;
const fractal = new FractalCanvas(64, 512, 320);
const view = new SpiralCanvasView(canvas, fractal, {
    autoStart: false,
    statsInterval: 125,
});

const hudContainer = document.querySelector("#hud") as HTMLElement;
const dashboard = new SpiralCanvasDashboard(hudContainer, view, {
    customPalettes: {
        dusk: {
            stops: [
                { offset: 0, color: "#021024" },
                { offset: 0.55, color: "#6f8dd6" },
                { offset: 1, color: "#f4bc6d" },
            ],
            gamma: 0.85,
        },
    },
    showRecorder: true,
    snapshotFilename: "latest-frame.png",
    onRecordingComplete: async (clip) => {
        // implement uploadBlobToS3 to push the recording to your backend
        await uploadBlobToS3(clip);
    },
});

view.start();
```

The dashboard ships lightweight glassmorphism-inspired defaults, can be embedded in any
layout, and exposes options for supplying custom palette presets or toggling controls on
and off. Since it only depends on `SpiralCanvasView`, it can be further wrapped inside
framework components as needed.

### Frame capture and recording

`SpiralCanvasView` now exposes helpers for exporting the currently rendered frame:

```ts
const pixels = view.capturePixels({ applyPalette: true }); // Uint8ClampedArray copy
const pngBlob = await view.toBlob("image/png");
const dataUrl = view.toDataURL("image/webp", 0.9);
const bitmap = await view.toImageBitmap();
const stream = view.createCaptureStream(60); // MediaStream for WebRTC/recording
```

For time-based captures hook the new `types/canvas-recorder.ts` helper. It wraps the
browser's `MediaRecorder` and automatically wires it to the canvas capture stream:

```ts
import { SpiralCanvasRecorder } from "./canvas-recorder";

const recorder = new SpiralCanvasRecorder(view, {
    mimeType: "video/webm;codecs=vp9",
    videoBitsPerSecond: 6_000_000,
});

recorder.start();

// ... wait for a few seconds ...
const clip = await recorder.stop();
```

The vanilla dashboard exposes snapshot/recording buttons out of the box. Provide
`snapshotFilename`, `onSnapshot`, or `onRecordingComplete` callbacks to integrate the
capture workflow with your own UX (e.g. uploading to a backend or pushing into a React
state store).

### Collaborative control between trainers, models, and humans

Real-time co-creation becomes much easier with `types/canvas-collab.ts`. The new
`SpiralCanvasCollabSession` peers a `SpiralCanvasView` across any number of browser tabs
or devices using the `BroadcastChannel` API with automatic fallbacks. Every participant –
whether they're a human artist, a trainer supervising gradients, or the training run
itself – has the same authority to steer palettes, zoom levels, navigation toggles,
stats sampling, and render loop state. Updates are batched into 16 ms micro-windows,
governed by a 20 diff/s token bucket, and pointer broadcasts are throttled to 30 Hz so
the UI keeps a steady 60 fps even under heavy interaction.

```ts
import { SpiralCanvasCollabSession } from "./canvas-collab";

const session = new SpiralCanvasCollabSession(view, {
    sessionId: "lab-floor-7", // pick any shared identifier for the room
    participant: {
        role: "trainer", // "trainer" | "model" | "human" or your own identifier
        label: "Curator A",
        color: "#facc15",
        capabilities: {
            wgpu: typeof navigator !== "undefined" && "gpu" in navigator,
            wasm: true,
            controlSurface: "palette",
        },
    },
    patchRateHz: 20,
    pointerRateHz: 30,
    pointerTrailMs: 240,
    replayWindowMs: 12_000,
    telemetry: (event) => console.debug("collab", event),
    attributionSink: (sample) => conductor.step(sample), // pipe into your ZConductor
    rolePolicies: {
        trainer: { canPatch: true, canState: true, rateLimitHz: 30, gain: 1.2 },
        model: { canPatch: false, canState: true, gain: 0.4 },
    },
    defaultRolePolicy: { canPatch: true, canState: true, rateLimitHz: 10, gain: 0.7 },
});

// Surface shared presence, last input timestamps, and pointer motions inside the HUD.
dashboard.attachCollaboration(session);

// React to remote updates (for example to log attribution or build custom UI chrome).
session.on("state", ({ participant, origin }) => {
    console.info(`%s adjusted the view (%s)`, participant.label ?? participant.role, origin);
});

session.on("pointer", ({ participant, event, origin }) => {
    // Mirror pointer navigation in a minimap, trigger haptics, etc.
    highlightParticipantCursor(participant.id, event.offset);
    console.debug("pointer", origin, participant.id);
});
```

When `BroadcastChannel` is unavailable the helper degrades to a
`localStorage`/`storage`-event transport with a safety-net polling loop, so you can still
wire it into dashboards without special casing. The session emits presence heartbeats at
1 Hz, records join/leave/suppression events via the optional `telemetry` hook, and pipes
every patch (local or remote) through the `attributionSink` so it can be fused straight
into your `ZConductor` dashboards. Each message carries schema version tags, participant
metadata (including the resolved `gain` for each participant), and size guards, making it
straightforward to colour-code the HUD or enforce your own policies on top of the
symmetric default. Declarative `rolePolicies` let you switch individual roles between
read-only, bursty, or high-authority modes: the token bucket honours the narrowest
`rateLimitHz`, the optional `gain` flows through to attribution samples, and the telemetry
hook reports `policy-blocked` events whenever a disallowed patch/state arrives.

Participants can now advertise a structured capability surface via the optional
`capabilities` object. Keys are automatically trimmed to 64 characters, values are limited
to simple JSON primitives (boolean, finite number, string, or null), and the helper keeps
at most 16 entries per participant (512 UTF‑8 bytes per value) by default. The advertised
set shows up on `session.participants`, in presence heartbeats, and in the payload passed
to `attributionSink`, enabling downstream dashboards to fan out richer context (e.g. GPU
availability, palette control preference, experiment tags). Call `session.setCapabilities`
at runtime to push an updated advertisement without waiting for the next presence tick:

```ts
session.setCapabilities({
    wgpu: typeof navigator !== "undefined" && "gpu" in navigator,
    wasm: true,
    sandbox: "beta-2025-10",
});

// Clearing the object broadcasts `null` to peers so they can retire cached badges.
session.setCapabilities(null);
```

Capability propagation keeps the “safe-by-default” posture from earlier hardening work.
Each `CollabRolePolicy` can now declare:

* `allowedCapabilities`: allow-list of keys that the role may broadcast (omit or `null`
  for "anything goes").
* `blockedCapabilities`: deny-list enforced before the allow-list.
* `maxCapabilityEntries`: per-participant ceiling (default 16, hard cap 64).
* `maxCapabilityValueBytes`: UTF‑8 byte budget per value (default 512, hard cap 4096).

Capabilities that miss the allow-list, hit a deny-list, overflow the quota, or include an
unsupported primitive are dropped locally and emit `policy-blocked` telemetry with the
reason code `capability:<context>:<constraint>:<key>`. That keeps innovation room wide
open—roles can still introduce new markers on the fly—while preserving a deterministic
boundary around what crosses the wire and what reaches downstream attribution sinks.

Pointer broadcasts now include two companion surfaces that make spectator UX and replay
dashboards effortless:

* Set `pointerTrailMs` (defaults to `200`) to accumulate per-participant cursor trails.
  Every time a pointer update lands, the session emits a `pointerTrail` event with the
  most recent positions, and you can query the latest trail with
  `session.getPointerTrail(participantId)`.
* Configure `replayWindowMs`/`replayMaxEntries` to bound an in-memory timeline that keeps
  recent pointer, patch, and full-state events. Call `session.replay({ windowMs: 3_000,
  kinds: ["pointer", "patch"] })` to obtain chronologically ordered `CollabReplayFrame`
  records for scrubbers, instant replays, or audit tooling.

```ts
session.on("pointerTrail", ({ participant, trail }) => {
    // Fade a ghost cursor using the recent positions.
    paintTrail(participant.id, trail);
});

const frames = session.replay({ windowMs: 5_000, participantId: "trainer-1" });
frames.forEach((frame) => {
    if (frame.kind === "pointer") {
        timeline.push({ type: "cursor", at: frame.timestamp, offset: frame.pointer!.offset });
    }
});
```

Replay frames preserve the `origin` (local vs. remote), Lamport `clock`, `kind`, and
sanitised payloads, so dashboards can interleave them with attribution reports without
touching the internal queues. Pointer trail emissions are equally policy-aware—roles that
cannot patch still broadcast trails for spectating, while deny-listed capabilities never
reach the queue that feeds the replay log.
