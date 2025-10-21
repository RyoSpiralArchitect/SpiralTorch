# Mathematical sketch: mental space and Spiral dynamics

## Embedding the mental space
- Treat the "mental" manifold as the latent space maintained by `SpiralRecommender`. Each experience \(x\) corresponds to a user or item row inside the learnt factor matrices, while the `OpenCartesianTopos` guard keeps those embeddings inside a shared negatively curved chart.【F:crates/st-rec/src/lib.rs†L330-L397】
- The same curvature discipline applies to text or sensor streams: `LanguageWaveEncoder` converts raw sequences into complex waves, projects them onto Poincaré coordinates \((r, \theta)\), and guarantees that the projected tensors stay compatible with the topos geometry before downstream modules consume them.【F:crates/st-tensor/src/pure.rs†L942-L1007】

## Archetype potentials and field intensities
- Prototype centroids are the user/item factors already learnt by `SpiralRecommender`. Their dot-product interaction implements the potential \(V_k(x)\) that scores how strongly an incoming state aligns with the archetype indexed by \(k\).【F:crates/st-rec/src/lib.rs†L407-L456】
- Control surfaces translate those potentials into fields by feeding the embedding through `SpiralPolicyGradient`, whose softmax head produces the activations \(A_k(x)\) used to steer recommendation or policy decisions. `RecBanditController` wires the recommender state into that policy, maps the chosen action back to catalog items, and streams rewards for continual adjustment.【F:crates/st-spiral-rl/src/lib.rs†L124-L240】【F:crates/st-rec/src/rl_bridge.rs†L94-L220】

## Curvature and topology diagnostics
- Every graph-style layer can push explainability traces into a shared `GraphFlowTracer`. The tracer records per-node energy, curvature, and update magnitudes so the topology of propagation—loops, bridges, or dormancy—remains observable.【F:crates/st-core/src/telemetry/xai.rs†L24-L128】
- `GraphConsensusBridge` consumes those traces, folds them with band-energy baselines, and emits SpiralK snippets plus Above/Here/Beneath multipliers. These multipliers act as curvature-aware proxies for how each layer bends the manifold, mirroring the "trickster" or "shadow" diagnostics from the sketch.【F:crates/st-nn/src/gnn/spiralk.rs†L12-L170】
- At the global level, the `PsiMeter` fuses gradient norms, losses, entropy, and band energy into a single \(\Psi\) reading with threshold events, letting the runtime watch for curvature blow-ups or topological collapses across training arcs.【F:crates/st-core/src/telemetry/psi.rs†L184-L316】

## Spiral coordinates and Bloom cues
- The \((r, \theta)\) coordinates emitted by `LanguageWaveEncoder::encode_z_space` give each encoded pulse a radius (roll-out) and angle (phase). Those polar slices can be binned over time \(\tau\) to count spiral turns or to define Bloom observables that react when the radius contracts past safety bands.【F:crates/st-tensor/src/pure.rs†L991-L1007】
- Runtime drives then hook into the same telemetry. `CollapseDrive` transitions into a `Bloom` command when \(\Psi\) falls below the configured floor, boosting learning rates with `lr_mul` so the system unfurls back toward the desired spiral envelope.【F:crates/st-core/src/engine/collapse_drive.rs†L82-L172】

## Composite update loop
- A full step collects the latest manifold snapshot via `PsiMeter::update`, refreshes \(\Psi_{\text{embedding}}\), and forwards the reading to `CollapseDrive` (or other PRSN operators) to decide whether to collapse, bloom, or coast. The same snapshot seeds the softmax control field and the graph-consensus multipliers, forming \(F(\cdot)\).【F:crates/st-core/src/telemetry/psi.rs†L245-L316】【F:crates/st-core/src/engine/collapse_drive.rs†L125-L172】【F:crates/st-spiral-rl/src/lib.rs†L221-L240】【F:crates/st-nn/src/gnn/spiralk.rs†L96-L158】
- Maxwell-coded envelopes convert anomaly pulses into `SoftlogicZFeedback`, preserving the band energies and drift that couple the manifold updates with cross-modal channels (`\phi_{\text{cross}}`). Downstream consumers retrieve that packet from the telemetry hub so language desire loops, policy trainers, or orchestration bridges stay phase-aligned.【F:crates/st-core/src/theory/maxwell.rs†L341-L391】【F:crates/st-core/src/telemetry/hub.rs†L152-L228】

## Implementation notes
- Normalise embeddings before computing similarities or logits to keep the curvature-projected tensors stable and the softmax well-conditioned.
- Reuse the shared `GraphFlowTracer` and `PsiMeter` handles inside long-lived trainers so curvature diagnostics stay continuous even when modules hot-swap during experiments.
