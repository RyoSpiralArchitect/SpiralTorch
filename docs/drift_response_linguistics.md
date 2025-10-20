# Drift-Response Linguistics for Z-space Language Training

The **Drift-Response Linguistics (DRL)** programme models how vocabulary reacts
when meanings begin to drift.  SpiralTorch exposes both Python and Rust helpers
so existential load, frame hazards, and safe radii can flow directly into
training and governance loops without reducing language science to ad-hoc
"semantics" heuristics.

This note unifies the research sketch shared in past discussions with the
concrete APIs available in `tools/python/drift_response_linguistics.py` and
`spiral_safety::drift_response` on the Rust side.

## 1. Linguistic Objects and Frames

DRL works with a vocabulary \(W\) embedded in a meaning space
\(\mathbb{R}^d\). Each word \(w\) is tracked across a family of **frames**
\(\mathcal{F}=\{\text{Physical},\text{Normative},\text{Social},\text{Protocol},\text{MetaLanguage},\text{Mythic}\}\).

For every frame the helper expects:

- **Containment weight** \(c_{w,f}\): how tightly the word binds to the frame.
- **Contained risk** via two mixed slopes (and optional quadratic terms):
  \[
    a_{w,f} = (1-\phi_{w,f}) a^{\text{den}}_{w,f} + \phi_{w,f} a^{\text{con}}_{w,f},\qquad
    b_{w,f} = (1-\phi_{w,f}) b^{\text{den}}_{w,f} + \phi_{w,f} b^{\text{con}}_{w,f}.
  \]
  Curvature coefficients \(a''\) and \(b''\) follow the same mixture to model
  how value or risk accelerates once the drift leaves the purely linear regime.
- **Comprehension slope** \(\kappa_{w,f}\) describing how quickly listeners
  lose the intended denotation under perturbation, plus an optional
  **comprehension curvature** \(\kappa'_{w,f}\) when that erosion itself
  accelerates.
- **Timing signal** \(s_w(t)\) that raises risk when the community or media
  cycle is hot.

In code these appear as `FrameState` fields, while the word-level container
holds the definition entropy \(H_{\text{def}}(w)\), timing signal, and the
trade-off constant \(\lambda\).【F:tools/python/drift_response_linguistics.py†L61-L115】

## 2. Existential Load and Safe Radius

The helper calculates two core quantities:

- **Existential load**
  \[
    E_w(t) = \sum_f c_{w,f}\,[-(a_{w,f}-\lambda b_{w,f} S_f(t))]_+\,(1+\beta H_{\text{def}}(w) \phi_{w,f}(t)).
  \]
  This grows when the drift creates more harm than value after weighting major
  frames.  The Python function `existence_load` mirrors the expression directly.
- **Safe radius**
  \[
    r_{w,f}(t) = \min\Big(\tfrac{1-\tau_f}{\kappa_{w,f}(t)},\ \tfrac{\rho_f}{b_{w,f}(t) S_f(t)}\Big),
  \]
  using per-frame comprehension and loss thresholds `FrameThreshold.tau` and
  `FrameThreshold.rho`.

These metrics emerge from `analyse_word`, which also measures the hazard count
\(\mathrm{CHI}_w\) and toggles strict mode when multiple frames flare up or the
safe radius collapses.  The call additionally emits a **frame signature** for
every frame, capturing the local slopes, curvatures, hazard multiplier, safe
radius, and comprehension curvature so downstream policy surfaces can see *why*
a frame is risky before acting.【F:tools/python/drift_response_linguistics.py†L117-L271】

### Signature Geometry

Each entry inside `DRLMetrics.frame_signatures` (mirrored by the Rust
`FrameSignature`) corresponds to the signature tuple \((S_w, C_w, r_{w,f}, \ldots)\)
from the theory note:

- `value_slope` \(= a_{w,f}\) and `risk_slope` \(= \lambda b_{w,f} S_f\) define the
  local tangent between creative value and harm.
- `value_curvature` and `risk_curvature` reuse the optional quadratic
  coefficients to measure how quickly the tangent changes.
- `net_slope` / `net_curvature` are the differences of the above, i.e. the
  signed quantities \(S_w\) and \(C_w\).
- `hazard_multiplier` exposes the triple-product amplifier, `safe_radius`
  mirrors \(r_{w,f}\), and `kappa_slope` keeps the comprehension curvature
  \(\kappa'_{w,f}\) intact.【F:crates/spiral-safety/src/drift_response.rs†L1-L236】

## 3. Triple-Product Amplifier

To capture the *ambiguous × implicature × timeliness* amplification (the
Japanese shorthand was 「曖昧 × 含意 × 時勢」), each frame hazard is scaled by
\(\exp(\beta H_{\text{def}} \phi_{w,f} s_w)\).  When timing is calm the
multiplier becomes 1.  Large timing spikes stay bounded via a \(\pm 30\)
clamp, matching the field-note "triple-product amplifier" description.
【F:tools/python/drift_response_linguistics.py†L96-L113】

## 4. Aggregation for Training

The summary object `DRLMetrics` contains the existential load, per-frame
hazards, safe radii, strict-mode flag, and the \(\mathrm{CHI}\) count.  A small
penalty helper translates the metrics into a scalar suitable for optimisation:

```python
metrics = analyse_word(word_state, DEFAULT_THRESHOLDS)
penalty = trainer_penalty(metrics)
```

The penalty adds the existential load, frame count, and a radius surcharge when
radii dip below the configured tolerance.  Strict mode applies a 1.25× boost so
schedulers can flip into hardened policies without rewiring the trainer.
Aggregating multiple words is a simple sum via `aggregate_penalty`.
【F:tools/python/drift_response_linguistics.py†L199-L230】

Rust services tap the exact same flow through
`spiral_safety::drift_response`, which mirrors the helper with serde-friendly
structs, signature extraction, lazy default thresholds, and unit tests pinned to
the Python reference.【F:crates/spiral-safety/src/drift_response.rs†L1-L357】

## 5. Feeding Z-space Trainers

`ZSpaceTrainer` now accepts a Drift-Response signal.  Pass the scalar under the
`drs` key (or via the `ZMetrics` dataclass) and weight it with `lam_drs`:

```python
from drift_response_linguistics import (
    DEFAULT_THRESHOLDS,
    FrameState,
    WordState,
    analyse_word,
    aggregate_penalty,
)
from zspace_trainer import ZSpaceTrainer

words = [
    WordState(
        name="AI",
        definition_entropy=0.72,
        timing_signal=1.4,
        frames={
            "Normative": FrameState(
                phi=0.65,
                c=0.9,
                S=0.8,
                a_den=-0.05,
                a_con=0.2,
                b_den=0.4,
                b_con=0.8,
                kappa=0.35,
            ),
        },
    ),
]
metrics = [analyse_word(word, DEFAULT_THRESHOLDS) for word in words]
penalty = aggregate_penalty(metrics)
trainer = ZSpaceTrainer(lam_drs=0.15)
loss = trainer.step({"speed": 0.7, "mem": 0.4, "stab": 0.6, "drs": penalty})
```

The trainer normalises the signal and folds it into the Adam update alongside
speed, memory, and stability terms.【F:tools/python/zspace_trainer.py†L12-L87】【F:tools/python/zspace_trainer.py†L133-L166】

## 6. Operational Guidance

1. **Frame labelling.** Tag corpora with the six frames so you can estimate
   \(c_{w,f}\), \(\phi_{w,f}\), and \(S_f(t)\) via light regressions or
   substitution experiments.
2. **Threshold design.** Start with `DEFAULT_THRESHOLDS` and tighten
   `tau`/`rho` for Protocol, Physical, or MetaLanguage frames if your
   application is safety-critical.
3. **Strict mode wiring.** When `DRLMetrics.strict_mode` (or the compatibility
   alias `DRSMetrics.strict_mode`) is true, enforce namespaces or definitions in
   the prompt stream before the next epoch.
4. **Bilingual monitoring.** Provide separate `WordState` entries for each
   orthography or locale to make the \(\Delta E\) comparison explicit.
5. **Telemetry.** Store `frame_summary(metrics)` alongside loss curves so you
   can audit which frames triggered the strict-mode latch after training runs.

## 7. Relation to Existing Artefacts

- The CSV/PNG timing studies generated earlier slot directly into the timing
  signal \(s_w(t)\) used by `_hazard_multiplier`.
- `docs/conceptual_entropy_qualia.md` already tracks conceptual drift for a
  single term; DRL generalises the approach to arbitrary vocabularies while
  providing a quantitative interface for SpiralTorch’s trainers.

By tying existential load and safe radius into the Z-space optimiser, the stack
can guard against hazardous linguistic drift without sacrificing creative drift
in low-risk frames.
