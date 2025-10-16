# Coded-Envelope Maxwell Model (M₀^code)

This technical note collects the master equations that link a coded-envelope
Maxwell drive to measurable neural responses, together with the operational
playbook for running those experiments. M₀ denotes the pure physical channel;
M_Ψ covers the minimal semantic extension that gates the same carrier with
meaning-aligned weights.

## 1. Transmission, tissue, and non-linearity (core model)

- **Coded envelope**: \(c(t) \in \{+1,-1\}\) is a staircase waveform with chip
  length \(1/f_{\text{chip}}\).
- **Carrier**: \(s_{\text{tx}}(t) = [1 + \alpha c(t)] \cos(\omega_c t + \phi)\).
- **Coupling into tissue**:

  \[
  E_{\text{ext}}(t) = \frac{G(\Omega)}{r} 10^{-S/20} |\cos \theta| s_{\text{tx}}(t)
  \]

  where \(S\) is shielding in dB, \(r\) is the transmitter–receiver distance,
  and \(\theta\) is the polarisation angle.
- **Linear tissue response**:
  \(h_{\text{tis}} \overset{\mathcal F}{\longleftrightarrow} H_{\text{tis}}(\omega)\).
- **Envelope extraction via second-order non-linearity**: the low-frequency drive
  \(u(t)\) follows

  \[
  u(t) \approx \gamma \alpha |H_{\text{tis}}(\omega_c)| \frac{G 10^{-S/20}
  |\cos \theta|}{r} c(t) = \lambda c(t),
  \]

  defining the positive Maxwell gain \(\lambda\).
- **Observation (EEG)**: \(y(t) = (h * u)(t) + x(t)\) with wide-sense stationary
  background \(x(t)\) uncorrelated with the code.

## 2. Block statistics and the CLT

- Split the recording into 2 s blocks \(b = 1, \dots, N\) and apply a matched
  filter: \(s_b = \langle y_b, c_b \rangle / \|c_b\|\).
- The moments are \(\mathbb E[s_b] = \kappa \lambda\) and
  \(\operatorname{Var}(s_b) = \sigma^2\).
- By the central limit theorem,

  \[
  Z_N = \frac{\overline s_N}{\hat\sigma / \sqrt{N}} \overset{H_0}{\approx}
  \mathcal N(0, 1).
  \]

- The block count needed to reach a target score \(z_\star\) is approximately

  \[
  N_{\text{req}} \approx \Bigl(\frac{z_\star \sigma}{\kappa \lambda}\Bigr)^2
  = \Bigl(\frac{z_\star \sigma}{\kappa \gamma \alpha
  |H_{\text{tis}}|}\Bigr)^2 \Bigl(\frac{r}{10^{-S/20} G |\cos \theta|}\Bigr)^2.
  \]

- Increased shielding, larger separations, or polarisation misalignment all push
  \(N_{\text{req}}\) upward and slow detection.

## 3. Maxwell fingerprint regressions

- A TRF or regression-derived amplitude \(A\) scales like
  \(A \propto \lambda \propto 10^{-S/20} |\cos \theta| / r\).
- Target slopes for experimental validation:
  - Shielding: \(\log_{10} A = C - S/20\).
  - Distance: \(\log_{10} A = C - \log_{10} r\).
  - Polarisation: \(A = a |\cos \theta| + b\).
- Slopes near \((-1/20, -1)\) with high \(R^2\) confirm a Maxwell origin.

## 4. Sequential tests

- **Sequential Z**: update \(Z_k\) block by block and stop once it exceeds the
  threshold \(z_\star\).
- **Sign-flip test**: randomly flip the signs of \(\{s_b\}\) to generate the
  null distribution, then compare the observed statistic to obtain
  \(p_{\text{flip}}\).
- **SPRT/Bayesian extensions**: reuse the framework to test \(\mu > 0\) once
  meaning gates are introduced.

## 5. Meaning-extended model (M_Ψ^code)

- Introduce semantic alignment \(\rho(t) \in [-1, 1]\) and coupling
  \(\mu \ge 0\):

  \[
  u_{\text{tot}}(t) = (\lambda + \mu \rho(t)) c(t).
  \]

- Block means become \(\mathbb E[s_b] = \kappa (\lambda + \mu \rho_b)\).
- Switching only \(\rho\) between two equal-energy, equal-spectrum conditions
  yields a mean difference of \(\kappa \mu \Delta \rho\).
- Orthogonal code families \(c_k(t)\) support parallel transmission and allow
  simultaneous regression of \(\{\lambda_k\}\) and \(\{\mu_k \rho_k\}\).

## 6. What can be estimated or identified?

- **Physical gain \(\lambda\)**: recoverable via shielding, distance, and
  polarisation slopes.
- **TRF latency**: invariance across conditions is predicted by M₀; small
  latency variance indicates the physical pathway dominates.
- **Semantic strength \(\mu\)**: estimated from condition differences in
  \(\Delta \rho\) combined with TRF/PLV/score deltas.
- **Detection time**: \(T_{\text{det}} \approx N_{\text{req}} \times
  \text{block\_sec}\).

## 7. Falsifiability checklist

- Increased shielding raises amplitudes, polarisation has no effect, or responses
  ignore distance → M₀ fails; suspect leakage.
- Claiming \(\mu > 0\) while equal-energy, equal-spectrum contrasts collapse to
  the null → reject the meaning extension.
- Condition-dependent latencies → revisit the physical channel assumptions.

## 8. Interpreting simulation outputs

- True codes climb in Z while mismatches and heavy shielding do not → evidence
  for \(\lambda > 0\) and intact shielding/polarisation dependencies.
- Small sign-flip \(p\)-values → statistics are biased even without relying on
  the CLT.
- Fingerprint regressions extract the \(\lambda\)-family constants from data.

## 9. Operational flow

1. Generate orthogonal codes \(c_k(t)\) (±1, 10 Hz chips).
2. Transmit with AM under equal energy and bandwidth while respecting safety.
3. Update \(s_{b,k} = \langle y_b, c_{k,b} \rangle / \|c_{k,b}\|\) every 2 s
   block.
4. Track sequential Z and sign-flip \(p\)-values online.
5. Sweep shielding, polarisation, and distance; regress the target slopes
   (−1/20, −1, |cos θ|).
6. Once the physical tests pass, toggle \(\rho\) to probe \(\mu > 0\) using
   difference tests or sequential Bayes factors.

## 10. Streaming Maxwell evidence into SpiralK and desire loops

- The Rust core exposes `MaxwellSpiralKBridge` to translate sequential pulses
  into SpiralK-ready `soft(...)` hints. Each registered channel is sanitised so
  KDSl programs can target it directly.
- Weights are derived from the Z magnitude, keeping hints between 0.55 and 0.95
  (tuneable) so strong detections carry proportionally stronger influence while
  weaker drifts stay gentle.
- Call `push_pulse(channel, &pulse)` for every matched-filter stream, then
  `script()` to obtain a KDSl snippet. Prepend an existing program via
  `with_base_program(...)` when the bridge needs to extend a live policy.
- The resulting string can be injected into the existing SpiralK orchestration
  so code-driven Z pulses bias Above/Here/Beneath steering without bespoke glue
  code.
- Enable the PSI feature to access `MaxwellPsiTelemetryBridge`. Each pulse is
  converted into a PSI reading, optional `PsiEvent::ThresholdCross` entries, and
  a `SoftlogicZFeedback` sample so the `DesirePsiBridge` captures the same
  evidence stream without manual hub plumbing.
- Use `MaxwellDesireBridge` to map channel labels to vocabulary windows. When a
  pulse arrives, call `hint_for(...)` to produce a `ConceptHint::Window` scaled
  by the observed Z magnitude and feed it directly into the `DesireLagrangian`.

---

## Reconstruction within M_Ψ⁺

- **Total input**:

  \[
  u_{\text{tot}}(t) = \sum_k [\lambda_k + \mu_k \rho_k(t)] (c_k * \kappa_k)(t
  - \tau_k).
  \]

- **Observation equation**:
  \(y(t) = \int h(\tau) u_{\text{tot}}(t - \tau) d\tau + \eta(t)\).
- Mobility terms \(D_k\) and similar factors can be absorbed into
  \(\mu_k \rho_k\).

### Mode-specific extensions

- **A — Explicit alignment (Z)**: \(g_A \equiv 1\).
- **B — Timing resonance**: sweep delays \(\tau\) and identify the peak in
  \(Z(\tau)\).
- **C — Affective gating**: regress
  \(g_C = \sigma(\beta_0 + \beta_1 A_s A_r)\).
- **D — Unconscious embedding**: regress latent behavioural parameters such as
  hierarchical DDM drifts.
- **E — Memory similarity**: correlate effect sizes with similarity indices
  \(G\).
- **F — Self-reinforcement (concept contagion)**: state update
  \(m(t+1) = (1-\eta)m(t) + \eta \phi(s_t)\) with
  \(\mu^{\text{eff}} = \mu^{(0)} + \omega m(t)\).

### External magnetic field \(\mathbf B_e\)

- Treat \(\mathbf B_e\) as a first-order perturbation on mobility or temporal
  kernels:

  \[
  \kappa_k(\mathbf B_e) \approx \kappa_k^{(0)} +
  \Bigl(\frac{\partial \kappa_k}{\partial \mathbf B}\Bigr)_0 \cdot
  \mathbf B_e,
  \quad D_k(\mathbf B_e) \approx D_k^{(0)} [1 + \chi_k \hat{\mathbf n}_k \cdot
  \mathbf B_e].
  \]

- Rotate the geomagnetic field and regress the interaction with the coded
  envelope; the Maxwell fingerprints should remain intact while the field term
  modulates amplitude.

## Estimators and “win conditions”

- Always report sequential Z (2 s blocks) and sign-flip \(p\)-values.
- Fingerprint targets: shielding slope −1/20, distance slope −1, polarisation
  \(|\cos \theta|\), latency SD ≲ 70 ms.
- Mode-specific detectors: affective interaction \(\beta_1 > 0\), similarity
  correlations, positive self-reinforcement \(\omega > 0\), etc.

## Experimental translation

1. **A (explicit alignment)**: already operational with the coded envelope,
   sequential Z, and sign-flip routines.
2. **B (timing resonance)**: sweep transmit delays ±hundreds of ms and map the
   \(Z(\tau)\) peak.
3. **C (affect)**: insert affective blocks on the transmitter side, keep the
   receiver blinded, and test the interaction term.
4. **E (similarity)**: choose high/low similarity pairs and design CLT-shortened
   blocks.
5. **Magnetic field**: rotate the geomagnetic vector while tracking the envelope
   interaction to confirm the modulation rides atop the Maxwell fingerprint.

---

## Summary

Injecting a coded physical envelope that obeys Maxwell’s law, then integrating
matched-filter statistics with the CLT and fingerprint regressions, recovers the
physical gain \(\lambda\). Layering the meaning gate \(\mu \rho(t)\) on top lets
us detect and estimate cross-effects between physics and semantics within the
same platform.

