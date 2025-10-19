# Invariant barrier gating and contraction notes

> “Try threading these ideas wherever the runtime can profit from them.”
>
> The memo distils the SpiralTorch dynamics notebook into a checklist that is
> immediately actionable inside the `st-core` control helpers implemented in
> `crates/st-core/src/theory/spiral_dynamics.rs`.

## A. Barrier design: making \( \mu_{\mathrm{eff}} \le 0 \) forward invariant

### A-0. Setup

The dissipative SpiralTorch core takes the form
\\[
\\begin{aligned}
\\dot z &= (\\mu_0 + \\gamma\\,\\hat c(u,s)) z - \\nu |z|^2 z + i\\omega z,\\\\
\\dot u &= \\kappa(\\alpha\\,\\Re z - \\beta\\,\\Im z - \\theta) - \\tau u,\\\\
\\dot s &= -\\lambda s + \\rho\\,\\Im z,
\\end{aligned}
\\]
where the forcing is a logistic gate
\\[
\\hat c(u,s) = \\frac{c_{\\max}}{1 + \\exp(-[u - \\sigma_s s])}.
\\]
The effective growth rate is \\(\\mu_{\\mathrm{eff}} = \\mu_0 + \\gamma\\,\\hat c\\).

### A-1. Hard barrier (sufficient condition)

Choose the gain ceiling such that
\\[
\\boxed{\\ \\mu_0 + \\gamma\\,c_{\\max} \\le 0\\ }
\\]
This yields \\(\\hat c \\le c_{\\max}\\) for all time, hence \\(\\mu_{\\mathrm{eff}} \\le 0\\) everywhere. The invariant set is the entire state space. The Lyapunov candidate \\(V=\\tfrac12|z|^2\\) obeys
\\[
\\dot V = (\\mu_0 + \\gamma\\hat c)|z|^2 - \\nu |z|^4 \\le -\\nu |z|^4 \\le 0,
\\]
so LaSalle guarantees global convergence to \\(z=0\\). This is the most robust option and should be preferred when the actuation budget can tolerate the strict cap.

### A-2. Soft barrier with container gating

To trade aggressiveness for responsiveness, cap the forcing with a state
modifier:
\\[
\\hat c(u,s) = \\frac{c_{\\text{base}}}{(1 + \\kappa_b s)\\,(1 + e^{-[u - \\sigma_s s]})}.
\\]
The worst-case forcing is now
\\[
\\sup_u \\hat c(u,s) = \\frac{c_{\\text{base}}}{1 + \\kappa_b s} \\le c_{\\text{base}}.
\\]
Picking
\\[
\\boxed{\\ c_{\\text{base}} \\le c_\\star := -\\frac{\\mu_0 + m}{\\gamma} \\quad (m > 0)\\ }
\\]
guarantees \\(\\hat c \\le c_\\star\\) for all \\(s \\ge 0\\), which in turn yields \\(\\mu_{\\mathrm{eff}} \\le -m < 0\\). As the container state \\(s\\) grows, the ceiling tightens further, gently nudging the dynamics back into the safe region while leaving additional headroom when \\(s\\) is small.

### A-3. Control-barrier-function check

Let \\(h(u,s) := - (\\mu_0 + \\gamma\\hat c(u,s))\\) and define the safe set
\\(\\mathcal{S} = \\{h \\ge 0\\} = \\{\\mu_{\\mathrm{eff}} \\le 0\\}\\). The logistic derivatives are
\\[
\\partial_u \\hat c = \\hat c\\Bigl(1 - \\frac{\\hat c}{c_{\\max}}\\Bigr), \\qquad
\\partial_s \\hat c = -\\sigma_s\\,\\partial_u \\hat c.
\\]
On the boundary where \\(h = 0\\), the barrier derivative satisfies
\\[
\\dot h = -\\gamma\\,\\partial_u \\hat c\\, (\\dot u - \\sigma_s \\dot s).
\\]
A sufficient condition for forward invariance is therefore
\\[
\\boxed{\\ \\dot u - \\sigma_s \\dot s \\le 0 \\quad (h = 0)\\ }
\\]
Substituting the dynamics gives
\\[
\\kappa(\\alpha\\,\\Re z - \\beta\\,\\Im z - \\theta) - \\tau u + \\sigma_s\\lambda s - \\sigma_s\\rho\\,\\Im z \\le 0.
\\]
Design tips: increase \\(\\tau\\), ensure \\(\\lambda \\le \\tau\\), reduce the product \\(\\sigma_s\\rho\\), and choose small \\(\\kappa\\) for large \\(|\\alpha|, |\\beta|\\). In practice we enforce the hard or soft barrier during synthesis and keep this check as a validation step when tuning the controller online.

## B. Closed form for the steady radius of the breathing mode

The breathing subsystem evolves \\(r = |z|\\) via
\\[
\\dot r = \\Bigl[\\underbrace{\\mu_0 - \\eta}_{A} + \\gamma\\,\\frac{c_1}{1 + q r^2} - \\nu r^2 - \\gamma\\sigma_s s\\Bigr] r.
\\]
Setting \\(y = r^2\\) gives the scalar equation
\\[
g(y) = A + \\frac{\\gamma c_1}{1 + q y} - \\nu y - \\gamma\\sigma_s s = 0.
\\]
Clearing the denominator produces the quadratic
\\[
a_2 y^2 + a_1 y + a_0 = 0,
\\]
with coefficients
\\[
\\begin{aligned}
a_2 &= q\\nu > 0,\\\\
a_1 &= \\nu - qA + q\\gamma\\sigma_s s,\\\\
a_0 &= (-A + \\gamma\\sigma_s s) - \\gamma c_1.
\\end{aligned}
\\]
The physically relevant root is
\\[
\\boxed{\\ r^{\\circ 2} = y^{\\circ} = \\frac{-a_1 + \\sqrt{a_1^2 - 4 a_2 a_0}}{2 a_2} \\ge 0.\\ }
\\]

Uniqueness and stability follow from the monotonic derivative
\\(g'(y) = -\\frac{\\gamma c_1 q}{(1 + qy)^2} - \\nu < 0\\), and the boundary condition
\\(g(0) = A + \\gamma c_1 - \\gamma\\sigma_s s\\). A strictly positive equilibrium exists iff
\\[
\\boxed{\\ g(0) > 0 \\iff \\mu_0 - \\eta + \\gamma c_1 > \\gamma\\sigma_s s.\\ }
\\]
Implicit differentiation shows the container suppresses the radius:
\\[
\\frac{\\mathrm{d} y^{\\circ}}{\\mathrm{d}s} = -\\frac{\\partial g / \\partial s}{\\partial g / \\partial y} = \\frac{\\gamma\\sigma_s}{-\\frac{\\gamma c_1 q}{(1 + q y^{\\circ})^2} - \\nu} < 0.
\\]
Hence larger container occupancy shrinks the breathing amplitude, making the gating knob an effective stabiliser.

## C. Lower bound on the contraction rate \\(\\varepsilon\\)

Linearising the contraction experiment around the origin yields the Jacobian
\\[
J = \\begin{bmatrix}
-a & -\\omega & \\gamma & 0\\\\
\\omega & -a & 0 & \\sigma_s\\\\
-\\kappa\\alpha & \\kappa\\beta & -\\tau & -\\sigma_s\\\\
0 & -\\rho & 0 & -\\lambda
\\end{bmatrix}.
\\]
The induced 2-norm matrix measure is \\(\\mu_2(J) = \\lambda_{\\max}((J + J^\\top)/2)\\). The symmetric part is
\\[
S = \\begin{bmatrix}
-a & 0 & \\tfrac{\\gamma - \\kappa\\alpha}{2} & 0\\\\
0 & -a & \\tfrac{\\kappa\\beta}{2} & \\tfrac{\\sigma_s - \\rho}{2}\\\\
\\tfrac{\\gamma - \\kappa\\alpha}{2} & \\tfrac{\\kappa\\beta}{2} & -\\tau & -\\tfrac{\\sigma_s}{2}\\\\
0 & \\tfrac{\\sigma_s - \\rho}{2} & -\\tfrac{\\sigma_s}{2} & -\\lambda
\\end{bmatrix}.
\\]
Applying Gershgorin gives the upper bound
\\[
\\lambda_{\\max}(S) \\le \\max_i (c_i + R_i),
\\]
with
\\[
\\begin{aligned}
c_1 &= -a, & R_1 &= \\tfrac12|\\gamma - \\kappa\\alpha|,\\\\
c_2 &= -a, & R_2 &= \\tfrac12(|\\kappa\\beta| + |\\sigma_s - \\rho|),\\\\
c_3 &= -\\tau, & R_3 &= \\tfrac12(|\\gamma - \\kappa\\alpha| + |\\kappa\\beta| + |\\sigma_s|),\\\\
c_4 &= -\\lambda, & R_4 &= \\tfrac12(|\\sigma_s - \\rho| + |\\sigma_s|).
\\end{aligned}
\\]
A sufficient contraction rate is therefore
\\[
\\boxed{\\ \\varepsilon \\ge -\\max(c_i + R_i) = \\min\\left\\{\\begin{aligned}
&a - \\tfrac12|\\gamma - \\kappa\\alpha|,\\\\
&a - \\tfrac12(|\\kappa\\beta| + |\\sigma_s - \\rho|),\\\\
&\\tau - \\tfrac12(|\\gamma - \\kappa\\alpha| + |\\kappa\\beta| + |\\sigma_s|),\\\\
&\\lambda - \\tfrac12(|\\sigma_s - \\rho| + |\\sigma_s|)
\\end{aligned}\\right\\}.\\ }
\\]
Keeping the minimum positive ensures \\(S \\preceq -\\varepsilon I\\), meaning the linearised system is contracting. The cubic damping in the \\((\\psi, \\phi)\\) block contributes an additional negative shift of at least \\(\\nu r^2\\) on the dominant eigenvalue, so the true contraction rate is no worse than
\\[
\\varepsilon_{\\text{actual}} \\ge \\varepsilon_{\\text{linear}} + \\nu r^2,
\\]
with \\(r^2 = \\psi^2 + \\phi^2\\). Even small oscillations therefore accelerate convergence when the linear margin is positive.

## Summary checklist

* **Barrier design**
  * Prefer the hard cap \\(\\mu_0 + \\gamma c_{\\max} \\le 0\\) when the actuation envelope allows it.
  * Otherwise enforce \\(c_{\\text{base}} \\le -\\tfrac{\\mu_0 + m}{\\gamma}\\) and \\(\\kappa_b > 0\\) so the container shrinks the forcing under load.
  * Validate tuning by checking \\(\\dot u - \\sigma_s \\dot s \\le 0\\) on the barrier surface.
* **Steady radius**
  * The quadratic coefficients are \\(a_2 = q\\nu\\), \\(a_1 = \\nu - qA + q\\gamma\\sigma_s s\\), \\(a_0 = (-A + \\gamma\\sigma_s s) - \\gamma c_1\\).
  * Existence condition: \\(\\mu_0 - \\eta + \\gamma c_1 > \\gamma\\sigma_s s\\).
  * Container influence: \\(\\mathrm{d} r^{\\circ 2}/\\mathrm{d}s < 0\\).
* **Contraction**
  * Maintain a positive margin in the minimum from the Gershgorin bound.
  * Leverage the cubic damping \\(\\nu r^2\\) to widen the contraction basin when designing experiments.
* **Hopf bookkeeping**
  * The equilibrium sits at \(z^\ast = 0,\; u^\ast = -\kappa\theta/\tau,\; s^\ast = 0\).
  * Use `hopf_normal_form` to recover \(\mu_{\mathrm{eff},0}\), the center-manifold correction \(C\), and the cubic coefficient \(\alpha_3 = \nu - \gamma C\).
  * `HopfRegime::Supercritical` iff \(\alpha_3 > 0\); strengthen the container (raise \(\sigma_s\), \(\rho\), or lower \(\lambda\)) if the routine flags a subcritical branch.
* **Dimensionless combos**
  * `dimensionless_parameters` maps the tuning set to \((\bar\mu, \bar\gamma, \bar\omega)\) and the two gain clusters \(\kappa a/(\tau\nu)\), \(\sigma_s\rho/(\lambda\nu)\).
  * These ratios index phase plots and reduce sweeping experiments to a handful of axes.
* **Noise tolerance**
  * `ito_mean_square_bound` yields the closed form upper bound on \(\mathbb{E}|z|^2\) for additive complex noise.
  * Keep \(\mu_{\mathrm{eff}} \le 0\); the bound contracts to zero as the noise power \(\sigma^2 \rightarrow 0\).
* **Audit vs. container gain**
  * `audit_container_balance` reports \(\kappa a/\tau - \sigma_s\rho/\lambda\).
  * A positive gap pushes the Hopf coefficient toward the subcritical side; tighten container feedback to cancel the excess.
