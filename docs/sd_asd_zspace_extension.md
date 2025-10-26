# sD/AsD Extension of Z-Space Relativity

The sD/AsD framework augments Z-space general relativity with a pair of complementary tensor sectors:

- **sD (symmetric dilation)** captures volumetric or trace-like deformations of the Z-space block metric.
- **AsD (antisymmetric dilation)** encodes rotational or flux-like excitations that do not contribute to the metric trace but influence torsion and mixed gauge couplings.

Together they provide a structured way to embed higher-form dynamics into the geometric data exported by `ZRelativityModel`.

## 1. Field Content and Geometric Placement

1. Equip the Z-space manifold \(Z\) with the Lorentzian metric \(g_{IJ}\) used in the base relativity construction.
2. Introduce an endomorphism-valued 1-form \(\mathcal{D}\) decomposed into symmetric and antisymmetric parts:
   \[
   \mathcal{D} = \mathcal{D}^{(s)} + \mathcal{D}^{(a)},\qquad \mathcal{D}^{(s)} = \frac{1}{2}(\mathcal{D}+\mathcal{D}^\top),\qquad \mathcal{D}^{(a)} = \frac{1}{2}(\mathcal{D}-\mathcal{D}^\top).
   \]
3. Define **sD** as the trace-adjusted symmetric component \(\mathrm{sD}_{IJ} = \mathcal{D}^{(s)}_{IJ} - \frac{1}{4} g_{IJ} \operatorname{tr}(\mathcal{D}^{(s)})\).
4. Define **AsD** as the antisymmetric 2-form \(\mathrm{AsD}_{IJ} = \mathcal{D}^{(a)}_{IJ}\), naturally interpreted as a Kalb–Ramond style flux.

## 2. Coupling to the Einstein Sector

- **Effective stress-energy.** Promote sD/AsD to sources through
  \[
  T^{\mathrm{sD}}_{IJ} = \lambda_s\Big(\mathrm{sD}_{IK}\mathrm{sD}^K_{\ J} - \tfrac{1}{4} g_{IJ}\mathrm{sD}_{KL}\mathrm{sD}^{KL}\Big),\qquad
  T^{\mathrm{AsD}}_{IJ} = \lambda_a\Big(\mathrm{AsD}_{IK}\mathrm{AsD}^K_{\ J} - \tfrac{1}{4} g_{IJ}\mathrm{AsD}_{KL}\mathrm{AsD}^{KL}\Big).
  \]
  Feed the combined tensor into `ZRelativityFieldEquation::with_source` to evaluate the modified Einstein residual.
- **Back-reaction control.** The couplings \(\lambda_s\) and \(\lambda_a\) expose knobs for how strongly each sector curves Z-space. Initialise them in SpiralTorch with `sd_asd::Couplings::default()` and override per experiment via `.with_lambda_s(value)` or `.with_lambda_a(value)`.

## 3. Dynamics and Constraints

1. **Kinetic terms.** Use
   \[
   \mathcal{L}_{\mathrm{sD}} = -\frac{1}{2} (\nabla_I \mathrm{sD}_{JK})(\nabla^I \mathrm{sD}^{JK}),\qquad
   \mathcal{L}_{\mathrm{AsD}} = -\frac{1}{12} H_{IJK} H^{IJK},\qquad H = \mathrm{d}\mathrm{AsD}.
   \]
   The SpiralTorch runtime exposes these via `sd_asd::lagrangian_density(z_slice)` which returns both contributions and their gradients.
2. **Gauge structure.** AsD inherits a 1-form gauge symmetry \(\mathrm{AsD} \mapsto \mathrm{AsD} + \mathrm{d}\Lambda\). The helper `sd_asd::GaugeFixer::lorenz(alpha)` enforces \(\nabla^I \mathrm{AsD}_{IJ} = 0\) for numerical stability.
3. **Constraint solver.** The mixed trace constraint for sD is implemented in `sd_asd::project_tracefree`, available on both Rust and Python bindings, ensuring the symmetric component remains trace-free at every integration step.

## 4. Integration Path with General Relativity

Follow these steps to link the sD/AsD sector with the existing GR tooling:

1. **Bundle assembly.** Call `sd_asd::assemble_bundle(z_metric, dilation_form)` to convert raw metric data and initial dilation forms into a coherent state vector. The bundle reuses `ProductGeometry::internal_volume_density` to stay consistent with the base Z-relativity volume form.
2. **Model augmentation.** Pass the bundle into `ZRelativityModel::with_sd_asd(bundle)` to obtain a `ZRelativityModel` extended with sD/AsD parameters. The resulting model keeps the legacy Einstein block while surfacing new accessors: `sd_tensor()`, `asd_form()`, and `sdasd_couplings()`.
3. **Einstein solver.** Invoke `spiraltorch.nn.ZRelativityModule.with_sd_asd(model)` when constructing neural surrogates so that training loops see the augmented field content. During PDE solves, `sd_asd::einstein_stepper` integrates the combined Einstein–dilation system using the same adaptive stepper configuration as `general_relativity::adm_stepper`.
4. **Diagnostics.** The vision stack adds `spiraltorch.vision.zrelativity_heatmap(..., field="sd")` and `field="asd"` to visualise energy densities. Curvature monitors accept an optional `sd_asd_energy` keyword that overlays the dilation contributions on top of the Kretschmann scalar plots.

## 5. Example Workflow

```python
from spiraltorch import geometry, sd_asd

# 1. Build a base Z-relativity model
product = geometry.ProductMetric.lorentzian_warp(
    warp_factor=geometry.WarpFactor.constant(1.0),
    external_metric=geometry.LorentzianMetric.friedmann(ro=1.0, curvature=0.0),
    internal_metric=geometry.InternalMetric.flat(dim=2)
)
model = geometry.ZRelativityModel.assemble(product)

# 2. Seed sD/AsD data
seed = sd_asd.DilationSeed.gaussian(patch_radius=0.2, amplitude=0.05)
bundle = sd_asd.assemble_bundle(model.metric_tensor(), seed)
model = model.with_sd_asd(bundle)

# 3. Evaluate the coupled Einstein residual
residual = model.field_equation_tensor(include_sd_asd=True)
print(residual.norm())
```

This script demonstrates how the SpiralTorch APIs interleave sD/AsD with the GR back-end so that new dilation fields feed into curvature computations transparently.

## 6. Relationship to Existing Documentation

- For the foundational GR framework on Z-space, see [Introducing General Relativity on an Abstract Z-Space Manifold](general_relativity_zspace.md).
- For the SpinoTensorVector perspective, consult [SpinoTensorVector Formalism in Z-Space](stv_z_space.md) to combine sD/AsD with fermionic/tensorial data.
- For introductory Z-space concepts, review [Z-Space Primer](zspace_intro.md).

---

With the sD/AsD extension in place, Z-space relativity inherits a richer set of geometric responses, allowing SpiralTorch experiments to probe how symmetric dilations and antisymmetric fluxes reshape curvature, energy transport, and neural surrogates alike.
