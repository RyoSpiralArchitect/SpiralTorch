//! Four row-local maps on one captured input and upstream adjoint; no counterfactual decoder passes.
use super::*;
use crate::geometry_artifact::WaveStageRecipe;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WaveFactorConfig {
    pub curvature: f32,
    pub saturation: f32,
    pub porosity: f32,
    pub log_radius: f32,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct WaveFactorEnergies {
    /// Sum of squared rounded output minus input, accumulated in f64.
    pub residual_sq: f64,
    /// Sum of squared rounded input adjoint minus the shared upstream.
    pub correction_sq: f64,
}

#[derive(Debug)]
pub struct ChannelParameterVjp {
    pub gate: Vec<f32>,
    pub bias: Vec<f32>,
}

enum LocalMap {
    Affine(ChannelAffineLearningBatch),
    Wave(WaveGateLearningBatch),
}

impl LocalMap {
    fn output(&self) -> &Tensor {
        match self {
            Self::Affine(b) => b.output(),
            Self::Wave(b) => b.output(),
        }
    }

    fn input_vjp(&self, seed: &Tensor) -> Result<Tensor> {
        Ok(match self {
            Self::Affine(b) => b.vjp_input(seed)?,
            Self::Wave(b) => b.vjp_input(seed)?,
        })
    }

    fn parameters(&self, seed: &Tensor) -> Result<ChannelParameterVjp> {
        let (g, b) = match self {
            Self::Affine(batch) => {
                let (_, g, b) = batch.vjp_requested(seed.data(), false)?;
                (g, b)
            }
            Self::Wave(batch) => {
                let (p, _) = batch.vjp_requested(seed.data(), false, false)?;
                (p.grad_gate, p.grad_bias)
            }
        };
        Ok(ChannelParameterVjp {
            gate: g.data().to_vec(),
            bias: b.data().to_vec(),
        })
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct WaveFactorObservation {
    pub schema: &'static str,
    pub config: WaveFactorConfig,
    pub factual: &'static str,
    pub strength: f32,
    pub elements: usize,
    pub input_sq: f64,
    pub upstream_sq: f64,
    pub affine: WaveFactorEnergies,
    pub saturation: WaveFactorEnergies,
    pub projection: WaveFactorEnergies,
    pub both: WaveFactorEnergies,
    pub gate_outside: usize,
    pub affine_outside: usize,
    /// Affine control energy on coordinates whose post-gate affine value crosses the rewrite limit.
    pub affine_on_outside: WaveFactorEnergies,
    pub factual_output_max_abs_diff: f64,
    pub factual_adjoint_max_abs_diff: f64,
}

impl WaveFactorConfig {
    fn local_map(
        &self,
        kind: &str,
        x: &Tensor,
        placement: &GeometryAdapterPlacement,
        max: usize,
    ) -> Result<LocalMap> {
        let gate = tensor(placement, "gate");
        let bias = tensor(placement, "bias");
        let (rows, cols) = x.shape();
        Ok(match kind {
            "affine" => LocalMap::Affine(ChannelAffineKernel::new(max)?.forward(
                x.data(),
                gate,
                bias,
                rows,
                cols,
            )?),
            "saturation" => LocalMap::Wave(
                WaveStageRecipe::Saturation {
                    saturation: self.saturation,
                    porosity: self.porosity,
                }
                .forward(x.data(), gate, bias, [rows, cols], max)?,
            ),
            "projection" => LocalMap::Wave(
                WaveStageRecipe::Projection {
                    curvature: self.curvature,
                    log_radius: self.log_radius,
                }
                .forward(x.data(), gate, bias, [rows, cols], max)?,
            ),
            "both" => LocalMap::Wave(
                WaveGateKernel::new(self.curvature, self.saturation, self.porosity, max)?
                    .forward_with_log_radius(x.data(), gate, bias, rows, cols, self.log_radius)?,
            ),
            _ => return Err("unknown local channel map".into()),
        })
    }

    pub(crate) fn admit(&self, p: &GeometryAdapterPlacement) -> Result<(&'static str, usize)> {
        use GeometryAdapterRecipe::*;
        let (kind, max, strength) = match &p.recipe {
            ChannelAffine {
                max_values,
                strength,
                ..
            } => ("affine", *max_values, *strength),
            WaveRadius {
                kernel, strength, ..
            } if (
                kernel.curvature,
                kernel.saturation,
                kernel.porosity,
                tensor(p, "log_radius")[0],
            ) == (
                self.curvature,
                self.saturation,
                self.porosity,
                self.log_radius,
            ) =>
            {
                ("both", kernel.max_values, *strength)
            }
            WaveStage {
                stage:
                    WaveStageRecipe::Saturation {
                        saturation,
                        porosity,
                    },
                max_values,
                strength,
                ..
            } if (*saturation, *porosity) == (self.saturation, self.porosity) => {
                ("saturation", *max_values, *strength)
            }
            WaveStage {
                stage:
                    WaveStageRecipe::Projection {
                        curvature,
                        log_radius,
                    },
                max_values,
                strength,
                ..
            } if (*curvature, *log_radius) == (self.curvature, self.log_radius) => {
                ("projection", *max_values, *strength)
            }
            _ => {
                return Err(
                    "wave factorial requires a matching affine, Wave v2 or single-stage source"
                        .into(),
                )
            }
        };
        if strength as f32 <= 0. {
            return Err("wave factorial requires a positive effective residual strength".into());
        }
        WaveGateKernel::new(self.curvature, self.saturation, self.porosity, max)?;
        WaveGateKernel::validate_log_radius(self.log_radius)?;
        Ok((kind, max))
    }
}

impl NativeGeometryBatch {
    pub(crate) fn wave_factor_input(&self) -> Result<&Tensor> {
        match &self.pullback {
            Pullback::Affine(b) => Ok(b.input()),
            Pullback::WaveRadius(b) | Pullback::WaveStage(b) => Ok(b.input()),
            _ => Err("wave factorial requires an active channel map".into()),
        }
    }

    /// Parameter pullbacks at one captured input. No optimizer, decoder rerun, or
    /// radius update; the factual arm reuses its actual forward snapshot.
    pub fn wave_parameter_pullbacks(
        &self,
        config: WaveFactorConfig,
        upstream: &[f32],
    ) -> Result<BTreeMap<&'static str, ChannelParameterVjp>> {
        let (factual, max) = config.admit(&self.placement)?;
        let x = self.wave_factor_input()?;
        let seed = self
            .direction(upstream)?
            .scale_with_backend(self.strength, TensorUtilBackend::Cpu)?;
        let mut out = BTreeMap::new();
        for kind in ["affine", "saturation", "projection", "both"] {
            let vjp = if kind == factual {
                let mut p = self.vjp_parameters(upstream)?;
                ChannelParameterVjp {
                    gate: p.remove("gate").ok_or("missing factual gate")?,
                    bias: p.remove("bias").ok_or("missing factual bias")?,
                }
            } else {
                config
                    .local_map(kind, x, &self.placement, max)?
                    .parameters(&seed)?
            };
            if vjp.gate.len() != x.shape().1
                || vjp.bias.len() != x.shape().1
                || vjp.gate.iter().chain(&vjp.bias).any(|v| !v.is_finite())
            {
                return Err("invalid local channel parameter pullback".into());
            }
            out.insert(kind, vjp);
        }
        Ok(out)
    }

    pub(crate) fn wave_factorial(
        &self,
        config: WaveFactorConfig,
        dy: &Tensor,
        factual_dx: &Tensor,
    ) -> Result<WaveFactorObservation> {
        let (factual, max) = config.admit(&self.placement)?;
        let x = self.wave_factor_input()?;
        if x.shape() != dy.shape() || x.shape() != factual_dx.shape() {
            return Err("wave factorial adjoint shape differs".into());
        }
        let (_, cols) = x.shape();
        let gate = tensor(&self.placement, "gate");
        let bias = tensor(&self.placement, "bias");
        let wave = WaveGateKernel::new(config.curvature, config.saturation, config.porosity, max)?;
        let backend = TensorUtilBackend::Cpu;
        let _policy = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
        let seed = dy.scale_with_backend(self.strength, backend)?;
        let gates: Vec<_> = gate.iter().map(|&g| wave.topos().saturate(g)).collect();
        let mut outside = WaveFactorEnergies::default();
        let mut count = 0;
        let (mut max_y, mut max_dx) = (0f64, 0f64);
        let mut run = |kind: &str| -> Result<WaveFactorEnergies> {
            let local = config.local_map(kind, x, &self.placement, max)?;
            let map = local.output();
            let vjp = local.input_vjp(&seed)?;
            let y =
                x.add_with_backend(&map.scale_with_backend(self.strength, backend)?, backend)?;
            let dx = dy.add_with_backend(&vjp, backend)?;
            let mut energy = WaveFactorEnergies::default();
            for i in 0..x.data().len() {
                let r = (f64::from(y.data()[i]) - f64::from(x.data()[i])).powi(2);
                let c = (f64::from(dx.data()[i]) - f64::from(dy.data()[i])).powi(2);
                energy.residual_sq += r;
                energy.correction_sq += c;
                if kind == "affine"
                    && (x.data()[i] * gates[i % cols] + bias[i % cols]).abs() > config.saturation
                {
                    count += 1;
                    outside.residual_sq += r;
                    outside.correction_sq += c;
                }
                if kind == factual {
                    max_y = max_y
                        .max((f64::from(y.data()[i]) - f64::from(self.output.data()[i])).abs());
                    max_dx = max_dx
                        .max((f64::from(dx.data()[i]) - f64::from(factual_dx.data()[i])).abs());
                }
            }
            if !energy.residual_sq.is_finite() || !energy.correction_sq.is_finite() {
                return Err("nonfinite counterfactual channel energy".into());
            }
            Ok(energy)
        };
        let (affine, saturation, projection, both) = (
            run("affine")?,
            run("saturation")?,
            run("projection")?,
            run("both")?,
        );
        if max_dx != 0. || max_y != 0. {
            return Err(
                "wave factorial factual arm differs from the observed forward or reverse".into(),
            );
        }
        let squares = |t: &Tensor| t.data().iter().map(|&v| f64::from(v).powi(2)).sum();
        Ok(WaveFactorObservation {
            schema: "spiraltorch.wave_factorial.v1",
            config,
            factual,
            strength: self.strength,
            elements: x.data().len(),
            input_sq: squares(x),
            upstream_sq: squares(dy),
            affine,
            saturation,
            projection,
            both,
            gate_outside: gate.iter().filter(|g| g.abs() > config.saturation).count(),
            affine_outside: count,
            affine_on_outside: outside,
            factual_output_max_abs_diff: max_y,
            factual_adjoint_max_abs_diff: max_dx,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn cfg() -> WaveFactorConfig {
        WaveFactorConfig {
            curvature: -0.75,
            saturation: 1.,
            porosity: 0.2,
            log_radius: 0.125,
        }
    }
    fn artifact(k: &str, gate: &[f32], bias: &[f32]) -> GeometryAdapterArtifact {
        let c = cfg();
        let stage = match k {
            "saturation" => {
                json!({"kind":"saturation","saturation":c.saturation,"porosity":c.porosity})
            }
            "projection" => {
                json!({"kind":"projection","curvature":c.curvature,"log_radius":c.log_radius})
            }
            _ => serde_json::Value::Null,
        };
        let mut tensors = vec![
            json!({"name":"gate","shape":[3],"values":gate,"requires_grad":true,"role":"parameter"}),
            json!({"name":"bias","shape":[3],"values":bias,"requires_grad":true,"role":"parameter"}),
        ];
        let recipe = match k {
            "affine" => {
                json!({"schema":"spiraltorch.channel_affine_adapter.v1","features":3,"strength":0.5,"max_values":16})
            }
            "both" => {
                tensors.push(json!({"name":"log_radius","shape":[],"values":[c.log_radius],"requires_grad":false,"role":"buffer"}));
                json!({"schema":"spiraltorch.wave_gate_adapter.v2","features":3,"strength":0.5,"learnable_radius":false,
                    "kernel":{"curvature":c.curvature,"saturation":c.saturation,"porosity":c.porosity,"max_values":16}})
            }
            _ => {
                json!({"schema":"spiraltorch.wave_stage_adapter.v1","features":3,"strength":0.5,"max_values":16,"stage":stage})
            }
        };
        GeometryAdapterArtifact::create_json(&serde_json::to_vec(&json!({"schema":"spiraltorch.geometry_adapter_artifact.v1",
            "base_model_identity":format!("sha256:{}","a".repeat(64)),"policy":"full_prefix_f32_no_cache.v1","training":true,
            "placements":[{"path":"site","training":true,"recipe":recipe,"tensors":tensors}]})).unwrap()).unwrap()
    }

    #[test]
    fn fixed_input_parameter_maps_recover_independent_native_arms() {
        let x = [1.8, -0.2, 0.7, -0.6, 1.4, 0.3];
        let dy = [0.3, -0.6, 0.7, 0.2, 0.8, -0.2];
        let batch = |kind| {
            NativeGeometryRuntime::new(artifact(kind, &[1.27, -0.9, 0.5], &[0.2, 0.03, -0.1]))
                .forward("site", &x, 2)
                .unwrap()
        };
        for factual in ["affine", "saturation", "projection", "both"] {
            let saved = batch(factual);
            let maps = saved.wave_parameter_pullbacks(cfg(), &dy).unwrap();
            assert_eq!(maps.len(), 4);
            for (kind, result) in maps {
                let expected = batch(kind).vjp_parameters(&dy).unwrap();
                assert_eq!(result.gate, expected["gate"], "{factual}/{kind}");
                assert_eq!(result.bias, expected["bias"], "{factual}/{kind}");
            }
            assert!(saved
                .wave_parameter_pullbacks(cfg(), &[f32::NAN; 6])
                .is_err());
        }
    }

    #[test]
    fn stage_parameters_and_input_vjps_match_independent_differences() {
        let x = [1.8, -0.2, 0.7, -0.6, 1.4, 0.3];
        let gate = [1.27, -0.9, 0.5];
        let bias = [0.2, 0.03, -0.1];
        let dy = [0.3, -0.6, 0.7, 0.2, 0.8, -0.2];
        for kind in ["saturation", "projection"] {
            let calc = |x: &[f32], g: &[f32], b: &[f32]| {
                NativeGeometryRuntime::new(artifact(kind, g, b))
                    .forward("site", x, 2)
                    .unwrap()
            };
            let batch = calc(&x, &gate, &bias);
            let grad = batch.vjp(&dy).unwrap();
            assert_eq!(grad.input.data(), batch.vjp_input(&dy).unwrap().data());
            assert_eq!(grad.parameters, batch.vjp_parameters(&dy).unwrap());
            for (j, expected) in grad
                .input
                .data()
                .iter()
                .chain(&grad.parameters["gate"])
                .chain(&grad.parameters["bias"])
                .enumerate()
            {
                let loss = |step| {
                    let mut v: Vec<_> = x.iter().chain(&gate).chain(&bias).copied().collect();
                    v[j] += step;
                    calc(&v[..6], &v[6..9], &v[9..])
                        .output()
                        .data()
                        .iter()
                        .zip(dy)
                        .map(|(&y, d)| f64::from(y) * f64::from(d))
                        .sum::<f64>()
                };
                let numeric = (loss(0.001) - loss(-0.001)) / 0.002;
                assert!(
                    (numeric - f64::from(*expected)).abs() < 9e-5,
                    "{kind}/{j}: {numeric} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn undefined_affine_alternative_does_not_invalidate_the_factual_wave() {
        let runtime = NativeGeometryRuntime::new(artifact("both", &[2., 1., 1.], &[0.; 3]));
        let pinned = runtime.artifact().artifact_id().to_owned();
        let batch = runtime
            .forward("site", &[f32::MAX * 0.8, 0., 0.], 1)
            .unwrap();
        let dy = Tensor::from_vec(1, 3, vec![0.5, -0.25, 0.125]).unwrap();
        let before = batch.vjp(dy.data()).unwrap();
        assert!(batch.wave_factorial(cfg(), &dy, &before.input).is_err());
        assert_eq!(runtime.artifact().artifact_id(), pinned);
        assert_eq!(batch.vjp(dy.data()).unwrap().parameters, before.parameters);
        assert_eq!(
            batch.vjp_input(dy.data()).unwrap().data(),
            before.input.data()
        );
    }

    #[test]
    fn four_factual_sources_recover_same_four_maps_with_one_captured_seed() {
        let x = [1.8, -0.2, 0.7, -0.6, 1.4, 0.3];
        let dy = Tensor::from_vec(2, 3, vec![0.3, -0.6, 0.7, 0.2, 0.8, -0.2]).unwrap();
        let mut reference = None;
        for kind in ["affine", "saturation", "projection", "both"] {
            let runtime =
                NativeGeometryRuntime::new(artifact(kind, &[1.27, -0.9, 0.5], &[0.2, 0.03, -0.1]));
            let batch = runtime.forward("site", &x, 2).unwrap();
            let dx = batch.vjp(dy.data()).unwrap().input;
            let r = batch.wave_factorial(cfg(), &dy, &dx).unwrap();
            assert_eq!((r.factual, r.gate_outside, r.affine_outside), (kind, 1, 2));
            assert_eq!(
                (
                    r.factual_output_max_abs_diff,
                    r.factual_adjoint_max_abs_diff
                ),
                (0., 0.)
            );
            let energies = [r.affine, r.saturation, r.projection, r.both]
                .map(|e| (e.residual_sq, e.correction_sq));
            assert!(energies.iter().all(|e| e.0 > 0. && e.1 > 0.));
            assert!(energies[1] != energies[2] && energies[0] != energies[3]);
            if let Some(expected) = reference {
                assert_eq!(energies, expected);
            } else {
                reference = Some(energies);
            }
            assert!(r.affine_on_outside.residual_sq < r.affine.residual_sq);
            if kind == "both" {
                let wrong = WaveFactorConfig {
                    log_radius: 1.,
                    ..cfg()
                };
                assert!(batch.wave_factorial(wrong, &dy, &dx).is_err());
            }
            assert!(batch
                .wave_factorial(cfg(), &dy, &Tensor::zeros(2, 3).unwrap())
                .is_err());
        }
    }
}
