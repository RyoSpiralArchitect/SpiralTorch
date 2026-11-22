//! A shared Euclidean budget on the selected LoRA factors and trainable geometry coordinates.
use super::*;
use crate::geometry_artifact::{GradientSize, ParameterRay, RayMotion};

#[derive(Clone, Debug, Serialize)]
pub struct CausalLmAdaptationRayGroup {
    pub origin_id: String,
    pub result_id: String,
    pub trainable_values: usize,
    pub gradient_l2: f64,
    pub realized_l2: f64,
    pub gradient_dot_step: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct CausalLmAdaptationRayStep {
    pub method: &'static str,
    pub mode: CausalLmAdaptationMode,
    pub requested_l2: f64,
    pub gradient_l2: f64,
    pub realized_l2: f64,
    /// L2 error of the realized displacement vector divided by the requested distance.
    pub relative_vector_error: f64,
    /// First-order prediction only. No post-update forward pass is performed.
    pub gradient_dot_step: f64,
    pub lora: Option<CausalLmAdaptationRayGroup>,
    pub geometry: Option<CausalLmAdaptationRayGroup>,
}

fn group(
    before: &str,
    after: &str,
    size: &GradientSize,
    movement: &RayMotion,
) -> CausalLmAdaptationRayGroup {
    CausalLmAdaptationRayGroup {
        origin_id: before.into(),
        result_id: after.into(),
        trainable_values: size.values,
        gradient_l2: size.norm(),
        realized_l2: movement.norm(),
        gradient_dot_step: movement.dot,
    }
}

impl NativeCausalLm {
    pub(super) fn admit_adaptation_distance(
        &self,
        mode: CausalLmAdaptationMode,
        distance: f64,
    ) -> CausalLmResult<()> {
        let lora = self
            .lora
            .as_ref()
            .filter(|_| mode != CausalLmAdaptationMode::Geometry)
            .map_or(0, |a| a.parameter_count());
        let geometry = self
            .geometry_artifact()
            .filter(|_| mode != CausalLmAdaptationMode::Lora)
            .map_or(0, GeometryAdapterArtifact::trainable_values);
        Ok(ParameterRay::admit_distance(distance, lora + geometry)?)
    }

    /// Normalize the sealed gradient in the concatenation of the selected groups,
    /// before any SGD proposal or f32 parameter rounding. A joint gradient may
    /// also be used for either individual control at the very same origin.
    pub fn step_adaptation_gradient_at_distance(
        &self,
        grads: &NativeCausalLmAdaptationGradients,
        mode: CausalLmAdaptationMode,
        target_l2: f64,
    ) -> CausalLmResult<(Self, CausalLmAdaptationRayStep)> {
        use CausalLmAdaptationMode::{Geometry, Lora};
        self.validate_adaptation_mode(mode)?;
        self.validate_adaptation_gradient_owner(grads)?;
        let params = grads.parameters();
        let mut ls = GradientSize::default();
        let lora = if mode != Geometry {
            let l = self.lora.as_ref().expect("admitted LoRA");
            let blocks = l.admit_ray(
                params
                    .get(&l.path())
                    .ok_or("missing selected LoRA gradients")?,
            )?;
            for b in blocks {
                ls.include(b);
            }
            Some((l, blocks))
        } else {
            None
        };
        let mut gs = GradientSize::default();
        let geometry = if mode != Lora {
            let rt = self.geometry.as_ref().expect("admitted geometry");
            let art = rt.artifact();
            // Validate exactly the selected placements, without copying channel vectors or LoRA blocks.
            let view = art
                .record()
                .placements
                .iter()
                .map(|p| {
                    Ok((
                        p.path.clone(),
                        params
                            .get(&p.path)
                            .ok_or("missing selected geometry gradient")?,
                    ))
                })
                .collect::<CausalLmResult<std::collections::BTreeMap<_, _>>>()?;
            rt.validate_gradients(art.artifact_id(), &view)?;
            gs = art.gradient_size(params);
            Some(art)
        } else {
            None
        };
        let gradient_l2 = ls.norm().hypot(gs.norm());
        let ray = ParameterRay::new(target_l2, gradient_l2, ls.values + gs.values)?;
        let (mut lm, mut gm) = (RayMotion::default(), RayMotion::default());
        let mut next = self.clone();
        if let Some((adapter, blocks)) = lora {
            if target_l2 != 0. && ls.norm() != 0. {
                next.lora = Some(Arc::new(adapter.project_ray(self, blocks, &ray, &mut lm)?));
            }
        }
        if let Some(art) = geometry {
            if target_l2 != 0. && gs.norm() != 0. {
                let candidate = art.project_gradient_ray(params, &ray, &mut gm)?;
                next.geometry = Some(Arc::new(NativeGeometryRuntime::new(candidate)));
            }
        }
        let realized_l2 = lm.norm().hypot(gm.norm());
        let relative_vector_error = lm.relative_error().hypot(gm.relative_error());
        ray.admit_motion(realized_l2, relative_vector_error)?;
        let r = CausalLmAdaptationRayStep {
            method: "selected_parameter_gradient_ray_f32.v1",
            mode,
            requested_l2: target_l2,
            gradient_l2,
            realized_l2,
            relative_vector_error,
            gradient_dot_step: lm.dot + gm.dot,
            lora: lora.map(|_| {
                group(
                    self.lora_artifact_id().unwrap(),
                    next.lora_artifact_id().unwrap(),
                    &ls,
                    &lm,
                )
            }),
            geometry: geometry.map(|g| {
                group(
                    g.artifact_id(),
                    next.geometry_artifact_id().unwrap(),
                    &gs,
                    &gm,
                )
            }),
        };
        Ok((next, r))
    }
}
