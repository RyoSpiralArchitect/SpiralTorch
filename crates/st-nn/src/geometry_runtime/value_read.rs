use super::*;

impl NativeGeometryRuntime {
    /// The second input is already routed: the decoder, not this pointwise map,
    /// owns Q/K, causal visibility and selection of its projected values.
    pub fn forward_value_read_sequence(
        &self,
        path: &str,
        input: &[f32],
        source: &[f32],
        shape: [usize; 2],
    ) -> Result<NativeGeometryBatch> {
        let (w, n) = self.sequence_dimensions(path, shape)?;
        let p = self.placement(path)?;
        let GeometryAdapterRecipe::AttentionValueGate {
            strength,
            max_values,
            ..
        } = p.recipe
        else {
            return Err("explicit value source requires an attention value gate".into());
        };
        if input.len() != n
            || source.len() != n
            || input.iter().chain(source).any(|v| !v.is_finite())
        {
            return Err("invalid value gate inputs".into());
        }
        let rows = n / w;
        let _policy = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
        let strength = strength as f32;
        let tape = if strength == 0. {
            Pullback::Identity
        } else {
            let branch = ChannelAffineKernel::new(max_values)?.forward(
                source,
                tensor(p, "gate"),
                tensor(p, "bias"),
                rows,
                w,
            )?;
            Pullback::ValueRead(Box::new(branch), input[n - w..].to_vec())
        };
        let mut batch = self.finish_forward(
            p,
            Tensor::from_vec(rows, w, input.to_vec())?,
            strength,
            tape,
        )?;
        batch.sequence_shape = Some([shape[0], shape[1], w]);
        Ok(batch)
    }
}

impl NativeGeometryBatch {
    pub(crate) fn value_read_route(&self) -> Option<crate::geometry_artifact::AttentionValueRoute> {
        self.placement.recipe.value_gate().map(|(r, _)| r)
    }
    pub(crate) fn last_value_site_input(&self) -> Result<&[f32]> {
        match (&self.pullback, self.sequence_shape) {
            (Pullback::ValueRead(_, raw), Some([1, _, _])) => Ok(raw),
            _ => Err("expected a single active value gate sequence".into()),
        }
    }

    /// Adjoint of the separately supplied value source; vjp_input is the direct site adjoint.
    pub fn vjp_value_source(&self, upstream: &[f32]) -> Result<Tensor> {
        let d = self.direction(upstream)?;
        match &self.pullback {
            Pullback::ValueRead(b, _) => {
                let dx =
                    b.vjp_input(&d.scale_with_backend(self.strength, TensorUtilBackend::Cpu)?)?;
                if dx.data().iter().any(|x| !x.is_finite()) {
                    return Err("nonfinite value source adjoint".into());
                }
                Ok(dx)
            }
            Pullback::Identity if self.placement.recipe.value_gate().is_some() => {
                Ok(Tensor::zeros(d.shape().0, d.shape().1)?)
            }
            _ => Err("no external value source on this batch".into()),
        }
    }
}
