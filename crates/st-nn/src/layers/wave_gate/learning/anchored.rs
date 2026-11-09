//! A projection chart with immutable f32 anchors and separately trainable f32 offsets.
use super::*;

#[derive(Debug)]
pub(crate) struct AnchoredProjectionBatch {
    input: Tensor,
    effective_gate: Vec<f64>,
    affine: Vec<f64>,
    row_gains: Vec<(f64, f64, f64)>,
    scale: f64,
    output: Tensor,
}

impl WaveGateKernel {
    pub(crate) fn forward_anchored_projection(
        &self,
        input: &Tensor,
        [gate, bias]: [&[f32]; 2],
        [dg, db]: [&[f32]; 2],
        log_radius: f32,
    ) -> PureResult<AnchoredProjectionBatch> {
        let (rows, cols) = input.shape();
        if cols == 0
            || cols > self.topos.max_volume()
            || [gate, bias, dg, db].iter().any(|p| p.len() != cols)
        {
            return Err(TensorError::InvalidDimensions { rows, cols });
        }
        self.topos
            .guard_tensor("anchored_projection_input", input)?;
        for s in [gate, bias, dg, db] {
            self.topos.guard_slice("anchored_projection_channels", s)?;
        }
        let radius = ProjectionRadius::new(log_radius)?;
        let scale = f64::from((-self.topos.curvature()).sqrt());
        let effective_gate = gate
            .iter()
            .zip(dg)
            .map(|(&a, &d)| {
                if d == 0. {
                    f64::from(a)
                } else {
                    f64::from(a) + f64::from(d)
                }
            })
            .collect();
        let mut affine = Vec::with_capacity(input.data().len());
        let mut values = Vec::with_capacity(input.data().len());
        let mut row_gains = Vec::with_capacity(rows);
        for row in input.data().chunks_exact(cols) {
            let start = affine.len();
            for (c, &x) in row.iter().enumerate() {
                // Preserve legacy multiply/add rounding at the origin. Never collapse
                // the offset back into a full f32 parameter before it acts on x.
                let anchor = f64::from(x * gate[c] + bias[c]);
                let low = f64::from(x) * f64::from(dg[c]) + f64::from(db[c]);
                affine.push(if low == 0. { anchor } else { anchor + low });
            }
            let z = &affine[start..];
            let norm = z.iter().map(|v| v * v).sum::<f64>().sqrt();
            if !norm.is_finite() {
                return Err(TensorError::InvalidValue {
                    label: "anchored_projection_affine",
                });
            }
            let (_, tangent, radial, _) = radius.row_gains(norm, scale);
            row_gains.push((norm, tangent, radial));
            values.extend(z.iter().map(|&v| (v * tangent / scale) as f32));
        }
        let output = Tensor::from_vec(rows, cols, values)?;
        self.topos
            .guard_tensor("anchored_projection_output", &output)?;
        Ok(AnchoredProjectionBatch {
            input: input.clone(),
            effective_gate,
            affine,
            row_gains,
            scale,
            output,
        })
    }
}

impl AnchoredProjectionBatch {
    pub(crate) fn output(&self) -> &Tensor {
        &self.output
    }

    fn visit(&self, seed: &Tensor, mut f: impl FnMut(usize, f64, f64)) -> PureResult<()> {
        if self.input.shape() != seed.shape() {
            return Err(TensorError::ShapeMismatch {
                left: self.input.shape(),
                right: seed.shape(),
            });
        }
        let (_, cols) = seed.shape();
        for (r, dy) in seed.data().chunks_exact(cols).enumerate() {
            let (norm, tangent, radial) = self.row_gains[r];
            let z = &self.affine[r * cols..(r + 1) * cols];
            let dot = if norm == 0. {
                0.
            } else {
                z.iter()
                    .zip(dy)
                    .map(|(&v, &u)| v / norm * f64::from(u))
                    .sum::<f64>()
            };
            for c in 0..cols {
                let aligned = if norm == 0. { 0. } else { z[c] / norm * dot };
                let da = (tangent * (f64::from(dy[c]) - aligned) + radial * aligned) / self.scale;
                f(c, da, f64::from(self.input.data()[r * cols + c]));
            }
        }
        Ok(())
    }

    fn checked(rows: usize, cols: usize, data: Vec<f32>) -> PureResult<Tensor> {
        if data.iter().any(|v| !v.is_finite()) {
            return Err(TensorError::InvalidValue {
                label: "anchored_projection_gradient",
            });
        }
        Tensor::from_vec(rows, cols, data)
    }

    pub(crate) fn vjp_input(&self, seed: &Tensor) -> PureResult<Tensor> {
        let (r, c) = self.input.shape();
        let mut dx = Vec::with_capacity(self.input.data().len());
        self.visit(seed, |i, da, _| {
            dx.push((da * self.effective_gate[i]) as f32)
        })?;
        Self::checked(r, c, dx)
    }

    pub(crate) fn vjp(
        &self,
        seed: &Tensor,
        need_input: bool,
    ) -> PureResult<(Option<Tensor>, Tensor, Tensor)> {
        let (rows, cols) = self.input.shape();
        let mut dx = need_input.then(|| Vec::with_capacity(self.input.data().len()));
        let mut dg = vec![0f64; cols];
        let mut db = vec![0f64; cols];
        self.visit(seed, |c, da, x| {
            if let Some(values) = dx.as_mut() {
                values.push((da * self.effective_gate[c]) as f32);
            }
            dg[c] += da * x;
            db[c] += da;
        })?;
        Ok((
            dx.map(|v| Self::checked(rows, cols, v)).transpose()?,
            Self::checked(1, cols, dg.into_iter().map(|v| v as f32).collect())?,
            Self::checked(1, cols, db.into_iter().map(|v| v as f32).collect())?,
        ))
    }
}
