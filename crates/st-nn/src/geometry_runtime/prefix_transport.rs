//! Relative-lag transport with an orthogonal step and no length-squared attention matrix.
use super::*;
use crate::geometry_artifact::{PrefixChannelMix, PrefixRotation, PrefixTransportSpec};

impl PrefixChannelMix {
    fn transpose(self) -> Self {
        match self {
            Self::CayleyTriples => Self::InverseCayleyTriples,
            Self::InverseCayleyTriples => Self::CayleyTriples,
        }
    }

    fn triple(self, [a, b, c]: [f64; 3]) -> [f64; 3] {
        // Each row sums to 19; row squared lengths are 361 and cross products vanish.
        let (l, r) = match self {
            Self::CayleyTriples => (-6., 10.),
            Self::InverseCayleyTriples => (10., -6.),
        };
        [
            (15. * a + l * b + r * c) / 19.,
            (r * a + 15. * b + l * c) / 19.,
            (l * a + r * b + 15. * c) / 19.,
        ]
    }
}

fn finish_branch(values: &mut [f32], w: usize, mix: Option<PrefixChannelMix>) -> Result<()> {
    if let Some(m) = mix {
        for row in values.chunks_exact_mut(w) {
            for p in row.chunks_exact_mut(3) {
                let out = m.triple([p[0].into(), p[1].into(), p[2].into()]);
                for (slot, val) in p.iter_mut().zip(out) {
                    *slot = val as f32;
                }
            }
        }
    }
    finite(values)
}

fn branch_adjoint(row: &[f32], mix: Option<PrefixChannelMix>, mut add: impl FnMut(usize, f64)) {
    let mut j = 0;
    if let Some(m) = mix {
        for p in row.chunks_exact(3) {
            for val in m
                .transpose()
                .triple([p[0].into(), p[1].into(), p[2].into()])
            {
                add(j, val);
                j += 1;
            }
        }
    }
    for (i, &v) in row[j..].iter().enumerate() {
        add(j + i, v.into());
    }
}

impl PrefixRotation {
    fn transpose(self) -> Self {
        match self {
            Self::Identity => Self::Identity,
            Self::QuarterTurn => Self::InverseQuarterTurn,
            Self::InverseQuarterTurn => Self::QuarterTurn,
        }
    }

    pub(crate) fn apply(self, values: &mut [f64]) {
        if self != Self::Identity {
            for p in values.chunks_exact_mut(2) {
                (p[0], p[1]) = if self == Self::QuarterTurn {
                    (-p[1], p[0])
                } else {
                    (p[1], -p[0])
                };
            }
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct TransportState {
    steps: usize,
    sum: Vec<f64>,
}

impl TransportState {
    fn new(width: usize) -> Self {
        Self {
            steps: 0,
            sum: vec![0.; width],
        }
    }

    pub(crate) fn cache_values(&self) -> usize {
        2 * self.sum.len() // cache admission counts f32-sized payload words
    }

    fn scan(&mut self, rows: &[f32], turn: PrefixRotation, mut emit: impl FnMut(usize, f32)) {
        let width = self.sum.len();
        for row in rows.chunks_exact(width) {
            turn.apply(&mut self.sum);
            self.steps += 1;
            for (i, (s, &x)) in self.sum.iter_mut().zip(row).enumerate() {
                *s += f64::from(x);
                emit(i, (*s / self.steps as f64) as f32);
            }
        }
    }
}

fn finite(v: &[f32]) -> Result<()> {
    if v.iter().all(|x| x.is_finite()) {
        Ok(())
    } else {
        Err("non-finite prefix transport result".into())
    }
}

#[derive(Debug)]
pub(super) struct TransportBatch {
    pub(super) output: Tensor,
    means: Vec<f32>,
    gate: Vec<f32>,
    steps: usize,
    turn: PrefixRotation,
    output_mix: Option<PrefixChannelMix>,
}

impl NativeGeometryBatch {
    pub(crate) fn single_prefix_last_mean(&self) -> Result<&[f32]> {
        match (&self.pullback, self.sequence_shape) {
            (Pullback::Transport(b), Some([1, _, w])) => Ok(&b.means[b.means.len() - w..]),
            _ => Err("expected one active prefix transport sequence".into()),
        }
    }
}

impl TransportBatch {
    pub(super) fn forward(
        p: &GeometryAdapterPlacement,
        spec: PrefixTransportSpec,
        x: &Tensor,
        [_, steps]: [usize; 2],
    ) -> Result<Self> {
        let gate = tensor(p, "gate");
        let bias = tensor(p, "bias");
        let (rows, w) = x.shape();
        let mut means = Vec::with_capacity(x.data().len());
        let mut values = Vec::with_capacity(x.data().len());
        for document in x.data().chunks_exact(steps * w) {
            TransportState::new(w).scan(document, spec.rotation, |i, m| {
                means.push(m);
                values.push(gate[i] * m + bias[i]);
            });
        }
        finish_branch(&mut values, w, spec.output_mix)?;
        Ok(Self {
            output: Tensor::from_vec(rows, w, values)?,
            means,
            gate: gate.to_vec(),
            steps,
            turn: spec.rotation,
            output_mix: spec.output_mix,
        })
    }

    pub(super) fn input_vjp(&self, dy: &Tensor) -> Result<Tensor> {
        let (rows, w) = dy.shape();
        let mut dx = vec![0.; rows * w];
        let mut adj = vec![0.; w];
        for (src, dst) in dy
            .data()
            .chunks_exact(self.steps * w)
            .zip(dx.chunks_exact_mut(self.steps * w))
        {
            adj.fill(0.);
            // Each S[t] has two uses: its own mean and R*S[t] in the next state.
            for (t, (d, row)) in src
                .chunks_exact(w)
                .zip(dst.chunks_exact_mut(w))
                .enumerate()
                .rev()
            {
                self.turn.transpose().apply(&mut adj);
                branch_adjoint(d, self.output_mix, |i, v| {
                    adj[i] += v * f64::from(self.gate[i]) / (t + 1) as f64;
                    row[i] = adj[i] as f32;
                });
            }
        }
        finite(&dx)?;
        Ok(Tensor::from_vec(rows, w, dx)?)
    }

    pub(super) fn vjp(&self, dy: &Tensor, need_input: bool) -> Result<NativeGeometryPullback> {
        let w = self.gate.len();
        let mut g = vec![0f64; w];
        let mut b = vec![0f64; w];
        for (grad, means) in dy.data().chunks_exact(w).zip(self.means.chunks_exact(w)) {
            branch_adjoint(grad, self.output_mix, |i, v| {
                g[i] += v * f64::from(means[i]);
                b[i] += v;
            });
        }
        let mut parameters = BTreeMap::new();
        for (name, v) in [("gate", g), ("bias", b)] {
            let v: Vec<f32> = v.into_iter().map(|x| x as f32).collect();
            finite(&v)?;
            parameters.insert(name.to_owned(), v);
        }
        Ok(NativeGeometryPullback {
            input: need_input.then(|| self.input_vjp(dy)).transpose()?,
            parameters,
        })
    }
}

impl NativeGeometryRuntime {
    /// Transactional prefill/append. Full sequence admission is unchanged; no old rows are replayed.
    pub(crate) fn extend_transport(
        &self,
        path: &str,
        x: Tensor,
        previous: Option<&TransportState>,
        steps: usize,
    ) -> Result<(Tensor, TransportState)> {
        let (width, _) = self.sequence_dimensions(path, [1, steps])?;
        let (rows, w) = x.shape();
        let start = steps
            .checked_sub(rows)
            .ok_or("invalid transport suffix length")?;
        if w != width
            || rows == 0
            || previous.map_or(0, |s| s.steps) != start
            || previous.is_some_and(|s| s.sum.len() != w)
        {
            return Err("prefix transport state/shape mismatch".into());
        }
        finite(x.data())?;
        let p = self.placement(path)?;
        let spec = p
            .recipe
            .prefix_transport()
            .ok_or("expected prefix transport geometry")?;
        let mut state = previous.cloned().unwrap_or_else(|| TransportState::new(w));
        let strength = spec.strength;
        if strength == 0. {
            state.scan(x.data(), spec.rotation, |_, _| {});
            return Ok((x, state));
        }
        let gate = tensor(p, "gate");
        let bias = tensor(p, "bias");
        let mut values = Vec::with_capacity(x.data().len());
        state.scan(x.data(), spec.rotation, |i, m| {
            values.push(gate[i] * m + bias[i])
        });
        finish_branch(&mut values, w, spec.output_mix)?;
        let branch = Tensor::from_vec(rows, w, values)?;
        let output = x.add_with_backend(
            &branch.scale_with_backend(strength, TensorUtilBackend::Cpu)?,
            TensorUtilBackend::Cpu,
        )?;
        finite(output.data())?;
        Ok((output, state))
    }
}
