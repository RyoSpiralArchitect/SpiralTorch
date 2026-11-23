//! Relative-lag transport with an orthogonal step and no length-squared attention matrix.
use super::*;
use crate::geometry_artifact::PrefixRotation;

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
}

impl TransportBatch {
    pub(super) fn forward(
        p: &GeometryAdapterPlacement,
        turn: PrefixRotation,
        x: &Tensor,
        [_, steps]: [usize; 2],
    ) -> Result<Self> {
        let gate = tensor(p, "gate");
        let bias = tensor(p, "bias");
        let (rows, w) = x.shape();
        let mut means = Vec::with_capacity(x.data().len());
        let mut values = Vec::with_capacity(x.data().len());
        for document in x.data().chunks_exact(steps * w) {
            TransportState::new(w).scan(document, turn, |i, m| {
                means.push(m);
                values.push(gate[i] * m + bias[i]);
            });
        }
        finite(&values)?;
        Ok(Self {
            output: Tensor::from_vec(rows, w, values)?,
            means,
            gate: gate.to_vec(),
            steps,
            turn,
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
                for i in 0..w {
                    adj[i] += f64::from(d[i]) * f64::from(self.gate[i]) / (t + 1) as f64;
                    row[i] = adj[i] as f32;
                }
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
            for i in 0..w {
                g[i] += f64::from(grad[i]) * f64::from(means[i]);
                b[i] += f64::from(grad[i]);
            }
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
        let GeometryAdapterRecipe::PrefixTransport {
            rotation, strength, ..
        } = p.recipe
        else {
            return Err("expected prefix transport geometry".into());
        };
        let mut state = previous.cloned().unwrap_or_else(|| TransportState::new(w));
        let strength = strength as f32;
        if strength == 0. {
            state.scan(x.data(), rotation, |_, _| {});
            return Ok((x, state));
        }
        let gate = tensor(p, "gate");
        let bias = tensor(p, "bias");
        let mut values = Vec::with_capacity(x.data().len());
        state.scan(x.data(), rotation, |i, m| {
            values.push(gate[i] * m + bias[i])
        });
        finite(&values)?;
        let branch = Tensor::from_vec(rows, w, values)?;
        let output = x.add_with_backend(
            &branch.scale_with_backend(strength, TensorUtilBackend::Cpu)?,
            TensorUtilBackend::Cpu,
        )?;
        finite(output.data())?;
        Ok((output, state))
    }
}
