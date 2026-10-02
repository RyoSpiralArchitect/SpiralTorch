use super::{finite, score, validate_inputs, AttentionError, AttentionMask, AttentionSpec};

/// Gradients have the exact input shapes; absent biases have no gradient buffer.
#[derive(Clone, Debug)]
pub struct AttentionGradients {
    pub query: Vec<f32>,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
    pub z_bias: Option<Vec<f32>>,
    pub pair_bias: Option<Vec<f32>>,
}

/// First-order VJP of the same scaled, biased, structurally masked softmax map
/// as `attention_reference`. Scores retain its checked f32 arithmetic; softmax
/// normalization and gradient accumulation use f64 before checked f32 output.
/// This is an algebraic derivative, not differentiation of floating-point
/// rounding. It uses O(K) row scratch, no retained quadratic probability matrix.
pub fn attention_vjp_reference(
    spec: AttentionSpec,
    query: &[f32],
    key: &[f32],
    value: &[f32],
    z_bias: Option<&[f32]>,
    pair_bias: Option<&[f32]>,
    upstream: &[f32],
) -> Result<AttentionGradients, AttentionError> {
    validate_inputs(spec, query, key, value, z_bias, pair_bias)?;
    if upstream.len() != query.len() {
        return Err(AttentionError::Length);
    }
    if upstream.iter().any(|v| !v.is_finite()) {
        return Err(AttentionError::NonFinite);
    }
    let mut dq = vec![0.0_f64; query.len()];
    let mut dk = vec![0.0_f64; key.len()];
    let mut dv = vec![0.0_f64; value.len()];
    let mut dz = z_bias.map(|v| vec![0.0_f64; v.len()]);
    let mut dpair = pair_bias.map(|v| vec![0.0_f64; v.len()]);
    let scratch = if query.is_empty() { 0 } else { spec.keys() };
    let mut probabilities = vec![0.0_f64; scratch];
    let mut dp = vec![0.0_f64; scratch];
    let d = spec.head_dim();
    for context in 0..spec.contexts() {
        for q in 0..spec.queries() {
            let row = context * spec.queries() + q;
            let visible = match spec.mask() {
                AttentionMask::None => spec.keys(),
                AttentionMask::Causal { query_offset } => query_offset + q + 1,
            };
            let mut maximum = f64::NEG_INFINITY;
            for (k, p) in probabilities[..visible].iter_mut().enumerate() {
                let key_row = context * spec.keys() + k;
                *p = f64::from(score(
                    spec,
                    query,
                    key,
                    (z_bias, pair_bias),
                    row,
                    key_row,
                    k,
                )?);
                maximum = maximum.max(*p);
            }
            let mut sum = 0.0;
            for p in &mut probabilities[..visible] {
                *p = (*p - maximum).exp();
                sum += *p;
            }
            let mut center = 0.0;
            for k in 0..visible {
                probabilities[k] /= sum;
                let key_row = context * spec.keys() + k;
                dp[k] = (0..d)
                    .map(|i| f64::from(upstream[row * d + i]) * f64::from(value[key_row * d + i]))
                    .sum();
                center += probabilities[k] * dp[k];
            }
            for k in 0..visible {
                let key_row = context * spec.keys() + k;
                let ds = probabilities[k] * (dp[k] - center);
                for i in 0..d {
                    dq[row * d + i] +=
                        f64::from(spec.scale()) * ds * f64::from(key[key_row * d + i]);
                    dk[key_row * d + i] +=
                        f64::from(spec.scale()) * ds * f64::from(query[row * d + i]);
                    dv[key_row * d + i] += probabilities[k] * f64::from(upstream[row * d + i]);
                }
                if let Some(values) = &mut dz {
                    values[key_row] += ds;
                }
                if let Some(values) = &mut dpair {
                    values[row * spec.keys() + k] = ds;
                }
            }
        }
    }
    fn narrow(values: Vec<f64>) -> Result<Vec<f32>, AttentionError> {
        values.into_iter().map(|v| finite(v as f32)).collect()
    }
    Ok(AttentionGradients {
        query: narrow(dq)?,
        key: narrow(dk)?,
        value: narrow(dv)?,
        z_bias: dz.map(narrow).transpose()?,
        pair_bias: dpair.map(narrow).transpose()?,
    })
}
