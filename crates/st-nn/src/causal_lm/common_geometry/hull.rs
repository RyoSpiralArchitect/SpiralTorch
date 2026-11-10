//! Nearest point to zero in a small convex hull of normalized gradients.
//! Only face selection uses the Gram matrix; certify the result in the actual coordinates.
use super::*;

pub(super) fn sum(values: impl Iterator<Item = f64>) -> f64 {
    let mut total = super::super::sum::Sum::default();
    for v in values {
        total.add(v);
    }
    total.value()
}
pub(in crate::causal_lm) fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    sum(a.iter().zip(b).map(|(a, b)| a * b))
}
pub(in crate::causal_lm) fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

fn face(gram: &[Vec<f64>], ids: &[usize]) -> Option<Vec<f64>> {
    let (&anchor, axes) = ids.split_last()?;
    let n = axes.len();
    let mut w = vec![0.; gram.len()];
    if n == 0 {
        w[anchor] = 1.;
        return Some(w);
    }
    // Orthogonal projection onto the affine span, in differences from the anchor.
    let mut a = [[0.; MAX_COMMON_GEOMETRY_GROUPS]; MAX_COMMON_GEOMETRY_GROUPS];
    let mut scale = 0f64;
    for (r, &i) in axes.iter().enumerate() {
        for (c, &j) in axes.iter().enumerate() {
            a[r][c] = (gram[i][j] - gram[i][anchor]) - (gram[anchor][j] - gram[anchor][anchor]);
            scale = scale.max(a[r][c].abs());
        }
        a[r][n] = gram[anchor][anchor] - gram[i][anchor];
    }
    for k in 0..n {
        let p = (k..n)
            .max_by(|&i, &j| a[i][k].abs().total_cmp(&a[j][k].abs()))
            .unwrap();
        if a[p][k].abs() <= 64. * f64::EPSILON * scale {
            return None;
        }
        a.swap(k, p);
        let pivot_row = a[k];
        for row in a.iter_mut().take(n).skip(k + 1) {
            let f = row[k] / pivot_row[k];
            for (value, pivot) in row[k + 1..=n].iter_mut().zip(&pivot_row[k + 1..=n]) {
                *value -= f * pivot;
            }
        }
    }
    for k in (0..n).rev() {
        w[axes[k]] = (a[k][n] - sum((k + 1..n).map(|j| a[k][j] * w[axes[j]]))) / a[k][k];
    }
    w[anchor] = 1. - sum(w.iter().copied());
    if w.iter().any(|x| !x.is_finite() || *x < -1e-12) {
        return None;
    }
    w.iter_mut().for_each(|v| *v = v.max(0.));
    let z = sum(w.iter().copied());
    w.iter_mut().for_each(|v| *v /= z);
    Some(w)
}

pub(in crate::causal_lm) struct Hull {
    pub norms: Vec<f64>,
    pub alpha: Vec<f64>,
    pub beta: Vec<f64>,
    pub vector: Vec<f64>,
    pub hull_norm: f64,
    pub margin: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CommonDirectionRefusal {
    ZeroMean,
    Unresolved,
    Uncertified,
}

impl std::fmt::Display for CommonDirectionRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::ZeroMean => "common descent requires nonzero finite group means",
            Self::Unresolved => "no numerically resolved strict common geometry direction",
            Self::Uncertified => {
                "common geometry direction failed its normalized margin certificate"
            }
        })
    }
}
impl std::error::Error for CommonDirectionRefusal {}

pub(in crate::causal_lm) fn solve(means: &[Vec<f64>]) -> CausalLmResult<Hull> {
    let n = means.len();
    if !(2..=MAX_COMMON_GEOMETRY_GROUPS).contains(&n)
        || means[0].is_empty()
        || means.iter().any(|g| g.len() != means[0].len())
    {
        return Err("invalid group gradient domain".into());
    }
    let norms: Vec<_> = means.iter().map(|g| norm(g)).collect();
    if norms.iter().any(|v| !v.is_finite()) {
        return Err("common descent requires nonzero finite group means".into());
    }
    if norms.contains(&0.) {
        return Err(CommonDirectionRefusal::ZeroMean.into());
    }
    let mut gram = vec![vec![0.; n]; n];
    for i in 0..n {
        for j in 0..=i {
            gram[i][j] = dot(&means[i], &means[j]) / (norms[i] * norms[j]);
            gram[j][i] = gram[i][j];
        }
    }
    let mut best = f64::INFINITY;
    let mut alpha = vec![0.; n];
    // A closest point has an affinely independent supporting face. Singular
    // faces can be skipped because all of their lower-dimensional faces are tried.
    for mask in 1..(1usize << n) {
        let face_ids: Vec<_> = (0..n).filter(|&i| mask & (1 << i) != 0).collect();
        if let Some(w) = face(&gram, &face_ids) {
            let square = sum((0..n).map(|i| w[i] * dot(&gram[i], &w)));
            if square < best {
                best = square;
                alpha = w;
            }
        }
    }
    let v: Vec<_> = (0..means[0].len())
        .map(|k| sum((0..n).map(|i| alpha[i] * (means[i][k] / norms[i]))))
        .collect();
    let hull_norm = norm(&v);
    if !hull_norm.is_finite() {
        return Err("nonfinite common geometry hull".into());
    }
    if hull_norm <= 1e-10 {
        return Err(CommonDirectionRefusal::Unresolved.into());
    }
    let margin = (0..n)
        .map(|i| dot(&means[i], &v) / (norms[i] * hull_norm))
        .fold(f64::INFINITY, f64::min);
    // This point supplies an upper bound; the worst group cosine of the chosen
    // direction supplies a lower bound on the optimal normalized margin.
    if !margin.is_finite() {
        return Err("nonfinite common geometry margin".into());
    }
    if margin <= 1e-10 || (hull_norm - margin).abs() > 1e-9 {
        return Err(CommonDirectionRefusal::Uncertified.into());
    }
    let mut beta: Vec<_> = (0..n).map(|i| alpha[i] / norms[i]).collect();
    let scale = sum(beta.iter().copied());
    beta.iter_mut().for_each(|b| *b /= scale);
    Ok(Hull {
        norms,
        alpha,
        beta,
        vector: v.into_iter().map(|x| x / scale).collect(),
        hull_norm,
        margin,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orthogonal_groups_use_unit_geometry_not_gradient_magnitude() {
        for n in 2..=8 {
            let v: Vec<_> = (0..n)
                .map(|i| {
                    let mut a = vec![0.; n];
                    a[i] = 2f64.powi(i as i32);
                    a
                })
                .collect();
            let s = solve(&v).unwrap();
            assert!((s.margin - 1. / (n as f64).sqrt()).abs() < 1e-12);
            assert!(s.alpha.iter().all(|x| (*x - 1. / n as f64).abs() < 1e-12));
            let z = sum(s.norms.iter().map(|x| 1. / x));
            for i in 0..n {
                assert!((s.beta[i] - 1. / (s.norms[i] * z)).abs() < 1e-12);
            }
        }
    }
    #[test]
    fn active_edge_permutation_duplicates_and_nearly_opposite_residual() {
        let a = vec![1., 0.];
        let b = vec![0., 1.];
        let c = vec![1., 1.];
        for v in [
            vec![a.clone(), b.clone(), c.clone()],
            vec![c, b.clone(), a.clone(), a],
        ] {
            let s = solve(&v).unwrap();
            assert!((s.margin - 0.5f64.sqrt()).abs() < 1e-12);
            assert!((s.vector[0] - s.vector[1]).abs() < 1e-12);
        }
        let s = solve(&[vec![1., 1e-7], vec![-1., 1e-7], b]).unwrap();
        assert!((s.hull_norm / 1e-7 - 1.).abs() < 1e-5 && s.margin > 9e-8);
    }
    #[test]
    fn infeasible_and_undefined_geometries_are_refused() {
        assert!(solve(&[vec![1., 0.], vec![-1., 0.], vec![0., 1.]]).is_err());
        assert!(solve(&[vec![1., 0.], vec![0., 1.], vec![-1., -1.]]).is_err());
        assert!(solve(&[vec![0., 0.], vec![0., 1.]]).is_err());
        assert!(solve(&[vec![1.], vec![f64::NAN]]).is_err());
    }
}
