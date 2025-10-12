use ndarray::{ArrayD, IxDyn};

/// In-place-free SGD: w_next = w - lr * g
pub fn sgd_update(w:&ArrayD<f32>, g:&ArrayD<f32>, lr:&ArrayD<f32>) -> ArrayD<f32> {
    let lw = if lr.shape().len()==0 { lr.clone().into_shape(IxDyn(w.shape())).unwrap() } else { lr.clone() };
    w - &(lw * g)
}

/// One-step unrolled hypergradient for SGD (first-order, no Hessian terms):
/// d L_val(w - lr*g_tr) / d lr  ≈  -⟨∇_w L_val(w'), g_tr⟩
pub fn hypergrad_sgd_one_step_dot(grad_train:&ArrayD<f32>, grad_val:&ArrayD<f32>) -> f32 {
    grad_train.iter().zip(grad_val.iter()).map(|(a,b)| -a*b).sum()
}

/// Implicit hypergradient surrogate: h ≈ -⟨g_val, g_tr⟩ / (1+λ)
pub fn hypergrad_sgd_implicit_surrogate(grad_train:&ArrayD<f32>, grad_val:&ArrayD<f32>, damping:f32) -> f32 {
    let dot: f32 = grad_train.iter().zip(grad_val.iter()).map(|(a,b)| a*b).sum();
    -dot / (1.0 + damping.max(0.0))
}
