use ndarray::{ArrayD, IxDyn};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

pub struct HardConcreteGate {
    pub log_alpha: ArrayD<f32>,
    pub temp: f32,
    pub seed: u64,
}

impl HardConcreteGate {
    pub fn forward(&self, shape:&[usize]) -> ArrayD<f32> {
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut g = ArrayD::<f32>::zeros(IxDyn(shape));
        let la = &self.log_alpha;
        let t = self.temp;
        for ((_, gg), a) in g.indexed_iter_mut().zip(la.iter()) {
            let mut u = rng.gen::<f32>();
            if u<=0.0 { u = 1e-6 } else if u>=1.0 { u = 1.0-1e-6 }
            let s = ((*a + u.ln() - (1.0-u).ln()) / t).sigmoid();
            *gg = s.clamp(0.0, 1.0);
        }
        g
    }
}

/// Approximate L0 surrogate: mean(sigmoid(log_alpha)).
pub fn l0_surrogate(log_alpha:&ArrayD<f32>) -> f32 {
    log_alpha.iter().map(|a| 1.0/(1.0+(-*a).exp()) ).sum::<f32>() / (log_alpha.len() as f32)
}

/// Integerize: threshold at tau (e.g., 0.5).
pub fn integerize_count(gates:&ArrayD<f32>, tau:f32) -> usize {
    gates.iter().filter(|&&g| g>=tau).count()
}

trait Sig { fn sigmoid(self)->f32; }
impl Sig for f32 { fn sigmoid(self)->f32 { 1.0/(1.0+(-self).exp()) } }
