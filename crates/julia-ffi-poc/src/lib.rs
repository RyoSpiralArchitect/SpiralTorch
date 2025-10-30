//! PoC crate for embedding Julia heuristics into Rust.
use anyhow::Result;

/// Pure Rust reference implementation that mirrors the Julia function used in the PoC.
/// It applies a cheap latency-score heuristic over the tile size and slack width.
pub fn rust_latency_score(tile: u32, slack: u32) -> f64 {
    let tile_term = (tile as f64).sqrt();
    let slack_term = (slack as f64) / 2.0;
    tile_term + slack_term
}

#[derive(Clone, Debug)]
pub struct ZTigerOptim {
    curvature: f64,
    gain: f64,
    history: Vec<f64>,
}

impl ZTigerOptim {
    pub fn new(curvature: f64) -> Self {
        let curvature = if curvature.is_finite() && curvature < 0.0 {
            curvature
        } else {
            -1.0
        };
        Self {
            curvature,
            gain: 1.0,
            history: Vec::new(),
        }
    }

    pub fn curvature(&self) -> f64 {
        self.curvature
    }

    pub fn gain(&self) -> f64 {
        self.gain
    }

    pub fn update(&mut self, lora_pid: f64, resonance: &[f64]) -> f64 {
        let hyper_boost = self.curvature.tanh().abs().max(1e-3);
        let pid = lora_pid.max(1e-6);
        let mut total = 0.0;
        for (idx, value) in resonance.iter().enumerate() {
            let phase = (idx as f64 + 1.0) / resonance.len().max(1) as f64;
            total += value.abs() * phase;
        }
        let mean = if resonance.is_empty() {
            0.0
        } else {
            total / resonance.len() as f64
        };
        self.history.push(mean);
        if self.history.len() > 32 {
            self.history.remove(0);
        }
        let smooth = if self.history.is_empty() {
            mean
        } else {
            self.history.iter().copied().sum::<f64>() / self.history.len() as f64
        };
        self.gain = (smooth * hyper_boost / pid).tanh().abs() + 1.0;
        self.gain
    }
}

#[cfg(feature = "with-julia")]
mod julia_impl {
    use super::*;
    use anyhow::anyhow;
    use jlrs::prelude::*;

    const JULIA_POC_MODULE: &str = r#"
module SpiralTempo
export tempo_score

function tempo_score(tile::UInt32, slack::UInt32)
    sqrt(float(tile)) + float(slack) / 2
end

end
"#;

    pub fn tempo_latency_score(tile: u32, slack: u32) -> Result<f64> {
        let mut julia = Builder::new()
            .start()
            .map_err(|err| anyhow!("failed to start Julia runtime: {err}"))?;

        let score = julia
            .scope(|mut frame| {
                Value::eval_string(&mut frame, JULIA_POC_MODULE)?;
                let module = Module::main(&frame).submodule_ref("SpiralTempo")?;
                let func = module.function("tempo_score")?;
                let tile_val = Value::new(&mut frame, tile);
                let slack_val = Value::new(&mut frame, slack);
                let result = func.call2(&mut frame, tile_val, slack_val)?;
                result.unbox::<f64>()
            })
            .map_err(|err| anyhow!("failed to execute Julia scope: {err}"))?;

        Ok(score)
    }
}

#[cfg(not(feature = "with-julia"))]
mod julia_impl {
    use super::*;

    pub fn tempo_latency_score(tile: u32, slack: u32) -> Result<f64> {
        Ok(rust_latency_score(tile, slack))
    }
}

pub use julia_impl::tempo_latency_score;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_reference_matches_stubbed_julia() {
        let tile = 256;
        let slack = 128;
        let rust_score = rust_latency_score(tile, slack);
        let julia_score = tempo_latency_score(tile, slack).expect("score should be available");
        assert!((rust_score - julia_score).abs() < 1e-6);
    }

    #[test]
    fn tiger_optim_updates_gain() {
        let mut optim = ZTigerOptim::new(-1.2);
        let gain = optim.update(0.5, &[0.2, 0.4, 0.8]);
        assert!(gain > 1.0);
        assert!(optim.gain() >= gain);
    }
}
