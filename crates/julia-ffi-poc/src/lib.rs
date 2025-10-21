//! PoC crate for embedding Julia heuristics into Rust.
use anyhow::Result;

/// Pure Rust reference implementation that mirrors the Julia function used in the PoC.
/// It applies a cheap latency-score heuristic over the tile size and slack width.
pub fn rust_latency_score(tile: u32, slack: u32) -> f64 {
    let tile_term = (tile as f64).sqrt();
    let slack_term = (slack as f64) / 2.0;
    tile_term + slack_term
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
}
