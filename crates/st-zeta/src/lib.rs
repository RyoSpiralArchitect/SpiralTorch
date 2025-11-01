//! Telemetry helpers for emitting SoftLogic feedback signals.

use anyhow::Result;
use st_softlogic::spiralk::ir::FeedbackBlock;
use tracing::event;
use tracing::Level;

#[derive(Clone, Debug)]
pub struct ZetaFeedback {
    run_id: String,
    export_path: String,
    metrics: Vec<String>,
}

impl ZetaFeedback {
    pub fn new(
        run_id: impl Into<String>,
        export_path: impl Into<String>,
        metrics: Vec<String>,
    ) -> Self {
        Self {
            run_id: run_id.into(),
            export_path: export_path.into(),
            metrics,
        }
    }

    pub fn from_block(run_id: impl Into<String>, block: &FeedbackBlock) -> Self {
        Self::new(run_id, block.export_path.clone(), block.metrics.clone())
    }

    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    pub fn export_path(&self) -> &str {
        &self.export_path
    }

    pub fn metrics(&self) -> &[String] {
        &self.metrics
    }

    pub fn emit_phase_deviation(&self, value: f32) {
        self.emit_numeric("phase_deviation", f64::from(value));
    }

    pub fn emit_collapse_resonance(&self, pattern: &str) {
        self.emit_pattern("collapse_resonance", pattern);
    }

    pub fn emit_kernel_cache_hits(&self, hits: u64) {
        self.emit_numeric("kernel_cache_hits", hits as f64);
    }

    pub fn emit_numeric(&self, metric_name: &str, value: f64) {
        event!(
            target: "spiraltorch::softlogic::zeta",
            Level::INFO,
            run = %self.run_id,
            export = %self.export_path,
            metric = %metric_name,
            value
        );
    }

    pub fn emit_pattern(&self, metric_name: &str, pattern: &str) {
        event!(
            target: "spiraltorch::softlogic::zeta",
            Level::INFO,
            run = %self.run_id,
            export = %self.export_path,
            metric = %metric_name,
            pattern = %pattern
        );
    }

    pub fn flush(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructs_from_feedback_block() {
        let block = FeedbackBlock {
            name: "fb".into(),
            export_path: "runs/1".into(),
            metrics: vec!["phase_deviation".into(), "collapse_resonance".into()],
        };
        let zeta = ZetaFeedback::from_block("session", &block);
        assert_eq!(zeta.run_id(), "session");
        assert_eq!(zeta.export_path(), "runs/1");
        assert_eq!(zeta.metrics().len(), 2);
    }
}
