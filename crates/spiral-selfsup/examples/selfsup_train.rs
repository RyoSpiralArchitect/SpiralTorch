use std::error::Error;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rand::{rngs::StdRng, Rng, SeedableRng};
use spiral_selfsup::contrastive::{info_nce_loss, InfoNCEResult};
use spiral_selfsup::metrics::{register_info_nce_descriptors, InfoNCEEpochMetrics};
use st_core::runtime::blackcat::StepMetrics;
use st_core::telemetry::chrono::{ChronoSummary, ChronoTimeline, ResonanceTemporalMetrics};
use st_core::telemetry::monitoring::{MetricsExporter, MonitoringHub};
use tensorboard_rs::summary_writer::SummaryWriter;

fn main() -> Result<(), Box<dyn Error>> {
    register_info_nce_descriptors();

    let logdir = std::env::var("SELF_SUP_LOGDIR").unwrap_or_else(|_| "runs/selfsup".into());
    std::fs::create_dir_all(&logdir)?;

    let exporter = Arc::new(TensorboardExporter::new(&logdir)?);
    let mut monitoring = MonitoringHub::default();
    monitoring.register_exporter(exporter.clone());

    let mut rng = StdRng::seed_from_u64(42);
    let mut timeline = ChronoTimeline::with_capacity(128);

    for step in 0..50 {
        let anchors = random_embeddings(&mut rng, 64, 128);
        let positives = random_embeddings(&mut rng, 64, 128);
        let result = info_nce_loss(&anchors, &positives, 0.2, true)?;
        let info_summary = InfoNCEMetricSummary::from_result(&result);
        let epoch_metrics = InfoNCEEpochMetrics {
            mean_loss: result.loss,
            batches: result.batch,
        };

        if let Some(chrono_summary) = record_timeline(&mut timeline, &mut rng) {
            exporter.log_chrono(step, &chrono_summary);
        }
        exporter.log_info_nce(step, &info_summary);

        let mut step_metrics = StepMetrics::default();
        step_metrics.step_time_ms = rng.gen_range(8.0..18.0);
        step_metrics.mem_peak_mb = rng.gen_range(256.0..640.0);
        step_metrics.retry_rate = rng.gen_range(0.0..0.05);
        for value in epoch_metrics.to_values() {
            step_metrics
                .extra
                .insert(value.name.to_string(), f64::from(value.value));
        }

        let reward = 1.0 - f64::from(info_summary.loss);
        let alerts = monitoring.observe(&step_metrics, reward);
        if !alerts.is_empty() {
            println!("alerts@step {step}: {alerts:?}");
        }

        std::thread::sleep(Duration::from_millis(10));
    }

    exporter.flush();
    println!("TensorBoard logs written to {logdir}");
    Ok(())
}

fn random_embeddings(rng: &mut StdRng, batch: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..batch)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect()
}

fn record_timeline(timeline: &mut ChronoTimeline, rng: &mut StdRng) -> Option<ChronoSummary> {
    let total_energy = rng.gen_range(0.6..3.2);
    let raw = [
        rng.gen_range(0.0..1.0),
        rng.gen_range(0.0..1.0),
        rng.gen_range(0.0..1.0),
        rng.gen_range(0.0..1.0),
        rng.gen_range(0.0..1.0),
    ];
    let sum = raw.iter().copied().sum::<f32>().max(1e-3);
    let scaled: Vec<f32> = raw.iter().map(|v| v / sum * total_energy).collect();
    let metrics = ResonanceTemporalMetrics {
        observed_curvature: rng.gen_range(-1.0..1.0),
        total_energy,
        homotopy_energy: scaled[0],
        functor_energy: scaled[1],
        recursive_energy: scaled[2],
        projection_energy: scaled[3],
        infinity_energy: scaled[4],
    }
    .sanitise();

    timeline.record(0.5, metrics);
    timeline.summarise(32)
}

#[derive(Debug, Clone, Copy)]
struct InfoNCEMetricSummary {
    loss: f32,
    top1_accuracy: f32,
    mean_positive_margin: f32,
    mean_positive_log_probability: f32,
}

impl InfoNCEMetricSummary {
    fn from_result(result: &InfoNCEResult) -> Self {
        let batch = result.batch;
        if batch == 0 {
            return Self {
                loss: result.loss,
                top1_accuracy: 0.0,
                mean_positive_margin: 0.0,
                mean_positive_log_probability: 0.0,
            };
        }

        let mut correct = 0usize;
        let mut margin_sum = 0.0f32;
        let mut log_prob_sum = 0.0f32;

        for (i, row) in result.logits.chunks(batch).enumerate() {
            if row.is_empty() {
                continue;
            }

            let positive_logit = row[i];
            let mut max_idx = 0usize;
            let mut max_value = row[0];
            for (idx, &value) in row.iter().enumerate() {
                if value > max_value {
                    max_idx = idx;
                    max_value = value;
                }
            }
            if max_idx == i {
                correct += 1;
            }

            if batch > 1 {
                let mut max_negative = f32::NEG_INFINITY;
                for (idx, &value) in row.iter().enumerate() {
                    if idx != i && value > max_negative {
                        max_negative = value;
                    }
                }
                if max_negative.is_finite() {
                    margin_sum += positive_logit - max_negative;
                }
            }

            let max_logit = row.iter().fold(f32::NEG_INFINITY, |acc, &v| acc.max(v));
            let exp_sum: f32 = row
                .iter()
                .map(|&v| ((v - max_logit) as f64).exp() as f32)
                .sum();
            let log_prob = positive_logit - max_logit - exp_sum.ln();
            log_prob_sum += log_prob;
        }

        let batch_f32 = batch as f32;
        let top1_accuracy = correct as f32 / batch_f32;
        let mean_positive_margin = if batch > 1 {
            margin_sum / batch_f32
        } else {
            0.0
        };
        let mean_positive_log_probability = log_prob_sum / batch_f32;

        Self {
            loss: result.loss,
            top1_accuracy,
            mean_positive_margin,
            mean_positive_log_probability,
        }
    }
}

struct TensorboardExporter {
    writer: Mutex<SummaryWriter>,
    gauge_step: AtomicUsize,
}

impl TensorboardExporter {
    fn new<P: AsRef<Path>>(logdir: P) -> Result<Self, Box<dyn Error>> {
        let writer = SummaryWriter::new(logdir.as_ref());
        Ok(Self {
            writer: Mutex::new(writer),
            gauge_step: AtomicUsize::new(0),
        })
    }

    fn log_info_nce(&self, step: usize, summary: &InfoNCEMetricSummary) {
        if let Ok(mut writer) = self.writer.lock() {
            writer.add_scalar("selfsup/loss", summary.loss, step);
            writer.add_scalar("selfsup/top1_accuracy", summary.top1_accuracy, step);
            writer.add_scalar(
                "selfsup/positive_margin",
                summary.mean_positive_margin,
                step,
            );
            writer.add_scalar(
                "selfsup/positive_log_prob",
                summary.mean_positive_log_probability,
                step,
            );
        }
    }

    fn log_chrono(&self, step: usize, summary: &ChronoSummary) {
        if let Ok(mut writer) = self.writer.lock() {
            writer.add_scalar("chrono/mean_drift", summary.mean_drift, step);
            writer.add_scalar("chrono/mean_energy", summary.mean_energy, step);
            writer.add_scalar("chrono/energy_std", summary.energy_std, step);
        }
    }

    fn flush(&self) {
        if let Ok(mut writer) = self.writer.lock() {
            writer.flush();
        }
    }
}

impl MetricsExporter for TensorboardExporter {
    fn record_gauge(&self, name: &str, value: f64, _labels: &[(&str, &str)]) {
        let step = self.gauge_step.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut writer) = self.writer.lock() {
            writer.add_scalar(&format!("runtime/{name}"), value as f32, step);
        }
    }

    fn record_counter(&self, name: &str, value: f64, _labels: &[(&str, &str)]) {
        let step = self.gauge_step.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut writer) = self.writer.lock() {
            writer.add_scalar(&format!("counter/{name}"), value as f32, step);
        }
    }
}
