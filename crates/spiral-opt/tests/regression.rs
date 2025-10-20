use spiral_opt::{
    CompressionReport, QatConfig, QatObserver, QuantizationLeveling, StructuredPruner,
    StructuredPruningConfig,
};

fn sample_weights() -> Vec<f32> {
    (0..128)
        .map(|i| ((i as f32).sin() * 0.5) + ((i as f32).cos() * 0.25))
        .collect()
}

#[test]
fn qat_and_pruning_pipeline_meets_budget() {
    let mut weights = sample_weights();
    let original_l2: f32 = weights.iter().map(|w| w * w).sum::<f32>().sqrt();

    let mut observer = QatObserver::new(QatConfig::default(), QuantizationLeveling::Symmetric);
    observer.observe(&weights);
    let quant_report = observer.quantize(&mut weights.clone());
    assert!(
        quant_report.quant_error < 0.2,
        "mse too large: {}",
        quant_report.quant_error
    );

    let mut pruned = weights.clone();
    let pruner = StructuredPruner::new();
    let pruning_report = pruner
        .apply(
            &mut pruned,
            StructuredPruningConfig {
                block_size: 8,
                target_sparsity: 0.5,
                min_l2_keep: 0.01,
            },
        )
        .expect("pruning should succeed");

    assert!(
        pruning_report.achieved_sparsity >= 0.3 && pruning_report.achieved_sparsity <= 0.7,
        "sparsity out of range: {}",
        pruning_report.achieved_sparsity
    );

    let remaining_params = pruned.iter().filter(|&&w| w != 0.0).count();
    let compression = CompressionReport::new(
        weights.len(),
        remaining_params,
        Some(quant_report),
        Some(pruning_report.clone()),
        0.35,
    );

    // Regression guard: ensure compression does not reduce representational
    // capacity too aggressively for the synthetic data.
    let pruned_l2: f32 = pruned.iter().map(|w| w * w).sum::<f32>().sqrt();
    assert!(
        pruned_l2 >= 0.25 * original_l2,
        "pruned weights lost too much energy"
    );
    assert!(compression.estimated_latency_reduction >= 0.25);
}
