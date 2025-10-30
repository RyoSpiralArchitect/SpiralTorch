use std::fs;
use std::process::Command;

use st_core::telemetry::xai_report::{AttributionMetadata, AttributionReport};
use tempfile::tempdir;

#[derive(serde::Serialize, serde::Deserialize)]
struct DiskTensor {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

fn write_tensor(path: &std::path::Path, tensor: &DiskTensor) {
    fs::write(path, serde_json::to_string(tensor).unwrap()).unwrap();
}

fn read_tensor(path: &std::path::Path) -> DiskTensor {
    serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

#[derive(serde::Deserialize)]
struct StatisticsFile {
    min: f32,
    max: f32,
    mean: f32,
    entropy: f32,
}

#[derive(serde::Deserialize)]
struct AuditSummaryFile {
    total_events: usize,
}

#[derive(serde::Deserialize)]
struct AuditCheckFile {
    name: String,
    passed: bool,
}

#[derive(serde::Deserialize)]
struct AuditReportFile {
    #[serde(default)]
    _events: Vec<serde_json::Value>,
    summary: AuditSummaryFile,
    #[serde(default)]
    self_checks: Vec<AuditCheckFile>,
}

#[allow(dead_code)]
#[derive(serde::Deserialize)]
struct StageDifferenceFile {
    stage: String,
    recorded: usize,
    recomputed: usize,
}

#[allow(dead_code)]
#[derive(serde::Deserialize)]
struct AuditCheckComparisonFile {
    name: String,
    matches: bool,
    #[serde(default)]
    recorded_passed: Option<bool>,
    #[serde(default)]
    recomputed_passed: Option<bool>,
}

#[derive(serde::Deserialize)]
struct AuditReviewFileOutput {
    observed_events: usize,
    summary_matches: bool,
    #[serde(default)]
    stage_differences: Vec<StageDifferenceFile>,
    #[serde(default)]
    issues: Vec<String>,
    #[serde(default)]
    check_comparisons: Vec<AuditCheckComparisonFile>,
}

#[derive(serde::Deserialize)]
struct AuditAnomalyFile {
    severity: String,
    message: String,
}

#[derive(serde::Deserialize)]
struct StageTransitionMetricFile {
    from: Option<String>,
    to: String,
    count: usize,
}

#[derive(serde::Deserialize)]
struct AuditIntrospectionFile {
    total_events: usize,
    unique_stages: usize,
    entropy: f64,
    #[serde(default)]
    loops: Vec<String>,
    #[serde(default)]
    transitions: Vec<StageTransitionMetricFile>,
    #[serde(default)]
    anomalies: Vec<AuditAnomalyFile>,
}

#[derive(serde::Deserialize)]
struct AuditIntrospectEntryFile {
    label: String,
    introspection: AuditIntrospectionFile,
}

#[derive(serde::Deserialize)]
struct AuditIntrospectReportFile {
    bundles: usize,
    aggregated: AuditIntrospectionFile,
    #[serde(default)]
    per_bundle: Vec<AuditIntrospectEntryFile>,
}

fn run_cli(args: &[&str]) {
    let status = Command::new(env!("CARGO_BIN_EXE_st-xai-cli"))
        .args(args)
        .status()
        .unwrap();
    assert!(status.success());
}

#[test]
fn grad_cam_cli_generates_report() {
    let dir = tempdir().unwrap();
    let activations_path = dir.path().join("activations.json");
    let gradients_path = dir.path().join("gradients.json");
    let output_path = dir.path().join("report.json");

    let activations = DiskTensor {
        rows: 2,
        cols: 4,
        data: vec![1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.5, 2.0],
    };
    let gradients = DiskTensor {
        rows: 2,
        cols: 4,
        data: vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.4],
    };

    write_tensor(&activations_path, &activations);
    write_tensor(&gradients_path, &gradients);

    run_cli(&[
        "grad-cam",
        "--activations",
        activations_path.to_str().unwrap(),
        "--gradients",
        gradients_path.to_str().unwrap(),
        "--height",
        "2",
        "--width",
        "2",
        "--layer",
        "conv1",
        "--output",
        output_path.to_str().unwrap(),
    ]);

    let report: AttributionReport =
        serde_json::from_str(&fs::read_to_string(&output_path).unwrap()).unwrap();
    assert_eq!(report.shape(), (2, 2));
    assert_eq!(report.metadata.algorithm, "grad-cam");
    assert_eq!(report.metadata.layer.as_deref(), Some("conv1"));
    assert!((report.values[3] - 1.0).abs() < 1e-6);
}

#[test]
fn grad_cam_cli_raw_heatmap() {
    let dir = tempdir().unwrap();
    let activations_path = dir.path().join("activations.json");
    let gradients_path = dir.path().join("gradients.json");
    let output_path = dir.path().join("report_raw.json");

    let activations = DiskTensor {
        rows: 2,
        cols: 4,
        data: vec![1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.5, 2.0],
    };
    let gradients = DiskTensor {
        rows: 2,
        cols: 4,
        data: vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.4],
    };

    write_tensor(&activations_path, &activations);
    write_tensor(&gradients_path, &gradients);

    run_cli(&[
        "grad-cam",
        "--activations",
        activations_path.to_str().unwrap(),
        "--gradients",
        gradients_path.to_str().unwrap(),
        "--height",
        "2",
        "--width",
        "2",
        "--layer",
        "conv1",
        "--raw-heatmap",
        "--output",
        output_path.to_str().unwrap(),
    ]);

    let report: AttributionReport =
        serde_json::from_str(&fs::read_to_string(&output_path).unwrap()).unwrap();
    assert_eq!(report.shape(), (2, 2));
    assert_eq!(report.metadata.algorithm, "grad-cam");
    assert_eq!(report.metadata.layer.as_deref(), Some("conv1"));
    assert_eq!(
        report
            .metadata
            .extras
            .get("normalise")
            .and_then(|value| value.as_bool()),
        Some(false)
    );
    assert!(report
        .values
        .iter()
        .zip([1.05_f32, 1.1_f32, 1.15_f32, 1.2_f32])
        .all(|(value, expected)| (value - expected).abs() < 1e-6));
}

#[test]
fn grad_cam_cli_writes_metadata_with_stats() {
    let dir = tempdir().unwrap();
    let activations_path = dir.path().join("activations.json");
    let gradients_path = dir.path().join("gradients.json");
    let output_path = dir.path().join("report_stats.json");
    let metadata_path = dir.path().join("metadata.json");

    let activations = DiskTensor {
        rows: 2,
        cols: 4,
        data: vec![1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.5, 2.0],
    };
    let gradients = DiskTensor {
        rows: 2,
        cols: 4,
        data: vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.4],
    };

    write_tensor(&activations_path, &activations);
    write_tensor(&gradients_path, &gradients);

    run_cli(&[
        "--include-stats",
        "--metadata-out",
        metadata_path.to_str().unwrap(),
        "grad-cam",
        "--activations",
        activations_path.to_str().unwrap(),
        "--gradients",
        gradients_path.to_str().unwrap(),
        "--height",
        "2",
        "--width",
        "2",
        "--output",
        output_path.to_str().unwrap(),
    ]);

    let report: AttributionReport =
        serde_json::from_str(&fs::read_to_string(&output_path).unwrap()).unwrap();
    let metadata: AttributionMetadata =
        serde_json::from_str(&fs::read_to_string(&metadata_path).unwrap()).unwrap();

    for extras in [&report.metadata.extras, &metadata.extras] {
        assert!(extras.contains_key("heatmap_min"));
        assert!(extras.contains_key("heatmap_max"));
        assert!(extras.contains_key("heatmap_mean"));
        assert!(extras.contains_key("heatmap_entropy"));
    }
}

#[test]
fn grad_cam_cli_applies_post_processing_and_focus_mask() {
    let dir = tempdir().unwrap();
    let activations_path = dir.path().join("activations.json");
    let gradients_path = dir.path().join("gradients.json");
    let output_path = dir.path().join("report_processed.json");
    let mask_path = dir.path().join("mask.json");

    let activations = DiskTensor {
        rows: 2,
        cols: 4,
        data: vec![1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.5, 2.0],
    };
    let gradients = DiskTensor {
        rows: 2,
        cols: 4,
        data: vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.4],
    };

    write_tensor(&activations_path, &activations);
    write_tensor(&gradients_path, &gradients);

    run_cli(&[
        "--smooth-kernel",
        "3",
        "--normalise-output",
        "--focus-mask-out",
        mask_path.to_str().unwrap(),
        "--focus-threshold",
        "0.0",
        "grad-cam",
        "--activations",
        activations_path.to_str().unwrap(),
        "--gradients",
        gradients_path.to_str().unwrap(),
        "--height",
        "2",
        "--width",
        "2",
        "--output",
        output_path.to_str().unwrap(),
    ]);

    let report: AttributionReport =
        serde_json::from_str(&fs::read_to_string(&output_path).unwrap()).unwrap();
    let smooth_kernel = report
        .metadata
        .extras
        .get("smooth_kernel")
        .and_then(|value| value.as_f64())
        .unwrap();
    assert!((smooth_kernel - 3.0).abs() < 1e-6);
    assert_eq!(
        report
            .metadata
            .extras
            .get("normalised_output")
            .and_then(|value| value.as_bool()),
        Some(true)
    );
    let threshold = report
        .metadata
        .extras
        .get("focus_threshold")
        .and_then(|value| value.as_f64())
        .unwrap();
    assert!(threshold.abs() < 1e-6);

    let mask: DiskTensor = serde_json::from_str(&fs::read_to_string(&mask_path).unwrap()).unwrap();
    assert_eq!(mask.rows, 2);
    assert_eq!(mask.cols, 2);
    assert!(mask.data.iter().any(|value| (*value - 1.0).abs() < 1e-6));
    assert!(mask
        .data
        .iter()
        .all(|value| (*value - 1.0).abs() < 1e-6 || value.abs() < 1e-6));
}

#[test]
fn grad_cam_cli_emits_overlays() {
    let dir = tempdir().unwrap();
    let activations_path = dir.path().join("activations.json");
    let gradients_path = dir.path().join("gradients.json");
    let output_path = dir.path().join("report_overlay.json");
    let base_path = dir.path().join("base.json");
    let overlay_path = dir.path().join("overlay.json");
    let gated_overlay_path = dir.path().join("gated_overlay.json");

    let activations = DiskTensor {
        rows: 2,
        cols: 4,
        data: vec![1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.5, 2.0],
    };
    let gradients = DiskTensor {
        rows: 2,
        cols: 4,
        data: vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.4],
    };
    let base = DiskTensor {
        rows: 2,
        cols: 2,
        data: vec![0.1, 0.3, 0.5, 0.7],
    };

    write_tensor(&activations_path, &activations);
    write_tensor(&gradients_path, &gradients);
    write_tensor(&base_path, &base);

    run_cli(&[
        "--overlay-base",
        base_path.to_str().unwrap(),
        "--overlay-alpha",
        "0.25",
        "--overlay-out",
        overlay_path.to_str().unwrap(),
        "--gated-overlay-out",
        gated_overlay_path.to_str().unwrap(),
        "--focus-threshold",
        "0.6",
        "grad-cam",
        "--activations",
        activations_path.to_str().unwrap(),
        "--gradients",
        gradients_path.to_str().unwrap(),
        "--height",
        "2",
        "--width",
        "2",
        "--output",
        output_path.to_str().unwrap(),
    ]);

    let report: AttributionReport =
        serde_json::from_str(&fs::read_to_string(&output_path).unwrap()).unwrap();
    let overlay_disk = read_tensor(&overlay_path);
    let gated_overlay_disk = read_tensor(&gated_overlay_path);

    assert_eq!(overlay_disk.rows, report.shape().0);
    assert_eq!(overlay_disk.cols, report.shape().1);
    assert_eq!(gated_overlay_disk.rows, report.shape().0);
    assert_eq!(gated_overlay_disk.cols, report.shape().1);

    let alpha = 0.25f32;
    for ((&heatmap, (&base_value, &overlay_value)), &gated_value) in report
        .values
        .iter()
        .zip(base.data.iter().zip(overlay_disk.data.iter()))
        .zip(gated_overlay_disk.data.iter())
    {
        let expected_overlay = base_value * (1.0 - alpha) + heatmap * alpha;
        assert!((overlay_value - expected_overlay).abs() < 1e-5);

        let mask = if heatmap >= 0.6 { 1.0 } else { 0.0 };
        let gated = heatmap * mask;
        let combined = base_value * (1.0 - alpha) + gated * alpha;
        let emphasised = heatmap.clamp(0.0, 1.0);
        let expected_gated = combined * (1.0 - alpha) + emphasised * alpha;
        assert!((gated_value - expected_gated).abs() < 1e-5);
    }
}

#[test]
fn grad_cam_cli_emits_heatmap_and_statistics_files() {
    let dir = tempdir().unwrap();
    let activations_path = dir.path().join("activations.json");
    let gradients_path = dir.path().join("gradients.json");
    let report_path = dir.path().join("report.json");
    let heatmap_path = dir.path().join("heatmap.json");
    let stats_path = dir.path().join("stats.json");

    let activations = DiskTensor {
        rows: 2,
        cols: 4,
        data: vec![1.0, 0.5, 0.25, 0.75, 0.8, 0.2, 0.3, 0.6],
    };
    let gradients = DiskTensor {
        rows: 2,
        cols: 4,
        data: vec![0.2, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4],
    };

    write_tensor(&activations_path, &activations);
    write_tensor(&gradients_path, &gradients);

    run_cli(&[
        "--heatmap-out",
        heatmap_path.to_str().unwrap(),
        "--stats-out",
        stats_path.to_str().unwrap(),
        "grad-cam",
        "--activations",
        activations_path.to_str().unwrap(),
        "--gradients",
        gradients_path.to_str().unwrap(),
        "--height",
        "2",
        "--width",
        "2",
        "--output",
        report_path.to_str().unwrap(),
    ]);

    let heatmap: DiskTensor =
        serde_json::from_str(&fs::read_to_string(&heatmap_path).unwrap()).unwrap();
    assert_eq!(heatmap.rows * heatmap.cols, heatmap.data.len());
    assert!(heatmap
        .data
        .iter()
        .any(|value| (*value - heatmap.data[0]).abs() > 1e-6));

    let stats: StatisticsFile =
        serde_json::from_str(&fs::read_to_string(&stats_path).unwrap()).unwrap();
    assert!(stats.max >= stats.min);
    assert!(stats.mean.is_finite());
    assert!(stats.entropy.is_finite());
}

#[test]
fn grad_cam_cli_writes_audit_report_and_summary() {
    let dir = tempdir().unwrap();
    let activations_path = dir.path().join("activations.json");
    let gradients_path = dir.path().join("gradients.json");
    let report_path = dir.path().join("report.json");
    let audit_path = dir.path().join("audit.json");

    let activations = DiskTensor {
        rows: 2,
        cols: 4,
        data: vec![1.0, 0.5, 0.25, 0.75, 0.8, 0.2, 0.3, 0.6],
    };
    let gradients = DiskTensor {
        rows: 2,
        cols: 4,
        data: vec![0.2, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4],
    };

    write_tensor(&activations_path, &activations);
    write_tensor(&gradients_path, &gradients);

    run_cli(&[
        "--embed-audit-summary",
        "--audit-out",
        audit_path.to_str().unwrap(),
        "grad-cam",
        "--activations",
        activations_path.to_str().unwrap(),
        "--gradients",
        gradients_path.to_str().unwrap(),
        "--height",
        "2",
        "--width",
        "2",
        "--output",
        report_path.to_str().unwrap(),
    ]);

    let report: AttributionReport =
        serde_json::from_str(&fs::read_to_string(&report_path).unwrap()).unwrap();
    let summary = report
        .metadata
        .extras
        .get("audit_summary")
        .and_then(|value| value.as_object())
        .cloned()
        .expect("audit summary embedded");
    let total_events = summary
        .get("total_events")
        .and_then(|value| value.as_u64())
        .expect("total events present");
    assert!(total_events > 0);

    let checks = report
        .metadata
        .extras
        .get("audit_self_checks")
        .and_then(|value| value.as_array())
        .cloned()
        .expect("audit checks embedded");
    assert!(!checks.is_empty());
    assert!(checks
        .iter()
        .all(|value| value.get("passed").and_then(|flag| flag.as_bool()) == Some(true)));

    let audit_report: AuditReportFile =
        serde_json::from_str(&fs::read_to_string(&audit_path).unwrap()).unwrap();
    assert!(audit_report.summary.total_events as u64 >= total_events);
    assert!(audit_report
        .self_checks
        .iter()
        .any(|check| check.name == "cli_parsed"));
    assert!(audit_report.self_checks.iter().all(|check| check.passed));
}

#[test]
fn audit_review_cli_reports_matches() {
    let dir = tempdir().unwrap();
    let activations_path = dir.path().join("activations.json");
    let gradients_path = dir.path().join("gradients.json");
    let report_path = dir.path().join("report.json");
    let audit_path = dir.path().join("audit.json");
    let review_path = dir.path().join("review.json");

    let activations = DiskTensor {
        rows: 2,
        cols: 4,
        data: vec![1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.0, 0.3],
    };
    let gradients = DiskTensor {
        rows: 2,
        cols: 4,
        data: vec![0.1, 0.2, 0.3, 0.4, 0.2, 0.1, 0.0, 0.5],
    };

    write_tensor(&activations_path, &activations);
    write_tensor(&gradients_path, &gradients);

    run_cli(&[
        "--audit-out",
        audit_path.to_str().unwrap(),
        "grad-cam",
        "--activations",
        activations_path.to_str().unwrap(),
        "--gradients",
        gradients_path.to_str().unwrap(),
        "--height",
        "2",
        "--width",
        "2",
        "--output",
        report_path.to_str().unwrap(),
    ]);

    run_cli(&[
        "audit-review",
        "--input",
        audit_path.to_str().unwrap(),
        "--output",
        review_path.to_str().unwrap(),
    ]);

    let review: AuditReviewFileOutput =
        serde_json::from_str(&fs::read_to_string(&review_path).unwrap()).unwrap();

    assert!(review.observed_events > 0);
    assert!(review.summary_matches);
    assert!(review.stage_differences.is_empty());
    assert!(review.issues.is_empty());
    assert!(review.check_comparisons.iter().all(|check| check.matches));
}

#[test]
fn audit_introspect_reports_structural_metrics() {
    let dir = tempdir().unwrap();
    let activations_path = dir.path().join("activations.json");
    let gradients_path = dir.path().join("gradients.json");
    let report_path = dir.path().join("report.json");
    let audit_path = dir.path().join("audit.json");
    let introspect_path = dir.path().join("introspect.json");

    let activations = DiskTensor {
        rows: 2,
        cols: 4,
        data: vec![1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.5, 2.0],
    };
    let gradients = DiskTensor {
        rows: 2,
        cols: 4,
        data: vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.4],
    };

    write_tensor(&activations_path, &activations);
    write_tensor(&gradients_path, &gradients);

    run_cli(&[
        "--audit-out",
        audit_path.to_str().unwrap(),
        "grad-cam",
        "--activations",
        activations_path.to_str().unwrap(),
        "--gradients",
        gradients_path.to_str().unwrap(),
        "--height",
        "2",
        "--width",
        "2",
        "--output",
        report_path.to_str().unwrap(),
    ]);

    run_cli(&[
        "audit-introspect",
        "--input",
        audit_path.to_str().unwrap(),
        "--per-bundle",
        "--output",
        introspect_path.to_str().unwrap(),
    ]);

    let report: AuditIntrospectReportFile =
        serde_json::from_str(&fs::read_to_string(&introspect_path).unwrap()).unwrap();

    assert_eq!(report.bundles, 1);
    assert!(report.aggregated.total_events > 0);
    assert!(report.aggregated.unique_stages > 0);
    assert!(!report.aggregated.transitions.is_empty());
    assert!(report.aggregated.entropy >= 0.0);
    assert!(report
        .aggregated
        .transitions
        .iter()
        .all(|transition| transition.count > 0));
    assert!(report
        .aggregated
        .transitions
        .iter()
        .any(|transition| transition.from.is_none()));
    assert!(report
        .aggregated
        .transitions
        .iter()
        .any(|transition| !transition.to.is_empty()));
    for stage in &report.aggregated.loops {
        assert!(!stage.is_empty());
    }
    assert!(report
        .aggregated
        .anomalies
        .iter()
        .all(|anomaly| {
            if anomaly.severity == "critical" {
                false
            } else {
                anomaly.message.is_empty() || !anomaly.message.trim().is_empty()
            }
        }));
    assert_eq!(report.per_bundle.len(), 1);
    assert_eq!(report.per_bundle[0].label, audit_path.to_str().unwrap());
    assert_eq!(
        report.per_bundle[0].introspection.total_events,
        report.aggregated.total_events
    );
}

#[test]
fn integrated_gradients_cli_identity_model() {
    let dir = tempdir().unwrap();
    let input_path = dir.path().join("input.json");
    let baseline_path = dir.path().join("baseline.json");
    let output_path = dir.path().join("ig.json");

    let baseline = DiskTensor {
        rows: 1,
        cols: 2,
        data: vec![0.0, 0.0],
    };
    let input = DiskTensor {
        rows: 1,
        cols: 2,
        data: vec![0.25, 0.75],
    };

    write_tensor(&baseline_path, &baseline);
    write_tensor(&input_path, &input);

    run_cli(&[
        "integrated-gradients",
        "--input",
        input_path.to_str().unwrap(),
        "--baseline",
        baseline_path.to_str().unwrap(),
        "--steps",
        "8",
        "--target",
        "0",
        "--target-label",
        "class0",
        "--output",
        output_path.to_str().unwrap(),
    ]);

    let report: AttributionReport =
        serde_json::from_str(&fs::read_to_string(&output_path).unwrap()).unwrap();
    assert_eq!(report.metadata.algorithm, "integrated-gradients");
    assert_eq!(report.metadata.target.as_deref(), Some("class0"));
    assert!((report.values[0] - 0.25).abs() < 1e-6);
    assert!(report.values[1].abs() < 1e-6);
}
