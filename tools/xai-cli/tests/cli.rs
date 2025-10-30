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
