use std::fs;
use std::process::Command;

use st_core::telemetry::xai_report::AttributionReport;
use tempfile::tempdir;

#[derive(serde::Serialize)]
struct DiskTensor {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
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

    fs::write(
        &activations_path,
        serde_json::to_string(&activations).unwrap(),
    )
    .unwrap();
    fs::write(&gradients_path, serde_json::to_string(&gradients).unwrap()).unwrap();

    let status = Command::new(env!("CARGO_BIN_EXE_st-xai-cli"))
        .args([
            "--algorithm",
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
        ])
        .status()
        .unwrap();
    assert!(status.success());

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

    fs::write(
        &activations_path,
        serde_json::to_string(&activations).unwrap(),
    )
    .unwrap();
    fs::write(&gradients_path, serde_json::to_string(&gradients).unwrap()).unwrap();

    let status = Command::new(env!("CARGO_BIN_EXE_st-xai-cli"))
        .args([
            "--algorithm",
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
        ])
        .status()
        .unwrap();
    assert!(status.success());

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

    fs::write(&baseline_path, serde_json::to_string(&baseline).unwrap()).unwrap();
    fs::write(&input_path, serde_json::to_string(&input).unwrap()).unwrap();

    let status = Command::new(env!("CARGO_BIN_EXE_st-xai-cli"))
        .args([
            "--algorithm",
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
        ])
        .status()
        .unwrap();
    assert!(status.success());

    let report: AttributionReport =
        serde_json::from_str(&fs::read_to_string(&output_path).unwrap()).unwrap();
    assert_eq!(report.metadata.algorithm, "integrated-gradients");
    assert_eq!(report.metadata.target.as_deref(), Some("class0"));
    assert!((report.values[0] - 0.25).abs() < 1e-6);
    assert!(report.values[1].abs() < 1e-6);
}
