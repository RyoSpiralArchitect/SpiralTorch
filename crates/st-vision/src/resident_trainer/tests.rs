use super::*;
use crate::{
    dataset_catalog, DatasetSample, ImageTensor, Normalize, RandomHorizontalFlip,
    TensorVisionDataset, TransformOperation, TransformPipeline,
};
use serde_json::{json, Value};
use st_tensor::Tensor;
use std::process::Command;

#[path = "control_tests.rs"]
mod control_tests;

const DATA_ID: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn device() -> Option<TensorDevice> {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return None;
    }
    let (runtime, _) =
        st_backend_wgpu::runtime::ensure_default_runtime_blocking("vision.training_boundary.test")
            .unwrap();
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
    Some(TensorDevice::new(runtime).unwrap())
}

fn model() -> ConvNeXtClassifier {
    ConvNeXtClassifier::new(
        ConvNeXtConfig {
            input_channels: 3,
            input_hw: (4, 4),
            stage_dims: vec![2, 3],
            stage_depths: vec![1, 1],
            patch_size: (2, 2),
            curvature: -1.,
            epsilon: 0.001,
        },
        2,
        43,
    )
    .unwrap()
}

fn loader(device: &TensorDevice, shape: usize, count: usize) -> DataLoader<TensorVisionDataset> {
    let samples = (0..count)
        .map(|i| {
            let image = ImageTensor::new(
                3,
                shape,
                shape,
                (0..3 * shape * shape)
                    .map(|j| ((i * 13 + j * 7) % 101) as f32 / 101.)
                    .collect(),
            )
            .unwrap();
            // Some immutable samples intentionally carry out-of-range class IDs.
            let target = if i == 7 { 2. } else { (i % 2) as f32 };
            DatasetSample::new(image)
                .with_label(i.to_string())
                .with_target(Tensor::from_vec(1, 1, vec![target]).unwrap())
        })
        .collect();
    let dataset =
        Arc::new(TensorVisionDataset::from_samples(dataset_catalog()[0].clone(), samples).unwrap());
    let mut pipeline = TransformPipeline::with_seed(29);
    pipeline
        .add(TransformOperation::RandomHorizontalFlip(
            RandomHorizontalFlip::new(0.5).unwrap(),
        ))
        .add(TransformOperation::Normalize(
            Normalize::new(vec![0.5], vec![0.25]).unwrap(),
        ));
    pipeline
        .set_gpu_dispatcher(crate::TransformDispatcher::from_runtime(device.runtime()).unwrap());
    let mut loader = DataLoader::new(dataset, 2, Some(17))
        .unwrap()
        .with_pipeline(pipeline);
    loader.enable_shuffle(true);
    loader
}

fn rate(scheduled: bool) -> ResidentLearningRate {
    if scheduled {
        ResidentLearningRate::WarmupCosine {
            state: WarmupCosineScheduler::new(0.002, 0.0001, 10, 100)
                .unwrap()
                .state(),
        }
    } else {
        ResidentLearningRate::Constant { rate: 0.001 }
    }
}

fn checkpoint(trainer: &ResidentVisionTrainer<TensorVisionDataset>) -> VisionTrainingCheckpoint {
    trainer.checkpoint_snapshot().unwrap().read().unwrap()
}

fn steps(trainer: &mut ResidentVisionTrainer<TensorVisionDataset>, count: usize) -> Vec<Value> {
    (0..count)
        .map(|_| {
            let before = trainer.state().clone();
            let submission = trainer.submit_next().unwrap();
            assert!(trainer.has_pending_update());
            assert!(trainer.checkpoint_snapshot().is_err());
            assert!(trainer.submit_next().is_err());
            let bits: Vec<_> = submission
                .images
                .snapshot()
                .unwrap()
                .read()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect();
            let outcome = trainer.settle().unwrap();
            assert_eq!(outcome.attempted_revision, submission.attempted_revision);
            assert!(!trainer.has_pending_update());
            if !outcome.accepted {
                assert_eq!(before.learning_rate, trainer.state.learning_rate);
            }
            json!({"labels": submission.labels, "images_bits": bits,
               "rate_bits": submission.learning_rate.to_bits(), "epoch": submission.epoch,
               "revision": outcome.attempted_revision, "accepted": outcome.accepted})
        })
        .collect()
}

#[test]
fn rate_checkpoint_clocks_are_validated_without_gpu() {
    assert!(rate(true).validate(0).is_ok());
    assert!(rate(true).validate(1).is_err());
    assert!(ResidentLearningRate::Constant {
        rate: f32::INFINITY
    }
    .validate(0)
    .is_err());
    assert!(ResidentLearningRate::Constant { rate: -0.1 }
        .validate(0)
        .is_err());
    assert!(ResidentLearningRate::Constant { rate: 0. }
        .validate(0)
        .is_ok());
}

#[test]
fn client_configuration_keeps_u64_seeds_lossless() {
    let config = ResidentVisionTrainerConfig {
        model_seed: u64::MAX,
        shuffle_seed: (1_u64 << 53) + 1,
        ..Default::default()
    };
    let json = config.to_json().unwrap();
    let restored = ResidentVisionTrainerConfig::from_json(&json).unwrap();
    assert_eq!(restored.model_seed, config.model_seed);
    assert_eq!(restored.shuffle_seed, config.shuffle_seed);
    for seed in [
        json!(0),
        json!("01"),
        json!("-1"),
        json!("18446744073709551616"),
    ] {
        let mut value: Value = serde_json::from_str(&json).unwrap();
        value["model_seed"] = seed;
        assert!(ResidentVisionTrainerConfig::from_json(&value.to_string()).is_err());
    }
    assert!(ResidentVisionTrainerConfig::from_json(&" ".repeat(65_537)).is_err());
}

#[test]
fn shared_client_factory_matches_explicit_loader_and_restores() {
    let Some(device) = device() else { return };
    let explicit = loader(&device, 4, 20);
    let dataset = Arc::clone(&explicit.dataset);
    let pipeline = explicit.pipeline.clone();
    let initial_pipeline = pipeline.as_ref().unwrap().checkpoint().unwrap();
    let config = ResidentVisionTrainerConfig {
        model: model().config().clone(),
        num_classes: 2,
        batch_size: 2,
        model_seed: 43,
        shuffle_seed: 17,
        shuffle: true,
        learning_rate: rate(true),
    };
    let mut client = ResidentVisionTrainer::from_dataset(
        &config,
        device.clone(),
        Arc::clone(&dataset),
        pipeline.clone(),
        DATA_ID,
    )
    .unwrap();
    let mut control =
        ResidentVisionTrainer::new(&model(), device.clone(), explicit, DATA_ID, rate(true))
            .unwrap();
    assert_eq!(
        checkpoint(&client).to_json().unwrap(),
        checkpoint(&control).to_json().unwrap()
    );
    assert_eq!(steps(&mut client, 13), steps(&mut control, 13));
    let saved = checkpoint(&client);
    let mut restored = ResidentVisionTrainer::from_dataset_checkpoint(
        device.clone(),
        Arc::clone(&dataset),
        pipeline.clone(),
        DATA_ID,
        &saved,
    )
    .unwrap();
    assert_eq!(steps(&mut client, 11), steps(&mut restored, 11));
    assert_eq!(
        checkpoint(&client).to_json().unwrap(),
        checkpoint(&restored).to_json().unwrap()
    );
    assert_eq!(
        pipeline.as_ref().unwrap().checkpoint().unwrap(),
        initial_pipeline
    );
    assert!(
        ResidentVisionTrainer::from_dataset_checkpoint(device, dataset, None, DATA_ID, &saved,)
            .is_err()
    );
}

#[test]
fn pending_settlement_errors_and_snapshot_lifetime() {
    let Some(device) = device() else {
        return;
    };
    let mut trainer = ResidentVisionTrainer::new(
        &model(),
        device.clone(),
        loader(&device, 4, 20),
        DATA_ID,
        rate(true),
    )
    .unwrap();
    assert!(trainer.settle().is_err());
    let frozen = trainer.checkpoint_snapshot().unwrap();
    let original = checkpoint(&trainer).to_json().unwrap();
    trainer.submit_next().unwrap();
    assert!(trainer
        .restore_checkpoint(&VisionTrainingCheckpoint::from_json(&original).unwrap())
        .is_err());
    let pending_state = trainer.state().clone();
    assert!(trainer
        .finish_settlement(Err(TrainingError::InvalidReadback))
        .is_err());
    assert!(trainer.has_pending_update());
    assert_eq!(trainer.state(), &pending_state);
    trainer.settle().unwrap();
    steps(&mut trainer, 10);
    assert_eq!(frozen.read().unwrap().to_json().unwrap(), original);
    assert_ne!(checkpoint(&trainer).to_json().unwrap(), original);
    trainer
        .restore_checkpoint(&VisionTrainingCheckpoint::from_json(&original).unwrap())
        .unwrap();
    assert_eq!(checkpoint(&trainer).to_json().unwrap(), original);
}

#[test]
fn rejected_update_preserves_all_weights_and_the_proposed_schedule() {
    let Some(device) = device() else {
        return;
    };
    let mut input = loader(&device, 4, 20);
    input.enable_shuffle(false);
    let mut trainer =
        ResidentVisionTrainer::new(&model(), device, input, DATA_ID, rate(true)).unwrap();
    steps(&mut trainer, 3);
    let before = checkpoint(&trainer);
    let submitted = trainer.submit_next().unwrap();
    assert_eq!(submitted.labels, vec![Some("6".into()), Some("7".into())]);
    assert!(!trainer.settle().unwrap().accepted);
    let after = checkpoint(&trainer);
    let before_model = serde_json::to_value(before.model()).unwrap();
    let after_model = serde_json::to_value(after.model()).unwrap();
    assert_eq!(
        before_model["backbone"]["parameters"],
        after_model["backbone"]["parameters"]
    );
    assert_eq!(before_model["head"], after_model["head"]);
    assert_eq!(
        before.trainer().learning_rate(),
        after.trainer().learning_rate()
    );
    assert_eq!(before.input().position(), 6);
    assert_eq!(after.input().position(), 8);
    assert_eq!(after.trainer().accepted_updates(), 3);
    assert_eq!(after.trainer().rejected_updates(), 1);
    assert_eq!(
        trainer.submit_next().unwrap().learning_rate.to_bits(),
        submitted.learning_rate.to_bits()
    );
    assert!(trainer.settle().unwrap().accepted);
}

#[test]
fn invalid_restore_and_failed_submission_preserve_entire_trainer() {
    let Some(device) = device() else {
        return;
    };
    let mut trainer = ResidentVisionTrainer::new(
        &model(),
        device.clone(),
        loader(&device, 4, 20),
        DATA_ID,
        rate(true),
    )
    .unwrap();
    steps(&mut trainer, 13);
    let good = checkpoint(&trainer);
    let before = good.to_json().unwrap();
    let mut cases = Vec::new();
    let mut bad = good.clone();
    bad.model_sha256 = "0".repeat(64);
    cases.push(bad);
    let mut bad = good.clone();
    bad.schema = "unknown".into();
    cases.push(bad);
    let mut bad = good.clone();
    bad.trainer.accepted_updates += 1;
    bad.trainer_sha256 = digest(&bad.trainer).unwrap();
    cases.push(bad);
    let mut bad = good.clone();
    bad.trainer.epoch += 1;
    bad.trainer_sha256 = digest(&bad.trainer).unwrap();
    cases.push(bad);
    let mut bad = good.clone();
    if let ResidentLearningRate::WarmupCosine { state } = &mut bad.trainer.learning_rate {
        state.step += 1;
    }
    bad.trainer_sha256 = digest(&bad.trainer).unwrap();
    cases.push(bad);
    let mut bad = good.clone();
    let mut input = serde_json::to_value(&bad.input).unwrap();
    input["dataset_sha256"] = json!("f".repeat(64));
    bad.input = serde_json::from_value(input).unwrap();
    bad.input_sha256 = digest(&bad.input).unwrap();
    cases.push(bad);
    for bad in cases {
        assert!(trainer.restore_checkpoint(&bad).is_err());
        assert_eq!(checkpoint(&trainer).to_json().unwrap(), before);
    }

    let mut wrong_shape = ResidentVisionTrainer::new(
        &model(),
        device.clone(),
        loader(&device, 5, 20),
        DATA_ID,
        rate(true),
    )
    .unwrap();
    let before = checkpoint(&wrong_shape).to_json().unwrap();
    assert!(wrong_shape.submit_next().is_err());
    assert!(!wrong_shape.has_pending_update());
    assert_eq!(checkpoint(&wrong_shape).to_json().unwrap(), before);
    assert!(ResidentVisionTrainer::new(
        &model(),
        device.clone(),
        loader(&device, 4, 19),
        DATA_ID,
        rate(false)
    )
    .is_err());
    assert!(ResidentVisionTrainer::new(
        &model(),
        device.clone(),
        loader(&device, 4, 0),
        DATA_ID,
        rate(false)
    )
    .is_err());
}

/// Spawned by the parent with this exact test filter, so no runtime/GPU owner is
/// inherited from the process that wrote the restart payload.
#[test]
fn resident_training_process_worker() {
    let Ok(mode) = std::env::var("SPIRALTORCH_TRAINING_BOUNDARY_WORKER") else {
        return;
    };
    let device = device().expect("worker requires real GPU execution");
    let root =
        std::path::PathBuf::from(std::env::var_os("SPIRALTORCH_TRAINING_BOUNDARY_DIR").unwrap());
    let scheduled = std::env::var("SPIRALTORCH_TRAINING_BOUNDARY_SCHEDULED").unwrap() == "1";
    let mut trainer = if mode == "resume" {
        let prefix: Value =
            serde_json::from_str(&std::fs::read_to_string(root.join("prefix.json")).unwrap())
                .unwrap();
        let state =
            VisionTrainingCheckpoint::from_json(prefix["checkpoint"].as_str().unwrap()).unwrap();
        ResidentVisionTrainer::from_checkpoint(
            device.clone(),
            loader(&device, 4, 20),
            DATA_ID,
            &state,
        )
        .unwrap()
    } else {
        ResidentVisionTrainer::new(
            &model(),
            device.clone(),
            loader(&device, 4, 20),
            DATA_ID,
            rate(scheduled),
        )
        .unwrap()
    };
    let count = match mode.as_str() {
        "control" => 100,
        "prefix" => 37,
        "resume" => 63,
        _ => panic!("unknown worker"),
    };
    let trace = steps(&mut trainer, count);
    let checkpoint = checkpoint(&trainer);
    assert!(checkpoint.trainer().accepted_updates() > 0);
    assert!(checkpoint.trainer().rejected_updates() > 0);
    let payload = json!({"checkpoint": checkpoint.to_json().unwrap(), "trace": trace});
    std::fs::write(
        root.join(format!("{mode}.json")),
        serde_json::to_vec(&payload).unwrap(),
    )
    .unwrap();
}

#[test]
fn actual_process_restart_matches_all_weights_input_rates_and_rejections() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    for scheduled in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        for mode in ["control", "prefix", "resume"] {
            let output = Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "resident_trainer::tests::resident_training_process_worker",
                    "--nocapture",
                ])
                .env("SPIRALTORCH_TRAINING_BOUNDARY_WORKER", mode)
                .env("SPIRALTORCH_TRAINING_BOUNDARY_DIR", directory.path())
                .env(
                    "SPIRALTORCH_TRAINING_BOUNDARY_SCHEDULED",
                    if scheduled { "1" } else { "0" },
                )
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "worker {mode}: {}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        let read = |mode| -> Value {
            serde_json::from_str(
                &std::fs::read_to_string(directory.path().join(format!("{mode}.json"))).unwrap(),
            )
            .unwrap()
        };
        let control = read("control");
        let prefix = read("prefix");
        let resumed = read("resume");
        assert_eq!(
            prefix["trace"].as_array().unwrap(),
            &control["trace"].as_array().unwrap()[..37]
        );
        assert_eq!(
            resumed["trace"].as_array().unwrap(),
            &control["trace"].as_array().unwrap()[37..]
        );
        assert_eq!(control["checkpoint"], resumed["checkpoint"]);
        let final_state =
            VisionTrainingCheckpoint::from_json(control["checkpoint"].as_str().unwrap()).unwrap();
        assert_eq!(final_state.model().attempted_updates(), 100);
        assert_eq!(final_state.trainer().accepted_updates(), 90);
        assert_eq!(final_state.trainer().rejected_updates(), 10);
        println!("process restart: scheduled={scheduled}, 100 attempts, 90 accepted, 10 rejected, exact final checkpoint and 63 resumed batches");
    }
}
