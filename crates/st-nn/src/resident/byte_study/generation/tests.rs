use super::*;

fn validate(prompts: &[Vec<u8>], shape: &[usize], count: usize) -> Result<()> {
    validate_lengths(prompts.iter().map(Vec::len), shape, count)
}

#[test]
fn borrowed_lengths_reject_without_materializing_oversized_prompts() {
    assert!(validate_lengths([usize::MAX; 32].into_iter(), &[32, 512, 8], 1).is_err());
    assert!(validate_lengths([2; 33].into_iter(), &[1, 2, 8], 1).is_err());
}

#[test]
fn decoding_is_bounded_and_never_pads_or_truncates_prompts() {
    assert!(validate(&[vec![0, 255], vec![128, 1]], &[2, 2, 8], 1).is_ok());
    assert!(validate(&vec![vec![255; 512]; 32], &[32, 512, 8], 256).is_ok());
    for count in [0, 257, usize::MAX] {
        assert!(validate(&[vec![0, 1]], &[1, 2, 8], count).is_err());
    }
    for prompts in [
        vec![],
        vec![vec![]],
        vec![vec![1]],
        vec![vec![1, 2, 3]],
        vec![vec![1, 2], vec![3, 4]],
    ] {
        assert!(validate(&prompts, &[1, 2, 8], 1).is_err());
    }
    for shape in [
        vec![],
        vec![1, 2],
        vec![1, 2, 8, 1],
        vec![0, 2, 8],
        vec![33, 2, 8],
        vec![1, 513, 8],
    ] {
        assert!(validate(&[vec![0, 1]], &shape, 1).is_err());
    }
}

#[test]
fn greedy_uses_lowest_byte_id_and_rejects_every_nonfinite_logit() {
    let mut scores = vec![-3.; 256];
    assert_eq!(greedy(&scores).unwrap(), 0);
    scores[255] = 2.;
    assert_eq!(greedy(&scores).unwrap(), 255);
    scores[17] = 2.;
    assert_eq!(greedy(&scores).unwrap(), 17);
    scores.fill(-0.);
    scores[3] = 0.;
    assert_eq!(greedy(&scores).unwrap(), 0);
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        scores[128] = bad;
        assert!(greedy(&scores).is_err());
    }
    for count in [0, 255, 257] {
        assert!(greedy(&vec![0.; count]).is_err());
    }
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
#[test]
fn batched_generation_matches_full_forward_oracle_and_preserves_owner() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) =
        st_backend_wgpu::runtime::ensure_default_runtime_blocking("byte.generation.test").unwrap();
    let mut request = super::super::tests::fixture();
    request["config"]["batch"] = json!(2);
    request["train_batches"] = json!([[[0, 0], [0, 2]], [[0, 2], [0, 0]]]);
    request["validation_batches"] = json!([[[0, 0], [0, 0]]]);
    let study = ByteCorpusStudy::from_json(request.to_string().as_bytes()).unwrap();
    let trained = pollster::block_on(study.advance(runtime.clone(), None, 2)).unwrap();
    for name in ["plain", "geometry"] {
        let artifact = study
            .case_artifact(
                &trained.checkpoint,
                name,
                ByteCorpusInferencePolicy::FixedWindowFrozenV1,
            )
            .unwrap();
        let model_before = artifact.checkpoint().to_json().unwrap();
        let mut owner = artifact.restore_wgpu(runtime.clone()).unwrap();
        let mut reference = artifact.restore_wgpu(runtime.clone()).unwrap();
        let prompts = vec![vec![0, 255], vec![128, 1]];
        let result = pollster::block_on(owner.generate_greedy(&prompts, 5)).unwrap();
        assert_eq!(result.artifact_id(), artifact.artifact_id());
        assert_eq!(result.revision, "2");
        assert_eq!(
            result.prompt_sha256[0],
            format!("{:x}", Sha256::digest(&prompts[0]))
        );
        assert!(result.serving.is_none());
        let mut windows = prompts.clone();
        let mut oracle = vec![Vec::new(); 2];
        for step in 0..5 {
            // A nonzero placeholder must not affect inputs or the selected logits.
            let mut rows = windows.clone();
            for row in &mut rows {
                row.push(255);
            }
            let host =
                ByteLmBatch::from_windows(&rows.iter().map(Vec::as_slice).collect::<Vec<_>>())
                    .unwrap();
            assert_eq!(host.input_bytes(), windows.concat());
            let batch = reference.prepare_batch(&host).unwrap();
            let forward = reference.forward(&batch).unwrap();
            let logits = forward
                .forward()
                .prediction()
                .snapshot()
                .unwrap()
                .read()
                .unwrap();
            let mut hash = Sha256::new();
            for row in 0..2 {
                let last = &logits[(row * 2 + 1) * 256..(row * 2 + 2) * 256];
                for value in last {
                    hash.update(value.to_le_bytes());
                }
                let highest = last.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let next = last.iter().position(|&v| v == highest).unwrap() as u8;
                oracle[row].push(next);
                windows[row] = vec![windows[row][1], next];
            }
            assert_eq!(
                result.next_logits_f32le_sha256[step],
                format!("{:x}", hash.finalize())
            );
        }
        assert_eq!(result.generated_bytes(), oracle);
        assert_eq!(
            pollster::block_on(owner.generate_greedy(&prompts, 5))
                .unwrap()
                .to_json()
                .unwrap(),
            result.to_json().unwrap()
        );
        assert!(pollster::block_on(owner.generate_greedy(&prompts[..1], 5)).is_err());
        assert_eq!(owner.attempted_revision(), 2);
        assert_eq!(artifact.checkpoint().to_json().unwrap(), model_before);
    }
}
