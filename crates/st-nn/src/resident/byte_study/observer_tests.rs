use super::*;

#[test]
fn observer_accepts_locally_owned_state_without_a_send_requirement() {
    let _compile_only = |study: &ByteCorpusStudy, runtime: WgpuRuntime| {
        let count = std::rc::Rc::new(std::cell::Cell::new(0));
        let _future = study.run_with_update_observer(runtime, move |_, _, _| {
            count.set(count.get() + 1);
            Ok(())
        });
    };
}

fn request(adamw: bool) -> ByteCorpusStudy {
    let mut value = optimizer::tests::designs()[1].clone();
    for case in value["cases"].as_array_mut().unwrap() {
        for parameter in case["parameters"].as_array_mut().unwrap() {
            for (i, number) in parameter["values"]
                .as_array_mut()
                .unwrap()
                .iter_mut()
                .enumerate()
            {
                *number = json!(0.1 + (i % 7) as f32 * 0.01);
            }
        }
    }
    if adamw {
        value = optimizer::tests::wrap(value, optimizer::tests::recipe());
    }
    ByteCorpusStudy::from_json(value.to_string().as_bytes()).unwrap()
}

#[test]
fn update_observer_freezes_first_boundary_without_reading_until_completion() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) =
        st_backend_wgpu::runtime::ensure_default_runtime_blocking("byte.observer.boundary")
            .unwrap();
    for adamw in [false, true] {
        let study = request(adamw);
        let mut events = Vec::new();
        let mut snapshots = Vec::new();
        let observed = pollster::block_on(study.run_with_update_observer(
            runtime.clone(),
            |name, step, model| {
                events.push((name.to_owned(), step));
                if step == 1 {
                    snapshots.push((name.to_owned(), model.checkpoint_snapshot()?));
                }
                Ok(())
            },
        ))
        .unwrap();
        let expected: Vec<_> = ["plain", "geometry", "frozen"]
            .into_iter()
            .flat_map(|name| [1, 2].map(|step| (name.to_owned(), step)))
            .collect();
        assert_eq!(events, expected);
        // No snapshot is mapped until every case has finished both updates.
        let snapshots: Vec<_> = snapshots
            .into_iter()
            .map(|(name, snapshot)| (name, snapshot.read().unwrap()))
            .collect();
        let ordinary = pollster::block_on(study.run(runtime.clone())).unwrap();
        assert_eq!(observed, ordinary, "instrumented fixture, adamw={adamw}");
        let first = pollster::block_on(study.advance(runtime.clone(), None, 1)).unwrap();
        for (name, snapshot) in snapshots {
            assert_eq!(snapshot.attempted_revision(), 1);
            assert_eq!(snapshot.adamw_state().is_some(), adamw);
            assert_eq!(
                snapshot.to_json().unwrap(),
                study
                    .case_checkpoint(&first.checkpoint, &name)
                    .unwrap()
                    .to_json()
                    .unwrap(),
                "first-boundary weights, moments and clocks, case={name}, adamw={adamw}"
            );
        }
    }
}

#[test]
fn update_observer_retains_first_boundary_without_copying_until_completion() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) =
        st_backend_wgpu::runtime::ensure_default_runtime_blocking("byte.observer.retained")
            .unwrap();
    for adamw in [false, true] {
        let study = request(adamw);
        let mut handles = Vec::new();
        let observed = pollster::block_on(study.run_with_update_observer(
            runtime.clone(),
            |name, step, model| {
                if step == 1 {
                    handles.push((name.to_owned(), model.retain_checkpoint()?));
                }
                Ok(())
            },
        ))
        .unwrap();
        assert_eq!(handles.len(), 3);
        // All model owners have been dropped, and each has submitted update 2.
        // Only now do we submit diagnostic copies, then map their results.
        let boundaries: Vec<_> = handles
            .into_iter()
            .map(|(name, handles)| (name, handles.snapshot().unwrap().read().unwrap()))
            .collect();
        assert_eq!(
            observed,
            pollster::block_on(study.run(runtime.clone())).unwrap(),
            "retained-handle fixture, adamw={adamw}"
        );
        let first = pollster::block_on(study.advance(runtime.clone(), None, 1)).unwrap();
        for (name, boundary) in boundaries {
            assert_eq!(boundary.attempted_revision(), 1);
            assert_eq!(boundary.adamw_state().is_some(), adamw);
            assert_eq!(
                boundary.to_json().unwrap(),
                study
                    .case_checkpoint(&first.checkpoint, &name)
                    .unwrap()
                    .to_json()
                    .unwrap(),
                "retained first-boundary weights, moments and clocks, case={name}, adamw={adamw}"
            );
        }
    }
}

#[test]
fn update_observer_error_aborts_before_the_next_callback() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) =
        st_backend_wgpu::runtime::ensure_default_runtime_blocking("byte.observer.error").unwrap();
    let study = request(true);
    let first = pollster::block_on(study.advance(runtime.clone(), None, 1)).unwrap();
    let before = first.checkpoint.to_json().unwrap();
    let mut events = Vec::new();
    let mut snapshot = None;
    let error = pollster::block_on(
        study.run_with_update_observer(runtime, |name, step, model| {
            events.push((name.to_owned(), step));
            snapshot = Some(model.checkpoint_snapshot()?);
            Err("intentional observer error".into())
        }),
    )
    .unwrap_err();
    assert_eq!(error.to_string(), "intentional observer error");
    assert_eq!(events, vec![("plain".to_owned(), 1)]);
    assert_eq!(
        snapshot.unwrap().read().unwrap().to_json().unwrap(),
        study
            .case_checkpoint(&first.checkpoint, "plain")
            .unwrap()
            .to_json()
            .unwrap()
    );
    assert_eq!(first.checkpoint.to_json().unwrap(), before);
}

#[test]
fn retained_boundary_client_preserves_outcome_and_matches_selected_first_state() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) =
        st_backend_wgpu::runtime::ensure_default_runtime_blocking("byte.boundary.client").unwrap();
    for adamw in [false, true] {
        let study = request(adamw);
        let first = pollster::block_on(study.advance(runtime.clone(), None, 1)).unwrap();
        let reference = study.case_checkpoint(&first.checkpoint, "frozen").unwrap();
        let saved = first.checkpoint.to_json().unwrap();
        let observed = pollster::block_on(study.run_with_retained_boundary(
            runtime.clone(),
            &first.checkpoint,
            "frozen",
        ))
        .unwrap();
        assert_eq!(observed.run_outcome["status"], "completed");
        assert_eq!(observed.run_outcome["all_cases_completed"], true);
        assert_eq!(observed.run_outcome["boundary_captured"], true);
        assert_eq!(observed.run_outcome["release_ready"], false);
        assert_eq!(observed.run_outcome["case"], "frozen");
        assert_eq!(
            observed.run_outcome["request_sha256"],
            study.request_sha256()
        );
        assert_eq!(
            observed.run_outcome["report"],
            pollster::block_on(study.run(runtime.clone())).unwrap()
        );
        assert_eq!(first.checkpoint.to_json().unwrap(), saved);
        drop(study);
        let boundary = observed
            .boundary
            .unwrap()
            .snapshot()
            .unwrap()
            .read()
            .unwrap();
        assert_eq!(boundary.to_json().unwrap(), reference.to_json().unwrap());
        assert_eq!(boundary.attempted_revision(), 1);
        assert_eq!(boundary.adamw_state().is_some(), adamw);
        // Reading diagnostic state cannot replace the completed study outcome.
        assert_eq!(observed.run_outcome["status"], "completed");
    }
}

#[test]
fn retained_boundary_keeps_real_study_errors_before_and_after_capture() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) =
        st_backend_wgpu::runtime::ensure_default_runtime_blocking("byte.boundary.failure").unwrap();
    for fail_before_capture in [true, false] {
        let mut value = tests::fixture();
        value["rate"] = json!(f32::MAX);
        // All-equal extreme head biases keep the initial softmax finite. SGD's
        // target-bias increase then overflows. Extreme head weights additionally
        // force initial evaluation to fail, before the selected update exists.
        for case in value["cases"].as_array_mut().unwrap() {
            let parameters = case["parameters"].as_array_mut().unwrap();
            let last = parameters.len() - 1;
            for slot in [last, last - 1] {
                if slot == last || fail_before_capture {
                    for number in parameters[slot]["values"].as_array_mut().unwrap() {
                        *number = json!(f32::MAX);
                    }
                }
            }
        }
        let study = ByteCorpusStudy::from_json(value.to_string().as_bytes()).unwrap();
        let reference = tests::saved(&study, 1); // Structural reference, not a training claim.
        let observed = pollster::block_on(study.run_with_retained_boundary(
            runtime.clone(),
            &reference,
            "plain",
        ))
        .unwrap();
        assert_eq!(
            observed.run_outcome["status"], "error",
            "before={fail_before_capture}"
        );
        assert_eq!(observed.run_outcome["all_cases_completed"], false);
        assert_eq!(
            observed.run_outcome["boundary_captured"],
            !fail_before_capture
        );
        assert!(observed.run_outcome["error"]
            .as_str()
            .is_some_and(|e| !e.is_empty()));
        let original = observed.run_outcome.to_string();
        eprintln!("diagnostic failure fixture before_capture={fail_before_capture}: {original}");
        if let Some(handles) = observed.boundary {
            // Rejected SGD preserves the old valid weights and still advances
            // the attempted revision. Readable state is NOT an accepted update.
            let boundary = handles
                .snapshot()
                .and_then(|capture| capture.read())
                .unwrap();
            assert_eq!(boundary.attempted_revision(), 1);
            assert_eq!(
                boundary.to_json().unwrap(),
                study
                    .case_checkpoint(&reference, "plain")
                    .unwrap()
                    .to_json()
                    .unwrap()
            );
        } else {
            assert!(fail_before_capture);
        }
        assert_eq!(observed.run_outcome.to_string(), original);
    }
}
