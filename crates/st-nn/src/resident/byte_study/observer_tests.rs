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
