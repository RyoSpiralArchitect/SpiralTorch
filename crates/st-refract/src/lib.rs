//! Apply SpiralK refract directives onto the kernel DSL backend.

#![cfg_attr(
    not(test),
    deny(clippy::expect_used, clippy::panic, clippy::unwrap_used)
)]

use std::any::Any;
use std::panic::{catch_unwind, AssertUnwindSafe};

use anyhow::{anyhow, bail, Context, Result};
use st_softlogic::spiralk::ir::{Backend, Layout, Precision, RefractBlock, TargetSpec};

/// Adapter trait that surfaces the knobs available in the kernel DSL compiler.
///
/// Implementations are expected to forward the calls to whichever backend owns
/// the actual state (e.g. the planner bridge or compiled session). Default
/// implementations are no-ops so callers can opt in gradually.
///
/// Backends that mutate state should override the transaction lifecycle hooks.
/// [`RefractLowering`] calls `rollback_refract` whenever a callback entered after
/// validation fails or panics, including `begin_refract` and `commit_refract`.
/// Rollback implementations must therefore tolerate a partially started
/// transaction.
pub trait KdslBackend {
    /// Capture any state needed to roll back this refract block.
    fn begin_refract(&mut self, _block: &RefractBlock) -> Result<()> {
        Ok(())
    }

    fn select_target(&mut self, _target: &TargetSpec) -> Result<()> {
        Ok(())
    }

    fn set_precision(&mut self, _precision: Precision) -> Result<()> {
        Ok(())
    }

    fn set_layout(&mut self, _layout: Layout) -> Result<()> {
        Ok(())
    }

    fn set_schedule(&mut self, _schedule: &str) -> Result<()> {
        Ok(())
    }

    fn select_backend(&mut self, _backend: Backend) -> Result<()> {
        Ok(())
    }

    fn tune_op(&mut self, _op: &str, _flags: &[String]) -> Result<()> {
        Ok(())
    }

    /// Make the staged refract changes visible.
    fn commit_refract(&mut self, _block: &RefractBlock) -> Result<()> {
        Ok(())
    }

    /// Restore the state captured by [`KdslBackend::begin_refract`].
    fn rollback_refract(&mut self, _block: &RefractBlock) -> Result<()> {
        Ok(())
    }
}

/// Validate a refract block without touching a backend.
///
/// Parsed SpiralK documents already satisfy these identifier rules. This check
/// also protects callers that construct or deserialize the public IR directly.
pub fn validate_refract_block(block: &RefractBlock) -> Result<()> {
    validate_identifier(&block.name, "block name")?;

    let (target_kind, target_name) = match &block.target {
        TargetSpec::Graph(name) => ("graph target", name),
        TargetSpec::Prsn(name) => ("prsn target", name),
    };
    validate_identifier(target_name, target_kind)?;

    if let Some(schedule) = block.schedule.as_deref() {
        validate_identifier(schedule, "schedule")?;
    }

    for (index, policy) in block.policies.iter().enumerate() {
        validate_identifier(&policy.op, &format!("policy {index} op"))?;
        for (flag_index, flag) in policy.flags.iter().enumerate() {
            validate_identifier(flag, &format!("policy {index} flag {flag_index}"))?;
        }
    }

    Ok(())
}

fn validate_identifier(value: &str, field: &str) -> Result<()> {
    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        bail!("refract {field} must not be empty");
    };
    if first != '_' && !first.is_alphabetic() {
        bail!("refract {field} must start with a letter or '_'");
    }
    if chars.any(|ch| ch != '_' && ch != '-' && !ch.is_alphanumeric()) {
        bail!("refract {field} contains an invalid character");
    }
    Ok(())
}

/// Lowering engine that converts parsed refract blocks into concrete calls on a
/// [`KdslBackend`].
pub struct RefractLowering<'a, B: KdslBackend> {
    backend: &'a mut B,
}

impl<'a, B: KdslBackend> RefractLowering<'a, B> {
    pub fn new(backend: &'a mut B) -> Self {
        Self { backend }
    }

    /// Validate and apply one refract block through the backend transaction hooks.
    pub fn apply(&mut self, block: &RefractBlock) -> Result<()> {
        validate_refract_block(block)?;

        if let Err(error) = call_backend("begin_refract", || self.backend.begin_refract(block)) {
            return Err(self.rollback_after_failure(block, error));
        }

        let result = self
            .apply_staged(block)
            .and_then(|()| call_backend("commit_refract", || self.backend.commit_refract(block)));

        match result {
            Ok(()) => Ok(()),
            Err(error) => Err(self.rollback_after_failure(block, error)),
        }
    }

    fn apply_staged(&mut self, block: &RefractBlock) -> Result<()> {
        call_backend("select_target", || {
            self.backend.select_target(&block.target)
        })?;

        if let Some(precision) = block.precision {
            call_backend("set_precision", || self.backend.set_precision(precision))?;
        }

        if let Some(layout) = block.layout {
            call_backend("set_layout", || self.backend.set_layout(layout))?;
        }

        if let Some(schedule) = block.schedule.as_deref() {
            call_backend("set_schedule", || self.backend.set_schedule(schedule))?;
        }

        if let Some(backend) = block.backend {
            call_backend("select_backend", || self.backend.select_backend(backend))?;
        }

        for policy in &block.policies {
            call_backend("tune_op", || {
                self.backend.tune_op(&policy.op, &policy.flags)
            })
            .with_context(|| format!("failed to tune refract op '{}'", policy.op))?;
        }

        Ok(())
    }

    fn rollback_after_failure(
        &mut self,
        block: &RefractBlock,
        error: anyhow::Error,
    ) -> anyhow::Error {
        match call_backend("rollback_refract", || self.backend.rollback_refract(block)) {
            Ok(()) => error,
            Err(rollback_error) => {
                anyhow!("refract lowering failed: {error:#}; rollback failed: {rollback_error:#}")
            }
        }
    }
}

fn call_backend<T>(stage: &str, callback: impl FnOnce() -> Result<T>) -> Result<T> {
    match catch_unwind(AssertUnwindSafe(callback)) {
        Ok(result) => result.with_context(|| format!("refract backend failed during {stage}")),
        Err(payload) => Err(anyhow!(
            "refract backend panicked during {stage}: {}",
            panic_payload_message(payload)
        )),
    }
}

fn panic_payload_message(payload: Box<dyn Any + Send>) -> String {
    let payload = match payload.downcast::<String>() {
        Ok(message) => return *message,
        Err(payload) => payload,
    };
    let payload = match payload.downcast::<&'static str>() {
        Ok(message) => return (*message).to_string(),
        Err(payload) => payload,
    };

    if let Err(secondary_payload) = catch_unwind(AssertUnwindSafe(|| drop(payload))) {
        std::mem::forget(secondary_payload);
    }
    "non-string panic payload".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use st_softlogic::spiralk::ir::RefractOpPolicy;

    #[derive(Clone, Debug, Default, PartialEq, Eq)]
    struct BackendState {
        precision: Option<Precision>,
        layout: Option<Layout>,
        schedule: Option<String>,
        backend: Option<Backend>,
        tuned: Vec<(String, Vec<String>)>,
        target: Option<TargetSpec>,
    }

    #[derive(Clone, Copy)]
    enum PanicMode {
        Message,
        HostilePayload,
    }

    struct FakeBackend {
        state: BackendState,
        snapshot: Option<BackendState>,
        events: Vec<&'static str>,
        fail_stage: Option<&'static str>,
        panic_stage: Option<(&'static str, PanicMode)>,
    }

    impl FakeBackend {
        fn new() -> Self {
            Self {
                state: BackendState::default(),
                snapshot: None,
                events: Vec::new(),
                fail_stage: None,
                panic_stage: None,
            }
        }

        fn finish_stage(&self, stage: &'static str) -> Result<()> {
            if let Some((panic_stage, mode)) = self.panic_stage {
                if panic_stage == stage {
                    match mode {
                        PanicMode::Message => panic!("{stage} panic"),
                        PanicMode::HostilePayload => std::panic::panic_any(PanicOnDrop),
                    }
                }
            }
            if self.fail_stage == Some(stage) {
                bail!("{stage} error");
            }
            Ok(())
        }
    }

    impl KdslBackend for FakeBackend {
        fn begin_refract(&mut self, _block: &RefractBlock) -> Result<()> {
            self.events.push("begin_refract");
            self.snapshot = Some(self.state.clone());
            self.finish_stage("begin_refract")
        }

        fn select_target(&mut self, target: &TargetSpec) -> Result<()> {
            self.events.push("select_target");
            self.state.target = Some(target.clone());
            self.finish_stage("select_target")
        }

        fn set_precision(&mut self, precision: Precision) -> Result<()> {
            self.events.push("set_precision");
            self.state.precision = Some(precision);
            self.finish_stage("set_precision")
        }

        fn set_layout(&mut self, layout: Layout) -> Result<()> {
            self.events.push("set_layout");
            self.state.layout = Some(layout);
            self.finish_stage("set_layout")
        }

        fn set_schedule(&mut self, schedule: &str) -> Result<()> {
            self.events.push("set_schedule");
            self.state.schedule = Some(schedule.to_owned());
            self.finish_stage("set_schedule")
        }

        fn select_backend(&mut self, backend: Backend) -> Result<()> {
            self.events.push("select_backend");
            self.state.backend = Some(backend);
            self.finish_stage("select_backend")
        }

        fn tune_op(&mut self, op: &str, flags: &[String]) -> Result<()> {
            self.events.push("tune_op");
            self.state.tuned.push((op.to_owned(), flags.to_vec()));
            self.finish_stage("tune_op")
        }

        fn commit_refract(&mut self, _block: &RefractBlock) -> Result<()> {
            self.events.push("commit_refract");
            self.finish_stage("commit_refract")?;
            self.snapshot = None;
            Ok(())
        }

        fn rollback_refract(&mut self, _block: &RefractBlock) -> Result<()> {
            self.events.push("rollback_refract");
            if let Some(snapshot) = self.snapshot.take() {
                self.state = snapshot;
            }
            self.finish_stage("rollback_refract")
        }
    }

    struct PanicOnDrop;

    impl Drop for PanicOnDrop {
        fn drop(&mut self) {
            panic!("panic payload drop");
        }
    }

    fn block() -> RefractBlock {
        RefractBlock {
            name: "main".into(),
            target: TargetSpec::Graph("Encoder".into()),
            precision: Some(Precision::Fp16),
            layout: Some(Layout::NHWC),
            schedule: Some("cooperative".into()),
            backend: Some(Backend::WGPU),
            policies: vec![RefractOpPolicy {
                op: "attention".into(),
                flags: vec!["fuse_softmax".into()],
            }],
        }
    }

    #[test]
    fn lowering_applies_all_fields_and_commits() {
        let mut backend = FakeBackend::new();
        RefractLowering::new(&mut backend).apply(&block()).unwrap();

        assert!(matches!(backend.state.precision, Some(Precision::Fp16)));
        assert!(matches!(backend.state.layout, Some(Layout::NHWC)));
        assert_eq!(backend.state.schedule.as_deref(), Some("cooperative"));
        assert!(matches!(backend.state.backend, Some(Backend::WGPU)));
        assert_eq!(backend.state.tuned.len(), 1);
        assert!(matches!(
            backend.state.target,
            Some(TargetSpec::Graph(ref graph)) if graph == "Encoder"
        ));
        assert_eq!(
            backend.events,
            [
                "begin_refract",
                "select_target",
                "set_precision",
                "set_layout",
                "set_schedule",
                "select_backend",
                "tune_op",
                "commit_refract",
            ]
        );
    }

    #[test]
    fn validation_rejects_invalid_ir_before_touching_backend() {
        let mut invalid_blocks = Vec::new();
        let mut invalid = block();
        invalid.name.clear();
        invalid_blocks.push(invalid);
        let mut invalid = block();
        invalid.target = TargetSpec::Prsn("9invalid".into());
        invalid_blocks.push(invalid);
        let mut invalid = block();
        invalid.schedule = Some("bad schedule".into());
        invalid_blocks.push(invalid);
        let mut invalid = block();
        invalid.policies[0].op.clear();
        invalid_blocks.push(invalid);
        let mut invalid = block();
        invalid.policies[0].flags[0] = "bad flag".into();
        invalid_blocks.push(invalid);

        for invalid in invalid_blocks {
            let mut backend = FakeBackend::new();
            let error = RefractLowering::new(&mut backend)
                .apply(&invalid)
                .unwrap_err();
            assert!(error.to_string().contains("refract"));
            assert!(backend.events.is_empty());
        }
    }

    #[test]
    fn every_failed_stage_rolls_back_to_the_original_state() {
        for stage in [
            "begin_refract",
            "select_target",
            "set_precision",
            "set_layout",
            "set_schedule",
            "select_backend",
            "tune_op",
            "commit_refract",
        ] {
            let original = BackendState {
                schedule: Some("original".into()),
                ..BackendState::default()
            };
            let mut backend = FakeBackend::new();
            backend.state = original.clone();
            backend.fail_stage = Some(stage);

            let error = RefractLowering::new(&mut backend)
                .apply(&block())
                .unwrap_err();

            assert!(format!("{error:#}").contains(stage), "{error:#}");
            assert_eq!(backend.state, original, "stage {stage}");
            assert_eq!(backend.events.last(), Some(&"rollback_refract"));
        }
    }

    #[test]
    fn backend_panic_is_reported_and_rolled_back() {
        let mut backend = FakeBackend::new();
        backend.panic_stage = Some(("set_layout", PanicMode::Message));

        let error = RefractLowering::new(&mut backend)
            .apply(&block())
            .unwrap_err();

        assert!(error.to_string().contains("panicked during set_layout"));
        assert_eq!(backend.state, BackendState::default());
        assert_eq!(backend.events.last(), Some(&"rollback_refract"));
    }

    #[test]
    fn hostile_panic_payload_is_contained() {
        let mut backend = FakeBackend::new();
        backend.panic_stage = Some(("tune_op", PanicMode::HostilePayload));

        let error = RefractLowering::new(&mut backend)
            .apply(&block())
            .unwrap_err();

        assert!(format!("{error:#}").contains("non-string panic payload"));
        assert_eq!(backend.state, BackendState::default());
    }

    #[test]
    fn rollback_failure_preserves_both_errors() {
        let mut backend = FakeBackend::new();
        backend.fail_stage = Some("set_layout");
        backend.panic_stage = Some(("rollback_refract", PanicMode::Message));

        let error = RefractLowering::new(&mut backend)
            .apply(&block())
            .unwrap_err();
        let message = format!("{error:#}");

        assert!(message.contains("set_layout error"));
        assert!(message.contains("rollback_refract panic"));
        assert_eq!(backend.state, BackendState::default());
    }
}
