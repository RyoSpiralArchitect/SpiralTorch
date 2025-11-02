//! Apply SpiralK refract directives onto the kernel DSL backend.

use anyhow::Result;
use st_softlogic::spiralk::ir::{Backend, Layout, Precision, RefractBlock, TargetSpec};

/// Adapter trait that surfaces the knobs available in the kernel DSL compiler.
///
/// Implementations are expected to forward the calls to whichever backend owns
/// the actual state (e.g. the planner bridge or compiled session).  Default
/// implementations are no-ops so callers can opt-in gradually.
pub trait KdslBackend {
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
}

/// Lowering engine that converts the parsed refract blocks into concrete calls
/// on a [`KdslBackend`].
pub struct RefractLowering<'a, B: KdslBackend> {
    backend: &'a mut B,
}

impl<'a, B: KdslBackend> RefractLowering<'a, B> {
    pub fn new(backend: &'a mut B) -> Self {
        Self { backend }
    }

    pub fn apply(&mut self, block: &RefractBlock) -> Result<()> {
        self.backend.select_target(&block.target)?;

        if let Some(precision) = block.precision {
            self.backend.set_precision(precision)?;
        }

        if let Some(layout) = block.layout {
            self.backend.set_layout(layout)?;
        }

        if let Some(schedule) = block.schedule.as_deref() {
            self.backend.set_schedule(schedule)?;
        }

        if let Some(backend) = block.backend {
            self.backend.select_backend(backend)?;
        }

        for policy in &block.policies {
            self.backend.tune_op(&policy.op, &policy.flags)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use st_softlogic::spiralk::ir::RefractOpPolicy;

    struct FakeBackend {
        last_precision: Option<Precision>,
        last_layout: Option<Layout>,
        schedule: Option<String>,
        backend: Option<Backend>,
        tuned: Vec<(String, Vec<String>)>,
        target: Option<TargetSpec>,
    }

    impl FakeBackend {
        fn new() -> Self {
            Self {
                last_precision: None,
                last_layout: None,
                schedule: None,
                backend: None,
                tuned: Vec::new(),
                target: None,
            }
        }
    }

    impl KdslBackend for FakeBackend {
        fn select_target(&mut self, target: &TargetSpec) -> Result<()> {
            self.target = Some(target.clone());
            Ok(())
        }

        fn set_precision(&mut self, precision: Precision) -> Result<()> {
            self.last_precision = Some(precision);
            Ok(())
        }

        fn set_layout(&mut self, layout: Layout) -> Result<()> {
            self.last_layout = Some(layout);
            Ok(())
        }

        fn set_schedule(&mut self, schedule: &str) -> Result<()> {
            self.schedule = Some(schedule.to_owned());
            Ok(())
        }

        fn select_backend(&mut self, backend: Backend) -> Result<()> {
            self.backend = Some(backend);
            Ok(())
        }

        fn tune_op(&mut self, op: &str, flags: &[String]) -> Result<()> {
            self.tuned.push((op.to_owned(), flags.to_vec()));
            Ok(())
        }
    }

    #[test]
    fn lowering_applies_all_fields() {
        let block = RefractBlock {
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
        };

        let mut backend = FakeBackend::new();
        let mut lowering = RefractLowering::new(&mut backend);
        lowering.apply(&block).unwrap();

        assert!(matches!(backend.last_precision, Some(Precision::Fp16)));
        assert!(matches!(backend.last_layout, Some(Layout::NHWC)));
        assert_eq!(backend.schedule.as_deref(), Some("cooperative"));
        assert!(matches!(backend.backend, Some(Backend::WGPU)));
        assert_eq!(backend.tuned.len(), 1);
        assert_eq!(backend.target, Some(TargetSpec::Graph("Encoder".into())));
    }
}
