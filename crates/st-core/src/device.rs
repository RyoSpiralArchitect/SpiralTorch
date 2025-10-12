
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device { Cpu, #[cfg(feature="wgpu")] Wgpu, #[cfg(feature="mps")] Mps, #[cfg(feature="cuda")] Cuda }
