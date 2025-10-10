#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Device {
    Cpu,
    /// Present even without the 'mps' feature so matches compile.
    Mps,
}
