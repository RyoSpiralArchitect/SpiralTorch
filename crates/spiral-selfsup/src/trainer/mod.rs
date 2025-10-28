//! Device abstractions and distributed synchronization helpers for self-supervised trainers.

mod device;
mod distributed;

pub use device::{CpuDevice, DistributedDevice, MetricReduce, TrainingDevice};

#[cfg(test)]
mod tests;
