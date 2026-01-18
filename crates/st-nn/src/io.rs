// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::module::Module;
use crate::{PureResult, Tensor, TensorError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct StoredTensor {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl StoredTensor {
    fn from_tensor(tensor: &Tensor) -> StoredTensor {
        StoredTensor {
            rows: tensor.shape().0,
            cols: tensor.shape().1,
            data: tensor.data().to_vec(),
        }
    }

    fn into_tensor(self) -> PureResult<Tensor> {
        Tensor::from_vec(self.rows, self.cols, self.data)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ModuleSnapshot {
    parameters: HashMap<String, StoredTensor>,
}

fn to_snapshot<M: Module + ?Sized>(module: &M) -> PureResult<ModuleSnapshot> {
    let state = module.state_dict()?;
    let mut parameters = HashMap::new();
    for (name, tensor) in state {
        parameters.insert(name, StoredTensor::from_tensor(&tensor));
    }
    Ok(ModuleSnapshot { parameters })
}

fn snapshot_from_state(state: &HashMap<String, Tensor>) -> ModuleSnapshot {
    let mut parameters = HashMap::new();
    for (name, tensor) in state {
        parameters.insert(name.clone(), StoredTensor::from_tensor(tensor));
    }
    ModuleSnapshot { parameters }
}

fn from_snapshot(snapshot: ModuleSnapshot) -> PureResult<HashMap<String, Tensor>> {
    let mut state = HashMap::new();
    for (name, tensor) in snapshot.parameters.into_iter() {
        state.insert(name, tensor.into_tensor()?);
    }
    Ok(state)
}

fn io_error(err: std::io::Error) -> TensorError {
    TensorError::IoError {
        message: err.to_string(),
    }
}

fn serde_error(err: impl ToString) -> TensorError {
    TensorError::SerializationError {
        message: err.to_string(),
    }
}

pub fn save_json<M: Module + ?Sized, P: AsRef<Path>>(module: &M, path: P) -> PureResult<()> {
    let snapshot = to_snapshot(module)?;
    let file = File::create(path.as_ref()).map_err(io_error)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &snapshot).map_err(serde_error)?;
    Ok(())
}

pub fn load_json<M: Module + ?Sized, P: AsRef<Path>>(
    module: &mut M,
    path: P,
) -> PureResult<()> {
    let file = File::open(path.as_ref()).map_err(io_error)?;
    let reader = BufReader::new(file);
    let snapshot: ModuleSnapshot = serde_json::from_reader(reader).map_err(serde_error)?;
    let state = from_snapshot(snapshot)?;
    module.load_state_dict(&state)
}

pub fn save_state_dict_json<P: AsRef<Path>>(
    state: &HashMap<String, Tensor>,
    path: P,
) -> PureResult<()> {
    let snapshot = snapshot_from_state(state);
    let file = File::create(path.as_ref()).map_err(io_error)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &snapshot).map_err(serde_error)?;
    Ok(())
}

pub fn load_state_dict_json<P: AsRef<Path>>(path: P) -> PureResult<HashMap<String, Tensor>> {
    let file = File::open(path.as_ref()).map_err(io_error)?;
    let reader = BufReader::new(file);
    let snapshot: ModuleSnapshot = serde_json::from_reader(reader).map_err(serde_error)?;
    from_snapshot(snapshot)
}

pub fn save_bincode<M: Module + ?Sized, P: AsRef<Path>>(module: &M, path: P) -> PureResult<()> {
    let snapshot = to_snapshot(module)?;
    let file = File::create(path.as_ref()).map_err(io_error)?;
    let writer = BufWriter::new(file);
    bincode::serialize_into(writer, &snapshot).map_err(serde_error)?;
    Ok(())
}

pub fn load_bincode<M: Module + ?Sized, P: AsRef<Path>>(
    module: &mut M,
    path: P,
) -> PureResult<()> {
    let file = File::open(path.as_ref()).map_err(io_error)?;
    let reader = BufReader::new(file);
    let snapshot: ModuleSnapshot = bincode::deserialize_from(reader).map_err(serde_error)?;
    let state = from_snapshot(snapshot)?;
    module.load_state_dict(&state)
}

pub fn save_state_dict_bincode<P: AsRef<Path>>(
    state: &HashMap<String, Tensor>,
    path: P,
) -> PureResult<()> {
    let snapshot = snapshot_from_state(state);
    let file = File::create(path.as_ref()).map_err(io_error)?;
    let writer = BufWriter::new(file);
    bincode::serialize_into(writer, &snapshot).map_err(serde_error)?;
    Ok(())
}

pub fn load_state_dict_bincode<P: AsRef<Path>>(path: P) -> PureResult<HashMap<String, Tensor>> {
    let file = File::open(path.as_ref()).map_err(io_error)?;
    let reader = BufReader::new(file);
    let snapshot: ModuleSnapshot = bincode::deserialize_from(reader).map_err(serde_error)?;
    from_snapshot(snapshot)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::linear::Linear;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn save_and_load_roundtrip_json() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("linear.json");
        let mut layer = Linear::new("io", 2, 2).unwrap();
        save_json(&layer, &path).unwrap();
        let before = layer.state_dict().unwrap();
        layer.apply_step(0.01).unwrap();
        load_json(&mut layer, &path).unwrap();
        let after = layer.state_dict().unwrap();
        assert_eq!(before, after);
    }

    #[test]
    fn save_and_load_roundtrip_bincode() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("linear.bin");
        let mut layer = Linear::new("io", 2, 2).unwrap();
        save_bincode(&layer, &path).unwrap();
        layer.apply_step(0.01).unwrap();
        load_bincode(&mut layer, &path).unwrap();
        let state = layer.state_dict().unwrap();
        assert!(fs::metadata(&path).unwrap().len() > 0);
        assert_eq!(state.len(), 2);
    }
}
