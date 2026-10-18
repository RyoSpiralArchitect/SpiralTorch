//! Bounded, owned safetensors ingress. This loads weights, not model code or graphs.
//!
//! Shapes and source dtypes remain inspectable. Selected F32/F16/BF16 weights
//! materialize as finite, host f32 N-D tensors; conversions never infer axes.

use half::{bf16, f16};
use safetensors::{tensor::Metadata, Dtype, SafeTensors};
use sha2::{Digest, Sha256};
use st_tensor::{NdTensor, NdTensorError};
use std::collections::BTreeMap;
use thiserror::Error;

mod header;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

#[derive(Clone, Copy, Debug)]
pub struct WeightLoadLimits {
    pub max_file_bytes: usize,
    pub max_header_bytes: usize,
    pub max_tensors: usize,
    pub max_rank: usize,
    /// Per requested materialization, not the sum of all tensors in the archive.
    pub max_tensor_elements: usize,
}

impl Default for WeightLoadLimits {
    fn default() -> Self {
        Self {
            max_file_bytes: 1 << 30,
            max_header_bytes: 8 << 20,
            max_tensors: 100_000,
            max_rank: 32,
            max_tensor_elements: 64 << 20,
        }
    }
}

impl WeightLoadLimits {
    pub fn validate(&self) -> Result<(), WeightLoadError> {
        if self.max_file_bytes < 8 || self.max_file_bytes >= isize::MAX as usize {
            return Err(WeightLoadError::Limit("invalid file byte budget"));
        }
        if self.max_header_bytes == 0 || self.max_header_bytes > 100_000_000 {
            return Err(WeightLoadError::Limit("invalid header byte budget"));
        }
        if self.max_tensors == 0 || self.max_rank == 0 || self.max_rank > 128 {
            return Err(WeightLoadError::Limit(
                "tensor count must be positive; rank limit must be 1..=128",
            ));
        }
        if self.max_tensor_elements > (isize::MAX as usize) / 4 {
            return Err(WeightLoadError::Limit("invalid f32 element budget"));
        }
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum WeightLoadError {
    #[error("safetensors limit: {0}")]
    Limit(&'static str),
    #[error("invalid safetensors archive: {0}")]
    Format(String),
    #[error("weight not found: {0}")]
    Missing(String),
    #[error("weight dtype {0} cannot materialize as f32; supported: F32, F16, BF16")]
    Dtype(String),
    #[error("non-finite weight {name} at flat index {index}")]
    NonFinite { name: String, index: usize },
    #[error("cannot allocate weight storage: {0}")]
    Allocation(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Tensor(#[from] NdTensorError),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub elements: usize,
    pub byte_len: usize,
}

/// Owns the original byte snapshot. No pickle, Python, mmap, network or device
/// initialization is involved. Each materialization creates independent storage.
#[derive(Debug)]
pub struct SafeTensorArchive {
    bytes: Vec<u8>,
    metadata: Metadata,
    data_start: usize,
    limits: WeightLoadLimits,
}

impl SafeTensorArchive {
    /// Validate the borrowed bytes before allocating their owned snapshot.
    pub fn from_bytes(bytes: &[u8], limits: WeightLoadLimits) -> Result<Self, WeightLoadError> {
        let (data_start, metadata) = validate(bytes, limits)?;
        let mut owned = Vec::new();
        owned.try_reserve_exact(bytes.len()).map_err(allocation)?;
        owned.extend_from_slice(bytes);
        Ok(Self {
            bytes: owned,
            metadata,
            data_start,
            limits,
        })
    }

    pub fn from_owned_bytes(
        bytes: Vec<u8>,
        limits: WeightLoadLimits,
    ) -> Result<Self, WeightLoadError> {
        let (data_start, metadata) = validate(&bytes, limits)?;
        Ok(Self {
            bytes,
            metadata,
            data_start,
            limits,
        })
    }

    /// Read one regular file through one descriptor, bounded before payload
    /// allocation. Shard indexes and model configuration are intentionally separate.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open(
        path: impl AsRef<std::path::Path>,
        limits: WeightLoadLimits,
    ) -> Result<Self, WeightLoadError> {
        limits.validate()?;
        let mut options = std::fs::OpenOptions::new();
        options.read(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            // Inspect the opened descriptor, but do not wait for a FIFO writer
            // before we can reject non-regular files. Regular files ignore this.
            options.custom_flags(libc::O_NONBLOCK);
        }
        let mut file = options.open(path)?;
        let metadata = file.metadata()?;
        if !metadata.is_file() {
            return Err(WeightLoadError::Format("expected a regular file".into()));
        }
        let len = usize::try_from(metadata.len())
            .map_err(|_| WeightLoadError::Limit("file too large"))?;
        Self::read_file(&mut file, len, limits)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn read_file(
        reader: &mut impl std::io::Read,
        len: usize,
        limits: WeightLoadLimits,
    ) -> Result<Self, WeightLoadError> {
        limits.validate()?;
        check_file_len(len, limits)?;
        let mut prefix = [0u8; 8];
        reader.read_exact(&mut prefix)?;
        let start = data_start(&prefix, len, limits)?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(start).map_err(allocation)?;
        bytes.extend_from_slice(&prefix);
        bytes.resize(start, 0);
        reader.read_exact(&mut bytes[8..])?;
        header::check(&bytes[8..], len - start, limits)?;
        bytes
            .try_reserve_exact(len - bytes.len())
            .map_err(allocation)?;
        bytes.resize(len, 0);
        reader.read_exact(&mut bytes[start..])?;
        // Probe growth outside the Vec, avoiding a capacity doubling on a
        // one-byte overrun. Truncation fails in read_exact above.
        if reader.read(&mut [0u8; 1])? != 0 {
            return Err(WeightLoadError::Format(
                "file size changed during reading".into(),
            ));
        }
        Self::from_owned_bytes(bytes, limits)
    }

    pub fn names(&self) -> Vec<String> {
        let mut names = self.metadata.offset_keys();
        names.sort();
        names
    }

    pub fn metadata(&self) -> BTreeMap<String, String> {
        self.metadata
            .metadata()
            .as_ref()
            .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default()
    }

    pub fn info(&self, name: &str) -> Result<WeightInfo, WeightLoadError> {
        let info = self
            .metadata
            .info(name)
            .ok_or_else(|| WeightLoadError::Missing(name.into()))?;
        Ok(WeightInfo {
            dtype: info.dtype.to_string(),
            shape: info.shape.clone(),
            elements: info.shape.iter().product(),
            byte_len: info.data_offsets.1 - info.data_offsets.0,
        })
    }

    /// Identity of the owned source bytes, not authentication or model semantics.
    pub fn content_id(&self) -> String {
        format!("sha256:{:x}", Sha256::digest(&self.bytes))
    }

    pub(crate) fn tensor_bytes(&self, name: &str) -> Result<&[u8], WeightLoadError> {
        let info = self
            .metadata
            .info(name)
            .ok_or_else(|| WeightLoadError::Missing(name.into()))?;
        Ok(&self.bytes
            [self.data_start + info.data_offsets.0..self.data_start + info.data_offsets.1])
    }

    pub fn tensor(&self, name: &str) -> Result<NdTensor, WeightLoadError> {
        let values = self.f32_values(name)?;
        let shape = &self.metadata.info(name).unwrap().shape;
        Ok(NdTensor::from_vec(shape, values)?)
    }

    pub(crate) fn f32_values(&self, name: &str) -> Result<Vec<f32>, WeightLoadError> {
        let info = self
            .metadata
            .info(name)
            .ok_or_else(|| WeightLoadError::Missing(name.into()))?;
        let width = match info.dtype {
            Dtype::F32 => 4,
            Dtype::F16 | Dtype::BF16 => 2,
            _ => return Err(WeightLoadError::Dtype(info.dtype.to_string())),
        };
        let count: usize = info.shape.iter().product();
        if count > self.limits.max_tensor_elements {
            return Err(WeightLoadError::Limit("tensor exceeds element budget"));
        }
        let raw = &self.bytes
            [self.data_start + info.data_offsets.0..self.data_start + info.data_offsets.1];
        let mut values = Vec::new();
        values.try_reserve_exact(count).map_err(allocation)?;
        for (index, chunk) in raw.chunks_exact(width).enumerate() {
            let value = match info.dtype {
                Dtype::F32 => f32::from_le_bytes(chunk.try_into().unwrap()),
                Dtype::F16 => {
                    f16::from_bits(u16::from_le_bytes(chunk.try_into().unwrap())).to_f32()
                }
                Dtype::BF16 => {
                    bf16::from_bits(u16::from_le_bytes(chunk.try_into().unwrap())).to_f32()
                }
                _ => unreachable!(),
            };
            if !value.is_finite() {
                return Err(WeightLoadError::NonFinite {
                    name: name.into(),
                    index,
                });
            }
            values.push(value);
        }
        Ok(values)
    }
}

fn allocation(error: std::collections::TryReserveError) -> WeightLoadError {
    WeightLoadError::Allocation(error.to_string())
}

fn check_file_len(len: usize, limits: WeightLoadLimits) -> Result<(), WeightLoadError> {
    if len < 8 {
        return Err(WeightLoadError::Format("missing header length".into()));
    }
    if len > limits.max_file_bytes {
        return Err(WeightLoadError::Limit("file exceeds byte budget"));
    }
    Ok(())
}

fn data_start(
    prefix: &[u8],
    len: usize,
    limits: WeightLoadLimits,
) -> Result<usize, WeightLoadError> {
    let size = u64::from_le_bytes(prefix[..8].try_into().unwrap());
    if size > limits.max_header_bytes as u64 {
        return Err(WeightLoadError::Limit("header exceeds byte budget"));
    }
    let start = 8usize
        .checked_add(size as usize)
        .filter(|&end| end <= len)
        .ok_or_else(|| WeightLoadError::Format("truncated header".into()))?;
    Ok(start)
}

fn validate(bytes: &[u8], limits: WeightLoadLimits) -> Result<(usize, Metadata), WeightLoadError> {
    limits.validate()?;
    check_file_len(bytes.len(), limits)?;
    let start = data_start(bytes, bytes.len(), limits)?;
    header::check(&bytes[8..start], bytes.len() - start, limits)?;
    let (_, metadata) = SafeTensors::read_metadata(bytes)
        .map_err(|err| WeightLoadError::Format(err.to_string()))?;
    Ok((start, metadata))
}
