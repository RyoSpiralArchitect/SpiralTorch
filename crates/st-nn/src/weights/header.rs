//! Strict admission ahead of the official parser: reject duplicate names and
//! fields, excessive metadata, and offsets outside the supplied payload.

use super::{WeightLoadError, WeightLoadLimits};
use safetensors::Dtype;
use serde::de::{self, DeserializeSeed, MapAccess, Visitor};
use serde::Deserialize;
use std::collections::HashSet;
use std::fmt;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TensorHeader {
    dtype: Dtype,
    shape: Vec<usize>,
    data_offsets: (usize, usize),
}

#[derive(Debug)]
struct StringMetadata;

impl<'de> Deserialize<'de> for StringMetadata {
    fn deserialize<D: de::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Strings;
        impl<'de> Visitor<'de> for Strings {
            type Value = StringMetadata;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("unique string metadata")
            }
            fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Self::Value, M::Error> {
                let mut seen = HashSet::new();
                while let Some(key) = map.next_key::<String>()? {
                    if !seen.insert(key) {
                        return Err(de::Error::custom("duplicate metadata key"));
                    }
                    let _: String = map.next_value()?;
                }
                Ok(StringMetadata)
            }
        }
        deserializer.deserialize_map(Strings)
    }
}

struct Header {
    payload_len: usize,
    limits: WeightLoadLimits,
}

impl<'de> DeserializeSeed<'de> for Header {
    type Value = ();
    fn deserialize<D: de::Deserializer<'de>>(self, deserializer: D) -> Result<(), D::Error> {
        deserializer.deserialize_map(self)
    }
}

impl<'de> Visitor<'de> for Header {
    type Value = ();
    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a safetensors header with unique tensor names")
    }
    fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<(), M::Error> {
        let mut seen = HashSet::new();
        let mut tensors = 0usize;
        while let Some(name) = map.next_key::<String>()? {
            if !seen.insert(name.clone()) {
                return Err(de::Error::custom("duplicate archive key"));
            }
            if name == "__metadata__" {
                let _: StringMetadata = map.next_value()?;
                continue;
            }
            tensors += 1;
            if tensors > self.limits.max_tensors {
                return Err(de::Error::custom("tensor count exceeds budget"));
            }
            let tensor: TensorHeader = map.next_value()?;
            if tensor.shape.len() > self.limits.max_rank {
                return Err(de::Error::custom("tensor rank exceeds budget"));
            }
            let elements = tensor
                .shape
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .ok_or_else(|| de::Error::custom("tensor shape overflow"))?;
            elements
                .checked_mul(tensor.dtype.bitsize())
                .ok_or_else(|| de::Error::custom("tensor bit length overflow"))?;
            if tensor.data_offsets.0 > tensor.data_offsets.1
                || tensor.data_offsets.1 > self.payload_len
            {
                return Err(de::Error::custom("tensor offsets outside payload"));
            }
        }
        Ok(())
    }
}

pub(super) fn check(
    bytes: &[u8],
    payload_len: usize,
    limits: WeightLoadLimits,
) -> Result<(), WeightLoadError> {
    if bytes.first() != Some(&b'{') {
        return Err(WeightLoadError::Format(
            "header must start with an object".into(),
        ));
    }
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    Header {
        payload_len,
        limits,
    }
    .deserialize(&mut deserializer)
    .map_err(|error| WeightLoadError::Format(error.to_string()))?;
    deserializer
        .end()
        .map_err(|error| WeightLoadError::Format(error.to_string()))
}
