use super::*;

/// Accumulation for one CPU matrix product. Input, weight and output storage stay f32.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CausalLmHeadPrecision {
    F32,
    F64,
}

impl CausalLmHeadPrecision {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }

    pub(super) fn multiply(self, lhs: &Tensor, rhs: &PackedB) -> CausalLmResult<Tensor> {
        Ok(match self {
            Self::F32 => lhs.matmul_prepacked_with_backend(rhs, MatmulBackend::CpuSimd)?,
            Self::F64 => lhs.matmul_prepacked_wide(rhs)?,
        })
    }
}

/// The complete vocabulary-head policy, used as a single identity in gradients and checkpoints.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum CausalLmReadoutAccumulator {
    #[default]
    F32,
    F64,
    #[serde(rename = "f32_wide_backward")]
    F32WideBackward,
}

impl CausalLmReadoutAccumulator {
    pub(super) fn is_f32(&self) -> bool {
        *self == Self::F32
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::F32WideBackward => "f32_wide_backward",
        }
    }

    pub fn forward_precision(self) -> CausalLmHeadPrecision {
        match self {
            Self::F32 | Self::F32WideBackward => CausalLmHeadPrecision::F32,
            Self::F64 => CausalLmHeadPrecision::F64,
        }
    }

    pub fn backward_precision(self) -> CausalLmHeadPrecision {
        match self {
            Self::F32 => CausalLmHeadPrecision::F32,
            Self::F64 | Self::F32WideBackward => CausalLmHeadPrecision::F64,
        }
    }
}

impl std::str::FromStr for CausalLmReadoutAccumulator {
    type Err = &'static str;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "f32" => Ok(Self::F32),
            "f64" => Ok(Self::F64),
            "f32_wide_backward" => Ok(Self::F32WideBackward),
            _ => Err("readout accumulator must be f32, f64 or f32_wide_backward"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn wire_identity_and_both_directions_are_explicit() {
        use CausalLmHeadPrecision::{F32, F64};
        for (token, forward, backward) in [
            ("f32", F32, F32),
            ("f64", F64, F64),
            ("f32_wide_backward", F32, F64),
        ] {
            let p: CausalLmReadoutAccumulator = token.parse().unwrap();
            assert_eq!(
                (p.as_str(), p.forward_precision(), p.backward_precision()),
                (token, forward, backward)
            );
            assert_eq!(serde_json::to_value(p).unwrap(), serde_json::json!(token));
            assert_eq!(
                serde_json::from_value::<CausalLmReadoutAccumulator>(serde_json::json!(token))
                    .unwrap(),
                p
            );
            assert_eq!(p.is_f32(), token == "f32");
        }
        assert!("wide".parse::<CausalLmReadoutAccumulator>().is_err());
        assert!(serde_json::from_str::<CausalLmReadoutAccumulator>("null").is_err());
    }
}
