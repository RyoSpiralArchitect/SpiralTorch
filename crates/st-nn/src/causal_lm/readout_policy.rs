use super::*;

#[derive(Debug)]
pub(super) enum TrainingHeadPack {
    Transpose(PackedB),
    Panels(st_tensor::CpuReadoutPanels),
}

/// Process-local traversal hint for the f32 vocabulary head. Never creates a cache.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CausalLmForwardStorage {
    #[default]
    Auto,
    OriginalPacked,
    PreferCachedTranspose,
}

impl CausalLmForwardStorage {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::OriginalPacked => "original_packed",
            Self::PreferCachedTranspose => "prefer_cached_transpose",
        }
    }

    fn borrows_existing_transpose(self, rows: usize, width: usize, vocabulary: usize) -> bool {
        match self {
            Self::OriginalPacked => false,
            Self::PreferCachedTranspose => true,
            // The larger heads qualified as pairs at seven rows, not as ranges.
            Self::Auto => {
                cfg!(all(target_arch = "aarch64", target_os = "macos"))
                    && matches!(
                        (rows, width, vocabulary),
                        (6..=7, 96..=128, 8192..=16_385) | (7, 384, 32_768) | (7, 768, 50_257)
                    )
            }
        }
    }
}

impl std::str::FromStr for CausalLmForwardStorage {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "auto" => Ok(Self::Auto),
            "original_packed" => Ok(Self::OriginalPacked),
            "prefer_cached_transpose" => Ok(Self::PreferCachedTranspose),
            _ => Err(
                "forward readout storage must be auto, original_packed or prefer_cached_transpose",
            ),
        }
    }
}

impl NativeCausalLm {
    pub fn forward_readout_storage(&self) -> CausalLmForwardStorage {
        self.forward_storage
    }

    /// Share existing weights and backward storage. Unlike the accumulator policy,
    /// this hint changes neither arithmetic precision nor checkpoint identity.
    pub fn with_forward_readout_storage(&self, storage: CausalLmForwardStorage) -> Self {
        let mut model = self.clone();
        model.forward_storage = storage;
        model
    }

    pub(super) fn project_readout(&self, lhs: &Tensor) -> CausalLmResult<Tensor> {
        let (rows, width) = lhs.shape();
        let precision = self.readout_accumulator.forward_precision();
        if precision == CausalLmHeadPrecision::F32
            && self.forward_storage != CausalLmForwardStorage::OriginalPacked
        {
            match self.backward_head.get() {
                Some(Ok(TrainingHeadPack::Panels(p))) if (1..=3).contains(&rows) => {
                    return Ok(p.forward(lhs)?)
                }
                Some(Ok(TrainingHeadPack::Transpose(tr)))
                    if self.forward_storage.borrows_existing_transpose(
                        rows,
                        width,
                        self.head.cols(),
                    ) =>
                {
                    // (W^T)^T uses the orientation already paid for by the first VJP.
                    return Ok(lhs.matmul_prepacked_transpose_cpu(tr)?);
                }
                _ => {}
            }
        }
        precision.multiply(lhs, &self.head)
    }

    pub(super) fn vocabulary_head_pullback_as(
        &self,
        seed: &Tensor,
        precision: CausalLmHeadPrecision,
    ) -> CausalLmResult<Tensor> {
        if seed.shape().1 != self.head.cols() {
            return Err(st_tensor::TensorError::ShapeMismatch {
                left: seed.shape(),
                right: (self.head.cols(), self.head.inner()),
            }
            .into());
        }
        if seed.layout() != st_tensor::Layout::RowMajor {
            return Err(st_tensor::TensorError::UnsupportedLayout {
                label: "vocabulary pullback requires a row-major seed",
            }
            .into());
        }
        // Reuse the old one-matrix transpose budget. The first pullback selects
        // one layout for all clones; inference alone never allocates it.
        let cache = self
            .backward_head
            .get_or_init(|| {
                let few_rows = (1..=3).contains(&seed.shape().0);
                let large_head = self.head.inner() >= 8
                    && self.head.cols() >= 8
                    && self.head.as_slice().len() >= 16 * 1024;
                if precision == CausalLmHeadPrecision::F32 && few_rows && large_head {
                    st_tensor::CpuReadoutPanels::from_packed(&self.head)
                        .map(TrainingHeadPack::Panels)
                        .map_err(|e| e.to_string())
                } else {
                    self.head
                        .transposed()
                        .map(TrainingHeadPack::Transpose)
                        .map_err(|e| e.to_string())
                }
            })
            .as_ref()
            .map_err(|e| e.clone())?;
        match cache {
            TrainingHeadPack::Transpose(weight) => precision.multiply(seed, weight),
            TrainingHeadPack::Panels(panels) => Ok(match precision {
                CausalLmHeadPrecision::F32 => panels.pullback(seed)?,
                CausalLmHeadPrecision::F64 => panels.pullback_wide(seed)?,
            }),
        }
    }
}

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

    fn matrix(rows: usize, cols: usize, salt: usize) -> Tensor {
        Tensor::from_vec(
            rows,
            cols,
            (0..rows * cols)
                .map(|i| (((i * 31 + salt) % 59) as f32 - 29.) / 17.)
                .collect(),
        )
        .unwrap()
    }

    fn head_fixture() -> NativeCausalLm {
        let mut model = super::super::tests::load();
        model.head =
            PackedB::from_tensor(&matrix(16, 1031, 7), st_tensor::Tile::col_major()).unwrap();
        model
    }

    fn same(actual: Tensor, expected: &Tensor) {
        assert_eq!(actual.shape(), expected.shape());
        assert!(actual
            .data()
            .iter()
            .zip(expected.data())
            .all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    #[test]
    fn automatic_transpose_keeps_exact_head_pairs_and_row_boundaries() {
        use CausalLmForwardStorage::{Auto, OriginalPacked, PreferCachedTranspose};
        let native = cfg!(all(target_arch = "aarch64", target_os = "macos"));
        for (width, vocabulary) in [(384usize, 32_768usize), (768, 50_257)] {
            for rows in [1, 3, 6, 7, 8, 9] {
                for w in [width - 1, width, width + 1] {
                    for v in [vocabulary - 1, vocabulary, vocabulary + 1] {
                        assert_eq!(
                            Auto.borrows_existing_transpose(rows, w, v),
                            native && rows == 7 && (w, v) == (width, vocabulary)
                        );
                        assert!(!OriginalPacked.borrows_existing_transpose(rows, w, v));
                        assert!(PreferCachedTranspose.borrows_existing_transpose(rows, w, v));
                    }
                }
            }
        }
        for (rows, width, vocab, qualified) in [
            (7, 384, 50_257, false),
            (7, 768, 32_768, false),
            (7, 512, 40_000, false),
            (6, 96, 8192, true),
            (7, 128, 16_385, true),
            (7, 129, 16_385, false),
            (6, 96, 8191, false),
            (8, 128, 16_385, false),
        ] {
            assert_eq!(
                Auto.borrows_existing_transpose(rows, width, vocab),
                native && qualified
            );
        }
    }

    fn observed(m: &NativeCausalLm, x: &Tensor) -> (Tensor, serde_json::Value) {
        let (tx, rx) = std::sync::mpsc::channel();
        let old = st_tensor::set_thread_meta_observer(Some(Arc::new(move |e| {
            if e.op_name == "matmul_prepacked" {
                tx.send(e.data.clone()).unwrap();
            }
        })));
        let result = m.project_readout(x);
        st_tensor::set_thread_meta_observer(old);
        let mut receipts = rx.try_iter();
        let receipt = receipts.next().unwrap();
        assert!(receipts.next().is_none());
        (result.unwrap(), receipt["kernel"].clone())
    }

    #[test]
    fn warm_forward_borrows_only_an_existing_compatible_transpose() {
        let mut m = super::super::tests::load();
        m.head = PackedB::from_tensor(&matrix(96, 8192, 9), st_tensor::Tile::col_major()).unwrap();
        let x = matrix(6, 96, 13);
        let (cold, kernel) = observed(&m, &x);
        assert!(kernel.is_null() && m.backward_head.get().is_none());
        use CausalLmForwardStorage::{Auto, OriginalPacked, PreferCachedTranspose};
        assert_eq!(m.forward_readout_storage(), Auto);
        let borrow = m.with_forward_readout_storage(PreferCachedTranspose);
        let packed = m.with_forward_readout_storage(OriginalPacked);
        same(observed(&borrow, &x).0, &cold);
        assert!(observed(&borrow, &x).1.is_null() && m.backward_head.get().is_none());
        let mixed = m.with_readout_accumulator(CausalLmReadoutAccumulator::F32WideBackward);
        mixed
            .vocabulary_head_pullback_as(&matrix(6, 8192, 5), CausalLmHeadPrecision::F64)
            .unwrap();
        assert!(matches!(
            m.backward_head.get(),
            Some(Ok(TrainingHeadPack::Transpose(_)))
        ));
        for model in [&m, &mixed] {
            let (warm, kernel) = observed(model, &x);
            same(warm, &cold);
            if cfg!(all(target_arch = "aarch64", target_os = "macos")) {
                assert_eq!(kernel, "cpu_readout_transpose_rows");
            } else {
                assert!(kernel.is_null());
            }
            assert!(observed(model, &matrix(3, 96, 7)).1.is_null());
            assert!(observed(model, &matrix(8, 96, 7)).1.is_null());
        }
        let wide = m.with_readout_accumulator(CausalLmReadoutAccumulator::F64);
        for rows in [1, 7, 8] {
            let x = matrix(rows, 96, 17);
            let (reference, k) = observed(&packed, &x);
            assert!(k.is_null());
            let (actual, k) = observed(&borrow, &x);
            same(actual, &reference);
            assert_eq!(k, "cpu_readout_transpose_rows");
        }
        let (output, kernel) = observed(
            &wide.with_forward_readout_storage(PreferCachedTranspose),
            &x,
        );
        same(
            output,
            &CausalLmHeadPrecision::F64.multiply(&x, &m.head).unwrap(),
        );
        assert_eq!(kernel, "packed_column_dot_wide_v1");
    }

    #[test]
    fn inference_does_not_pack_and_clones_share_the_first_small_training_pack() {
        use CausalLmHeadPrecision::{F32, F64};
        let model = head_fixture();
        let x = matrix(2, 16, 13);
        let f = model.project_readout(&x).unwrap();
        let tr = model.head.transposed().unwrap();
        assert!(model.vocabulary_head_pullback_as(&x, F32).is_err());
        assert!(model.backward_head.get().is_none());
        let seed = matrix(2, 1031, 3);
        let expected = F32.multiply(&seed, &tr).unwrap();
        let barrier = std::sync::Barrier::new(2);
        std::thread::scope(|scope| {
            let m = model.clone();
            let (b, s) = (&barrier, &seed);
            let t = scope.spawn(move || {
                b.wait();
                m.vocabulary_head_pullback_as(s, F32).unwrap()
            });
            barrier.wait();
            same(
                model.vocabulary_head_pullback_as(&seed, F32).unwrap(),
                &expected,
            );
            same(t.join().unwrap(), &expected);
        });
        assert!(matches!(
            model.backward_head.get(),
            Some(Ok(TrainingHeadPack::Panels(_)))
        ));
        for (hint, expected_kernel) in [
            (CausalLmForwardStorage::Auto, Some("cpu_readout_panel8")),
            (CausalLmForwardStorage::OriginalPacked, None),
            (
                CausalLmForwardStorage::PreferCachedTranspose,
                Some("cpu_readout_panel8"),
            ),
        ] {
            let changed = model.with_forward_readout_storage(hint);
            let (actual, kernel) = observed(&changed, &x);
            same(actual, &f);
            assert_eq!(kernel.as_str(), expected_kernel);
            assert!(observed(&changed, &matrix(4, 16, 5)).1.is_null());
            assert!(matches!(
                changed.backward_head.get(),
                Some(Ok(TrainingHeadPack::Panels(_)))
            ));
        }
        let wide = model.with_readout_accumulator(CausalLmReadoutAccumulator::F64);
        assert!(Arc::ptr_eq(&model.backward_head, &wide.backward_head));
        same(
            wide.project_readout(&x).unwrap(),
            &F64.multiply(&x, &model.head).unwrap(),
        );
        for rows in [1, 3, 9] {
            let seed = matrix(rows, 1031, 5);
            for prec in [F64, F32] {
                same(
                    wide.vocabulary_head_pullback_as(&seed, prec).unwrap(),
                    &prec.multiply(&seed, &tr).unwrap(),
                );
            }
        }
    }

    #[test]
    fn wide_or_larger_batch_first_keeps_the_original_transpose() {
        use CausalLmHeadPrecision::{F32, F64};
        for (rows, precision) in [(2, F64), (4, F32)] {
            let m = head_fixture();
            let tr = m.head.transposed().unwrap();
            for (rows, prec) in [(rows, precision), (1, F32)] {
                let seed = matrix(rows, 1031, 11);
                same(
                    m.vocabulary_head_pullback_as(&seed, prec).unwrap(),
                    &prec.multiply(&seed, &tr).unwrap(),
                );
                assert!(matches!(
                    m.backward_head.get(),
                    Some(Ok(TrainingHeadPack::Transpose(_)))
                ));
            }
        }
    }

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
