//! CPU vocabulary projection with one cached orientation and a zero-copy pullback source.
use super::*;

const LANES: usize = 8;

/// The source is shared, not copied. Only the interleaved forward panels
/// allocate: exactly one f32 per weight, including a compact final panel.
#[derive(Debug)]
pub struct CpuReadoutPanels {
    source: PackedB,
    forward: AlignedVec,
}

#[derive(Clone, Copy)]
enum Product<'a> {
    Forward(&'a CpuReadoutPanels),
    TransposeRows(&'a PackedB),
    TransposeDense(&'a PackedB),
    WidePanels(&'a CpuReadoutPanels),
    WideSource(&'a PackedB),
}

fn source_stripe_shape(rows: usize, features: usize, vocab: usize) -> bool {
    (1..=3).contains(&rows)
        && (384..=1024).contains(&features)
        && features.is_multiple_of(64)
        && (8192..=50_257).contains(&vocab)
}

impl CpuReadoutPanels {
    pub fn from_packed(source: &PackedB) -> PureResult<Self> {
        if source.layout() != PackedLayout::ColMajor {
            return Err(TensorError::UnsupportedLayout {
                label: "CPU readout panels require column-major source weights",
            });
        }
        let (inner, cols) = (source.inner(), source.cols());
        let mut forward = aligned_with_capacity(source.as_slice().len());
        if inner != 0 {
            for col in (0..cols).step_by(LANES) {
                let width = (cols - col).min(LANES);
                let panel = &source.as_slice()[col * inner..(col + width) * inner];
                match width {
                    LANES => pack_eight_columns(&mut forward, panel, inner),
                    1 => forward.extend_from_slice(panel),
                    _ => {
                        for k in 0..inner {
                            for column in panel.chunks_exact(inner) {
                                forward.push(column[k]);
                            }
                        }
                    }
                }
            }
        }
        Ok(Self {
            source: source.clone(),
            forward,
        })
    }

    pub fn forward(&self, lhs: &Tensor) -> PureResult<Tensor> {
        Product::Forward(self).apply(lhs)
    }

    pub fn pullback(&self, seed: &Tensor) -> PureResult<Tensor> {
        if seed.rows <= 3 {
            Product::TransposeRows(&self.source).apply(seed)
        } else {
            Product::TransposeDense(&self.source).apply(seed)
        }
    }

    /// Preserves the four f64 partial sums of the ordinary wide readout.
    pub fn pullback_wide(&self, seed: &Tensor) -> PureResult<Tensor> {
        // Only the measured Apple Silicon shape range switches weight traversal.
        // Both views already belong to this cache and sum in the same order.
        let product = if cfg!(all(target_arch = "aarch64", target_os = "macos"))
            && source_stripe_shape(seed.rows, self.source.inner(), self.source.cols())
        {
            Product::WideSource(&self.source)
        } else {
            Product::WidePanels(self)
        };
        product.apply(seed)
    }
}

impl Tensor {
    /// CPU-only X * B^T, reading the columns of B as contiguous rows without
    /// packing again. The f32 products and additions follow sequential K order.
    pub fn matmul_prepacked_transpose_cpu(&self, rhs: &PackedB) -> PureResult<Tensor> {
        if rhs.layout() != PackedLayout::ColMajor {
            return Err(TensorError::UnsupportedLayout {
                label: "transposed CPU readout requires column-major packed weights",
            });
        }
        Product::TransposeRows(rhs).apply(self)
    }
}

impl Product<'_> {
    fn apply(self, lhs: &Tensor) -> PureResult<Tensor> {
        let source = match self {
            Self::Forward(p) | Self::WidePanels(p) => &p.source,
            Self::TransposeRows(s) | Self::TransposeDense(s) | Self::WideSource(s) => s,
        };
        let transpose = !matches!(self, Self::Forward(_));
        let (rows, inner, cols) = if transpose {
            (lhs.rows, source.cols(), source.inner())
        } else {
            (lhs.rows, source.inner(), source.cols())
        };
        if lhs.cols != inner {
            return Err(TensorError::ShapeMismatch {
                left: lhs.shape(),
                right: (inner, cols),
            });
        }
        lhs.layout.expect_row_major("CPU readout lhs")?;
        rows.checked_mul(cols)
            .and_then(AlignedVec::checked_layout)
            .ok_or(TensorError::InvalidValue {
                label: "CPU readout output exceeds addressable storage",
            })?;
        let execution = prepare_component_execution(
            TensorExecutionWorkload::PrepackedMatmul {
                rows: execution_dimension(rows)?,
                inner: execution_dimension(inner)?,
                cols: execution_dimension(cols)?,
                bias: false,
            },
            "matmul_prepacked",
            TensorExecutionBackend::CpuSimd,
        )?;
        let mut output = Tensor::zeros(rows, cols)?;
        let completion = if rows == 0 || cols == 0 {
            completed_no_op_execution(execution)?
        } else {
            if inner != 0 {
                let dst = output.data_mut();
                match self {
                    Self::Forward(p) => {
                        forward_panels(dst, lhs.data(), p.forward.as_slice(), inner, cols)
                    }
                    Self::TransposeRows(s) => {
                        pullback_rows(dst, lhs.data(), s.as_slice(), inner, cols)
                    }
                    Self::TransposeDense(s) => {
                        cpu_dense::matmul_into(dst, lhs.data(), s.as_slice(), rows, inner, cols)
                            .map_err(|message| TensorError::BackendFailure {
                                backend: "cpu_simd",
                                message,
                            })?
                    }
                    Self::WidePanels(p) => {
                        wide_panel_pullback(dst, lhs.data(), p.forward.as_slice(), inner, cols)
                    }
                    Self::WideSource(s) => {
                        wide_source_pullback(dst, lhs.data(), s.as_slice(), inner, cols)
                    }
                }
            }
            Tensor::validate_finite_tensor_util_slice("matmul_prepacked_output", output.data())?;
            completed_component_execution(execution, "cpu_simd", false)?
        };
        crate::emit_tensor_op(
            "matmul_prepacked",
            &[rows, inner, inner, cols],
            &[rows, cols],
        );
        emit_tensor_execution_receipt(completion, |data| {
            let (kernel, dtype, layout) = match self {
                Self::Forward(_) => ("cpu_readout_panel8", "f32", "output_panel8"),
                Self::TransposeRows(_) => ("cpu_readout_transpose_rows", "f32", "col_major"),
                Self::TransposeDense(_) => ("cpu_readout_transpose_dense", "f32", "col_major"),
                Self::WidePanels(_) => {
                    ("cpu_readout_panel8_transpose_wide", "f64", "output_panel8")
                }
                Self::WideSource(_) => ("cpu_readout_source_stripes_wide", "f64", "col_major"),
            };
            data.insert("rows".into(), rows.into());
            data.insert("inner".into(), inner.into());
            data.insert("cols".into(), cols.into());
            data.insert("lhs_layout".into(), "row_major".into());
            data.insert("rhs_layout".into(), "packed".into());
            data.insert("packed_layout".into(), layout.into());
            data.insert("rhs_transpose".into(), transpose.into());
            data.insert("kernel".into(), kernel.into());
            data.insert("accumulator_dtype".into(), dtype.into());
        });
        Ok(output)
    }
}

fn pack_eight_columns(dst: &mut AlignedVec, source: &[f32], inner: usize) {
    let mut columns = source.chunks_exact(inner);
    let lanes: [&[f32]; LANES] = std::array::from_fn(|_| columns.next().unwrap());
    // Copy four adjacent weights from each column; append the transposed tile
    // in one write without first zeroing a second vocabulary-sized buffer.
    const HEIGHT: usize = 4;
    let full = inner / HEIGHT * HEIGHT;
    for k in (0..full).step_by(HEIGHT) {
        let mut tile = [0.; HEIGHT * LANES];
        for (j, column) in lanes.iter().enumerate() {
            for (i, &value) in column[k..k + HEIGHT].iter().enumerate() {
                tile[i * LANES + j] = value;
            }
        }
        dst.extend_from_slice(&tile);
    }
    for k in full..inner {
        dst.extend_from_slice(&lanes.map(|column| column[k]));
    }
}

fn wide_panel_pullback(
    dst: &mut [f32],
    lhs: &[f32],
    panels: &[f32],
    vocab: usize,
    features: usize,
) {
    // A panel stores eight vocabulary terms per feature: two successive
    // contributions to each of the four fixed f64 partial sums.
    const FEATURES_PER_TILE: usize = 128;
    let packed_tail = &panels[(vocab / LANES * LANES) * features..];
    for (out, row) in dst.chunks_exact_mut(features).zip(lhs.chunks_exact(vocab)) {
        let (blocks, tail) = row.as_chunks::<LANES>();
        for (tile, cells) in out.chunks_mut(FEATURES_PER_TILE).enumerate() {
            let first = tile * FEATURES_PER_TILE;
            let mut sums = [[0.0f64; 4]; FEATURES_PER_TILE];
            for (block, g) in blocks.iter().enumerate() {
                let begin = (block * features + first) * LANES;
                let (weights, _) = panels[begin..begin + cells.len() * LANES].as_chunks::<LANES>();
                let g = g.map(f64::from);
                for (s, w) in sums.iter_mut().zip(weights) {
                    for lane in 0..4 {
                        s[lane] += g[lane] * f64::from(w[lane]);
                        s[lane] += g[4 + lane] * f64::from(w[4 + lane]);
                    }
                }
            }
            let group = tail.len() / 4 * 4;
            for (j, (cell, s)) in cells.iter_mut().zip(&mut sums).enumerate() {
                let start = (first + j) * tail.len();
                let w = &packed_tail[start..start + tail.len()];
                for lane in 0..group {
                    s[lane] += f64::from(tail[lane]) * f64::from(w[lane]);
                }
                *cell = cpu_dense::finish_wide_dot(
                    *s,
                    tail[group..]
                        .iter()
                        .copied()
                        .zip(w[group..].iter().copied()),
                );
            }
        }
    }
}

fn wide_source_pullback(
    dst: &mut [f32],
    lhs: &[f32],
    source: &[f32],
    vocab: usize,
    features: usize,
) {
    // Each original column is contiguous across features. Stream four of them
    // into four separate strips so adjacent features can accumulate in parallel.
    const TILE: usize = 1024;
    let full = vocab / 4 * 4;
    let (blocks, tail) = source.split_at(full * features);
    for (out, row) in dst.chunks_exact_mut(features).zip(lhs.chunks_exact(vocab)) {
        for (tile, cells) in out.chunks_mut(TILE).enumerate() {
            let first = tile * TILE;
            let n = cells.len();
            let mut sums = [[0.0f64; TILE]; 4];
            let [s0, s1, s2, s3] = &mut sums;
            for (g, weights) in row[..full]
                .as_chunks::<4>()
                .0
                .iter()
                .zip(blocks.chunks_exact(4 * features))
            {
                let [g0, g1, g2, g3] = g.map(f64::from);
                let (a, rest) = weights.split_at(features);
                let (b, rest) = rest.split_at(features);
                let (c, d) = rest.split_at(features);
                let [a, b, c, d] = [a, b, c, d].map(|column| &column[first..first + n]);
                for j in 0..n {
                    s0[j] += g0 * f64::from(a[j]);
                    s1[j] += g1 * f64::from(b[j]);
                    s2[j] += g2 * f64::from(c[j]);
                    s3[j] += g3 * f64::from(d[j]);
                }
            }
            for (j, cell) in cells.iter_mut().enumerate() {
                *cell = cpu_dense::finish_wide_dot(
                    [s0[j], s1[j], s2[j], s3[j]],
                    row[full..]
                        .iter()
                        .copied()
                        .zip(tail.chunks_exact(features).map(|w| w[first + j])),
                );
            }
        }
    }
}

fn forward_panels(dst: &mut [f32], lhs: &[f32], panels: &[f32], inner: usize, cols: usize) {
    let full_cols = cols / LANES * LANES;
    let (full, tail) = panels.split_at(full_cols * inner);
    for (out, row) in dst.chunks_exact_mut(cols).zip(lhs.chunks_exact(inner)) {
        let (prefix, rest) = out.split_at_mut(full_cols);
        if !prefix.is_empty() {
            for (cells, weights) in prefix
                .as_chunks_mut::<LANES>()
                .0
                .iter_mut()
                .zip(full.chunks_exact(LANES * inner))
            {
                let mut sums = [0.0; LANES];
                for (weights, &a) in weights.as_chunks::<LANES>().0.iter().zip(row) {
                    for (sum, &b) in sums.iter_mut().zip(weights) {
                        *sum += a * b;
                    }
                }
                cells.copy_from_slice(&sums);
            }
        }
        if !rest.is_empty() {
            for (weights, &a) in tail.chunks_exact(rest.len()).zip(row) {
                for (sum, &b) in rest.iter_mut().zip(weights) {
                    *sum += a * b;
                }
            }
        }
    }
}

fn pullback_rows(dst: &mut [f32], lhs: &[f32], source: &[f32], inner: usize, cols: usize) {
    // Each original packed column is a contiguous row of the transposed RHS.
    // Update independent features together; do not reassociate the K reduction.
    for (out, row) in dst.chunks_exact_mut(cols).zip(lhs.chunks_exact(inner)) {
        let full = inner / 4 * 4;
        if full != 0 {
            for (g, weights) in row[..full]
                .as_chunks::<4>()
                .0
                .iter()
                .zip(source.chunks_exact(4 * cols))
            {
                let (a, rest) = weights.split_at(cols);
                let (b, rest) = rest.split_at(cols);
                let (c, d) = rest.split_at(cols);
                for j in 0..cols {
                    out[j] = (((out[j] + g[0] * a[j]) + g[1] * b[j]) + g[2] * c[j]) + g[3] * d[j];
                }
            }
        }
        for (&g, weights) in row[full..]
            .iter()
            .zip(source[full * cols..].chunks_exact(cols))
        {
            for (sum, &weight) in out.iter_mut().zip(weights) {
                *sum += g * weight;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(rows: usize, cols: usize, salt: usize) -> Tensor {
        let scale = [1e4, 0.03125, -1., 0.125, -1e4, 0., 1.25];
        Tensor::from_vec(
            rows,
            cols,
            (0..rows * cols)
                .map(|i| ((i * 7 + salt) % 43) as f32 / 11. * scale[(i / 5 + salt) % scale.len()])
                .collect(),
        )
        .unwrap()
    }

    fn bits(t: &Tensor) -> Vec<u32> {
        t.data().iter().map(|v| v.to_bits()).collect()
    }

    #[test]
    fn transposed_pack_reads_non_square_operands_in_sequential_f32_order() {
        for width in 1..=9 {
            for vocab in [1, 5, 16, 33] {
                let w = data(width, vocab, 1);
                let tr = PackedB::from_tensor_transpose(&w, Tile::col_major()).unwrap();
                for rows in [1, 4, 7, 9] {
                    let x = data(rows, width, 2);
                    let y = x.matmul_prepacked_transpose_cpu(&tr).unwrap();
                    assert_eq!(y.shape(), (rows, vocab));
                    for (r, row) in x.data().chunks_exact(width).enumerate() {
                        for c in 0..vocab {
                            let gold = row
                                .iter()
                                .enumerate()
                                .fold(0.0f32, |s, (k, a)| s + a * w.data()[k * vocab + c]);
                            assert_eq!(gold.to_bits(), y.data()[r * vocab + c].to_bits());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn source_stripes_have_bounded_shape_and_alignment() {
        for s in [(1, 384, 8192), (3, 1024, 50_257), (2, 704, 29_225)] {
            assert!(source_stripe_shape(s.0, s.1, s.2));
        }
        for s in [
            (0, 384, 8192),
            (4, 384, 8192),
            (2, 320, 8192),
            (2, 385, 8192),
            (1, 1088, 8192),
            (1, 384, 8191),
            (1, 384, 50_258),
        ] {
            assert!(!source_stripe_shape(s.0, s.1, s.2));
        }
    }

    #[test]
    fn wide_source_preserves_the_packed_dot_across_both_strip_tails() {
        for w in [1, 63, 384, 1023, 1024, 1025, 2051] {
            for v in [1, 2, 3, 4, 5, 6, 7, 12, 31] {
                let source =
                    PackedB::from_tensor_transpose(&data(v, w, 5), Tile::col_major()).unwrap();
                let g = data(2, v, 3);
                let reference = g
                    .matmul_prepacked_wide(&source.transposed().unwrap())
                    .unwrap();
                let mut result = vec![f32::NAN; 2 * w];
                wide_source_pullback(&mut result, g.data(), source.as_slice(), v, w);
                assert!(result
                    .iter()
                    .zip(reference.data())
                    .all(|(x, y)| x.to_bits() == y.to_bits()));
            }
        }
    }

    #[test]
    fn wide_receipt_reports_which_weight_storage_the_host_actually_read() {
        let (w, v) = (384, 8192);
        let source =
            PackedB::from_tensor_transpose(&Tensor::zeros(v, w).unwrap(), Tile::col_major())
                .unwrap();
        let panels = CpuReadoutPanels::from_packed(&source).unwrap();
        let seed = data(1, v, 3);
        let (tx, rx) = std::sync::mpsc::channel();
        let last = crate::set_thread_meta_observer(Some(Arc::new(move |event| {
            if event.op_name == "matmul_prepacked" {
                tx.send(event.data.clone()).unwrap();
            }
        })));
        let out = panels.pullback_wide(&seed);
        crate::set_thread_meta_observer(last);
        assert!(out.unwrap().data().iter().all(|&x| x == 0.0));
        let receipt = rx
            .try_iter()
            .find(|d| d["accumulator_dtype"] == "f64")
            .unwrap();
        let (kernel, layout) = if cfg!(all(target_arch = "aarch64", target_os = "macos")) {
            ("cpu_readout_source_stripes_wide", "col_major")
        } else {
            ("cpu_readout_panel8_transpose_wide", "output_panel8")
        };
        assert_eq!(receipt["kernel"], kernel);
        assert_eq!(receipt["packed_layout"], layout);
        assert_eq!(receipt["rhs_transpose"], true);
        assert_eq!(receipt["executed_backend"], "cpu_simd");
    }

    #[test]
    fn wide_pullback_matches_four_stripes_for_every_panel_tail_and_multiple_feature_tiles() {
        for features in [1, 7, 64, 127, 128, 129, 257, 769] {
            for vocab in (1..=16).chain([33, 131]) {
                let source =
                    PackedB::from_tensor_transpose(&data(vocab, features, 1), Tile::col_major())
                        .unwrap();
                let g = data(2, vocab, 3);
                let got = CpuReadoutPanels::from_packed(&source)
                    .unwrap()
                    .pullback_wide(&g)
                    .unwrap();
                for (row, output) in g
                    .data()
                    .chunks_exact(vocab)
                    .zip(got.data().chunks_exact(features))
                {
                    for (j, &actual) in output.iter().enumerate() {
                        let term = |k: usize| {
                            f64::from(row[k]) * f64::from(source.as_slice()[k * features + j])
                        };
                        let full = vocab / 4 * 4;
                        let mut lanes = [0.; 4];
                        for k in 0..full {
                            lanes[k % 4] += term(k);
                        }
                        let mut expected = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
                        for k in full..vocab {
                            expected += term(k);
                        }
                        assert_eq!(
                            actual.to_bits(),
                            (expected as f32).to_bits(),
                            "{features}x{vocab} feature {j}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn wide_combines_four_sums_before_tail_and_reports_the_panel_that_was_read() {
        let big = (1u64 << 60) as f32;
        let w = [big, 1., -big, 1., 0., 0., 0., 0., 1.];
        assert_eq!(w.iter().fold(0f64, |sum, &x| sum + f64::from(x)), 2.);
        let source = PackedB::from_tensor(
            &Tensor::from_vec(1, 9, w.to_vec()).unwrap(),
            Tile::col_major(),
        )
        .unwrap();
        let panels = CpuReadoutPanels::from_packed(&source).unwrap();
        let seed = Tensor::from_vec(1, 9, vec![1.; 9]).unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        let previous = crate::set_thread_meta_observer(Some(Arc::new(move |event| {
            if event.op_name == "matmul_prepacked" {
                tx.send(event.data.clone()).unwrap();
            }
        })));
        let out = panels.pullback_wide(&seed);
        crate::set_thread_meta_observer(previous);
        assert_eq!(out.unwrap().data(), &[1.]);
        let receipt = rx
            .try_iter()
            .find(|d| d["accumulator_dtype"] == "f64")
            .unwrap();
        assert_eq!(receipt["rhs_transpose"], true);
        assert_eq!(receipt["packed_layout"], "output_panel8");
        assert_eq!(receipt["kernel"], "cpu_readout_panel8_transpose_wide");
        assert_eq!(receipt["execution_receipt"]["executed_backend"], "cpu_simd");
    }

    #[test]
    fn panel_pack_preserves_every_weight_bit_without_padding_either_tail() {
        for inner in [0, 1, 2, 3, 4, 5, 7, 8, 9, 63, 64, 65, 767, 768, 769] {
            for cols in 0..=17 {
                let words: Vec<_> = (0..inner * cols)
                    .map(|i| match i % 9 {
                        0 => 0x8000_0000,
                        1 => 0x7f80_0000,
                        2 => 0x7fc0_0000 | (i as u32 + 1),
                        3 => 0xff80_0000 | (i as u32 + 1),
                        4 => i as u32 + 1,
                        _ => 0x3f00_0000 + i as u32,
                    })
                    .collect();
                let raw = Tensor::from_vec(
                    cols,
                    inner,
                    words.iter().copied().map(f32::from_bits).collect(),
                )
                .unwrap();
                let source = PackedB::from_tensor_transpose(&raw, Tile::col_major()).unwrap();
                let packed = CpuReadoutPanels::from_packed(&source).unwrap();
                assert_eq!(packed.forward.len(), inner * cols);
                for col in 0..cols {
                    let start = col / LANES * LANES;
                    let span = (cols - start).min(LANES);
                    for k in 0..inner {
                        assert_eq!(
                            packed.forward[start * inner + k * span + col - start].to_bits(),
                            words[col * inner + k]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn both_orientations_match_packed_order_including_tails() {
        for (width, vocab) in [(1, 1), (3, 7), (8, 8), (9, 15), (17, 16), (32, 65)] {
            let source = PackedB::from_tensor(&data(width, vocab, 5), Tile::col_major()).unwrap();
            let tr = source.transposed().unwrap();
            let p = CpuReadoutPanels::from_packed(&source).unwrap();
            assert!(Arc::ptr_eq(&p.source.buf, &source.buf));
            assert_eq!(p.forward.len(), source.as_slice().len());
            for rows in [1, 2, 3, 5, 9] {
                let x = data(rows, width, 2);
                let g = data(rows, vocab, 3);
                let mul = |lhs: &Tensor, rhs| {
                    lhs.matmul_prepacked_with_backend(rhs, MatmulBackend::CpuSimd)
                        .unwrap()
                };
                assert_eq!(bits(&p.forward(&x).unwrap()), bits(&mul(&x, &source)));
                assert_eq!(bits(&p.pullback(&g).unwrap()), bits(&mul(&g, &tr)));
                assert_eq!(
                    bits(&p.pullback_wide(&g).unwrap()),
                    bits(&g.matmul_prepacked_wide(&tr).unwrap())
                );
            }
        }
    }

    #[test]
    fn cancellation_keeps_wide_and_default_policies_distinct() {
        let w = Tensor::from_vec(1, 4, vec![1e8, 1., -1e8, 1.]).unwrap();
        let p =
            CpuReadoutPanels::from_packed(&PackedB::from_tensor(&w, Tile::col_major()).unwrap())
                .unwrap();
        let g = Tensor::from_vec(1, 4, vec![1.; 4]).unwrap();
        assert_eq!(p.pullback(&g).unwrap().data(), &[1.]);
        assert_eq!(p.pullback_wide(&g).unwrap().data(), &[2.]);
    }

    #[test]
    fn contracts_cover_zero_extents_bad_layout_shape_and_overflow() {
        for (rows, width, vocab) in [(0, 7, 8), (3, 0, 8), (3, 7, 0)] {
            let source = PackedB::from_tensor(&data(width, vocab, 0), Tile::col_major()).unwrap();
            let p = CpuReadoutPanels::from_packed(&source).unwrap();
            let g = data(rows, vocab, 1);
            for (out, cols) in [
                (p.forward(&data(rows, width, 1)), vocab),
                (p.pullback(&g), width),
                (g.matmul_prepacked_transpose_cpu(&source), width),
                (p.pullback_wide(&g), width),
            ] {
                let out = out.unwrap();
                assert_eq!(out.shape(), (rows, cols));
                assert!(out.data().iter().all(|&x| x == 0.));
            }
        }
        let mut w = PackedB::from_tensor(
            &Tensor::from_vec(1, 1, vec![f32::MAX]).unwrap(),
            Tile::col_major(),
        )
        .unwrap();
        let p = CpuReadoutPanels::from_packed(&w).unwrap();
        assert!(p.forward(&data(1, 2, 2)).is_err());
        assert!(p.pullback(&data(1, 2, 2)).is_err());
        assert!(data(1, 2, 2).matmul_prepacked_transpose_cpu(&w).is_err());
        let mut g = Tensor::from_vec(1, 1, vec![2.]).unwrap();
        assert!(g.matmul_prepacked_transpose_cpu(&w).is_err());
        assert!(p.forward(&g).is_err() && p.pullback(&g).is_err() && p.pullback_wide(&g).is_err());
        g.layout = Layout::ColMajor;
        assert!(matches!(
            g.matmul_prepacked_transpose_cpu(&w),
            Err(TensorError::UnsupportedLayout { .. })
        ));
        assert!(matches!(
            p.forward(&g),
            Err(TensorError::UnsupportedLayout { .. })
        ));
        w.layout = PackedLayout::Tiled {
            tm: 1,
            tn: 1,
            tk: 1,
        };
        assert!(CpuReadoutPanels::from_packed(&w).is_err());
        assert!(data(1, 1, 1).matmul_prepacked_transpose_cpu(&w).is_err());
        let empty_tr = PackedB::from_tensor(&data(2, 0, 0), Tile::col_major()).unwrap();
        assert!(Tensor::zeros(usize::MAX, 0)
            .unwrap()
            .matmul_prepacked_transpose_cpu(&empty_tr)
            .is_err());
        let empty = PackedB::from_tensor(&data(0, 2, 0), Tile::col_major()).unwrap();
        assert!(CpuReadoutPanels::from_packed(&empty)
            .unwrap()
            .forward(&Tensor::zeros(usize::MAX, 0).unwrap())
            .is_err());
    }
}
