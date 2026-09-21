//! Shared CPU layout conversion; only copies bits, never performs arithmetic.

pub(crate) fn transpose_into(src: &[f32], dst: &mut [f32], rows: usize, cols: usize) {
    let len = rows.checked_mul(cols).expect("validated tensor dimensions");
    assert_eq!(src.len(), len);
    assert_eq!(dst.len(), len);
    if len == 0 {
        return;
    }
    if rows == 1 || cols == 1 {
        dst.copy_from_slice(src);
        return;
    }
    // Keep both sides of a transpose tile hot instead of striding the full output.
    const TILE: usize = 32;
    for row_start in (0..rows).step_by(TILE) {
        let row_end = row_start.saturating_add(TILE).min(rows);
        for col_start in (0..cols).step_by(TILE) {
            let col_end = col_start.saturating_add(TILE).min(cols);
            for row in row_start..row_end {
                let source = &src[row * cols + col_start..row * cols + col_end];
                for (offset, &value) in source.iter().enumerate() {
                    dst[(col_start + offset) * rows + row] = value;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rectangular_tiles_preserve_all_float_bits_and_guards() {
        let bits = [
            0,
            0x8000_0000,
            0x7fc0_0042,
            0x7f80_0000,
            0xff80_0000,
            1,
            0x3f80_0000,
        ];
        for rows in [0, 1, 2, 7, 31, 32, 33, 65, 137] {
            for cols in [0, 1, 3, 16, 31, 32, 33, 97] {
                let source: Vec<_> = (0..rows * cols)
                    .map(|i| f32::from_bits(bits[i % bits.len()]))
                    .collect();
                let mut destination = vec![f32::from_bits(0x7fc0_0100); source.len() + 2];
                transpose_into(&source, &mut destination[1..=source.len()], rows, cols);
                assert_eq!(destination[0].to_bits(), 0x7fc0_0100);
                assert_eq!(destination[source.len() + 1].to_bits(), 0x7fc0_0100);
                for row in 0..rows {
                    for col in 0..cols {
                        assert_eq!(
                            source[row * cols + col].to_bits(),
                            destination[1 + col * rows + row].to_bits()
                        );
                    }
                }
            }
        }
        transpose_into(&[], &mut [], usize::MAX, 0);
        transpose_into(&[], &mut [], 0, usize::MAX);
    }
}
