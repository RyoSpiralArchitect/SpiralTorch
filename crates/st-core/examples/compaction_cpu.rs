use st_core::ops::compaction::{compact_below, compact_between};

fn main() {
    // Two rows of toy data.
    let rows = 2u32;
    let cols = 8u32;
    let row_stride = cols;
    let x: Vec<f32> = vec![
        0.0, 2.0, 1.0, 4.0, 3.0, 9.0, 8.0, 7.0, // row 0
        3.0, 2.0, 1.0, 0.0, -1.0, -2.0, 5.0, 6.0, // row 1
    ];

    let between = compact_between(&x, rows, cols, row_stride, 2.0, 4.0).unwrap();
    println!("between counts={:?}", between.counts);
    println!("between row0 values={:?}", &between.values[0..cols as usize]);
    println!("between row0 indices={:?}", &between.indices[0..cols as usize]);

    let below = compact_below(&x, rows, cols, row_stride, 1.0).unwrap();
    println!("below counts={:?}", below.counts);
    println!(
        "below row1 values={:?}",
        &below.values[cols as usize..2 * cols as usize]
    );
    println!(
        "below row1 indices={:?}",
        &below.indices[cols as usize..2 * cols as usize]
    );
}
