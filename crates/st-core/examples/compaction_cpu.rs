use st_core::ops::midk::{bottomk_compact_below, midk_compact_between};

fn main() {
    // Two rows of toy data.
    let rows = 2u32;
    let cols = 8u32;
    let row_stride = cols;
    let x: Vec<f32> = vec![
        0.0, 2.0, 1.0, 4.0, 3.0, 9.0, 8.0, 7.0, // row 0
        3.0, 2.0, 1.0, 0.0, -1.0, -2.0, 5.0, 6.0, // row 1
    ];

    let mid = midk_compact_between(&x, rows, cols, row_stride, 2.0, 4.0).unwrap();
    println!("midk counts={:?}", mid.counts);
    println!("midk row0 values={:?}", &mid.values[0..cols as usize]);
    println!("midk row0 indices={:?}", &mid.indices[0..cols as usize]);

    let bot = bottomk_compact_below(&x, rows, cols, row_stride, 1.0).unwrap();
    println!("bottomk counts={:?}", bot.counts);
    println!(
        "bottomk row1 values={:?}",
        &bot.values[cols as usize..2 * cols as usize]
    );
    println!(
        "bottomk row1 indices={:?}",
        &bot.indices[cols as usize..2 * cols as usize]
    );
}
