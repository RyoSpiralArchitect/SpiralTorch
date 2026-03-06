use st_core::backend::cpu_exec::CpuExecutor;
use st_core::backend::device_caps::DeviceCaps;
use st_core::ops::rank_entry::{execute_rank, plan_rank, RankKind};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rows = 2u32;
    let cols = 5u32;
    let k = 2u32;
    let row_stride = cols;

    // Two rows of toy data (row-major).
    let x: Vec<f32> = vec![
        1.0, 3.0, 2.0, 3.0, -1.0, // row 0
        0.0, -2.0, 5.0, 4.0, 5.0, // row 1
    ];

    let mut out_vals = vec![0.0f32; (rows * k) as usize];
    let mut out_idx = vec![0u32; (rows * k) as usize];

    let caps = DeviceCaps::cpu();
    let plan = plan_rank(RankKind::TopK, rows, cols, k, caps);
    let mut exec = CpuExecutor::new(&x, row_stride, &mut out_vals, &mut out_idx);
    execute_rank(&mut exec, &plan)?;

    println!("TopK:");
    for r in 0..rows as usize {
        let base = r * k as usize;
        println!(
            "  row{} values={:?} idx={:?}",
            r,
            &out_vals[base..base + k as usize],
            &out_idx[base..base + k as usize]
        );
    }

    let plan = plan_rank(RankKind::BottomK, rows, cols, k, caps);
    let mut exec = CpuExecutor::new(&x, row_stride, &mut out_vals, &mut out_idx);
    execute_rank(&mut exec, &plan)?;

    println!("BottomK:");
    for r in 0..rows as usize {
        let base = r * k as usize;
        println!(
            "  row{} values={:?} idx={:?}",
            r,
            &out_vals[base..base + k as usize],
            &out_idx[base..base + k as usize]
        );
    }

    let plan = plan_rank(RankKind::MidK, rows, cols, k, caps);
    let mut exec = CpuExecutor::new(&x, row_stride, &mut out_vals, &mut out_idx);
    execute_rank(&mut exec, &plan)?;

    println!("MidK:");
    for r in 0..rows as usize {
        let base = r * k as usize;
        println!(
            "  row{} values={:?} idx={:?}",
            r,
            &out_vals[base..base + k as usize],
            &out_idx[base..base + k as usize]
        );
    }

    Ok(())
}
