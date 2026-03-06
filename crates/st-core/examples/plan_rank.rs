use st_core::backend::device_caps::DeviceCaps;
use st_core::ops::rank_entry::{plan_rank, RankKind};

fn main() {
    let caps = DeviceCaps::mps(32, 256, Some(32 * 1024));
    let plan = plan_rank(RankKind::TopK, 128, 8192, 64, caps);

    println!(
        "kind={:?} rows={} cols={} k={}",
        plan.kind, plan.rows, plan.cols, plan.k
    );
    println!("{}", plan.choice.to_unison_script(plan.kind));
    println!("fft_hint: {}", plan.fft_spiralk_hint().replace('\n', "; "));
}
