// crates/st-core/src/ops/ablog.rs  (v1.8.7)
use once_cell::sync::OnceCell;
use crate::heur::self_rewrite::{SR, ABKey};

static SR_G: OnceCell<std::sync::Mutex<SR>> = OnceCell::new();

fn sr() -> &'static std::sync::Mutex<SR> {
    SR_G.get_or_init(|| std::sync::Mutex::new(SR::from_env()))
}

pub fn ab_log_topk(cols:u32, k:u32, variant:&'static str, win:bool){
    let mut sr = sr().lock().unwrap();
    sr.log(&ABKey{ kind:"topk", rows:1, cols, k, variant }, win);
}
pub fn ab_log_midk(cols:u32, k:u32, variant:&'static str, win:bool){
    let mut sr = sr().lock().unwrap();
    sr.log(&ABKey{ kind:"midk", rows:1, cols, k, variant }, win);
}
pub fn ab_log_bottomk(cols:u32, k:u32, variant:&'static str, win:bool){
    let mut sr = sr().lock().unwrap();
    sr.log(&ABKey{ kind:"bottomk", rows:1, cols, k, variant }, win);
}
